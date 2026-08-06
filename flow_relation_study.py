"""외국인 수급과 지수 수익의 관계가 시간에 따라 변했는지 검정한다.

배경 가설: 최근 코스피에서 외국인이 순매수하면 지수가 오르는 경향이 강해졌다.
예전에는 외국인이 팔아도 개인이 물량을 받아 올렸는데 이제 그 여력이 줄었다는 것이다.

이 스크립트가 지키는 규칙:

1. 동시성과 예측력을 반드시 나눠 본다. 외국인이 산 날 지수가 올랐다는 것은 대부분
   동어반복이다. 어드바이스에 쓰려면 다음 거래일 수익을 맞혀야 한다.
2. 수익은 두 가지로 잰다. 종가-종가는 실제로 잡을 수 없는 수익이고, 신호를 확인한
   다음 거래일 시가에 사서 종가에 파는 것이 실행 가능한 형태다.
3. 일별 수급은 자기상관이 강하므로 신뢰구간은 원형 블록 부트스트랩으로 만든다.
4. 여러 투자 주체를 동시에 시험하므로 FDR 보정 q값을 함께 낸다.
5. 위약 신호로 같은 검정을 돌려 귀무분포가 어디쯤인지 확인한다.

주의: 연도별 분할 결과를 먼저 눈으로 본 뒤에 이 검정을 짰다. 따라서 아래 수치는
확증이 아니라 사후 검정이다. 진짜 확증은 앞으로 쌓일 자료로 해야 한다.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from foreign_flow_validation import benjamini_hochberg
from kis import PanelStore, get_dataset

# 수급 규모는 시장이 커지면 같이 커진다. 절대 금액이 아니라 최근 변동성 대비 크기로 본다.
SCALE_WINDOW = 60
ROLLING_WINDOW = 250
BLOCK_SIZE = 5
BOOTSTRAP_DRAWS = 2000
SEED = 20260806

PARTIES = {
    "foreign_net_value": "외국인",
    "individual_net_value": "개인",
    "institution_net_value": "기관계",
    "pension_net_value": "연기금",
    "private_fund_net_value": "사모펀드",
    "invtrust_net_value": "투신",
    "etc_corp_net_value": "기타법인",
}


def build_frame(store: PanelStore, index_code: str, start: str | None) -> pd.DataFrame:
    frame = store.read(
        get_dataset("market_investor_flow_daily"), tickers=[index_code], start=start
    )
    if frame.empty:
        raise SystemExit(
            f"{index_code} 시장 수급이 없습니다. collect_kis_flow.py로 먼저 수집하세요."
        )
    frame = frame.sort_values("date").reset_index(drop=True)

    # 당일 수익. 동시성 확인용.
    frame["ret"] = frame["index_change_pct"]
    # 다음 거래일 종가-종가. 실제로는 잡을 수 없는 수익이다.
    frame["ret_next_cc"] = frame["ret"].shift(-1)
    # 다음 거래일 시가 매수, 종가 매도. 신호를 확인한 뒤 실행 가능한 형태다.
    intraday = (frame["index_close"] / frame["index_open"] - 1) * 100
    frame["ret_next_oc"] = intraday.shift(-1)

    for column in PARTIES:
        if column not in frame.columns:
            continue
        scale = frame[column].rolling(SCALE_WINDOW).std()
        frame[f"z_{column}"] = frame[column] / scale.replace(0, np.nan)
    return frame


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    """단순 회귀 기울기. 수급 1표준편차당 다음날 수익률(%p)."""
    if len(x) < 3:
        return math.nan
    variance = x.var()
    return float(np.cov(x, y, bias=True)[0, 1] / variance) if variance else math.nan


def block_bootstrap(
    x: pd.Series,
    y: pd.Series,
    statistic,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = SEED,
) -> tuple[float, float, float, float]:
    """원형 블록 부트스트랩으로 통계량의 95% 구간과 양측 p값을 만든다."""
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < BLOCK_SIZE * 4:
        return math.nan, math.nan, math.nan, math.nan
    xs = pair.iloc[:, 0].to_numpy(dtype=float)
    ys = pair.iloc[:, 1].to_numpy(dtype=float)
    observed = statistic(xs, ys)

    rng = np.random.default_rng(seed)
    blocks_needed = math.ceil(len(xs) / BLOCK_SIZE)
    offsets = np.arange(BLOCK_SIZE)
    values = []
    for _ in range(draws):
        starts = rng.integers(0, len(xs), size=blocks_needed)
        idx = ((starts[:, None] + offsets) % len(xs)).ravel()[: len(xs)]
        value = statistic(xs[idx], ys[idx])
        if not math.isnan(value):
            values.append(value)
    if not values:
        return observed, math.nan, math.nan, math.nan
    distribution = np.asarray(values)
    low, high = np.quantile(distribution, [0.025, 0.975])
    p_value = min(
        1.0,
        2
        * min(
            (np.count_nonzero(distribution <= 0) + 1) / (len(distribution) + 1),
            (np.count_nonzero(distribution >= 0) + 1) / (len(distribution) + 1),
        ),
    )
    return float(observed), float(low), float(high), float(p_value)


def interaction_slope(pairs: np.ndarray, y: np.ndarray) -> float:
    """ret = a + b*fx + c*(fx*t) 에서 c. 예측력이 시간에 따라 변했는지 본다."""
    design = np.column_stack([np.ones(len(y)), pairs[:, 0], pairs[:, 0] * pairs[:, 1]])
    try:
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return math.nan
    return float(coefficients[2])


def study_interaction(frame: pd.DataFrame, target: str) -> dict:
    """사전 지정한 1차 검정. 외국인 수급의 예측력이 시간에 따라 커졌는가."""
    data = frame[["z_foreign_net_value", target]].dropna()
    if data.empty:
        return {}
    tau = np.linspace(0.0, 1.0, len(data))
    pairs = pd.Series(list(zip(data["z_foreign_net_value"], tau)), index=data.index)
    stacked = np.column_stack([data["z_foreign_net_value"].to_numpy(), tau])

    def statistic(x_index: np.ndarray, y: np.ndarray) -> float:
        return interaction_slope(x_index, y)

    # 블록 부트스트랩이 쌍을 통째로 다시 뽑도록 인덱스를 함께 넘긴다.
    holder = pd.DataFrame({"i": np.arange(len(data)), "y": data[target].to_numpy()})

    def wrapped(idx: np.ndarray, y: np.ndarray) -> float:
        return interaction_slope(stacked[idx.astype(int)], y)

    observed, low, high, p_value = block_bootstrap(holder["i"], holder["y"], wrapped)
    del pairs
    return {
        "지표": "시간 상호작용 계수",
        "수익 정의": target,
        "관측": observed,
        "하한": low,
        "상한": high,
        "p": p_value,
        "표본": len(data),
    }


def study_parties(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """주체별 다음날 예측 기울기. 여러 가설을 동시에 보므로 FDR 보정을 붙인다."""
    rows = []
    for column, label in PARTIES.items():
        key = f"z_{column}"
        if key not in frame.columns:
            continue
        observed, low, high, p_value = block_bootstrap(frame[key], frame[target], _slope)
        rows.append(
            {
                "주체": label,
                "기울기(%p/1σ)": observed,
                "하한": low,
                "상한": high,
                "p": p_value,
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["q(FDR)"] = benjamini_hochberg(result["p"])
    return result


def study_rolling(frame: pd.DataFrame) -> pd.DataFrame:
    """250일 롤링으로 동시성과 예측력을 따로 그린다."""
    z = frame["z_foreign_net_value"]
    result = pd.DataFrame({"date": frame["date"]})
    result["동시_상관"] = z.rolling(ROLLING_WINDOW).corr(frame["ret"])
    result["익일_상관_종가"] = z.rolling(ROLLING_WINDOW).corr(frame["ret_next_cc"])
    result["익일_상관_시가진입"] = z.rolling(ROLLING_WINDOW).corr(frame["ret_next_oc"])
    return result.dropna(subset=["동시_상관"])


def study_robustness(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """극단값에 기대고 있는 결과인지 확인한다."""
    z = frame["z_foreign_net_value"]
    rows = []

    def add(label: str, mask: pd.Series | None, winsor: float | None = None) -> None:
        x, y = z.copy(), frame[target].copy()
        if mask is not None:
            x, y = x[mask], y[mask]
        if winsor:
            lo, hi = y.quantile([winsor, 1 - winsor])
            y = y.clip(lo, hi)
        observed, low, high, p_value = block_bootstrap(x, y, _slope)
        rows.append(
            {"조건": label, "기울기": observed, "하한": low, "상한": high, "p": p_value,
             "표본": int(pd.concat([x, y], axis=1).dropna().shape[0])}
        )

    add("전체", None)
    add("수익 1% 윈저화", None, winsor=0.01)
    add("일간 등락 ±5% 이내", frame[target].abs() <= 5)
    add("전반기", frame.index < len(frame) // 2)
    add("후반기", frame.index >= len(frame) // 2)
    return pd.DataFrame(rows)


def study_placebo(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """수급을 블록 단위로 섞은 위약 신호. 같은 검정이 얼마나 쉽게 유의해지는지 본다."""
    rng = np.random.default_rng(SEED)
    z = frame["z_foreign_net_value"].to_numpy(dtype=float)
    y = frame[target]
    rows = []
    for trial in range(20):
        blocks = [z[i : i + BLOCK_SIZE] for i in range(0, len(z), BLOCK_SIZE)]
        order = np.arange(len(blocks))
        rng.shuffle(order)
        shuffled = np.concatenate([blocks[i] for i in order])[: len(z)]
        fake = pd.Series(shuffled, index=frame.index)
        pair = pd.concat([fake, y], axis=1).dropna()
        if len(pair) < 10:
            continue
        rows.append(
            {
                "시행": trial + 1,
                "기울기": _slope(pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def study_absorption(frame: pd.DataFrame) -> pd.DataFrame:
    """개인이 외국인 물량을 받아내는 정도가 변했는지 본다.

    개인-외인 상관이 -1에 가까워지는 것만으로는 개인 자금 고갈의 증거가 되지 못한다.
    5주체 순매수 합이 0이라는 항등식 때문에, 기관이 완충을 줄이기만 해도 개인은
    자동으로 외국인의 거울상이 된다. 그래서 기관 몫도 같이 본다.
    """
    rows = []
    for year, part in frame.assign(year=frame["date"].str[:4]).groupby("year"):
        part = part.dropna(subset=["foreign_net_value"])
        if len(part) < 60:
            continue
        foreign = part["foreign_net_value"]
        absorbed = -foreign
        individual = part["individual_net_value"]
        institution = part["institution_net_value"]
        # 외국인이 판 물량 중 개인이 받은 비중. 매도일만 본다.
        selling = foreign < 0
        share_individual = (
            individual[selling].sum() / absorbed[selling].sum() if selling.any() else math.nan
        )
        share_institution = (
            institution[selling].sum() / absorbed[selling].sum() if selling.any() else math.nan
        )
        rows.append(
            {
                "연도": year,
                "일수": len(part),
                "외인 순매도일": int(selling.sum()),
                "개인 흡수 비중": share_individual,
                "기관 흡수 비중": share_institution,
                "개인 순매수 절대규모 중앙값": individual.abs().median(),
                # 명목 금액은 지수가 오르면 같이 커진다. 지수로 나눠 규모를 공정하게 본다.
                "지수 중앙값": part["index_close"].median(),
                "개인 규모/지수": individual.abs().median() / part["index_close"].median(),
                "개인-외인 상관": individual.corr(foreign),
                "기관-외인 상관": institution.corr(foreign),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="외국인 수급과 지수 수익의 관계 검정")
    parser.add_argument("--index", default="0001", help="지수 코드. 기본 0001(코스피)")
    parser.add_argument("--start", default=None, help="시작일 YYYY-MM-DD")
    parser.add_argument("--db", type=Path, default=Path("data/kis_panel.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = PanelStore(args.db)
    frame = build_frame(store, args.index, args.start)
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"지수 {args.index}  {frame['date'].min()} ~ {frame['date'].max()}  {len(frame)}일")
    print("수급은 60일 표준편차로 규격화했고, 신뢰구간은 블록 부트스트랩 2000회입니다.\n")

    print("=" * 78)
    print("1. 동시성 vs 예측력  (외국인 수급 1σ당 지수 수익 %p)")
    print("=" * 78)
    same_day = block_bootstrap(frame["z_foreign_net_value"], frame["ret"], _slope)
    rows = [{"수익 정의": "당일 (동시성)", "기울기": same_day[0], "하한": same_day[1],
             "상한": same_day[2], "p": same_day[3]}]
    for target, label in [("ret_next_cc", "다음날 종가-종가"),
                          ("ret_next_oc", "다음날 시가매수-종가매도 (실행 가능)")]:
        observed, low, high, p_value = block_bootstrap(
            frame["z_foreign_net_value"], frame[target], _slope
        )
        rows.append({"수익 정의": label, "기울기": observed, "하한": low,
                     "상한": high, "p": p_value})
    headline = pd.DataFrame(rows).round(4)
    print(headline.to_string(index=False))

    print("\n" + "=" * 78)
    print("2. 1차 검정: 예측력이 시간에 따라 커졌는가 (시간 상호작용 계수)")
    print("=" * 78)
    interactions = pd.DataFrame(
        [study_interaction(frame, target) for target in ["ret_next_cc", "ret_next_oc"]]
    )
    print(interactions.round(4).to_string(index=False))
    print("  계수가 0보다 크고 구간이 0을 넘지 않으면 관계가 강해졌다는 뜻입니다.")

    print("\n" + "=" * 78)
    print("3. 주체별 다음날 예측력 (시가매수-종가매도 기준, FDR 보정)")
    print("=" * 78)
    parties = study_parties(frame, "ret_next_oc")
    print(parties.round(4).to_string(index=False))

    print("\n" + "=" * 78)
    print("4. 강건성 점검 (시가매수-종가매도 기준)")
    print("=" * 78)
    robustness = study_robustness(frame, "ret_next_oc")
    print(robustness.round(4).to_string(index=False))

    print("\n" + "=" * 78)
    print("5. 위약 검정 (수급을 블록 단위로 섞음)")
    print("=" * 78)
    placebo = study_placebo(frame, "ret_next_oc")
    real = block_bootstrap(frame["z_foreign_net_value"], frame["ret_next_oc"], _slope)[0]
    print(f"  위약 기울기 범위 {placebo['기울기'].min():.4f} ~ {placebo['기울기'].max():.4f}  "
          f"절대값 중앙 {placebo['기울기'].abs().median():.4f}")
    print(f"  실제 기울기 {real:.4f}")

    print("\n" + "=" * 78)
    print("6. 개인 흡수 여력 (외국인이 순매도한 날, 그 물량을 누가 받았나)")
    print("=" * 78)
    absorption = study_absorption(frame)
    print(absorption.round(3).to_string(index=False))

    print("\n" + "=" * 78)
    print("7. 250일 롤링 관계")
    print("=" * 78)
    rolling = study_rolling(frame)
    sample = rolling.iloc[:: max(1, len(rolling) // 8)]
    print(sample.round(3).to_string(index=False))

    for name, table in [
        ("flow_relation_headline", headline),
        ("flow_relation_interaction", interactions),
        ("flow_relation_parties", parties),
        ("flow_relation_robustness", robustness),
        ("flow_relation_absorption", absorption),
        ("flow_relation_rolling", rolling),
    ]:
        path = args.output / f"{name}.csv"
        table.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n표는 {args.output.resolve()}에 저장했습니다.")
    print("연도별 결과를 먼저 본 뒤 설계한 검정이므로 확증이 아니라 사후 검정입니다.")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
