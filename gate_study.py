"""돈치안 매수 신호에 위험 게이트를 덧대면 손실이 줄어드는지 검정한다.

앞선 `flow_relation_study.py`에서 외국인 순매수의 진입 예측력은 통계적으로 약하다는
결론이 났다. 이 스크립트는 다른 질문을 한다. 진입을 맞히는 것이 아니라, 이미 나온
매수 신호 중 나쁜 것을 걸러낼 수 있는가.

평가 기준은 수익이 아니다. 목적이 손해 최소화이므로 아래를 본다.

- CVaR5: 하위 5% 날의 평균 수익. 최악이 얼마나 나쁜가
- 손실일 비율
- MDD와 최장 손실 구간
- 수익은 얼마나 깎이는가 (게이트의 비용)

방법상 지키는 것:

1. 게이트는 사전에 4개만 정했다. 여러 개를 만들어 좋은 것을 고르면 안 된다.
2. 네 개를 동시에 시험하므로 FDR 보정 q값을 낸다.
3. 자기상관 때문에 신뢰구간은 원형 블록 부트스트랩으로 만든다.
4. 판단 시점 이후 정보를 쓰지 않는다. 게이트는 전일 종가까지의 자료로만 만들고,
   공개가 늦는 대차잔고는 하루 더 미룬다.
5. 매매는 신호 다음 거래일 시가에 실행한다고 본다. 수익도 시가-시가로 잰다.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from foreign_flow_validation import benjamini_hochberg
from kis import PanelStore, get_dataset

DC_WINDOW = 20
VOL_WINDOW = 20
FLOW_WINDOW = 5
SBL_WINDOW = 20
RANK_WINDOW = 250
RANK_CUT = 0.80
BLOCK_SIZE = 5
BOOTSTRAP_DRAWS = 2000
SEED = 20260806
ROUND_TRIP_COST_BPS = 25.0


def load_index(store: PanelStore, index_code: str, start: str | None) -> pd.DataFrame:
    flow = store.read(
        get_dataset("market_investor_flow_daily"), tickers=[index_code], start=start
    )
    if flow.empty:
        raise SystemExit(f"{index_code} 시장 수급이 없습니다. 먼저 수집하세요.")
    flow = flow.sort_values("date").reset_index(drop=True)

    loan = store.read(get_dataset("market_loan_trans_daily"), tickers=[index_code], start=start)
    if not loan.empty:
        loan = loan[["date", "sbl_balance_qty"]].sort_values("date")
        flow = flow.merge(loan, on="date", how="left")
    else:
        flow["sbl_balance_qty"] = np.nan
    return flow


def donchian_position(
    frame: pd.DataFrame, high_column: str = "index_high", low_column: str = "index_low"
) -> pd.Series:
    """donchian_kospi_daily.py와 같은 규칙. 어제까지의 채널로 오늘을 판단한다.

    지수와 ETF 어느 쪽 가격으로도 신호를 낼 수 있게 열 이름을 받는다.
    """
    high, low = frame[high_column], frame[low_column]
    channel_high = high.rolling(DC_WINDOW).max().shift(1)
    channel_low = low.rolling(DC_WINDOW).min().shift(1)
    entry = high > channel_high
    exit_ = low < channel_low

    position = 0
    values = []
    for i in range(len(frame)):
        if pd.isna(channel_high.iloc[i]):
            values.append(0)
            continue
        if position == 0 and entry.iloc[i]:
            position = 1
        elif position == 1 and exit_.iloc[i]:
            position = 0
        values.append(position)
    return pd.Series(values, index=frame.index, dtype=int)


def _trailing_rank(series: pd.Series, window: int = RANK_WINDOW) -> pd.Series:
    """과거 window일 안에서 오늘 값이 어느 분위인지. 미래를 보지 않는다."""
    return series.rolling(window).rank(pct=True)


def build_gates(frame: pd.DataFrame) -> pd.DataFrame:
    """사전에 정한 위험 게이트 4개. True면 위험 신호가 켜진 것이다."""
    ret = frame["index_change_pct"]
    gates = pd.DataFrame(index=frame.index)

    # 1. 변동성 과열. 극단일에 효과가 몰려 있었다는 앞선 결과에서 나온 후보다.
    realized = ret.rolling(VOL_WINDOW).std()
    gates["변동성 과열"] = _trailing_rank(realized) >= RANK_CUT

    # 2. 외국인 이탈. 5일 누적 순매수가 마이너스인 상태.
    foreign = frame["foreign_net_value"].rolling(FLOW_WINDOW).sum()
    gates["외국인 이탈"] = foreign < 0

    # 3. 개인 과열. 개인이 대규모로 받아내는 국면은 고점 신호일 수 있다는 반대 가설.
    individual = frame["individual_net_value"].rolling(FLOW_WINDOW).sum()
    gates["개인 과열"] = _trailing_rank(individual) >= RANK_CUT

    # 4. 대차잔고 증가. 공매도 대기 물량이 쌓이는 국면.
    #    이 자료는 다음날 아침에 공개되므로 하루 더 미뤄 판단 시점을 맞춘다.
    balance = frame["sbl_balance_qty"]
    growth = balance / balance.shift(SBL_WINDOW) - 1
    gates["대차잔고 급증"] = (_trailing_rank(growth) >= RANK_CUT).shift(1).astype("boolean").fillna(False).astype(bool)

    return gates.fillna(False)


def build_returns(frame: pd.DataFrame) -> pd.Series:
    """오늘 시가에 사서 다음 거래일 시가에 파는 수익. 실행 가능한 형태다."""
    open_ = frame["index_open"]
    return (open_.shift(-1) / open_ - 1) * 100


def cvar(values: pd.Series, level: float = 0.05) -> float:
    clean = values.dropna()
    if clean.empty:
        return math.nan
    cut = clean.quantile(level)
    tail = clean[clean <= cut]
    return float(tail.mean()) if len(tail) else math.nan


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else math.nan


def _cvar(values: np.ndarray, level: float = 0.05) -> float:
    if len(values) < 5:
        return math.nan
    cut = np.quantile(values, level)
    tail = values[values <= cut]
    return float(tail.mean()) if len(tail) else math.nan


def bootstrap_difference(
    values: pd.Series,
    flag: pd.Series,
    statistic=_mean,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = SEED,
) -> tuple[float, float, float, float]:
    """게이트가 켜진 날과 꺼진 날의 통계량 차이. 블록 부트스트랩 구간.

    평균만 봐서는 부족하다. 손해 최소화가 목적이면 꼬리가 얼마나 두꺼워지는지가
    본질이므로 CVaR 차이도 같은 방식으로 검정한다.
    """
    pair = pd.concat([values, flag], axis=1).dropna()
    if len(pair) < BLOCK_SIZE * 4:
        return math.nan, math.nan, math.nan, math.nan
    sample = pair.iloc[:, 0].to_numpy(dtype=float)
    selected = pair.iloc[:, 1].to_numpy(dtype=bool)
    if not selected.any() or selected.all():
        return math.nan, math.nan, math.nan, math.nan
    observed = statistic(sample[selected]) - statistic(sample[~selected])

    rng = np.random.default_rng(seed)
    blocks_needed = math.ceil(len(sample) / BLOCK_SIZE)
    offsets = np.arange(BLOCK_SIZE)
    draws_list = []
    for _ in range(draws):
        starts = rng.integers(0, len(sample), size=blocks_needed)
        idx = ((starts[:, None] + offsets) % len(sample)).ravel()[: len(sample)]
        picked = selected[idx]
        if picked.any() and not picked.all():
            value = statistic(sample[idx][picked]) - statistic(sample[idx][~picked])
            if not math.isnan(value):
                draws_list.append(value)
    if not draws_list:
        return observed, math.nan, math.nan, math.nan
    distribution = np.asarray(draws_list)
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


def study_gates(returns: pd.Series, gates: pd.DataFrame, active: pd.Series) -> pd.DataFrame:
    """보유 중인 날만 놓고, 게이트가 켜진 날이 정말 더 나빴는지 본다."""
    held = returns[active]
    rows = []
    for name in gates.columns:
        flag = gates[name][active]
        risky, clear = held[flag], held[~flag]
        observed, low, high, p_value = bootstrap_difference(held, flag, _mean)
        tail, tail_low, tail_high, tail_p = bootstrap_difference(held, flag, _cvar)
        rows.append(
            {
                "게이트": name,
                "위험일": int(flag.sum()),
                "정상일": int((~flag).sum()),
                "위험일 평균%": risky.mean(),
                "정상일 평균%": clear.mean(),
                "평균차%": observed,
                "평균 하한": low,
                "평균 상한": high,
                "평균 p": p_value,
                "위험일 CVaR5": cvar(risky),
                "정상일 CVaR5": cvar(clear),
                "CVaR차%": tail,
                "CVaR 하한": tail_low,
                "CVaR 상한": tail_high,
                "CVaR p": tail_p,
                "위험일 손실비율": (risky < 0).mean(),
                "정상일 손실비율": (clear < 0).mean(),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["평균 q(FDR)"] = benjamini_hochberg(result["평균 p"])
        result["CVaR q(FDR)"] = benjamini_hochberg(result["CVaR p"])
    return result


def equity_stats(returns: pd.Series, name: str, exposure: pd.Series) -> dict:
    """손실 쪽 지표 위주로 성과를 요약한다."""
    clean = returns.fillna(0.0) / 100
    equity = (1 + clean).cumprod()
    drawdown = equity / equity.cummax() - 1
    years = max(len(clean) / 252, 1e-9)

    underwater = 0
    longest = 0
    for value in drawdown:
        underwater = underwater + 1 if value < -1e-12 else 0
        longest = max(longest, underwater)

    volatility = clean.std()
    return {
        "전략": name,
        "누적수익%": (equity.iloc[-1] - 1) * 100,
        "연수익%": (equity.iloc[-1] ** (1 / years) - 1) * 100,
        "MDD%": drawdown.min() * 100,
        "최장 손실구간(일)": longest,
        "CVaR5%": cvar(returns),
        "손실일 비율": (clean < 0).mean(),
        "샤프": clean.mean() / volatility * math.sqrt(252) if volatility else math.nan,
        "평균 노출": exposure.mean(),
    }


def apply_costs(weight: pd.Series, returns: pd.Series, cost_bps: float) -> pd.Series:
    """비중이 바뀐 만큼 왕복비용을 나눠 부과한다."""
    turnover = weight.diff().abs().fillna(weight.abs())
    cost = turnover * (cost_bps / 2) / 100
    return weight * returns - cost


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="돈치안 신호에 위험 게이트를 덧댄 효과 검정")
    parser.add_argument("--index", default="0001")
    parser.add_argument("--start", default=None)
    parser.add_argument("--db", type=Path, default=Path("data/kis_panel.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--round-trip-cost-bps", type=float, default=ROUND_TRIP_COST_BPS)
    parser.add_argument(
        "--risky-weight",
        type=float,
        default=0.5,
        help="게이트가 켜졌을 때 남길 비중. 0이면 전량 보류",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = PanelStore(args.db)
    frame = load_index(store, args.index, args.start)

    position = donchian_position(frame)
    gates = build_gates(frame)
    returns = build_returns(frame)

    # 어제 종가로 판단하고 오늘 시가에 실행한다. 게이트도 같은 시점의 정보만 쓴다.
    active = position.shift(1).fillna(0).astype(bool)
    gates_lagged = gates.shift(1).astype("boolean").fillna(False).astype(bool)
    usable = returns.notna() & frame["index_open"].notna()
    active = active & usable

    print(f"지수 {args.index}  {frame['date'].min()} ~ {frame['date'].max()}  {len(frame)}일")
    trades = int((position.diff() == 1).sum())
    print(f"돈치안 보유일 {int(active.sum())}일  진입 횟수 {trades}회")
    print(f"진입 횟수가 적어 거래 단위 비교는 힘이 없다. 보유일 단위로 검정한다.\n")

    print("=" * 80)
    print("1. 게이트별 효과 (보유일만, 시가-시가 수익, FDR 보정)")
    print("=" * 80)
    table = study_gates(returns, gates_lagged, active)
    print("  평균 수익 차이")
    print(table[["게이트", "위험일", "정상일", "위험일 평균%", "정상일 평균%",
                 "평균차%", "평균 하한", "평균 상한", "평균 p",
                 "평균 q(FDR)"]].round(4).to_string(index=False))
    print("\n  꼬리 위험 차이 (CVaR5, 하위 5% 날의 평균)")
    print(table[["게이트", "위험일 CVaR5", "정상일 CVaR5", "CVaR차%",
                 "CVaR 하한", "CVaR 상한", "CVaR p",
                 "CVaR q(FDR)"]].round(4).to_string(index=False))
    print("\n  손실일 비율")
    print(table[["게이트", "위험일 손실비율", "정상일 손실비율"]].round(4).to_string(index=False))

    print("\n" + "=" * 80)
    print(f"2. 게이트를 실제로 적용했을 때 (위험일 비중 {args.risky_weight:.0%})")
    print("=" * 80)
    base_weight = active.astype(float)
    rows = [
        equity_stats(
            apply_costs(base_weight, returns, args.round_trip_cost_bps),
            "돈치안 단독",
            base_weight,
        )
    ]
    for name in gates.columns:
        weight = base_weight.where(~gates_lagged[name], base_weight * args.risky_weight)
        rows.append(
            equity_stats(
                apply_costs(weight, returns, args.round_trip_cost_bps),
                f"+ {name}",
                weight,
            )
        )
    any_gate = gates_lagged.any(axis=1)
    weight = base_weight.where(~any_gate, base_weight * args.risky_weight)
    rows.append(
        equity_stats(
            apply_costs(weight, returns, args.round_trip_cost_bps), "+ 하나라도 켜지면", weight
        )
    )
    count = gates_lagged.sum(axis=1)
    weight = base_weight.where(count < 2, base_weight * args.risky_weight)
    rows.append(
        equity_stats(
            apply_costs(weight, returns, args.round_trip_cost_bps), "+ 둘 이상 켜지면", weight
        )
    )
    equity = pd.DataFrame(rows)
    print(equity.round(3).to_string(index=False))

    print("\n" + "=" * 80)
    print("3. 게이트가 켜진 빈도와 겹침")
    print("=" * 80)
    held_gates = gates_lagged[active]
    frequency = pd.DataFrame(
        {
            "켠 비율": held_gates.mean(),
            "보유일 중 일수": held_gates.sum(),
        }
    )
    print(frequency.round(3).to_string())
    print("\n  게이트 사이 상관")
    print(held_gates.astype(float).corr().round(3).to_string())

    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "gate_study_effects.csv", index=False, encoding="utf-8-sig")
    equity.to_csv(args.output / "gate_study_equity.csv", index=False, encoding="utf-8-sig")
    print(f"\n표는 {args.output.resolve()}에 저장했습니다.")
    print("표본이 3년 남짓이라 어떤 결과든 확정으로 다루면 안 됩니다.")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
