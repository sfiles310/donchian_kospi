# -*- coding: utf-8 -*-
"""돌파일 거래대금이 돈치안 위신호를 걸러내는지 검정한다.

발단은 이 전략의 유일한 구조적 약점이다. 채널 파라미터를 어떻게 바꿔도
2008~2013 횡보장에서 CAGR이 3% 근처로 죽는다. 20일 신고가를 뚫었다가 되밀리는
위신호가 그 구간에 몰려 있기 때문이다.

FTD 규칙은 거래량을 본다. 돈치안은 안 본다. 같은 20일 신고가라도 거래대금이
평소보다 적은 날의 돌파는 참여가 없는 돌파이므로 되밀릴 것이라는 가설이다.

세 가지를 차례로 묻는다.

1. 돌파일 거래대금 비율을 그 시점에 알 수 있는가   -> 룩어헤드
2. 그 비율이 이후 매매 수익을 예측하는가            -> 예측력 (구간별 부호)
3. 예측한다면 전략 성과가 개선되는가                -> 진입 차단

판정 기준은 돌리기 전에 못박는다. 아래 VERDICT_* 상수가 그것이고 결론은 사람이
표를 보고 정하지 않고 그 상수로 계산한다. 구간마다 부호가 뒤집히면 폐기한다.
종가확인 규칙이 그렇게 폐기됐다.

    python volume_confirm_study.py

로컬 전용 연구 스크립트다. 매일 돌리는 알림 경로에는 관여하지 않는다.
실제 운용 수치는 ETF(148020) 시초가로 다시 재야 한다.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import ftd_study as F

warnings.filterwarnings("ignore")

START = "1995-01-01"
COST_BPS = 10.0            # 비중이 1 바뀔 때마다. 왕복이면 20bp
AMT_WINDOW = 20            # 돌파일 거래대금을 직전 며칠 평균과 견주는가
CASH_RATES = (0.0, 0.03)   # 현금 구간 이자. 필터는 현금 시간을 늘리므로 둘 다 본다
SEGMENTS = (("1995", "2007"), ("2008", "2013"), ("2014", "2026"))
RATIO_BUCKETS = [0, 0.8, 1.0, 1.25, 1.6, 100]
RATIO_LABELS = ["0.8 미만", "0.8~1.0", "1.0~1.25", "1.25~1.6", "1.6 이상"]

# ---- 판정 기준. 돌리기 전에 고정한다 -------------------------------------
VERDICT_MAIN = 1.0                # 본 규칙: 돌파일 거래대금 >= 직전 20일 평균
VERDICT_GRID = (0.8, 1.0, 1.25)   # 이 전부에서 같은 부호가 나와야 한다
VERDICT_MIN_TRADES = 5            # 구간별 매매가 이보다 적으면 판정 불가로 둔다
# 통과 조건 (넷 다 만족해야 채택)
#   A. 본 임계에서 구간별 ΔCAGR이 세 구간 모두 악화가 아니다
#   B. 본 임계에서 구간별 ΔMDD가 세 구간 모두 악화가 아니다 (MDD는 음수, 클수록 좋다)
#   C. A와 B가 VERDICT_GRID 임계값 전부에서 성립한다 (하나만 되면 과적합)
#   D. 저거래대금 매매가 고거래대금 매매보다 나쁘다, 그것도 세 구간 모두에서
#      (전체 평균만 좋은 것은 근거가 아니다)
VERDICT_TOL_CAGR = -0.10   # %p. 이만큼의 악화는 노이즈로 본다
VERDICT_TOL_MDD = -0.50    # %p


def load_kospi(start: str) -> pd.DataFrame:
    import FinanceDataReader as fdr

    frame = fdr.DataReader("KS11", start)
    frame.index = pd.to_datetime(frame.index)
    return frame.dropna(subset=["Open", "High", "Low", "Close", "Amount"])


def donchian_parts(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """진입은 돈치안 상단 돌파만 쓴다.

    FTD 결합을 뺀 이유는 1995~2007에서 MDD를 -36.6%에서 -54.9%로 키웠기 때문이다.
    구간마다 부호가 뒤집히는 규칙은 기준선이 못 된다.
    """
    table = pd.DataFrame(
        {"index_high": frame["High"].values, "index_low": frame["Low"].values}
    )
    entry, exit_, warmup = F.donchian_parts(table)
    for series in (entry, exit_, warmup):
        series.index = frame.index
    return entry, exit_, warmup


def amount_ratio(amount: pd.Series, window: int = AMT_WINDOW) -> pd.Series:
    """돌파일 거래대금 / 직전 window일 평균.

    평균에서 당일을 뺀다. 넣으면 자기 자신이 분모를 부풀려 비율이 눌린다.
    거래대금은 31년 사이 스무 배가 되므로 절대값은 쓸 수 없고 비율만 쓴다.

    룩어헤드는 없다. 돌파일 t의 거래대금은 t일 15:30에 확정되고 이 규칙이 바꾸는
    것은 t+1일 08:50의 매수 여부다. 밤사이 해외 자료와 달리 잘라낼 것이 없다.
    """
    return amount / amount.rolling(window).mean().shift(1)


def position_with_filter(entry: pd.Series, exit_: pd.Series, warmup: pd.Series,
                         allow: pd.Series | None) -> pd.Series:
    """진입에만 조건을 건다. 청산은 손대지 않는다.

    걸러진 돌파는 사라지는 것이 아니라 미뤄진다. 상승이 이어지면 다음 날도 20일
    신고가라 진입 신호가 다시 뜨기 때문이다. 그래서 이 필터의 실제 효과는
    '진입 포기'가 아니라 '거래대금이 붙을 때까지 대기'다.
    """
    gated = entry if allow is None else (entry & allow.fillna(False))
    return F.position_from_entries(gated, exit_, warmup)


def trades(position: pd.Series, frame: pd.DataFrame, ratio: pd.Series) -> pd.DataFrame:
    """매매 단위 표. 신호일 t의 다음 시가에 사고 청산 신호일의 다음 시가에 판다."""
    pos = position.to_numpy()
    open_ = frame["Open"].to_numpy()
    rows = []
    i = 0
    while i < len(pos) - 1:
        if pos[i] == 1 and (i == 0 or pos[i - 1] == 0):
            j = i
            while j < len(pos) - 1 and pos[j] == 1:
                j += 1
            if j >= len(pos) - 1:
                break
            rows.append({
                "신호일": position.index[i],
                "매수일": position.index[i + 1],
                "매도일": position.index[j + 1],
                "보유일": j - i,
                "수익%": (open_[j + 1] / open_[i + 1] - 1) * 100,
                "돌파일거래대금비": ratio.iloc[i],
            })
            i = j + 1
        else:
            i += 1
    return pd.DataFrame(rows)


def segment_of(dates: pd.Series) -> pd.Series:
    out = pd.Series("기타", index=dates.index)
    for a, b in SEGMENTS:
        out[(dates >= f"{a}-01-01") & (dates <= f"{b}-12-31")] = f"{a}~{b}"
    return out


def daily_returns(position: pd.Series, frame: pd.DataFrame, cash_rate: float) -> pd.Series:
    """보유일은 시가에서 다음 시가, 현금일은 이자. 비용은 비중 변화량마다."""
    held = position.shift(1).fillna(0).astype(float)
    ret = (frame["Open"].shift(-1) / frame["Open"] - 1).fillna(0)
    cash = (1 + cash_rate) ** (1 / 252) - 1
    cost = held.diff().abs().fillna(0) * (COST_BPS / 10000)
    return held * ret + (1 - held) * cash - cost


def stats(daily: pd.Series, held: pd.Series, label: str) -> dict:
    eq = (1 + daily).cumprod()
    return {
        "전략": label,
        "CAGR%": round((eq.iloc[-1] ** (252 / len(daily)) - 1) * 100, 2),
        "MDD%": round((eq / eq.cummax() - 1).min() * 100, 2),
        "샤프": round(daily.mean() / daily.std() * np.sqrt(252), 3),
        "노출": f"{held.mean() * 100:.0f}%",
        "배수": round(eq.iloc[-1], 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="돌파일 거래대금 확인 필터 검정")
    parser.add_argument("--start", default=START)
    parser.add_argument("--outdir", default="output")
    args = parser.parse_args()

    kospi = load_kospi(args.start)
    print(f"코스피 {kospi.index[0].date()} ~ {kospi.index[-1].date()}, {len(kospi)}일")
    print(f"거래대금 결측 {int(kospi['Amount'].isna().sum())}일, "
          f"0원 {int((kospi['Amount'] == 0).sum())}일\n")

    entry, exit_, warmup = donchian_parts(kospi)
    ratio = amount_ratio(kospi["Amount"])
    base = position_with_filter(entry, exit_, warmup, None)
    seg_names = [f"{a}~{b}" for a, b in SEGMENTS]

    # ---- 질문 2. 예측력 ---------------------------------------------------
    print("=" * 78)
    print("질문 2. 돌파일 거래대금 비율이 그 매매의 수익을 예측하는가")
    print("=" * 78)
    tr = trades(base, kospi, ratio).dropna(subset=["돌파일거래대금비"])
    tr["구간"] = segment_of(tr["신호일"])
    tr["대금구간"] = pd.cut(tr["돌파일거래대금비"], RATIO_BUCKETS, labels=RATIO_LABELS)
    print(f"돈치안 단독 매매 {len(tr)}회. "
          f"돌파일 거래대금비 중앙값 {tr['돌파일거래대금비'].median():.2f}\n")

    print("[전체]")
    print(tr.groupby("대금구간", observed=True).agg(
        횟수=("수익%", "size"), 평균수익=("수익%", "mean"), 중앙=("수익%", "median"),
        승률=("수익%", lambda x: (x > 0).mean() * 100), 평균보유일=("보유일", "mean"),
    ).round(2).to_string())

    print("\n[구간별] 저대금(1.0배 미만) 대 고대금(1.0배 이상)")
    rows_lh = []
    for seg in seg_names:
        sub = tr[tr["구간"] == seg]
        lo = sub[sub["돌파일거래대금비"] < VERDICT_MAIN]
        hi = sub[sub["돌파일거래대금비"] >= VERDICT_MAIN]
        rows_lh.append({
            "구간": seg,
            "저대금n": len(lo), "저대금평균%": round(lo["수익%"].mean(), 2),
            "저대금승률%": round((lo["수익%"] > 0).mean() * 100, 1) if len(lo) else np.nan,
            "고대금n": len(hi), "고대금평균%": round(hi["수익%"].mean(), 2),
            "고대금승률%": round((hi["수익%"] > 0).mean() * 100, 1) if len(hi) else np.nan,
            "차이(고-저)%p": round(hi["수익%"].mean() - lo["수익%"].mean(), 2),
        })
    lh = pd.DataFrame(rows_lh)
    print(lh.to_string(index=False))

    judged = lh[(lh["저대금n"] >= VERDICT_MIN_TRADES) & (lh["고대금n"] >= VERDICT_MIN_TRADES)]
    pass_d = len(judged) == len(SEGMENTS) and bool((judged["차이(고-저)%p"] > 0).all())
    print(f"\n  판정 D 부호 {list(np.sign(lh['차이(고-저)%p'].fillna(0)).astype(int))}"
          f" -> {'통과' if pass_d else '탈락'}")

    # ---- 질문 3. 전략 -----------------------------------------------------
    print("\n" + "=" * 78)
    print("질문 3. 진입을 차단하면 성과가 개선되는가")
    print("=" * 78)
    rows = []
    for cash in CASH_RATES:
        rows.append(stats(daily_returns(base, kospi, cash), base.shift(1).fillna(0),
                          f"돈치안 단독 (현금 {cash * 100:.0f}%)"))
        for theta in VERDICT_GRID:
            pos = position_with_filter(entry, exit_, warmup, ratio >= theta)
            rows.append(stats(daily_returns(pos, kospi, cash), pos.shift(1).fillna(0),
                              f"대금 >= {theta:g}배 (현금 {cash * 100:.0f}%)"))
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n구간별 기준선 대비 증감 (현금 0% 가정)")
    base_daily = daily_returns(base, kospi, 0.0)
    base_held = base.shift(1).fillna(0)
    deltas = []
    for theta in VERDICT_GRID:
        pos = position_with_filter(entry, exit_, warmup, ratio >= theta)
        filt_daily = daily_returns(pos, kospi, 0.0)
        filt_held = pos.shift(1).fillna(0)
        parts = []
        for a, b in SEGMENTS:
            mask = (kospi.index >= f"{a}-01-01") & (kospi.index <= f"{b}-12-31")
            bs = stats(base_daily[mask], base_held[mask], "")
            fs = stats(filt_daily[mask], filt_held[mask], "")
            dc = fs["CAGR%"] - bs["CAGR%"]
            dm = fs["MDD%"] - bs["MDD%"]
            deltas.append({
                "임계": theta, "구간": f"{a}~{b}",
                "기준CAGR%": bs["CAGR%"], "필터CAGR%": fs["CAGR%"], "ΔCAGR%p": round(dc, 2),
                "기준MDD%": bs["MDD%"], "필터MDD%": fs["MDD%"], "ΔMDD%p": round(dm, 2),
            })
            parts.append(f"{a}~{b} ΔCAGR {dc:+6.2f} ΔMDD {dm:+6.2f}")
        print(f"  대금 >= {theta:<5g} | " + " | ".join(parts))

    dl = pd.DataFrame(deltas)
    main_rows = dl[dl["임계"] == VERDICT_MAIN]
    pass_a = bool((main_rows["ΔCAGR%p"] >= VERDICT_TOL_CAGR).all())
    pass_b = bool((main_rows["ΔMDD%p"] >= VERDICT_TOL_MDD).all())
    pass_c = bool((dl["ΔCAGR%p"] >= VERDICT_TOL_CAGR).all()
                  and (dl["ΔMDD%p"] >= VERDICT_TOL_MDD).all())

    # ---- 필터가 실제로 무엇을 바꾸는가 -------------------------------------
    print("\n필터가 실제로 바꾸는 것 (임계 "
          f"{VERDICT_MAIN:g}배)")
    blocked = int((entry & warmup & (ratio < VERDICT_MAIN)).sum())
    print(f"  돌파 신호일 중 거래대금 미달로 막힌 날: {blocked}일 "
          f"(전체 돌파 신호일 {int((entry & warmup).sum())}일)")
    gated_pos = position_with_filter(entry, exit_, warmup, ratio >= VERDICT_MAIN)
    gt = trades(gated_pos, kospi, ratio)
    same = len(set(tr['매수일']) & set(gt['매수일']))
    print(f"  매매 수 {len(tr)}회 -> {len(gt)}회. 매수일이 같은 매매 {same}회")
    print(f"  즉 실제로 달라지는 매매는 {len(tr) - same}회뿐이다. "
          "돌파는 다음 날 다시 뜨므로 필터는 진입을 미룰 뿐 대부분 막지 못한다.")

    # 미뤄진 진입이 얼마나 비싸졌는가. 이것이 예측력이 성과로 넘어가지 못하는 이유다
    delays = []
    for _, row in tr.iterrows():
        later = gt[(gt["매수일"] >= row["매수일"]) & (gt["매수일"] <= row["매도일"])]
        if len(later) and later.iloc[0]["매수일"] != row["매수일"]:
            days = int((later.iloc[0]["매수일"] - row["매수일"]).days)
            slip = (kospi["Open"].loc[later.iloc[0]["매수일"]]
                    / kospi["Open"].loc[row["매수일"]] - 1) * 100
            delays.append({"지연일": days, "진입가차이%": slip})
    if delays:
        dd = pd.DataFrame(delays)
        print(f"  미뤄진 진입 {len(dd)}회: 지연 중앙 {dd['지연일'].median():.0f}일, "
              f"진입가가 평균 {dd['진입가차이%'].mean():+.2f}% 비싸진다")
        print("  거래대금이 붙기를 기다리면 그만큼 높은 값에 산다. "
              "매매 단위 예측력이 성과로 넘어가지 못하는 이유다.")

    # ---- 위신호 -----------------------------------------------------------
    print("\n" + "=" * 78)
    print("2008~2013 위신호를 실제로 잡는가")
    print("=" * 78)
    for theta in (None, VERDICT_MAIN):
        pos = position_with_filter(entry, exit_, warmup,
                                   None if theta is None else ratio >= theta)
        table = trades(pos, kospi, ratio)
        table["구간"] = segment_of(table["신호일"])
        print(f"\n[{'돈치안 단독' if theta is None else f'대금 >= {theta:g}배'}]")
        print(table.groupby("구간", observed=True).agg(
            매매=("수익%", "size"),
            손실매매=("수익%", lambda x: int((x <= 0).sum())),
            위신호율=("수익%", lambda x: round((x <= 0).mean() * 100, 1)),
            평균수익=("수익%", "mean"), 평균보유일=("보유일", "mean"),
        ).round(2).to_string())

    # ---- 판정 -------------------------------------------------------------
    print("\n" + "=" * 78)
    print("판정 (기준은 실행 전에 고정했다)")
    print("=" * 78)
    checks = [
        (f"A. 임계 {VERDICT_MAIN:g}배에서 세 구간 ΔCAGR 악화 없음", pass_a),
        (f"B. 임계 {VERDICT_MAIN:g}배에서 세 구간 ΔMDD 악화 없음", pass_b),
        ("C. 임계값 " + "·".join(f"{t:g}" for t in VERDICT_GRID) + " 전부에서 A·B 성립", pass_c),
        ("D. 세 구간 모두 고대금 매매가 저대금보다 우위", pass_d),
    ]
    for name, ok in checks:
        print(f"  {'O' if ok else 'X'}  {name}")
    verdict = all(ok for _, ok in checks)
    print("\n  결론: " + ("채택 후보 - 운용 반영 검토"
                        if verdict else "폐기 - 구간별 부호가 일관되지 않는다"))

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    for frame, name in ((tr, "trades"), (lh, "predictive"),
                        (pd.DataFrame(rows), "strategy"), (dl, "segments")):
        path = outdir / f"volume_confirm_{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        print(path)


if __name__ == "__main__":
    main()
