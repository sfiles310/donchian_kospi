# -*- coding: utf-8 -*-
"""위험을 같게 맞춘 뒤 수익이 올라가는지 본다.

지금까지의 검정은 전부 "규칙을 하나 더 얹으면 좋아지는가"였고 전부 실패했다.
이 스크립트는 다른 것을 묻는다. **위험축을 이동시킨 뒤 되돌리는 것**이다.

2026-08-19에 채택된 변동성타겟 사이징은 MDD를 -36.57%에서 -29.64%로 낮추지만
CAGR도 8.76%에서 7.15%로 깎는다. 핸드오프에도 적어 뒀듯 이것은 알파가 아니라
위험축 이동이다. 그렇다면 낮아진 위험만큼 레버리지로 되돌렸을 때 원래보다
높은 수익이 남는가? 남는다면 그것은 진짜 개선이다. 남지 않으면 변동성타겟은
MDD를 감당 못 할 때만 쓰는 도구일 뿐이다.

비교는 반드시 **위험을 맞춘 뒤에** 한다. CAGR만 비교하면 레버리지를 키울수록
좋아 보이는 것이 당연하므로 아무 말도 하지 않는 것과 같다.

    같은 MDD에서 CAGR이 더 높은가
    같은 CAGR에서 MDD가 더 낮은가

현금 이자도 같이 넣는다. 이 전략은 시간의 47%를 현금으로 있고, 레버리지를 쓰면
차입 비용이 붙으므로 둘을 같이 봐야 비교가 정직하다.

    python leverage_study.py

로컬 전용 연구 스크립트다. 레버리지는 실행 가능성(증거금·일일정산·추적오차)을
전혀 반영하지 않은 이론값이다. 결론이 나오더라도 그대로 운용할 수 없다.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

START = "1995-01-01"
COST_BPS = 10.0
SEGMENTS = (("1995", "2007"), ("2008", "2013"), ("2014", "2026"))
ENTRY_WINDOW = EXIT_WINDOW = 20
VOL_WINDOW = 20
MIN_ENTRIES = 10
CASH_RATE = 0.035          # 파킹형 ETF. 2026-08-19 부수 발견 C
BORROW_SPREAD = 0.02       # 차입은 현금 이자보다 이만큼 비싸다고 본다
LEVERAGE_GRID = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0)
MAX_WEIGHT = 2.0
WALK_START_YEAR = 2005
TOL_MDD = -0.50


def load_kospi(start: str) -> pd.DataFrame:
    import FinanceDataReader as fdr

    frame = fdr.DataReader("KS11", start)
    frame.index = pd.to_datetime(frame.index)
    return frame.dropna(subset=["Open", "High", "Low", "Close"])


def donchian(k: pd.DataFrame) -> pd.Series:
    ch_high = k["High"].rolling(ENTRY_WINDOW).max().shift(1)
    ch_low = k["Low"].rolling(EXIT_WINDOW).min().shift(1)
    entry = (k["High"] > ch_high).to_numpy()
    exit_ = (k["Low"] < ch_low).to_numpy()
    warm = ch_high.notna().to_numpy() & ch_low.notna().to_numpy()
    pos, state = np.zeros(len(k), dtype=np.int8), 0
    for i in range(len(k)):
        if not warm[i]:
            continue
        if state == 0 and entry[i]:
            state = 1
        elif state == 1 and exit_[i]:
            state = 0
        pos[i] = state
    return pd.Series(pos, index=k.index, dtype=int)


def expanding_target_vol(vol: pd.Series, position: pd.Series) -> pd.Series:
    entry_day = (position == 1) & (position.shift(1).fillna(0) == 0)
    at_entry = vol.where(entry_day.shift(1).fillna(False))
    return at_entry.expanding(min_periods=MIN_ENTRIES).median().ffill().shift(1).ffill()


def daily_returns(k: pd.DataFrame, position: pd.Series, weight: pd.Series,
                  cash_rate: float = CASH_RATE):
    """비중이 1을 넘으면 초과분에 차입 비용을 물린다.

    레버리지를 공짜로 두면 결과가 전부 거짓이 된다. 남는 현금에는 이자가 붙고
    모자란 현금에는 이자가 나간다. 같은 식으로 처리한다.
    """
    held = position.shift(1).fillna(0).astype(float) * weight.shift(1).fillna(0).clip(0, MAX_WEIGHT)
    ret = (k["Open"].shift(-1) / k["Open"] - 1).fillna(0)
    cash_daily = (1 + cash_rate) ** (1 / 252) - 1
    borrow_daily = (1 + cash_rate + BORROW_SPREAD) ** (1 / 252) - 1
    idle = (1 - held).clip(lower=0)
    debt = (held - 1).clip(lower=0)
    cost = held.diff().abs().fillna(0) * (COST_BPS / 10000)
    return held * ret + idle * cash_daily - debt * borrow_daily - cost, held


def stats(daily: pd.Series, held: pd.Series, label: str = "") -> dict:
    eq = (1 + daily).cumprod()
    live = daily[held > 0]
    return {
        "전략": label,
        "CAGR%": round((eq.iloc[-1] ** (252 / len(daily)) - 1) * 100, 2),
        "MDD%": round((eq / eq.cummax() - 1).min() * 100, 2),
        "샤프": round(daily.mean() / daily.std() * np.sqrt(252), 3),
        "CVaR5%": round(live[live <= live.quantile(0.05)].mean() * 100, 2) if len(live) else np.nan,
        "평균노출%": round(held.mean() * 100, 0),
        "최대노출": round(held.max(), 2),
        "배수": round(eq.iloc[-1], 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="위험을 맞춘 뒤 수익이 올라가는지")
    parser.add_argument("--start", default=START)
    parser.add_argument("--outdir", default="output")
    args = parser.parse_args()

    k = load_kospi(args.start)
    print(f"코스피 {k.index[0].date()} ~ {k.index[-1].date()}, {len(k)}일")
    print(f"현금 이자 {CASH_RATE:.1%}, 차입 가산 {BORROW_SPREAD:.1%}, "
          f"비중 상한 {MAX_WEIGHT:g}배\n")

    position = donchian(k)
    vol = (k["Close"].pct_change() * 100).rolling(VOL_WINDOW).std().shift(1)
    target = expanding_target_vol(vol, position)
    # 채택된 규칙은 1.0에서 자른다. 자르지 않으면 저변동 국면에 저절로 레버리지가
    # 걸려 MDD가 -29.6%에서 -51.5%로 터진다. 변동성타겟의 MDD 이득은 스케일링이
    # 아니라 이 절단에서 나온다. 배수는 절단한 뒤에 곱해야 물음이 성립한다.
    vol_weight = (target / vol).clip(0, 1.0)
    ones = pd.Series(1.0, index=k.index)

    base_daily, base_held = daily_returns(k, position, ones)
    base = stats(base_daily, base_held, "기준선 100% 고정")

    print("=" * 78)
    print("레버리지 격자")
    print("=" * 78)
    rows = [base]
    curves = {"기준선 100% 고정": (base_daily, base_held)}
    for lev in LEVERAGE_GRID:
        name = f"변동성타겟 x{lev:g}"
        d, h = daily_returns(k, position, vol_weight * lev)
        rows.append(stats(d, h, name))
        curves[name] = (d, h)
    for lev in LEVERAGE_GRID[1:]:
        name = f"고정비중 x{lev:g}"
        d, h = daily_returns(k, position, ones * lev)
        rows.append(stats(d, h, name))
        curves[name] = (d, h)
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    # ---- 위험을 맞춘 비교 --------------------------------------------------
    print("\n" + "=" * 78)
    print("위험을 맞춘 비교 (이것만이 의미가 있다)")
    print("=" * 78)
    print(f"기준선: CAGR {base['CAGR%']}%  MDD {base['MDD%']}%  샤프 {base['샤프']}")

    vt = table[table["전략"].str.startswith("변동성타겟")]
    fx = table[table["전략"].str.startswith("고정비중")]
    for label, sub in (("변동성타겟", vt), ("단순 고정비중", fx)):
        safer = sub[sub["MDD%"] >= base["MDD%"]]
        if len(safer):
            best = safer.loc[safer["CAGR%"].idxmax()]
            print(f"\n[{label}] 기준선보다 MDD가 나쁘지 않은 것 중 CAGR 최고")
            print(f"  {best['전략']}: CAGR {best['CAGR%']}% "
                  f"({best['CAGR%'] - base['CAGR%']:+.2f}%p)  "
                  f"MDD {best['MDD%']}% ({best['MDD%'] - base['MDD%']:+.2f}%p)  "
                  f"샤프 {best['샤프']}  평균노출 {best['평균노출%']}%")
        else:
            print(f"\n[{label}] 기준선보다 MDD가 나쁘지 않은 조합이 없다")

    # ---- 구간별 안정성 -----------------------------------------------------
    print("\n" + "=" * 78)
    print("구간별 안정성 (전체가 좋아도 구간에서 뒤집히면 못 쓴다)")
    print("=" * 78)
    seg_rows = []
    for name, (d, h) in curves.items():
        if name == "기준선 100% 고정":
            continue
        row = {"전략": name}
        ok = True
        for a, b in SEGMENTS:
            mask = (k.index >= f"{a}-01-01") & (k.index <= f"{b}-12-31")
            s = stats(d[mask], h[mask])
            bs = stats(base_daily[mask], base_held[mask])
            row[f"ΔC {a}"] = round(s["CAGR%"] - bs["CAGR%"], 2)
            row[f"ΔM {a}"] = round(s["MDD%"] - bs["MDD%"], 2)
            if s["CAGR%"] - bs["CAGR%"] <= 0 or s["MDD%"] - bs["MDD%"] < TOL_MDD:
                ok = False
        row["판정"] = "통과" if ok else ""
        seg_rows.append(row)
    seg = pd.DataFrame(seg_rows)
    print(seg.to_string(index=False))
    winners = seg[seg["판정"] == "통과"]["전략"].tolist()
    print(f"\n  세 구간 모두 CAGR 개선 + MDD 악화 없음: {winners if winners else '없음'}")

    # ---- 워크포워드 --------------------------------------------------------
    print("\n" + "=" * 78)
    print("워크포워드 (매년 말에 과거만 보고 배수를 고른다)")
    print("=" * 78)
    print("  표본 전체를 보고 배수를 고르면 반칙이다. 그해까지의 자료로만 고른다.")
    names = [n for n in curves if n != "기준선 100% 고정"]
    wf_daily = pd.Series(0.0, index=k.index)
    wf_held = pd.Series(0.0, index=k.index)
    picks = []
    for year in range(WALK_START_YEAR, k.index[-1].year + 1):
        past = k.index < f"{year}-01-01"
        if past.sum() < 252 * 5:
            continue
        scored = {}
        for n in names:
            d, h = curves[n]
            eq = (1 + d[past]).cumprod()
            mdd = (eq / eq.cummax() - 1).min() * 100
            if mdd < base["MDD%"]:          # 그때까지 기준선보다 위험했으면 제외
                continue
            scored[n] = eq.iloc[-1]
        pick = max(scored, key=scored.get) if scored else "기준선 100% 고정"
        this = (k.index >= f"{year}-01-01") & (k.index <= f"{year}-12-31")
        d, h = curves[pick]
        wf_daily[this], wf_held[this] = d[this], h[this]
        picks.append((year, pick))
    print("  선택: " + ", ".join(f"{y}:{p.replace('변동성타겟 ', 'VT').replace('고정비중 ', 'FX')}"
                               for y, p in picks))
    live = k.index >= f"{picks[0][0]}-01-01"
    wf = pd.DataFrame([
        stats(base_daily[live], base_held[live], "기준선 100% 고정"),
        stats(wf_daily[live], wf_held[live], "워크포워드 선택"),
    ])
    print(f"\n  {picks[0][0]}~{k.index[-1].year}")
    print(wf.to_string(index=False))
    b, w = wf.iloc[0], wf.iloc[1]
    final = w["CAGR%"] > b["CAGR%"] and w["MDD%"] - b["MDD%"] >= TOL_MDD
    print(f"\n  ΔCAGR {w['CAGR%'] - b['CAGR%']:+.2f}%p  "
          f"ΔMDD {w['MDD%'] - b['MDD%']:+.2f}%p  -> {'통과' if final else '탈락'}")

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    table.to_csv(outdir / "leverage_grid.csv", index=False, encoding="utf-8-sig")
    seg.to_csv(outdir / "leverage_segments.csv", index=False, encoding="utf-8-sig")
    wf.to_csv(outdir / "leverage_walkforward.csv", index=False, encoding="utf-8-sig")
    print(f"\n{outdir / 'leverage_grid.csv'}")


if __name__ == "__main__":
    main()
