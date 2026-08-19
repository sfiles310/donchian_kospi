# -*- coding: utf-8 -*-
"""고전 보조지표를 돈치안에 얹어 도움이 되는지 일괄 검정한다.

`improvement_sweep.py`가 검정한 29종은 이동평균·ATR·채널 계열이었다. 여기서는
증권사 차트에 기본으로 들어 있는 **고전 보조지표**를 같은 검정대에 올린다.
RSI, MACD, 볼린저밴드, 스토캐스틱, ADX, 일목균형표, CCI, MFI, OBV, Williams %R,
SuperTrend, 파라볼릭 SAR, 켈트너 채널.

지표를 많이 시도할수록 우연히 통과하는 것이 생긴다. 그래서 관문은 그대로 둔다.

    1관문  전체 CAGR 개선 + 세 구간 모두 CAGR·MDD 악화 없음
    2관문  워크포워드 - 매년 말에 그해까지의 자료만 보고 채택 여부를 정한다

워크포워드는 1관문 통과 여부와 무관하게 **전 후보에 균일 적용**한다.

쓰는 방식은 세 가지다.

    진입 필터  - 지표가 조건을 만족할 때만 돈치안 돌파를 받아들인다
    청산 규칙  - 지표가 꺾이면 돈치안 하단을 기다리지 않고 판다
    단독 진입  - 돈치안 대신 지표만으로 사고판다 (참고용 기준선)

모든 지표는 `.shift(1)`로 전날까지의 값만 쓴다. 오늘 종가로 계산한 지표를 오늘
판단에 쓰면 룩어헤드다.

    python indicator_sweep.py

로컬 전용 연구 스크립트다. 매일 돌리는 알림 경로에는 관여하지 않는다.
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
WALK_START_YEAR = 2005
TOL_CAGR = -0.10
TOL_MDD = -0.50


# ─────────────────────────────────────────────────────────────────
# 지표. 전부 마지막에 shift(1) 한다
# ─────────────────────────────────────────────────────────────────
def wilder(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(alpha=1 / n, adjust=False).mean()


def indicators(k: pd.DataFrame) -> dict:
    h, l, c, v = k["High"], k["Low"], k["Close"], k["Volume"]
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    atr14 = wilder(tr, 14)
    tp = (h + l + c) / 3

    ind = {}
    ind["ATR14"] = atr14

    # RSI(14) - Wilder
    delta = c.diff()
    ind["RSI14"] = 100 - 100 / (1 + wilder(delta.clip(lower=0), 14)
                                / wilder(-delta.clip(upper=0), 14).replace(0, np.nan))

    # MACD(12,26,9)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    ind["MACD"] = macd
    ind["MACD시그널"] = macd.ewm(span=9, adjust=False).mean()

    # 볼린저(20,2)
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    ind["볼린저상단"] = ma20 + 2 * sd20
    ind["볼린저하단"] = ma20 - 2 * sd20
    ind["밴드폭"] = (4 * sd20 / ma20) * 100

    # 스토캐스틱(14,3,3) 슬로우
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    fast_k = (c - ll) / (hh - ll).replace(0, np.nan) * 100
    ind["스토%K"] = fast_k.rolling(3).mean()
    ind["스토%D"] = ind["스토%K"].rolling(3).mean()

    # ADX(14) - Wilder DMI
    up, dn = h.diff(), -l.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr_s = wilder(tr, 14)
    pdi = 100 * wilder(pd.Series(plus, index=k.index), 14) / atr_s
    mdi = 100 * wilder(pd.Series(minus, index=k.index), 14) / atr_s
    ind["ADX14"] = wilder((pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan) * 100, 14)
    ind["PDI"] = pdi
    ind["MDI"] = mdi

    # 일목균형표
    conv = (h.rolling(9).max() + l.rolling(9).min()) / 2
    base = (h.rolling(26).max() + l.rolling(26).min()) / 2
    ind["전환선"] = conv
    ind["기준선"] = base
    ind["선행A"] = ((conv + base) / 2).shift(26)
    ind["선행B"] = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)

    # CCI(20)
    ma_tp = tp.rolling(20).mean()
    md = (tp - ma_tp).abs().rolling(20).mean()
    ind["CCI20"] = (tp - ma_tp) / (0.015 * md.replace(0, np.nan))

    # MFI(14)
    mf = tp * v
    pos_mf = mf.where(tp > tp.shift(1), 0.0).rolling(14).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0.0).rolling(14).sum()
    ind["MFI14"] = 100 - 100 / (1 + pos_mf / neg_mf.replace(0, np.nan))

    # OBV와 그 20일 이평
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    ind["OBV"] = obv
    ind["OBV_MA20"] = obv.rolling(20).mean()

    # Williams %R(14)
    ind["윌리엄스R"] = (hh - c) / (hh - ll).replace(0, np.nan) * -100

    # 켈트너 채널(20, 2xATR10)
    ema20 = c.ewm(span=20, adjust=False).mean()
    ind["켈트너상단"] = ema20 + 2 * wilder(tr, 10)
    ind["켈트너하단"] = ema20 - 2 * wilder(tr, 10)

    # SuperTrend(10, 3) - 방향만 쓴다
    atr10 = wilder(tr, 10)
    hl2 = (h + l) / 2
    ub, lb = hl2 + 3 * atr10, hl2 - 3 * atr10
    st_dir = np.ones(len(k))
    ubv, lbv, cv = ub.to_numpy(), lb.to_numpy(), c.to_numpy()
    fu, fl = ubv[0], lbv[0]
    for i in range(1, len(k)):
        fu = min(ubv[i], fu) if cv[i - 1] <= fu else ubv[i]
        fl = max(lbv[i], fl) if cv[i - 1] >= fl else lbv[i]
        st_dir[i] = 1 if cv[i] > fu else (-1 if cv[i] < fl else st_dir[i - 1])
    ind["슈퍼트렌드"] = pd.Series(st_dir, index=k.index)

    # 이격도(20)
    ind["이격도20"] = c / ma20 * 100

    return {name: s.shift(1) for name, s in ind.items()}


# ─────────────────────────────────────────────────────────────────
# 엔진
# ─────────────────────────────────────────────────────────────────
def simulate(k: pd.DataFrame, *, allow: pd.Series | None = None,
             extra_exit: pd.Series | None = None) -> pd.Series:
    ch_high = k["High"].rolling(ENTRY_WINDOW).max().shift(1).to_numpy()
    ch_low = k["Low"].rolling(EXIT_WINDOW).min().shift(1).to_numpy()
    high, low = k["High"].to_numpy(), k["Low"].to_numpy()
    ok = (allow.fillna(False).to_numpy() if allow is not None
          else np.ones(len(k), dtype=bool))
    xx = (extra_exit.fillna(False).to_numpy() if extra_exit is not None
          else np.zeros(len(k), dtype=bool))
    pos = np.zeros(len(k), dtype=np.int8)
    state = 0
    for i in range(len(k)):
        if np.isnan(ch_high[i]) or np.isnan(ch_low[i]):
            continue
        if state == 0:
            if high[i] > ch_high[i] and ok[i]:
                state = 1
        elif low[i] < ch_low[i] or xx[i]:
            state = 0
        pos[i] = state
    return pd.Series(pos, index=k.index, dtype=int)


def standalone(k: pd.DataFrame, buy: pd.Series, sell: pd.Series) -> pd.Series:
    """돈치안 없이 지표만으로 사고판다. 비교 기준선용."""
    b = buy.fillna(False).to_numpy()
    s = sell.fillna(False).to_numpy()
    pos = np.zeros(len(k), dtype=np.int8)
    state = 0
    for i in range(len(k)):
        if state == 0 and b[i]:
            state = 1
        elif state == 1 and s[i]:
            state = 0
        pos[i] = state
    return pd.Series(pos, index=k.index, dtype=int)


def daily_returns(k: pd.DataFrame, pos: pd.Series):
    held = pos.shift(1).fillna(0).astype(float)
    ret = (k["Open"].shift(-1) / k["Open"] - 1).fillna(0)
    return held * ret - held.diff().abs().fillna(0) * (COST_BPS / 10000), held


def stats(d: pd.Series, h: pd.Series) -> dict:
    eq = (1 + d).cumprod()
    return {"CAGR%": round((eq.iloc[-1] ** (252 / len(d)) - 1) * 100, 2),
            "MDD%": round((eq / eq.cummax() - 1).min() * 100, 2),
            "샤프": round(d.mean() / d.std() * np.sqrt(252), 3),
            "노출%": round(h.mean() * 100, 0)}


def walk_forward(k, base_d, base_h, cand_d, cand_h):
    d = pd.Series(0.0, index=k.index)
    h = pd.Series(0.0, index=k.index)
    picks = 0
    total = 0
    for year in range(WALK_START_YEAR, k.index[-1].year + 1):
        past = k.index < f"{year}-01-01"
        if past.sum() < 252 * 5:
            continue
        use = ((1 + cand_d[past]).cumprod().iloc[-1]
               > (1 + base_d[past]).cumprod().iloc[-1])
        m = (k.index >= f"{year}-01-01") & (k.index <= f"{year}-12-31")
        d[m] = (cand_d if use else base_d)[m]
        h[m] = (cand_h if use else base_h)[m]
        picks += int(use)
        total += 1
    live = k.index >= f"{WALK_START_YEAR}-01-01"
    return stats(d[live], h[live]), stats(base_d[live], base_h[live]), f"{picks}/{total}"


def build(k: pd.DataFrame, I: dict) -> dict:
    c = k["Close"]
    out = {}

    # ---- 진입 필터 -------------------------------------------------------
    out["RSI14 < 70 (과열 회피)"] = simulate(k, allow=I["RSI14"] < 70)
    out["RSI14 > 50 (모멘텀 확인)"] = simulate(k, allow=I["RSI14"] > 50)
    out["MACD > 시그널"] = simulate(k, allow=I["MACD"] > I["MACD시그널"])
    out["MACD > 0"] = simulate(k, allow=I["MACD"] > 0)
    out["ADX14 > 20 (추세 있을 때만)"] = simulate(k, allow=I["ADX14"] > 20)
    out["ADX14 > 25"] = simulate(k, allow=I["ADX14"] > 25)
    out["PDI > MDI (상승 우위)"] = simulate(k, allow=I["PDI"] > I["MDI"])
    out["일목 구름 위"] = simulate(k, allow=(c.shift(1) > I["선행A"])
                              & (c.shift(1) > I["선행B"]))
    out["일목 전환선 > 기준선"] = simulate(k, allow=I["전환선"] > I["기준선"])
    out["스토캐스틱 < 80"] = simulate(k, allow=I["스토%K"] < 80)
    out["스토캐스틱 %K > %D"] = simulate(k, allow=I["스토%K"] > I["스토%D"])
    out["CCI20 > 0"] = simulate(k, allow=I["CCI20"] > 0)
    out["CCI20 < 200 (과열 회피)"] = simulate(k, allow=I["CCI20"] < 200)
    out["MFI14 < 80"] = simulate(k, allow=I["MFI14"] < 80)
    out["MFI14 > 50"] = simulate(k, allow=I["MFI14"] > 50)
    out["OBV > OBV20이평"] = simulate(k, allow=I["OBV"] > I["OBV_MA20"])
    out["윌리엄스R > -80"] = simulate(k, allow=I["윌리엄스R"] > -80)
    out["슈퍼트렌드 상승"] = simulate(k, allow=I["슈퍼트렌드"] > 0)
    out["볼린저 밴드폭 하위 (스퀴즈)"] = simulate(
        k, allow=I["밴드폭"] < I["밴드폭"].expanding(252).quantile(0.5))
    out["이격도 < 110 (과열 회피)"] = simulate(k, allow=I["이격도20"] < 110)

    # ---- 청산 규칙 -------------------------------------------------------
    out["청산: MACD < 시그널"] = simulate(k, extra_exit=I["MACD"] < I["MACD시그널"])
    out["청산: RSI14 < 40"] = simulate(k, extra_exit=I["RSI14"] < 40)
    out["청산: RSI14 > 80 (익절)"] = simulate(k, extra_exit=I["RSI14"] > 80)
    out["청산: 슈퍼트렌드 하락"] = simulate(k, extra_exit=I["슈퍼트렌드"] < 0)
    out["청산: 켈트너 하단 이탈"] = simulate(k, extra_exit=c.shift(1) < I["켈트너하단"])
    out["청산: 일목 기준선 이탈"] = simulate(k, extra_exit=c.shift(1) < I["기준선"])
    out["청산: 볼린저 하단 이탈"] = simulate(k, extra_exit=c.shift(1) < I["볼린저하단"])
    out["청산: PDI < MDI"] = simulate(k, extra_exit=I["PDI"] < I["MDI"])
    out["청산: 스토%K < %D"] = simulate(k, extra_exit=I["스토%K"] < I["스토%D"])

    # ---- 조합 (개별로 유망한 것 겹치기) ----------------------------------
    out["ADX>20 + 청산 슈퍼트렌드"] = simulate(k, allow=I["ADX14"] > 20,
                                        extra_exit=I["슈퍼트렌드"] < 0)
    out["일목 구름위 + PDI>MDI"] = simulate(
        k, allow=(c.shift(1) > I["선행A"]) & (c.shift(1) > I["선행B"])
        & (I["PDI"] > I["MDI"]))

    # ---- 단독 진입 (참고용, 돈치안을 아예 안 씀) --------------------------
    out["[단독] 볼린저 상단 돌파"] = standalone(
        k, c.shift(1) > I["볼린저상단"], c.shift(1) < I["볼린저하단"])
    out["[단독] MACD 골든/데드크로스"] = standalone(
        k, (I["MACD"] > I["MACD시그널"]) & (I["MACD"].shift(1) <= I["MACD시그널"].shift(1)),
        (I["MACD"] < I["MACD시그널"]) & (I["MACD"].shift(1) >= I["MACD시그널"].shift(1)))
    out["[단독] 슈퍼트렌드"] = standalone(
        k, I["슈퍼트렌드"] > 0, I["슈퍼트렌드"] < 0)
    out["[단독] 일목 구름 돌파"] = standalone(
        k, (c.shift(1) > I["선행A"]) & (c.shift(1) > I["선행B"]),
        (c.shift(1) < I["선행A"]) | (c.shift(1) < I["선행B"]))
    out["[단독] 켈트너 채널"] = standalone(
        k, c.shift(1) > I["켈트너상단"], c.shift(1) < I["켈트너하단"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="고전 보조지표 일괄 검정")
    ap.add_argument("--start", default=START)
    ap.add_argument("--outdir", default="output")
    a = ap.parse_args()

    import FinanceDataReader as fdr
    k = fdr.DataReader("KS11", a.start)
    k.index = pd.to_datetime(k.index)
    k = k.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    print(f"코스피 {k.index[0].date()} ~ {k.index[-1].date()}, {len(k)}일")

    I = indicators(k)
    base_pos = simulate(k)
    bd, bh = daily_returns(k, base_pos)
    b_all = stats(bd, bh)
    b_seg = {f"{x}~{y}": stats(bd[(k.index >= f"{x}-01-01") & (k.index <= f"{y}-12-31")],
                               bh[(k.index >= f"{x}-01-01") & (k.index <= f"{y}-12-31")])
             for x, y in SEGMENTS}
    print(f"기준선 돈치안 단독: CAGR {b_all['CAGR%']}%  MDD {b_all['MDD%']}%  "
          f"샤프 {b_all['샤프']}\n")

    cands = build(k, I)
    print(f"보조지표 후보 {len(cands)}개\n")

    rows = []
    for name, pos in cands.items():
        d, h = daily_returns(k, pos)
        s = stats(d, h)
        dc, dm = [], []
        for x, y in SEGMENTS:
            m = (k.index >= f"{x}-01-01") & (k.index <= f"{y}-12-31")
            ss = stats(d[m], h[m])
            dc.append(round(ss["CAGR%"] - b_seg[f"{x}~{y}"]["CAGR%"], 2))
            dm.append(round(ss["MDD%"] - b_seg[f"{x}~{y}"]["MDD%"], 2))
        g1 = (s["CAGR%"] > b_all["CAGR%"] and all(v >= TOL_CAGR for v in dc)
              and all(v >= TOL_MDD for v in dm))
        w, bb, picks = walk_forward(k, bd, bh, d, h)
        g2 = (w["CAGR%"] - bb["CAGR%"] > 0 and w["MDD%"] - bb["MDD%"] >= TOL_MDD)
        rows.append({"후보": name, "CAGR%": s["CAGR%"], "MDD%": s["MDD%"],
                     "샤프": s["샤프"], "노출%": s["노출%"],
                     "ΔC95": dc[0], "ΔC08": dc[1], "ΔC14": dc[2],
                     "ΔM95": dm[0], "ΔM08": dm[1], "ΔM14": dm[2],
                     "WF채택": picks, "WFΔC": round(w["CAGR%"] - bb["CAGR%"], 2),
                     "1관문": "O" if g1 else "", "2관문": "O" if g2 else "",
                     "최종": "통과" if (g1 and g2) else ""})

    t = pd.DataFrame(rows).sort_values("CAGR%", ascending=False)
    pd.set_option("display.width", 250)
    print(t.to_string(index=False))
    final = t[t["최종"] == "통과"]["후보"].tolist()
    print(f"\n1관문 통과 {int((t['1관문'] == 'O').sum())}개, "
          f"2관문 통과 {int((t['2관문'] == 'O').sum())}개")
    print(f"두 관문 모두 통과: {final if final else '없음'}")

    outdir = Path(a.outdir)
    outdir.mkdir(exist_ok=True)
    t.to_csv(outdir / "indicator_sweep.csv", index=False, encoding="utf-8-sig")
    print(f"\n{outdir / 'indicator_sweep.csv'}")


if __name__ == "__main__":
    main()
