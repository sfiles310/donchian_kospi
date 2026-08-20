# -*- coding: utf-8 -*-
"""이름 붙은 매매 시스템들을 코스피에 붙여 검정한다.

돈치안도 리처드 돈치안이 만든 시스템이고 FTD도 윌리엄 오닐이 만든 규칙이다.
같은 계보의 것들이 더 있다. 트레이더나 투자회사가 자기 이름을 걸고 내놓은
완결된 매매 규칙들이다. `indicator_sweep.py`가 검정한 RSI·MACD 같은 **보조지표**와
달리, 이것들은 진입·청산·필터가 한 벌로 묶인 **시스템**이다.

검정 대상.

    터틀 시스템 1·2      리처드 데니스. 돈치안의 직계 후손. 20/10과 55/20
    래리 윌리엄스 변동성돌파  시가 + k x 전일 변동폭 돌파 매수, 당일 종가 청산
    다바스 박스          니콜라스 다바스. 박스 상단 돌파 매수, 하단 이탈 청산
    와인스타인 국면분석     스탠 와인스타인. 30주 이평이 상승 중이고 그 위일 때만
    파버 GTAA          멥 파버. 200일 이평 위에서만 보유
    절대 모멘텀          게리 안토나치. 12개월 수익이 양수일 때만 보유
    52주 신고가 모멘텀     제가디시·티트먼 계보. 52주 고점 근처에서만 산다
    샹들리에 청산         척 르보. 최고가 - 3 x ATR
    코너스 RSI2         래리 코너스. 과매도 반등 (평균회귀 - 반대 진영 대조군)
    엘더 삼중창          알렉산더 엘더. 주봉 추세 + 일봉 진입
    미너비니 VCP 근사      마크 미너비니. 변동성 수축 후 돌파
    할로윈 효과          11월~4월만 보유 (계절성 대조군)

두 가지 방식으로 본다.

    [단독]  그 시스템만으로 사고판다. 돈치안과 정면 비교
    [겹침]  돈치안에 그 시스템의 필터나 청산을 얹는다

판정은 앞선 두 스윕과 같다. 후보를 많이 시도할수록 우연히 통과하는 것이 생기므로
관문을 바꾸지 않는다.

    1관문  전체 CAGR 개선 + 세 구간 모두 CAGR·MDD 악화 없음
    2관문  워크포워드 - 매년 말에 그해까지의 자료만 보고 채택 여부를 정한다

    python system_sweep.py

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
WALK_START_YEAR = 2005
TOL_CAGR = -0.10
TOL_MDD = -0.50


def wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False).mean()


def prep(k: pd.DataFrame) -> dict:
    h, l, c = k["High"], k["Low"], k["Close"]
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    d = {
        "ATR22": wilder(tr, 22).shift(1),
        "MA150": c.rolling(150).mean().shift(1),
        "MA200": c.rolling(200).mean().shift(1),
        "MA5": c.rolling(5).mean().shift(1),
        "고52주": h.rolling(252).max().shift(1),
        "전일변동폭": (h - l).shift(1),
    }
    # 코너스 RSI(2)
    delta = c.diff()
    d["RSI2"] = (100 - 100 / (1 + wilder(delta.clip(lower=0), 2)
                              / wilder(-delta.clip(upper=0), 2).replace(0, np.nan))).shift(1)
    # 주봉 MACD 히스토그램 (엘더 삼중창의 첫 화면)
    wk = c.resample("W-FRI").last()
    macd_w = wk.ewm(span=12, adjust=False).mean() - wk.ewm(span=26, adjust=False).mean()
    hist = macd_w - macd_w.ewm(span=9, adjust=False).mean()
    d["주봉히스토상승"] = (hist > hist.shift(1)).reindex(c.index, method="ffill").shift(1)
    return d


def channel_system(k: pd.DataFrame, n_in: int, n_out: int,
                   allow: pd.Series | None = None,
                   extra_exit: pd.Series | None = None) -> pd.Series:
    """채널 돌파 진입 + 채널 이탈 청산. 돈치안·터틀이 전부 이 틀이다."""
    ch_h = k["High"].rolling(n_in).max().shift(1).to_numpy()
    ch_l = k["Low"].rolling(n_out).min().shift(1).to_numpy()
    hi, lo = k["High"].to_numpy(), k["Low"].to_numpy()
    ok = (allow.fillna(False).to_numpy() if allow is not None
          else np.ones(len(k), dtype=bool))
    xx = (extra_exit.fillna(False).to_numpy() if extra_exit is not None
          else np.zeros(len(k), dtype=bool))
    pos = np.zeros(len(k), dtype=np.int8)
    st = 0
    for i in range(len(k)):
        if np.isnan(ch_h[i]) or np.isnan(ch_l[i]):
            continue
        if st == 0:
            if hi[i] > ch_h[i] and ok[i]:
                st = 1
        elif lo[i] < ch_l[i] or xx[i]:
            st = 0
        pos[i] = st
    return pd.Series(pos, index=k.index, dtype=int)


def state_system(buy: pd.Series, sell: pd.Series) -> pd.Series:
    b = buy.fillna(False).to_numpy()
    s = sell.fillna(False).to_numpy()
    pos = np.zeros(len(b), dtype=np.int8)
    st = 0
    for i in range(len(b)):
        if st == 0 and b[i]:
            st = 1
        elif st == 1 and s[i]:
            st = 0
        pos[i] = st
    return pd.Series(pos, index=buy.index, dtype=int)


def darvas_box(k: pd.DataFrame, look: int = 20, hold: int = 3) -> pd.Series:
    """다바스 박스. 신고가가 hold일 동안 안 깨지면 박스가 굳고 그 위를 뚫으면 산다.

    다바스는 1950년대에 전보로 종가만 받아 보며 이 규칙을 만들었다. 박스 상단은
    '더 못 올라간 자리', 하단은 '더 안 내려간 자리'다.
    """
    h, l = k["High"].to_numpy(), k["Low"].to_numpy()
    n = len(k)
    top = np.full(n, np.nan)
    bot = np.full(n, np.nan)
    cur_top = cur_bot = np.nan
    since_high = 0
    run_high = -np.inf
    run_low = np.inf
    for i in range(look, n):
        if h[i] > run_high:
            run_high = h[i]
            run_low = l[i]
            since_high = 0
        else:
            since_high += 1
            run_low = min(run_low, l[i])
        if since_high >= hold and np.isfinite(run_high):
            cur_top, cur_bot = run_high, run_low
        top[i], bot[i] = cur_top, cur_bot
    top_s = pd.Series(top, index=k.index).shift(1)
    bot_s = pd.Series(bot, index=k.index).shift(1)
    return state_system(k["High"] > top_s, k["Low"] < bot_s)


def volatility_breakout(k: pd.DataFrame, kf: float = 0.5) -> pd.Series:
    """래리 윌리엄스 변동성 돌파. 당일 시가 + k x 전일 변동폭을 뚫으면 사고 종가에 판다.

    하루짜리 매매라 이 저장소의 시가-다음시가 틀과 맞지 않는다. 그래서 이 후보만
    당일 시가->종가 수익으로 따로 계산하고, 비교표에는 참고용으로만 올린다.
    """
    target = k["Open"] + kf * (k["High"] - k["Low"]).shift(1)
    return (k["High"] >= target).astype(int)


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


def walk_forward(k, bd, bh, cd, ch):
    d = pd.Series(0.0, index=k.index)
    h = pd.Series(0.0, index=k.index)
    picks = tot = 0
    for y in range(WALK_START_YEAR, k.index[-1].year + 1):
        past = k.index < f"{y}-01-01"
        if past.sum() < 252 * 5:
            continue
        use = ((1 + cd[past]).cumprod().iloc[-1] > (1 + bd[past]).cumprod().iloc[-1])
        m = (k.index >= f"{y}-01-01") & (k.index <= f"{y}-12-31")
        d[m] = (cd if use else bd)[m]
        h[m] = (ch if use else bh)[m]
        picks += int(use)
        tot += 1
    live = k.index >= f"{WALK_START_YEAR}-01-01"
    return stats(d[live], h[live]), stats(bd[live], bh[live]), f"{picks}/{tot}"


def build(k: pd.DataFrame, P: dict) -> dict:
    c, h = k["Close"], k["High"]
    out = {}

    # ---- 채널 계열: 돈치안의 직계 -----------------------------------------
    out["[단독] 터틀 시스템1 (20/10)"] = channel_system(k, 20, 10)
    out["[단독] 터틀 시스템2 (55/20)"] = channel_system(k, 55, 20)
    out["[단독] 터틀 55/20 + 샹들리에"] = channel_system(
        k, 55, 20, extra_exit=c.shift(1) < h.rolling(22).max().shift(1) - 3 * P["ATR22"])
    out["[단독] 다바스 박스"] = darvas_box(k)

    # ---- 추세 국면 판정 계열 -----------------------------------------------
    out["[단독] 파버 GTAA (MA200)"] = state_system(c.shift(1) > P["MA200"],
                                                c.shift(1) < P["MA200"])
    out["[단독] 절대모멘텀 (12개월)"] = state_system(
        c.shift(1) > c.shift(253), c.shift(1) <= c.shift(253))
    out["[단독] 와인스타인 2국면"] = state_system(
        (c.shift(1) > P["MA150"]) & (P["MA150"] > P["MA150"].shift(21)),
        (c.shift(1) < P["MA150"]) | (P["MA150"] < P["MA150"].shift(21)))
    out["[단독] 할로윈 (11~4월)"] = pd.Series(
        np.isin(k.index.month, [11, 12, 1, 2, 3, 4]).astype(int), index=k.index)

    # ---- 평균회귀 계열 (반대 진영 대조군) ----------------------------------
    out["[단독] 코너스 RSI2"] = state_system(
        (P["RSI2"] < 10) & (c.shift(1) > P["MA200"]), c.shift(1) > P["MA5"])

    # ---- 돈치안에 겹치기 ---------------------------------------------------
    out["[겹침] 와인스타인 2국면 필터"] = channel_system(
        k, 20, 20, allow=(c.shift(1) > P["MA150"]) & (P["MA150"] > P["MA150"].shift(21)))
    out["[겹침] 절대모멘텀 필터"] = channel_system(k, 20, 20, allow=c.shift(1) > c.shift(253))
    out["[겹침] 52주 고점 -5% 이내"] = channel_system(
        k, 20, 20, allow=c.shift(1) > P["고52주"] * 0.95)
    out["[겹침] 엘더 주봉 히스토 상승"] = channel_system(
        k, 20, 20, allow=P["주봉히스토상승"] == True)
    out["[겹침] 미너비니 VCP 근사"] = channel_system(
        k, 20, 20,
        allow=(c.shift(1) > P["MA150"]) & (P["MA150"] > P["MA200"])
        & ((k["High"] - k["Low"]).rolling(10).mean().shift(1)
           < (k["High"] - k["Low"]).rolling(50).mean().shift(1)))
    out["[겹침] 샹들리에 청산 (3xATR22)"] = channel_system(
        k, 20, 20, extra_exit=c.shift(1) < h.rolling(22).max().shift(1) - 3 * P["ATR22"])
    out["[겹침] 샹들리에 청산 (2xATR22)"] = channel_system(
        k, 20, 20, extra_exit=c.shift(1) < h.rolling(22).max().shift(1) - 2 * P["ATR22"])
    out["[겹침] 터틀 청산 (하단10일)"] = channel_system(k, 20, 10)
    out["[겹침] 다바스 하단 청산"] = channel_system(
        k, 20, 20, extra_exit=darvas_box(k).diff() < 0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="이름 붙은 매매 시스템 일괄 검정")
    ap.add_argument("--start", default=START)
    ap.add_argument("--outdir", default="output")
    a = ap.parse_args()

    import FinanceDataReader as fdr
    k = fdr.DataReader("KS11", a.start)
    k.index = pd.to_datetime(k.index)
    k = k.dropna(subset=["Open", "High", "Low", "Close"])
    print(f"코스피 {k.index[0].date()} ~ {k.index[-1].date()}, {len(k)}일")

    P = prep(k)
    base = channel_system(k, 20, 20)
    bd, bh = daily_returns(k, base)
    b_all = stats(bd, bh)
    b_seg = {}
    for x, y in SEGMENTS:
        m = (k.index >= f"{x}-01-01") & (k.index <= f"{y}-12-31")
        b_seg[f"{x}~{y}"] = stats(bd[m], bh[m])
    print(f"기준선 돈치안 20/20: CAGR {b_all['CAGR%']}%  MDD {b_all['MDD%']}%  "
          f"샤프 {b_all['샤프']}\n")

    cands = build(k, P)
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
        rows.append({"시스템": name, "CAGR%": s["CAGR%"], "MDD%": s["MDD%"],
                     "샤프": s["샤프"], "노출%": s["노출%"],
                     "ΔC95": dc[0], "ΔC08": dc[1], "ΔC14": dc[2],
                     "ΔM95": dm[0], "ΔM08": dm[1], "ΔM14": dm[2],
                     "WF채택": picks, "WFΔC": round(w["CAGR%"] - bb["CAGR%"], 2),
                     "WFΔM": round(w["MDD%"] - bb["MDD%"], 2),
                     "1관문": "O" if g1 else "", "2관문": "O" if g2 else "",
                     "최종": "통과" if (g1 and g2) else ""})
    t = pd.DataFrame(rows).sort_values("CAGR%", ascending=False)
    pd.set_option("display.width", 250)
    print(t.to_string(index=False))

    # 래리 윌리엄스 변동성 돌파는 하루짜리라 따로 잰다
    print("\n" + "-" * 78)
    print("래리 윌리엄스 변동성 돌파 (당일 시가->종가, 틀이 달라 별도 계산)")
    print("-" * 78)
    for kf in (0.3, 0.5, 0.7):
        sig = volatility_breakout(k, kf)
        # 갭상승으로 시가가 이미 목표가 위면 목표가에 살 수 없다. 시가에 산다
        entry = np.maximum(k["Open"] + kf * (k["High"] - k["Low"]).shift(1), k["Open"])
        r = np.where(sig == 1, k["Close"] / entry - 1, 0.0)
        r = pd.Series(r, index=k.index) - sig * (COST_BPS / 10000) * 2
        eq = (1 + r).cumprod()
        print(f"  k={kf}: CAGR {(eq.iloc[-1] ** (252 / len(r)) - 1) * 100:6.2f}%  "
              f"MDD {(eq / eq.cummax() - 1).min() * 100:7.2f}%  "
              f"샤프 {r.mean() / r.std() * np.sqrt(252):5.3f}  "
              f"매매일 {sig.mean() * 100:.0f}%  연 {sig.sum() / (len(k) / 252):.0f}회")

    final = t[t["최종"] == "통과"]["시스템"].tolist()
    print(f"\n1관문 통과 {int((t['1관문'] == 'O').sum())}개, "
          f"2관문 통과 {int((t['2관문'] == 'O').sum())}개")
    print(f"두 관문 모두 통과: {final if final else '없음'}")

    outdir = Path(a.outdir)
    outdir.mkdir(exist_ok=True)
    t.to_csv(outdir / "system_sweep.csv", index=False, encoding="utf-8-sig")
    print(f"\n{outdir / 'system_sweep.csv'}")


if __name__ == "__main__":
    main()
