# -*- coding: utf-8 -*-
"""수익을 올리고 손실을 줄이는 규칙 후보를 한 검정대에 올려 일괄 판정한다.

지금까지 개별 가설을 하나씩 검정해 왔다. 밤사이 해외 충격, 돌파일 거래대금,
미국 장기금리, 엔 캐리, FTD 절단점. 전부 폐기됐다. 폐기 사유는 매번 같았다 -
전체 성과는 좋아 보이는데 구간을 나누면 부호가 뒤집힌다.

그래서 여기서는 후보를 하나씩 보지 않고 **같은 잣대에 한꺼번에 올린다.** 후보를
많이 시도할수록 우연히 통과하는 것이 생기므로, 통과 기준을 전체 성과가 아니라
아래 넷을 동시에 만족하는 것으로 둔다.

    1. 전체 CAGR이 기준선보다 높다
    2. 세 구간(1995~2007, 2008~2013, 2014~2026) 모두 CAGR이 악화되지 않는다
    3. 세 구간 모두 MDD가 악화되지 않는다
    4. 워크포워드(매년 말에 과거만 보고 선택)로도 기준선보다 낫다

1~3을 통과한 것만 4로 보낸다. 4까지 통과한 것만 보고할 가치가 있다.

후보군은 세 갈래다.

    진입 필터  - 언제 사지 않을 것인가
    청산 규칙  - 언제 팔 것인가
    사이징     - 얼마나 살 것인가

    python improvement_sweep.py

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
ENTRY_WINDOW = 20
EXIT_WINDOW = 20
VOL_WINDOW = 20
MIN_ENTRIES = 10
WALK_START_YEAR = 2005

# ---- 판정 기준. 돌리기 전에 고정한다 -------------------------------------
TOL_CAGR = -0.10   # %p. 이만큼의 악화는 노이즈로 본다
TOL_MDD = -0.50    # %p


# ─────────────────────────────────────────────────────────────────
# 자료와 기본 신호
# ─────────────────────────────────────────────────────────────────
def load_kospi(start: str) -> pd.DataFrame:
    import FinanceDataReader as fdr

    frame = fdr.DataReader("KS11", start)
    frame.index = pd.to_datetime(frame.index)
    return frame.dropna(subset=["Open", "High", "Low", "Close", "Volume"])


def indicators(k: pd.DataFrame) -> dict:
    """모든 후보가 공유하는 지표. 전부 shift로 룩어헤드를 막는다."""
    high, low, close = k["High"], k["Low"], k["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return {
        "채널상단": high.rolling(ENTRY_WINDOW).max().shift(1),
        "ATR20": tr.rolling(20).mean().shift(1),
        "MA100": close.rolling(100).mean().shift(1),
        "MA150": close.rolling(150).mean().shift(1),
        "MA200": close.rolling(200).mean().shift(1),
        "MA50": close.rolling(50).mean().shift(1),
        "vol20": (close.pct_change() * 100).rolling(VOL_WINDOW).std().shift(1),
        "거래대금비": (k["Amount"] / k["Amount"].rolling(20).mean().shift(1)
                  if "Amount" in k else pd.Series(1.0, index=k.index)),
    }


def simulate(k: pd.DataFrame, ind: dict, *, exit_window: int = EXIT_WINDOW,
             allow: pd.Series | None = None, trail_atr: float | None = None,
             ma_exit: str | None = None) -> pd.Series:
    """진입 신호와 청산 규칙으로 보유 상태를 만든다.

    ATR 트레일링은 진입 이후의 최고가를 따라가므로 경로 의존이다. 벡터로는
    표현할 수 없어 루프로 돈다. 청산 조건은 OR로 겹쳐 쓴다 - 어느 하나라도
    걸리면 판다.
    """
    high = k["High"].to_numpy()
    low = k["Low"].to_numpy()
    close = k["Close"].to_numpy()
    ch_high = ind["채널상단"].to_numpy()
    ch_low = k["Low"].rolling(exit_window).min().shift(1).to_numpy()
    atr = ind["ATR20"].to_numpy()
    ma = ind[ma_exit].to_numpy() if ma_exit else None
    ok = allow.to_numpy() if allow is not None else np.ones(len(k), dtype=bool)

    pos = np.zeros(len(k), dtype=np.int8)
    state = 0
    peak = -np.inf
    for i in range(len(k)):
        if np.isnan(ch_high[i]) or np.isnan(ch_low[i]):
            continue
        if state == 0:
            if high[i] > ch_high[i] and ok[i]:
                state = 1
                peak = high[i]
        else:
            peak = max(peak, high[i])
            hit = low[i] < ch_low[i]
            if not hit and trail_atr is not None and not np.isnan(atr[i]):
                hit = low[i] < peak - trail_atr * atr[i]
            if not hit and ma is not None and not np.isnan(ma[i]):
                hit = close[i] < ma[i]
            if hit:
                state = 0
                peak = -np.inf
        pos[i] = state
    return pd.Series(pos, index=k.index, dtype=int)


def expanding_target_vol(vol: pd.Series, position: pd.Series) -> pd.Series:
    """진입일 직전 변동성의 확장 중앙값. 2026-08-19에 채택된 사이징 규칙이다."""
    entry_day = (position == 1) & (position.shift(1).fillna(0) == 0)
    at_entry = vol.where(entry_day.shift(1).fillna(False))
    return at_entry.expanding(min_periods=MIN_ENTRIES).median().ffill().shift(1).ffill()


def daily_returns(k: pd.DataFrame, position: pd.Series,
                  weight: pd.Series | None = None, cash_rate: float = 0.0):
    held = position.shift(1).fillna(0).astype(float)
    if weight is not None:
        held = held * weight.shift(1).fillna(1.0).clip(0, 1)
    ret = (k["Open"].shift(-1) / k["Open"] - 1).fillna(0)
    cash = (1 + cash_rate) ** (1 / 252) - 1
    cost = held.diff().abs().fillna(0) * (COST_BPS / 10000)
    return held * ret + (1 - held) * cash - cost, held


def stats(daily: pd.Series, held: pd.Series) -> dict:
    eq = (1 + daily).cumprod()
    live = daily[held > 0]
    return {
        "CAGR%": round((eq.iloc[-1] ** (252 / len(daily)) - 1) * 100, 2),
        "MDD%": round((eq / eq.cummax() - 1).min() * 100, 2),
        "샤프": round(daily.mean() / daily.std() * np.sqrt(252), 3),
        "CVaR5%": round(live[live <= live.quantile(0.05)].mean() * 100, 2) if len(live) else np.nan,
        "노출%": round(held.mean() * 100, 0),
        "배수": round(eq.iloc[-1], 1),
    }


def segment_stats(daily: pd.Series, held: pd.Series, index) -> dict:
    out = {}
    for a, b in SEGMENTS:
        mask = (index >= f"{a}-01-01") & (index <= f"{b}-12-31")
        out[f"{a}~{b}"] = stats(daily[mask], held[mask])
    return out


# ─────────────────────────────────────────────────────────────────
# 후보군
# ─────────────────────────────────────────────────────────────────
def build_candidates(k: pd.DataFrame, ind: dict) -> dict:
    """이름 -> (position, weight, cash_rate). weight None이면 100%."""
    close = k["Close"]
    vol = ind["vol20"]
    base_pos = simulate(k, ind)
    target = expanding_target_vol(vol, base_pos)
    vol_weight = (target / vol).clip(0, 1)

    cands = {}

    # -- 진입 필터: 언제 사지 않을 것인가 --------------------------------
    for span in (100, 150, 200):
        cands[f"진입필터 종가>MA{span}"] = (
            simulate(k, ind, allow=close > ind[f"MA{span}"]), None, 0.0)
    cands["진입필터 MA50>MA200"] = (
        simulate(k, ind, allow=ind["MA50"] > ind["MA200"]), None, 0.0)
    # 변동성이 그때까지 역사의 상위 N%면 사지 않는다
    rank = vol.expanding(min_periods=252).apply(lambda x: (x[-1] >= x).mean(), raw=True)
    for q in (0.80, 0.90):
        cands[f"진입필터 변동성<{q:.0%}분위"] = (
            simulate(k, ind, allow=rank < q), None, 0.0)

    # -- 청산 규칙: 언제 팔 것인가 ---------------------------------------
    for w in (10, 12, 15, 18, 25, 30, 40):
        cands[f"청산 하단{w}일"] = (simulate(k, ind, exit_window=w), None, 0.0)
    for mult in (2.0, 3.0, 4.0, 5.0):
        cands[f"청산 ATR{mult:g}N 트레일링"] = (
            simulate(k, ind, trail_atr=mult), None, 0.0)
    for span in (50, 100):
        cands[f"청산 종가<MA{span}"] = (simulate(k, ind, ma_exit=f"MA{span}"), None, 0.0)
    # 채널 청산에 ATR 트레일링을 겹쳐 쓴다. 어느 하나라도 걸리면 판다
    for mult in (3.0, 4.0):
        cands[f"청산 하단20일+ATR{mult:g}N"] = (
            simulate(k, ind, trail_atr=mult), None, 0.0) if False else (
            simulate(k, ind, exit_window=EXIT_WINDOW, trail_atr=mult), None, 0.0)

    # -- 사이징: 얼마나 살 것인가 -----------------------------------------
    cands["사이징 변동성타겟"] = (base_pos, vol_weight, 0.0)
    cands["사이징 변동성타겟(진입시고정)"] = (
        base_pos, vol_weight.where((base_pos == 1) & (base_pos.shift(1) == 0)).ffill(), 0.0)
    cands["사이징 절반"] = (base_pos, pd.Series(0.5, index=k.index), 0.0)

    # -- 조합: 개별로 좋았던 것을 겹친다 ----------------------------------
    fast_exit = simulate(k, ind, exit_window=15)
    fast_target = expanding_target_vol(vol, fast_exit)
    cands["조합 하단15일+변동성타겟"] = (
        fast_exit, (fast_target / vol).clip(0, 1), 0.0)
    cands["조합 하단15일+MA150필터"] = (
        simulate(k, ind, exit_window=15, allow=close > ind["MA150"]), None, 0.0)
    cands["조합 하단15일+변동성<80%"] = (
        simulate(k, ind, exit_window=15, allow=rank < 0.80), None, 0.0)

    # -- 현금 이자: 규칙이 아니라 가정 변경이다 ----------------------------
    # 이 검정대에서는 같은 이자를 기준선에도 줘야 공정하므로 증감이 0으로 나온다.
    # 이득은 표의 CAGR 절대값에서 읽는다 (8.76 -> 10.52).
    cands["현금 3.5% 파킹"] = (base_pos, None, 0.035)
    cands["조합 하단15일+현금3.5%"] = (fast_exit, None, 0.035)

    return cands, base_pos, vol_weight


def walk_forward(k: pd.DataFrame, base_daily, base_held,
                 cand_daily, cand_held) -> dict:
    """매년 말에 그때까지의 자료만 보고 다음 해에 후보를 쓸지 정한다.

    구간별 부호를 통과해도 그것은 표본 전체를 본 뒤의 판단이다. 실제로는 그해
    연말에 아는 것만으로 골라야 한다. 이 검사를 통과하지 못하면 쓸 수 없다.
    """
    daily = pd.Series(0.0, index=k.index)
    held = pd.Series(0.0, index=k.index)
    picks = []
    for year in range(WALK_START_YEAR, k.index[-1].year + 1):
        past = k.index < f"{year}-01-01"
        if past.sum() < 252 * 5:
            continue
        use = (1 + cand_daily[past]).cumprod().iloc[-1] > (1 + base_daily[past]).cumprod().iloc[-1]
        this = (k.index >= f"{year}-01-01") & (k.index <= f"{year}-12-31")
        daily[this] = (cand_daily if use else base_daily)[this]
        held[this] = (cand_held if use else base_held)[this]
        picks.append(use)
    live = k.index >= f"{WALK_START_YEAR}-01-01"
    return {
        "선택비율": f"{int(sum(picks))}/{len(picks)}",
        "워크포워드": stats(daily[live], held[live]),
        "기준선동구간": stats(base_daily[live], base_held[live]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="개선 후보 일괄 검정")
    parser.add_argument("--start", default=START)
    parser.add_argument("--outdir", default="output")
    args = parser.parse_args()

    k = load_kospi(args.start)
    ind = indicators(k)
    print(f"코스피 {k.index[0].date()} ~ {k.index[-1].date()}, {len(k)}일")

    cands, base_pos, _ = build_candidates(k, ind)
    base_daily, base_held = daily_returns(k, base_pos)
    base_all = stats(base_daily, base_held)
    base_seg = segment_stats(base_daily, base_held, k.index)

    print(f"\n기준선 (돈치안 단독, 현금 0%): "
          f"CAGR {base_all['CAGR%']}%  MDD {base_all['MDD%']}%  "
          f"샤프 {base_all['샤프']}  배수 {base_all['배수']}")
    print(f"후보 {len(cands)}개를 같은 잣대에 올린다\n")

    rows = []
    passers = {}
    for name, (pos, weight, cash) in cands.items():
        daily, held = daily_returns(k, pos, weight, cash)
        # 현금 이자를 쓰는 후보는 기준선도 같은 이자로 맞춰야 공정하다
        if cash:
            b_daily, b_held = daily_returns(k, base_pos, None, cash)
            b_all, b_seg = stats(b_daily, b_held), segment_stats(b_daily, b_held, k.index)
        else:
            b_daily, b_held, b_all, b_seg = base_daily, base_held, base_all, base_seg
        allst = stats(daily, held)
        seg = segment_stats(daily, held, k.index)
        dc = {s: round(seg[s]["CAGR%"] - b_seg[s]["CAGR%"], 2) for s in seg}
        dm = {s: round(seg[s]["MDD%"] - b_seg[s]["MDD%"], 2) for s in seg}
        ok = (allst["CAGR%"] > b_all["CAGR%"]
              and all(v >= TOL_CAGR for v in dc.values())
              and all(v >= TOL_MDD for v in dm.values()))
        row = {"후보": name, "CAGR%": allst["CAGR%"], "MDD%": allst["MDD%"],
               "샤프": allst["샤프"], "배수": allst["배수"]}
        for s in seg:
            row[f"ΔC {s[:4]}"] = dc[s]
            row[f"ΔM {s[:4]}"] = dm[s]
        row["구간판정"] = "통과" if ok else ""
        rows.append(row)
        passers[name] = (daily, held, b_daily, b_held, ok)

    table = pd.DataFrame(rows).sort_values("CAGR%", ascending=False)
    print(table.to_string(index=False))
    n_seg = sum(1 for v in passers.values() if v[4])
    print(f"\n구간 부호 판정 통과: {n_seg}개 / {len(passers)}개")

    # ---- 워크포워드는 통과 여부와 무관하게 전 후보에 균일 적용한다 --------
    # 통과한 것에만 돌리면 그 자체가 선별이 된다. 같은 잣대를 모두에게 댄다.
    print("\n" + "=" * 78)
    print("워크포워드 (매년 말에 과거만 보고 채택 여부를 정한다) - 전 후보")
    print("=" * 78)
    wf_rows = []
    for name, (daily, held, b_daily, b_held, seg_ok) in passers.items():
        res = walk_forward(k, b_daily, b_held, daily, held)
        w, b = res["워크포워드"], res["기준선동구간"]
        wf_ok = (w["CAGR%"] - b["CAGR%"] > 0 and w["MDD%"] - b["MDD%"] >= TOL_MDD)
        wf_rows.append({"후보": name, "채택연수": res["선택비율"],
                        "WF CAGR%": w["CAGR%"], "WF MDD%": w["MDD%"],
                        "WF 샤프": w["샤프"],
                        "ΔCAGR%p": round(w["CAGR%"] - b["CAGR%"], 2),
                        "ΔMDD%p": round(w["MDD%"] - b["MDD%"], 2),
                        "구간": "O" if seg_ok else "X",
                        "WF": "O" if wf_ok else "X",
                        "최종": "통과" if (seg_ok and wf_ok) else ""})
    wf = pd.DataFrame(wf_rows).sort_values("ΔCAGR%p", ascending=False)
    print(wf.to_string(index=False))
    final = [r["후보"] for r in wf_rows if r["최종"] == "통과"]
    print(f"\n  두 관문 모두 통과: {final if final else '없음'}")
    near = [r["후보"] for r in wf_rows if r["최종"] != "통과" and r["WF"] == "O"]
    if near:
        print(f"  워크포워드만 통과(구간 판정 탈락): {near}")

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    table.to_csv(outdir / "improvement_sweep.csv", index=False, encoding="utf-8-sig")
    if wf_rows:
        pd.DataFrame(wf_rows).to_csv(outdir / "improvement_walkforward.csv",
                                     index=False, encoding="utf-8-sig")
    print(f"\n{outdir / 'improvement_sweep.csv'}")


if __name__ == "__main__":
    main()
