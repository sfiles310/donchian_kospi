# -*- coding: utf-8 -*-
"""FTD 결합의 '2014년 이후에만 좋다'가 진짜 구조변화인지 검정한다.

발단은 2026-08-19에 나온 부수 발견이다. 돈치안에 FTD 진입을 OR로 얹으면

    1995~2007  CAGR 10.99 -> 7.05,  MDD -36.57 -> -54.85
    2008~2013  CAGR  1.67 -> 0.37,  MDD -22.00 -> -32.01
    2014~2026  CAGR  9.95 -> 12.38, MDD -19.28 -> -17.00

2014년 이후에서만 좋다. 그런데 이 2014라는 절단점은 결과를 보고 고른 것이다.
31년 아무 데나 절단점을 그으면 어느 한쪽은 좋아 보이게 마련이다. 그러니 물어야 할
것은 "2014 이후가 좋은가"가 아니라 **"2014가 특별한가"**다.

세 가지를 묻는다.

1. 가능한 모든 절단점 중 최적이 어디인가, 그 최적이 우연으로 설명되는가  -> sup-t
2. 그 구조변화를 실시간으로 잡아낼 수 있었는가                            -> 워크포워드
3. 2014년 이후 자체가 안정적인가                                          -> 하위구간

판정 기준은 돌리기 전에 못박는다. 통과 못 하면 운용 진입 규칙
(`donchian_kospi_daily.py`의 `dc_hit or ftd_hit`)을 돈치안 단독으로 되돌린다.
이 결론은 연구 스크립트가 아니라 매일 돌아가는 알림 경로를 바꾸는 것이므로
사용자 확인 없이 코드를 고치지 않는다. 여기서는 근거만 만든다.

    python ftd_breakpoint_study.py

로컬 전용 연구 스크립트다.
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
COST_BPS = 10.0
GAIN_THRESHOLD = 2.0       # 운용값(ftd_signal.GAIN_THRESHOLD)과 같아야 한다
TRIM = 0.15                # 절단점 후보를 표본 앞뒤 이만큼씩 잘라낸다
BLOCK = 63                 # 블록 부트스트랩 블록 길이. 한 분기
BOOTSTRAP_DRAWS = 2000
SEED = 20260820
WALK_START_YEAR = 2005     # 이 해 말부터 실시간 판단을 시작한다 (10년 관측 뒤)

# ---- 판정 기준. 돌리기 전에 고정한다 -------------------------------------
VERDICT_P = 0.05
VERDICT_BREAK_RANGE = (2011, 2017)   # 최적 절단점이 2014 +- 3년 안에 있어야 한다
VERDICT_TOL_CAGR = -0.10
VERDICT_TOL_MDD = -0.50
SUB_SEGMENTS = (("2014", "2019"), ("2020", "2026"))
# 통과 조건 (넷 다 만족해야 현행 OR FTD 유지)
#   A. sup-t 블록 부트스트랩 p < 0.05 (구조변화가 우연이 아니다)
#   B. 최적 절단점이 2011~2017 (주장하는 2014 근처다)
#   C. 워크포워드 선택이 돈치안 단독보다 CAGR·MDD 둘 다 악화가 아니다
#   D. 2014~2019, 2020~2026 둘 다에서 OR FTD가 돈치안 단독보다 낫다


def load_kospi(start: str) -> pd.DataFrame:
    import FinanceDataReader as fdr

    frame = fdr.DataReader("KS11", start)
    frame.index = pd.to_datetime(frame.index)
    return frame.dropna(subset=["Open", "High", "Low", "Close", "Volume"])


def build_signals(kospi: pd.DataFrame):
    """돈치안 진입·청산과 FTD 진입을 만든다. 운용 코드와 같은 정의를 쓴다."""
    table = pd.DataFrame({
        "date": kospi.index,
        "index_open": kospi["Open"].values, "index_high": kospi["High"].values,
        "index_low": kospi["Low"].values, "index_close": kospi["Close"].values,
        "volume": kospi["Volume"].values,
    }).reset_index(drop=True)
    entry, exit_, warmup = F.donchian_parts(table)
    ftd = F.ftd_signals(table, GAIN_THRESHOLD)["ftd"]
    for series in (entry, exit_, warmup, ftd):
        series.index = kospi.index
    return entry, exit_, warmup, ftd


def daily_returns(position: pd.Series, frame: pd.DataFrame):
    held = position.shift(1).fillna(0).astype(float)
    ret = (frame["Open"].shift(-1) / frame["Open"] - 1).fillna(0)
    cost = held.diff().abs().fillna(0) * (COST_BPS / 10000)
    return held * ret - cost, held


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


def sup_t(diff: np.ndarray, trim: float = TRIM) -> tuple[float, int]:
    """모든 절단점에서 전후 평균 차의 t값을 구하고 그 최대를 돌려준다.

    절단점을 결과를 보고 고른다는 것이 문제의 핵심이므로, 통계량 자체를
    '최대를 고르는 행위'로 정의해야 한다. 그래야 부트스트랩이 그 행위까지
    포함해서 우연 분포를 만든다.
    """
    n = len(diff)
    lo, hi = int(n * trim), int(n * (1 - trim))
    csum = np.concatenate([[0.0], np.cumsum(diff)])
    csq = np.concatenate([[0.0], np.cumsum(diff ** 2)])
    best, best_i = 0.0, lo
    for i in range(lo, hi):
        na, nb = i, n - i
        ma, mb = csum[i] / na, (csum[n] - csum[i]) / nb
        va = max(csq[i] / na - ma ** 2, 1e-30) * na / (na - 1)
        vb = max((csq[n] - csq[i]) / nb - mb ** 2, 1e-30) * nb / (nb - 1)
        t = abs(mb - ma) / np.sqrt(va / na + vb / nb)
        if t > best:
            best, best_i = t, i
    return best, best_i


def block_bootstrap_null(diff: np.ndarray, draws: int = BOOTSTRAP_DRAWS) -> np.ndarray:
    """귀무가설(구조변화 없음)에서 sup-t가 어디까지 나오는지 만든다.

    일별로 섞으면 안 된다. 두 전략의 차이는 몇 달씩 이어지는 매매 단위로 뭉치고,
    그 뭉침 자체가 가짜 구조변화를 만들어내기 때문이다. 블록으로 뽑는다.
    """
    n = len(diff)
    rng = np.random.default_rng(SEED)
    n_blocks = int(np.ceil(n / BLOCK))
    out = np.empty(draws)
    for b in range(draws):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(BLOCK)[None, :]).ravel()[:n] % n
        out[b] = sup_t(diff[idx])[0]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="FTD 결합 절단점의 구조변화 검정")
    parser.add_argument("--start", default=START)
    parser.add_argument("--outdir", default="output")
    args = parser.parse_args()

    kospi = load_kospi(args.start)
    print(f"코스피 {kospi.index[0].date()} ~ {kospi.index[-1].date()}, {len(kospi)}일\n")

    entry, exit_, warmup, ftd = build_signals(kospi)
    pos_don = F.position_from_entries(entry, exit_, warmup)
    pos_or = F.position_from_entries(entry | ftd, exit_, warmup)

    don_daily, don_held = daily_returns(pos_don, kospi)
    or_daily, or_held = daily_returns(pos_or, kospi)

    # ---- FTD가 실제로 몇 번 다르게 만드는가 --------------------------------
    print("=" * 78)
    print("FTD가 단독으로 만드는 진입은 몇 번인가")
    print("=" * 78)
    don_entry_day = (pos_don == 1) & (pos_don.shift(1).fillna(0) == 0)
    or_entry_day = (pos_or == 1) & (pos_or.shift(1).fillna(0) == 0)
    ftd_only = or_entry_day & ~don_entry_day & ftd
    by_year = ftd_only.groupby(ftd_only.index.year).sum()
    by_year = by_year[by_year > 0]
    print(f"돈치안 진입 {int(don_entry_day.sum())}회, OR FTD 진입 {int(or_entry_day.sum())}회")
    print(f"FTD가 단독으로 여는 진입 {int(ftd_only.sum())}회")
    print("  연도별: " + ", ".join(f"{y}년 {int(c)}회" for y, c in by_year.items()))
    before = int(ftd_only[ftd_only.index < "2014-01-01"].sum())
    print(f"  2014년 이전 {before}회 / 이후 {int(ftd_only.sum()) - before}회")
    print("\n  검정의 유효 표본은 7,886일이 아니라 이 진입 횟수다. 미리 새겨둔다.")

    # ---- 전체·구간별 성과 ---------------------------------------------------
    print("\n" + "=" * 78)
    print("구간별 성과 (기존 주장 재현)")
    print("=" * 78)
    rows = []
    spans = [("1995", "2007"), ("2008", "2013"), ("2014", "2026"), ("1995", "2026")]
    for a, b in spans:
        mask = (kospi.index >= f"{a}-01-01") & (kospi.index <= f"{b}-12-31")
        ds = stats(don_daily[mask], don_held[mask], "")
        os_ = stats(or_daily[mask], or_held[mask], "")
        rows.append({"구간": f"{a}~{b}",
                     "돈치안 CAGR%": ds["CAGR%"], "OR FTD CAGR%": os_["CAGR%"],
                     "ΔCAGR%p": round(os_["CAGR%"] - ds["CAGR%"], 2),
                     "돈치안 MDD%": ds["MDD%"], "OR FTD MDD%": os_["MDD%"],
                     "ΔMDD%p": round(os_["MDD%"] - ds["MDD%"], 2)})
    print(pd.DataFrame(rows).to_string(index=False))

    # ---- 질문 1. 절단점이 특별한가 ------------------------------------------
    print("\n" + "=" * 78)
    print("질문 1. 가능한 모든 절단점 중 최적이 어디인가")
    print("=" * 78)
    diff = (or_daily - don_daily).to_numpy()
    observed, at = sup_t(diff)
    break_date = kospi.index[at]
    print(f"  최적 절단점: {break_date.date()}  (sup-t = {observed:.2f})")
    print(f"  절단점 후보는 앞뒤 {TRIM:.0%}를 잘라낸 "
          f"{kospi.index[int(len(diff) * TRIM)].date()} ~ "
          f"{kospi.index[int(len(diff) * (1 - TRIM))].date()}")

    print(f"\n  블록 부트스트랩 {BOOTSTRAP_DRAWS}회 (블록 {BLOCK}일)로 우연 분포를 만든다")
    null = block_bootstrap_null(diff)
    pval = float((null >= observed).mean())
    print(f"  귀무분포 중앙 {np.median(null):.2f}, 95분위 {np.percentile(null, 95):.2f}, "
          f"99분위 {np.percentile(null, 99):.2f}")
    print(f"  p = {pval:.3f}")
    pass_a = pval < VERDICT_P
    pass_b = VERDICT_BREAK_RANGE[0] <= break_date.year <= VERDICT_BREAK_RANGE[1]
    print(f"\n  판정 A (p < {VERDICT_P}): {'통과' if pass_a else '탈락'}")
    print(f"  판정 B (최적 절단점이 {VERDICT_BREAK_RANGE[0]}~{VERDICT_BREAK_RANGE[1]}): "
          f"{'통과' if pass_b else '탈락'}  실제 {break_date.year}년")

    # 절단점별 t값 모양도 남긴다. 봉우리가 뾰족한지 평평한지가 중요하다
    curve = []
    n = len(diff)
    for year in range(1999, 2023):
        mark = kospi.index.searchsorted(pd.Timestamp(f"{year}-01-01"))
        if int(n * TRIM) <= mark < int(n * (1 - TRIM)):
            a_, b_ = diff[:mark], diff[mark:]
            t = abs(b_.mean() - a_.mean()) / np.sqrt(a_.var(ddof=1) / len(a_)
                                                     + b_.var(ddof=1) / len(b_))
            curve.append((year, round(float(t), 2)))
    print("\n  연초를 절단점으로 놓았을 때의 t값 (봉우리가 뾰족해야 진짜다)")
    for i in range(0, len(curve), 8):
        print("    " + "  ".join(f"{y}:{t:.2f}" for y, t in curve[i:i + 8]))

    # ---- 질문 2. 실시간으로 잡을 수 있었는가 --------------------------------
    print("\n" + "=" * 78)
    print("질문 2. 그 구조변화를 실시간으로 잡아낼 수 있었는가 (워크포워드)")
    print("=" * 78)
    print(f"  매년 말에 그때까지의 자료만 보고 다음 해에 FTD를 쓸지 정한다")
    choice_daily = pd.Series(0.0, index=kospi.index)
    choice_held = pd.Series(0.0, index=kospi.index)
    picks = []
    for year in range(WALK_START_YEAR, kospi.index[-1].year + 1):
        past = kospi.index < f"{year}-01-01"
        if past.sum() < 252 * 5:
            continue
        d_eq = (1 + don_daily[past]).cumprod().iloc[-1]
        o_eq = (1 + or_daily[past]).cumprod().iloc[-1]
        use_ftd = o_eq > d_eq
        this = (kospi.index >= f"{year}-01-01") & (kospi.index <= f"{year}-12-31")
        choice_daily[this] = (or_daily if use_ftd else don_daily)[this]
        choice_held[this] = (or_held if use_ftd else don_held)[this]
        picks.append((year, "FTD" if use_ftd else "돈치안"))
    print("  선택: " + ", ".join(f"{y}:{p}" for y, p in picks))
    live = kospi.index >= f"{picks[0][0]}-01-01"
    wf = [
        stats(don_daily[live], don_held[live], "항상 돈치안 단독"),
        stats(or_daily[live], or_held[live], "항상 OR FTD"),
        stats(choice_daily[live], choice_held[live], "워크포워드 선택"),
    ]
    print(f"\n  {picks[0][0]}~{kospi.index[-1].year} 성과")
    print(pd.DataFrame(wf).to_string(index=False))
    base, pick = wf[0], wf[2]
    pass_c = (pick["CAGR%"] - base["CAGR%"] >= VERDICT_TOL_CAGR
              and pick["MDD%"] - base["MDD%"] >= VERDICT_TOL_MDD)
    print(f"\n  판정 C (워크포워드가 돈치안 단독보다 악화 아님): "
          f"{'통과' if pass_c else '탈락'}"
          f"  ΔCAGR {pick['CAGR%'] - base['CAGR%']:+.2f}%p"
          f"  ΔMDD {pick['MDD%'] - base['MDD%']:+.2f}%p")

    # ---- 질문 3. 2014년 이후는 안정적인가 -----------------------------------
    print("\n" + "=" * 78)
    print("질문 3. 2014년 이후 자체가 안정적인가")
    print("=" * 78)
    sub_rows = []
    for a, b in SUB_SEGMENTS:
        mask = (kospi.index >= f"{a}-01-01") & (kospi.index <= f"{b}-12-31")
        ds = stats(don_daily[mask], don_held[mask], "")
        os_ = stats(or_daily[mask], or_held[mask], "")
        sub_rows.append({"구간": f"{a}~{b}", "돈치안 CAGR%": ds["CAGR%"],
                         "OR FTD CAGR%": os_["CAGR%"],
                         "ΔCAGR%p": round(os_["CAGR%"] - ds["CAGR%"], 2),
                         "ΔMDD%p": round(os_["MDD%"] - ds["MDD%"], 2),
                         "FTD단독진입": int(ftd_only[mask].sum())})
    sub = pd.DataFrame(sub_rows)
    print(sub.to_string(index=False))
    pass_d = bool((sub["ΔCAGR%p"] >= VERDICT_TOL_CAGR).all()
                  and (sub["ΔMDD%p"] >= VERDICT_TOL_MDD).all())
    print(f"\n  판정 D (두 하위구간 모두 악화 아님): {'통과' if pass_d else '탈락'}")

    # ---- 판정 --------------------------------------------------------------
    print("\n" + "=" * 78)
    print("판정 (기준은 실행 전에 고정했다)")
    print("=" * 78)
    checks = [
        (f"A. sup-t 부트스트랩 p < {VERDICT_P}", pass_a),
        (f"B. 최적 절단점이 {VERDICT_BREAK_RANGE[0]}~{VERDICT_BREAK_RANGE[1]}년", pass_b),
        ("C. 워크포워드 선택이 돈치안 단독보다 악화 아님", pass_c),
        ("D. 2014~2019, 2020~2026 둘 다에서 악화 아님", pass_d),
    ]
    for name, ok in checks:
        print(f"  {'O' if ok else 'X'}  {name}")
    verdict = all(ok for _, ok in checks)
    print("\n  결론: " + ("현행 OR FTD 유지" if verdict else
                        "구조변화 근거 없음 - 운용 진입을 돈치안 단독으로 되돌릴 것을 권고"))
    if not verdict:
        print("  (알림 경로 변경이므로 사용자 확인 전에는 코드를 고치지 않는다)")

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(outdir / "ftd_breakpoint_segments.csv",
                              index=False, encoding="utf-8-sig")
    pd.DataFrame(wf).to_csv(outdir / "ftd_breakpoint_walkforward.csv",
                            index=False, encoding="utf-8-sig")
    pd.DataFrame(curve, columns=["절단연도", "t값"]).to_csv(
        outdir / "ftd_breakpoint_curve.csv", index=False, encoding="utf-8-sig")
    sub.to_csv(outdir / "ftd_breakpoint_sub.csv", index=False, encoding="utf-8-sig")
    for name in ("segments", "walkforward", "curve", "sub"):
        print(outdir / f"ftd_breakpoint_{name}.csv")


if __name__ == "__main__":
    main()
