# -*- coding: utf-8 -*-
"""미국 장기금리 상승 국면이 코스피에 영향을 주는지 검정한다.

발단은 널리 퍼진 통념이다. 미국 30년물이 계속 오르면 국장이 눌린다는 것이다.
2026-08-19 기준으로 30년물은 5.19%이고 7월 말 5.12%에서 3주간 올라왔다.

어제(2026-08-19) 검정에서 밤사이 해외 충격은 사이징 근거가 못 된다는 결론이 났다.
충격은 그날 시가 갭을 설명하지만 갭 이후 수익은 설명하지 못했다. 금리를 같은
'밤사이 충격' 통로에 넣으면 같은 벽에 부딪힌다. 그래서 여기서는 충격이 아니라
**국면 변수**로 건다. 하루치 금리 변동이 아니라 3개월에 걸친 추세를 본다.

네 가지를 차례로 묻는다.

1. 30년물 상태를 한국 08:50에 알 수 있는가        -> 룩어헤드
2. 금리 상승이 코스피 동시 수익을 설명하는가       -> 이미 반영된 것인가
3. 금리 상승이 코스피 미래 수익을 예측하는가       -> 쓸 수 있는가 (구간별 부호)
4. 예측한다면 전략 성과가 개선되는가                -> 진입 차단·비중 축소

판정 기준은 돌리기 전에 못박는다. 아래 VERDICT_* 상수가 그것이다. 구간마다 부호가
뒤집히면 폐기한다. 종가확인 규칙과 거래대금 필터가 그렇게 폐기됐다.

주의: 이 변수는 어제 실패한 7변수 스크린에 이어 붙는 여덟 번째다. 다중검정 보정을
하지 않았으므로 통과하더라도 그것만으로 운용에 넣지 않는다.

    python bond_regime_study.py

로컬 전용 연구 스크립트다. 매일 돌리는 알림 경로에는 관여하지 않는다.
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
SEGMENTS = (("1995", "2007"), ("2008", "2013"), ("2014", "2026"))
TREND_WINDOW = 63          # 3개월. '계속 오른다'를 이 창의 변화량으로 정의한다
FORWARD = 21               # 1개월 앞 수익
EPISODE_GAP_DAYS = 30      # 이만큼 떨어지면 별개 사건으로 센다
BOOTSTRAP_DRAWS = 20000
SEED = 20260820

# 30년물 3개월 변화(bp) 구간. 좌우 대칭으로 미리 잘라둔다
TREND_BUCKETS = [-10000, -50, -20, 20, 50, 10000]
TREND_LABELS = ["-50bp 이하", "-50~-20bp", "-20~+20bp", "+20~+50bp", "+50bp 이상"]

# 미국 금리는 한국 장 마감 뒤에 정해진다. 라벨이 t보다 엄격히 앞선 마지막 봉만 안다
YIELDS = {
    "미30년": "^TYX",       # CBOE 30년물. 1990년부터 전 구간 있고 실시간이다
    "미10년": "^TNX",       # 어제 'FDR에서 못 받는다'고 적었는데 이 티커로는 받힌다
}
CURVE = "FRED:T10Y2Y"      # 장단기 스프레드. 공표가 늦어 참고용으로만 본다

# ---- 판정 기준. 돌리기 전에 고정한다 -------------------------------------
VERDICT_MIN_DAYS = 200     # 구간별 표본이 이보다 적으면 판정 불가
VERDICT_MIN_TRADES = 5
VERDICT_TOL_CAGR = -0.10   # %p
VERDICT_TOL_MDD = -0.50    # %p
# 통과 조건 (넷 다 만족해야 채택)
#   A. 금리 급등 국면(+50bp 이상)의 코스피 미래 수익이 그 밖보다 낮다, 세 구간 모두
#   B. 돈치안 진입일 기준으로도 같은 부호, 세 구간 모두
#   C. 급등 국면 회피 전략의 ΔCAGR·ΔMDD가 세 구간 모두 악화가 아니다
#   D. 임계를 +30bp/+50bp/+70bp로 바꿔도 C가 유지된다 (하나만 되면 과적합)
GATE_GRID = (30.0, 50.0, 70.0)
GATE_MAIN = 50.0


def load_kospi(start: str) -> pd.DataFrame:
    import FinanceDataReader as fdr

    frame = fdr.DataReader("KS11", start)
    frame.index = pd.to_datetime(frame.index)
    return frame.dropna(subset=["Open", "High", "Low", "Close"])


def strictly_prev(series: pd.Series, target: pd.DatetimeIndex) -> pd.Series:
    """한국 t일 08:50에 아는 미국 봉은 라벨이 t보다 엄격히 앞선 마지막 봉이다.

    날짜로 그냥 붙이면 안 된다. 미국 t일 세션은 한국 t일 밤 22:30에 시작하므로
    같은 라벨의 봉은 그 시점에 아직 없다. 그런데도 자료에는 들어 있다.
    """
    pos = series.index.searchsorted(target, side="left") - 1
    out = pd.Series(np.nan, index=target)
    ok = pos >= 0
    out[ok] = series.values[pos[ok]]
    return out


def load_yields(target: pd.DatetimeIndex) -> pd.DataFrame:
    import FinanceDataReader as fdr

    out = {}
    for name, code in list(YIELDS.items()) + [("10년-2년", CURVE)]:
        try:
            frame = fdr.DataReader(code, "1990-01-01")
        except Exception as exc:
            print(f"  {name}({code}) 수신 실패: {type(exc).__name__}")
            continue
        series = (frame["Close"] if "Close" in frame else frame.iloc[:, 0]).dropna()
        series.index = pd.to_datetime(series.index)
        out[name] = strictly_prev(series, target)
    return pd.DataFrame(out, index=target)


def donchian_position(frame: pd.DataFrame) -> pd.Series:
    """FTD 결합은 구간마다 부호가 뒤집혀 기준선이 못 된다. 돈치안 단독을 쓴다."""
    table = pd.DataFrame(
        {"index_high": frame["High"].values, "index_low": frame["Low"].values}
    )
    entry, exit_, warmup = F.donchian_parts(table)
    for series in (entry, exit_, warmup):
        series.index = frame.index
    position = F.position_from_entries(entry, exit_, warmup)
    return position, entry, exit_, warmup


def trades(position: pd.Series, frame: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
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
            row = {
                "신호일": position.index[i],
                "보유일": j - i,
                "수익%": (open_[j + 1] / open_[i + 1] - 1) * 100,
            }
            for col in extra.columns:
                row[col] = extra[col].iloc[i]
            rows.append(row)
            i = j + 1
        else:
            i += 1
    return pd.DataFrame(rows)


def segment_of(dates) -> pd.Series:
    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    out = pd.Series("기타", index=dates.index)
    for a, b in SEGMENTS:
        out[(dates >= f"{a}-01-01") & (dates <= f"{b}-12-31")] = f"{a}~{b}"
    return out


def episode_bootstrap(values: pd.Series) -> dict:
    """일별 표본을 그대로 세면 안 된다. 금리 국면은 몇 달씩 이어져 뭉친다."""
    gaps = pd.Series(values.index).diff().dt.days.fillna(10 ** 6)
    episode = (gaps.values > EPISODE_GAP_DAYS).cumsum()
    means = values.groupby(episode).mean()
    rng = np.random.default_rng(SEED)
    draws = rng.choice(means.values, size=(BOOTSTRAP_DRAWS, len(means)), replace=True).mean(axis=1)
    return {
        "에피소드": len(means),
        "일수": len(values),
        "에피소드평균의평균%": round(float(means.mean()), 2),
        "음수에피소드": f"{int((means < 0).sum())}/{len(means)}",
        "95%하한": round(float(np.percentile(draws, 2.5)), 2),
        "95%상한": round(float(np.percentile(draws, 97.5)), 2),
        "평균이 0보다 작을 확률": round(float((draws < 0).mean()), 3),
    }


def daily_returns(position: pd.Series, frame: pd.DataFrame,
                  weight: pd.Series | None = None) -> pd.Series:
    held = position.shift(1).fillna(0).astype(float)
    if weight is not None:
        held = held * weight.shift(1).fillna(1.0).clip(0, 1)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="미국 장기금리 국면이 코스피에 주는 영향")
    parser.add_argument("--start", default=START)
    parser.add_argument("--outdir", default="output")
    args = parser.parse_args()

    kospi = load_kospi(args.start)
    print(f"코스피 {kospi.index[0].date()} ~ {kospi.index[-1].date()}, {len(kospi)}일")
    print("금리 자료 수신")
    yields = load_yields(kospi.index)
    print(f"  결측: " + ", ".join(f"{c} {int(yields[c].isna().sum())}일" for c in yields))

    y30 = yields["미30년"]
    trend = (y30 - y30.shift(TREND_WINDOW)) * 100          # bp
    ret1 = kospi["Close"].pct_change() * 100                 # 동시 수익
    fwd = (kospi["Close"].shift(-FORWARD) / kospi["Close"] - 1) * 100
    d30 = y30.diff() * 100                                   # 하루 변화 bp

    panel = pd.DataFrame({
        "30년물%": y30, "30년물3개월변화bp": trend, "30년물일변화bp": d30,
        "코스피당일%": ret1, f"코스피{FORWARD}일선행%": fwd,
    })
    panel["구간"] = segment_of(panel.index).values
    panel["추세구간"] = pd.cut(trend, TREND_BUCKETS, labels=TREND_LABELS)

    print(f"\n최근 상태: 30년물 {y30.iloc[-1]:.2f}%, "
          f"3개월 변화 {trend.iloc[-1]:+.0f}bp, 10년물 {yields['미10년'].iloc[-1]:.2f}%")

    # ---- 질문 2. 동시 수익 -------------------------------------------------
    print("\n" + "=" * 78)
    print("질문 2. 금리 상승이 코스피 당일 수익을 설명하는가 (동시성)")
    print("=" * 78)
    corr = pd.DataFrame({
        "당일 상관": [panel["코스피당일%"].corr(panel["30년물일변화bp"]),
                   panel["코스피당일%"].corr(panel["30년물3개월변화bp"])],
        f"{FORWARD}일 선행 상관": [panel[f"코스피{FORWARD}일선행%"].corr(panel["30년물일변화bp"]),
                            panel[f"코스피{FORWARD}일선행%"].corr(panel["30년물3개월변화bp"])],
    }, index=["30년물 일변화", "30년물 3개월변화"])
    print(corr.round(3).to_string())

    print("\n연도별 당일 상관 (부호가 뒤집히는지 본다)")
    yearly = panel.groupby(panel.index.year).apply(
        lambda g: g["코스피당일%"].corr(g["30년물일변화bp"]) if len(g) > 50 else np.nan)
    line = []
    for year, value in yearly.dropna().items():
        line.append(f"{year}:{value:+.2f}")
    for i in range(0, len(line), 8):
        print("  " + "  ".join(line[i:i + 8]))
    print(f"  양수 {int((yearly > 0).sum())}년 / 음수 {int((yearly < 0).sum())}년"
          "  -- 통념대로면 전부 음수여야 한다")

    # ---- 질문 3. 미래 수익 -------------------------------------------------
    print("\n" + "=" * 78)
    print(f"질문 3. 금리 상승 국면이 코스피 {FORWARD}일 선행 수익을 예측하는가")
    print("=" * 78)
    valid = panel.dropna(subset=["30년물3개월변화bp", f"코스피{FORWARD}일선행%"])
    print("[전체]")
    print(valid.groupby("추세구간", observed=True).agg(
        일수=(f"코스피{FORWARD}일선행%", "size"),
        평균선행=(f"코스피{FORWARD}일선행%", "mean"),
        중앙=(f"코스피{FORWARD}일선행%", "median"),
        승률=(f"코스피{FORWARD}일선행%", lambda x: (x > 0).mean() * 100),
        평균당일=("코스피당일%", "mean"),
    ).round(2).to_string())

    print(f"\n[구간별] 급등 국면(+{GATE_MAIN:.0f}bp 이상) 대 그 밖")
    rows_a = []
    for seg in [f"{a}~{b}" for a, b in SEGMENTS]:
        sub = valid[valid["구간"] == seg]
        hot = sub[sub["30년물3개월변화bp"] >= GATE_MAIN]
        rest = sub[sub["30년물3개월변화bp"] < GATE_MAIN]
        rows_a.append({
            "구간": seg, "급등일수": len(hot),
            "급등선행%": round(hot[f"코스피{FORWARD}일선행%"].mean(), 2),
            "그밖일수": len(rest),
            "그밖선행%": round(rest[f"코스피{FORWARD}일선행%"].mean(), 2),
            "차이(급등-그밖)%p": round(hot[f"코스피{FORWARD}일선행%"].mean()
                                 - rest[f"코스피{FORWARD}일선행%"].mean(), 2),
        })
    ta = pd.DataFrame(rows_a)
    print(ta.to_string(index=False))
    judged_a = ta[(ta["급등일수"] >= VERDICT_MIN_DAYS) & (ta["그밖일수"] >= VERDICT_MIN_DAYS)]
    pass_a = len(judged_a) == len(SEGMENTS) and bool((judged_a["차이(급등-그밖)%p"] < 0).all())
    print(f"\n  판정 A 부호 {list(np.sign(ta['차이(급등-그밖)%p'].fillna(0)).astype(int))}"
          f" -> {'통과' if pass_a else '탈락'}  (통념대로면 전부 -1)")

    hot_all = valid[valid["30년물3개월변화bp"] >= GATE_MAIN][f"코스피{FORWARD}일선행%"]
    if len(hot_all) >= 50:
        print(f"\n급등 국면의 에피소드 단위 검정 (겹치는 {FORWARD}일 창이라 일수는 못 믿는다)")
        for key, value in episode_bootstrap(hot_all).items():
            print(f"  {key}: {value}")

    # ---- 두 번째 정의: 수준 -----------------------------------------------
    # '계속 오른다'는 추세로도, '높다'는 수준으로도 읽을 수 있다. 둘 다 실행 전에
    # 정해뒀다. 수준은 비정상이라 절대값을 못 쓰고 그때까지의 확장 백분위를 쓴다.
    print("\n" + "=" * 78)
    print("두 번째 정의: 추세가 아니라 수준이 높을 때 (확장 백분위)")
    print("=" * 78)
    rank = y30.expanding(min_periods=756).apply(lambda x: (x[-1] >= x).mean(), raw=True)
    lv = panel.assign(수준백분위=rank).dropna(
        subset=["수준백분위", f"코스피{FORWARD}일선행%"])
    lv["수준구간"] = pd.cut(lv["수준백분위"], [0, .2, .4, .6, .8, 1.0],
                        labels=["하위20%", "20~40%", "40~60%", "60~80%", "상위20%"])
    print(lv.groupby("수준구간", observed=True).agg(
        일수=(f"코스피{FORWARD}일선행%", "size"),
        평균선행=(f"코스피{FORWARD}일선행%", "mean"),
        승률=(f"코스피{FORWARD}일선행%", lambda x: (x > 0).mean() * 100),
    ).round(2).to_string())
    rows_lv = []
    for seg in [f"{a}~{b}" for a, b in SEGMENTS]:
        sub = lv[lv["구간"] == seg]
        hi = sub[sub["수준백분위"] >= 0.8]
        rest = sub[sub["수준백분위"] < 0.8]
        rows_lv.append({"구간": seg, "상위20%일수": len(hi),
                        "상위20%선행%": round(hi[f"코스피{FORWARD}일선행%"].mean(), 2),
                        "그밖일수": len(rest),
                        "그밖선행%": round(rest[f"코스피{FORWARD}일선행%"].mean(), 2),
                        "차이%p": round(hi[f"코스피{FORWARD}일선행%"].mean()
                                      - rest[f"코스피{FORWARD}일선행%"].mean(), 2)})
    tl = pd.DataFrame(rows_lv)
    print("\n" + tl.to_string(index=False))
    if int(tl["상위20%일수"].sum()) == 0:
        print(f"  상위20% 표본이 0일이다. 판정 불가.")
        print(f"  1998년 이후 30년물이 그때까지 역사의 상위 20%에 든 날이 하루도 없다.")
        print(f"  1995~2000년에 6~7%였기 때문이다. 오늘 {y30.iloc[-1]:.2f}%는 "
              "31년 표본에서 높은 금리가 아니다.")
        print("  '수준이 높다'는 정의는 이 표본으로 검정할 수 없다. 추세 정의만 유효하다.")
    else:
        print(f"  수준 정의 부호 {list(np.sign(tl['차이%p'].fillna(0)).astype(int))}"
              "  (통념대로면 전부 -1)")

    # ---- 통념이 실제로 맞았던 구간 -----------------------------------------
    print("\n" + "=" * 78)
    print("통념이 맞았던 때는 언제인가 (왜 그렇게 알려졌는지)")
    print("=" * 78)
    for lo, hi in (("2021-01-01", "2023-12-31"), ("2024-01-01", "2026-12-31")):
        sub = panel.loc[lo:hi].dropna(subset=["코스피당일%", "30년물일변화bp"])
        c = sub["코스피당일%"].corr(sub["30년물일변화bp"])
        move = (y30.loc[lo:hi].iloc[-1] - y30.loc[lo:hi].iloc[0]) * 100
        ks = (kospi["Close"].loc[lo:hi].iloc[-1] / kospi["Close"].loc[lo:hi].iloc[0] - 1) * 100
        print(f"  {lo[:4]}~{hi[:4]}: 30년물 {move:+.0f}bp, 코스피 {ks:+.1f}%, 당일상관 {c:+.2f}")

    # ---- 돈치안 진입일 기준 -----------------------------------------------
    print("\n" + "=" * 78)
    print("돈치안이 실제로 사는 날에도 같은가")
    print("=" * 78)
    position, entry, exit_, warmup = donchian_position(kospi)
    tr = trades(position, kospi, panel[["30년물3개월변화bp"]]).dropna(subset=["30년물3개월변화bp"])
    tr["구간"] = segment_of(tr["신호일"]).values
    rows_b = []
    for seg in [f"{a}~{b}" for a, b in SEGMENTS]:
        sub = tr[tr["구간"] == seg]
        hot = sub[sub["30년물3개월변화bp"] >= GATE_MAIN]
        rest = sub[sub["30년물3개월변화bp"] < GATE_MAIN]
        rows_b.append({
            "구간": seg, "급등진입n": len(hot), "급등수익%": round(hot["수익%"].mean(), 2),
            "그밖진입n": len(rest), "그밖수익%": round(rest["수익%"].mean(), 2),
            "차이%p": round(hot["수익%"].mean() - rest["수익%"].mean(), 2),
        })
    tb = pd.DataFrame(rows_b)
    print(tb.to_string(index=False))
    judged_b = tb[(tb["급등진입n"] >= VERDICT_MIN_TRADES) & (tb["그밖진입n"] >= VERDICT_MIN_TRADES)]
    pass_b = len(judged_b) == len(SEGMENTS) and bool((judged_b["차이%p"] < 0).all())
    print(f"\n  판정 B 부호 {list(np.sign(tb['차이%p'].fillna(0)).astype(int))}"
          f" -> {'통과' if pass_b else '탈락'}")

    # ---- 질문 4. 전략 ------------------------------------------------------
    print("\n" + "=" * 78)
    print("질문 4. 급등 국면을 피하면 성과가 개선되는가")
    print("=" * 78)
    base_daily, base_held = daily_returns(position, kospi)
    rows = [stats(base_daily, base_held, "돈치안 단독")]
    for gate in GATE_GRID:
        allow = (trend < gate) | trend.isna()
        blocked = F.position_from_entries(entry & allow, exit_, warmup)
        d, h = daily_returns(blocked, kospi)
        rows.append(stats(d, h, f"+{gate:.0f}bp 이상 진입 차단"))
        d2, h2 = daily_returns(position, kospi, pd.Series(np.where(allow, 1.0, 0.5), index=kospi.index))
        rows.append(stats(d2, h2, f"+{gate:.0f}bp 이상 비중 50%"))
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n구간별 기준선 대비 증감 (진입 차단)")
    deltas = []
    for gate in GATE_GRID:
        allow = (trend < gate) | trend.isna()
        blocked = F.position_from_entries(entry & allow, exit_, warmup)
        fd, fh = daily_returns(blocked, kospi)
        parts = []
        for a, b in SEGMENTS:
            mask = (kospi.index >= f"{a}-01-01") & (kospi.index <= f"{b}-12-31")
            bs = stats(base_daily[mask], base_held[mask], "")
            fs = stats(fd[mask], fh[mask], "")
            dc, dm = fs["CAGR%"] - bs["CAGR%"], fs["MDD%"] - bs["MDD%"]
            deltas.append({"임계bp": gate, "구간": f"{a}~{b}", "기준CAGR%": bs["CAGR%"],
                           "필터CAGR%": fs["CAGR%"], "ΔCAGR%p": round(dc, 2),
                           "기준MDD%": bs["MDD%"], "필터MDD%": fs["MDD%"], "ΔMDD%p": round(dm, 2)})
            parts.append(f"{a}~{b} ΔCAGR {dc:+6.2f} ΔMDD {dm:+6.2f}")
        print(f"  +{gate:>3.0f}bp | " + " | ".join(parts))

    dl = pd.DataFrame(deltas)
    main_rows = dl[dl["임계bp"] == GATE_MAIN]
    pass_c = bool((main_rows["ΔCAGR%p"] >= VERDICT_TOL_CAGR).all()
                  and (main_rows["ΔMDD%p"] >= VERDICT_TOL_MDD).all())
    pass_d = bool((dl["ΔCAGR%p"] >= VERDICT_TOL_CAGR).all()
                  and (dl["ΔMDD%p"] >= VERDICT_TOL_MDD).all())

    # ---- 판정 --------------------------------------------------------------
    print("\n" + "=" * 78)
    print("판정 (기준은 실행 전에 고정했다)")
    print("=" * 78)
    checks = [
        (f"A. 급등 국면 {FORWARD}일 선행 수익이 세 구간 모두 더 낮다", pass_a),
        ("B. 돈치안 진입일 기준으로도 세 구간 모두 더 낮다", pass_b),
        (f"C. +{GATE_MAIN:.0f}bp 차단의 ΔCAGR·ΔMDD가 세 구간 모두 악화 아님", pass_c),
        ("D. 임계 " + "·".join(f"+{g:.0f}" for g in GATE_GRID) + "bp 전부에서 C 성립", pass_d),
    ]
    for name, ok in checks:
        print(f"  {'O' if ok else 'X'}  {name}")
    print("\n  결론: " + ("채택 후보 - 다중검정 보정 후 재판단"
                        if all(ok for _, ok in checks) else "폐기 - 구간별 부호가 일관되지 않는다"))

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    for frame, name in ((panel.reset_index(), "panel"), (ta, "forward"),
                        (tb, "trades"), (pd.DataFrame(rows), "strategy"), (dl, "segments")):
        path = outdir / f"bond_regime_{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        print(path)


if __name__ == "__main__":
    main()
