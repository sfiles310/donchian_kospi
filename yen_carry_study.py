# -*- coding: utf-8 -*-
"""엔 캐리 청산이 코스피를 누르는지 검정한다.

발단은 2026년 7월 말이다. USD/JPY가 7/28 163.86에서 8/3 157.53으로 나흘 만에
3.9% 떨어졌고(엔 강세) 같은 기간 코스피가 6,755에서 5,663으로 무너졌다. 정렬이
맞아 보였다. 눈으로 본 것을 30년 표본에 대고 확인한다.

가설은 캐리 청산이다. 엔을 빌려 위험자산을 산 자금이 엔 강세에 되감기면서 신흥국
주식을 판다는 것이다. 그렇다면 엔 강세는 코스피 하락에 선행하거나 최소한 동행해야
한다.

앞선 두 검정에서 배운 것을 적용한다.

- 밤사이 충격은 갭을 설명하고 갭 이후를 설명하지 못한다(2026-08-19). 그래서 여기서도
  갭과 갭 이후를 갈라서 본다. 둘을 합치면 설명력이 있는 것처럼 보인다
- 30년물은 통념과 부호가 반대였고 최근 구간에서만 맞았다(2026-08-20). 그래서 여기서도
  전체 평균이 아니라 구간별 부호로 판정한다

자료 하자가 하나 있다. FDR의 USD/JPY는 금요일 행이 649개인데 일요일 행이 900개다.
금요일 종가가 일요일로 라벨된 주가 많다. 그래서 변화율을 원본에서 계산하지 않고
코스피 거래일에 strictly_prev로 먼저 정렬한 뒤 계산한다.

    python yen_carry_study.py

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
FAST = 5                   # 캐리 청산은 몇 달이 아니라 며칠에 걸쳐 일어난다
SLOW = 21
FORWARD = 21
EPISODE_GAP_DAYS = 30
BOOTSTRAP_DRAWS = 20000
SEED = 20260820

# USD/JPY 5일 변화% 구간. 내려가면 엔 강세다
JPY_BUCKETS = [-100, -3, -1.5, 1.5, 3, 100]
JPY_LABELS = ["엔 3%+ 강세", "엔 1.5~3% 강세", "횡보", "엔 1.5~3% 약세", "엔 3%+ 약세"]

# ---- 판정 기준. 돌리기 전에 고정한다 -------------------------------------
GATE_MAIN = -2.0           # 5일 USD/JPY 변화가 이 아래면 캐리 청산 국면으로 본다
GATE_GRID = (-1.5, -2.0, -3.0)
VERDICT_MIN_DAYS = 150
VERDICT_MIN_TRADES = 5
VERDICT_TOL_CAGR = -0.10   # %p
VERDICT_TOL_MDD = -0.50    # %p
# 통과 조건 (넷 다 만족해야 채택)
#   A. 엔 강세 국면의 코스피 21일 선행 수익이 그 밖보다 낮다, 세 구간 모두
#   B. 돈치안 진입일 기준으로도 같은 부호, 세 구간 모두
#   C. 엔 강세 국면 회피의 ΔCAGR·ΔMDD가 세 구간 모두 악화가 아니다
#   D. 임계 -1.5/-2/-3% 전부에서 C가 유지된다


def load_kospi(start: str) -> pd.DataFrame:
    import FinanceDataReader as fdr

    frame = fdr.DataReader("KS11", start)
    frame.index = pd.to_datetime(frame.index)
    return frame.dropna(subset=["Open", "High", "Low", "Close"])


def strictly_prev(series: pd.Series, target: pd.DatetimeIndex) -> pd.Series:
    """한국 t일 08:50에 아는 봉은 라벨이 t보다 엄격히 앞선 마지막 봉이다.

    FX는 24시간 돌기 때문에 라벨이 t인 봉은 그 시점에 아직 진행 중이다.
    주말 라벨이 섞여 있어도 이 규칙 아래에서는 안전하다. 일요일 라벨 값은
    금요일 종가이거나 일요일 개장가이고 둘 다 월요일 08:50 이전에 존재한다.
    """
    pos = series.index.searchsorted(target, side="left") - 1
    out = pd.Series(np.nan, index=target)
    ok = pos >= 0
    out[ok] = series.values[pos[ok]]
    return out


def load_fx(target: pd.DatetimeIndex) -> pd.DataFrame:
    import FinanceDataReader as fdr

    out = {}
    for name, code in (("USDJPY", "USD/JPY"), ("JPYKRW", "JPY/KRW"), ("닛케이", "^N225")):
        try:
            frame = fdr.DataReader(code, "1990-01-01")
        except Exception as exc:
            print(f"  {name}({code}) 수신 실패: {type(exc).__name__}")
            continue
        series = (frame["Close"] if "Close" in frame else frame.iloc[:, 0]).dropna()
        series.index = pd.to_datetime(series.index)
        out[name] = strictly_prev(series, target)
    return pd.DataFrame(out, index=target)


def donchian_position(frame: pd.DataFrame):
    table = pd.DataFrame(
        {"index_high": frame["High"].values, "index_low": frame["Low"].values}
    )
    entry, exit_, warmup = F.donchian_parts(table)
    for series in (entry, exit_, warmup):
        series.index = frame.index
    return F.position_from_entries(entry, exit_, warmup), entry, exit_, warmup


def trades(position: pd.Series, frame: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    pos = position.to_numpy()
    open_ = frame["Open"].to_numpy()
    rows, i = [], 0
    while i < len(pos) - 1:
        if pos[i] == 1 and (i == 0 or pos[i - 1] == 0):
            j = i
            while j < len(pos) - 1 and pos[j] == 1:
                j += 1
            if j >= len(pos) - 1:
                break
            row = {"신호일": position.index[i], "보유일": j - i,
                   "수익%": (open_[j + 1] / open_[i + 1] - 1) * 100}
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
    gaps = pd.Series(values.index).diff().dt.days.fillna(10 ** 6)
    episode = (gaps.values > EPISODE_GAP_DAYS).cumsum()
    means = values.groupby(episode).mean()
    rng = np.random.default_rng(SEED)
    draws = rng.choice(means.values, size=(BOOTSTRAP_DRAWS, len(means)), replace=True).mean(axis=1)
    return {
        "에피소드": len(means), "일수": len(values),
        "에피소드평균의평균%": round(float(means.mean()), 2),
        "음수에피소드": f"{int((means < 0).sum())}/{len(means)}",
        "95%하한": round(float(np.percentile(draws, 2.5)), 2),
        "95%상한": round(float(np.percentile(draws, 97.5)), 2),
        "평균이 0보다 작을 확률": round(float((draws < 0).mean()), 3),
    }


def daily_returns(position: pd.Series, frame: pd.DataFrame,
                  weight: pd.Series | None = None):
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
    parser = argparse.ArgumentParser(description="엔 캐리 청산이 코스피에 주는 영향")
    parser.add_argument("--start", default=START)
    parser.add_argument("--outdir", default="output")
    args = parser.parse_args()

    kospi = load_kospi(args.start)
    print(f"코스피 {kospi.index[0].date()} ~ {kospi.index[-1].date()}, {len(kospi)}일")
    print("환율 자료 수신")
    fx = load_fx(kospi.index)
    jpy = fx["USDJPY"]
    first = jpy.first_valid_index()
    print(f"  USD/JPY 유효 시작 {first.date()} - 1995~1996은 표본 밖이다")
    print(f"  결측: " + ", ".join(f"{c} {int(fx[c].isna().sum())}일" for c in fx))

    # 정렬 뒤에 변화율을 계산한다. 원본 캘린더에는 주말 라벨이 섞여 있다
    fast = (jpy / jpy.shift(FAST) - 1) * 100
    slow = (jpy / jpy.shift(SLOW) - 1) * 100
    d1 = (jpy / jpy.shift(1) - 1) * 100

    ret1 = kospi["Close"].pct_change() * 100
    gap = (kospi["Open"] / kospi["Close"].shift(1) - 1) * 100
    after_gap = (kospi["Close"] / kospi["Open"] - 1) * 100
    fwd = (kospi["Close"].shift(-FORWARD) / kospi["Close"] - 1) * 100

    panel = pd.DataFrame({
        "USDJPY": jpy, f"엔{FAST}일변화%": fast, f"엔{SLOW}일변화%": slow,
        "엔일변화%": d1, "코스피당일%": ret1, "코스피갭%": gap,
        "코스피갭이후%": after_gap, f"코스피{FORWARD}일선행%": fwd,
    })
    panel["구간"] = segment_of(panel.index).values
    panel["엔구간"] = pd.cut(fast, JPY_BUCKETS, labels=JPY_LABELS)

    print(f"\n최근 상태: USD/JPY {jpy.iloc[-1]:.2f}, "
          f"{FAST}일 {fast.iloc[-1]:+.2f}%, {SLOW}일 {slow.iloc[-1]:+.2f}%")

    # ---- 발단이 된 국면 ----------------------------------------------------
    print("\n" + "=" * 78)
    print("발단: 2026년 7~8월 실제로 무슨 일이 있었는가")
    print("=" * 78)
    window = panel.loc["2026-07-24":"2026-08-19",
                       ["USDJPY", f"엔{FAST}일변화%", "코스피당일%", "코스피갭%", "코스피갭이후%"]]
    print(window.round(2).to_string())

    # ---- 갭과 갭 이후를 가른다 ---------------------------------------------
    print("\n" + "=" * 78)
    print("엔 변화가 무엇을 설명하는가 (갭 대 갭 이후)")
    print("=" * 78)
    valid = panel.dropna(subset=["엔일변화%", "코스피갭%"])
    corr = pd.DataFrame({
        "코스피 갭 상관": [valid["코스피갭%"].corr(valid["엔일변화%"]),
                     valid["코스피갭%"].corr(valid[f"엔{FAST}일변화%"])],
        "갭 이후 상관": [valid["코스피갭이후%"].corr(valid["엔일변화%"]),
                    valid["코스피갭이후%"].corr(valid[f"엔{FAST}일변화%"])],
        f"{FORWARD}일 선행 상관": [valid[f"코스피{FORWARD}일선행%"].corr(valid["엔일변화%"]),
                            valid[f"코스피{FORWARD}일선행%"].corr(valid[f"엔{FAST}일변화%"])],
    }, index=["엔 일변화", f"엔 {FAST}일변화"])
    print(corr.round(3).to_string())
    print("\n부호에 주의한다. USD/JPY가 오르면 엔 약세다. 캐리 가설대로면")
    print("엔 약세(양수)에 코스피가 오르므로 상관은 양수여야 한다.")

    print("\n연도별 갭 상관 (부호가 유지되는지)")
    yearly = valid.groupby(valid.index.year).apply(
        lambda g: g["코스피갭%"].corr(g["엔일변화%"]) if len(g) > 50 else np.nan).dropna()
    line = [f"{y}:{v:+.2f}" for y, v in yearly.items()]
    for i in range(0, len(line), 8):
        print("  " + "  ".join(line[i:i + 8]))
    print(f"  양수 {int((yearly > 0).sum())}년 / 음수 {int((yearly < 0).sum())}년")

    # ---- 판정 A ------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"판정 A. 엔 강세 국면이 코스피 {FORWARD}일 선행 수익을 예측하는가")
    print("=" * 78)
    fv = panel.dropna(subset=[f"엔{FAST}일변화%", f"코스피{FORWARD}일선행%"])
    print("[전체]")
    print(fv.groupby("엔구간", observed=True).agg(
        일수=(f"코스피{FORWARD}일선행%", "size"),
        평균선행=(f"코스피{FORWARD}일선행%", "mean"),
        중앙=(f"코스피{FORWARD}일선행%", "median"),
        승률=(f"코스피{FORWARD}일선행%", lambda x: (x > 0).mean() * 100),
        평균당일=("코스피당일%", "mean"),
    ).round(2).to_string())

    print(f"\n[구간별] 엔 강세 국면({FAST}일 {GATE_MAIN:+.1f}% 이하) 대 그 밖")
    rows_a = []
    for seg in [f"{a}~{b}" for a, b in SEGMENTS]:
        sub = fv[fv["구간"] == seg]
        hot = sub[sub[f"엔{FAST}일변화%"] <= GATE_MAIN]
        rest = sub[sub[f"엔{FAST}일변화%"] > GATE_MAIN]
        rows_a.append({
            "구간": seg, "엔강세일수": len(hot),
            "엔강세선행%": round(hot[f"코스피{FORWARD}일선행%"].mean(), 2),
            "그밖일수": len(rest),
            "그밖선행%": round(rest[f"코스피{FORWARD}일선행%"].mean(), 2),
            "차이%p": round(hot[f"코스피{FORWARD}일선행%"].mean()
                          - rest[f"코스피{FORWARD}일선행%"].mean(), 2)})
    ta = pd.DataFrame(rows_a)
    print(ta.to_string(index=False))
    judged = ta[(ta["엔강세일수"] >= VERDICT_MIN_DAYS) & (ta["그밖일수"] >= VERDICT_MIN_DAYS)]
    pass_a = len(judged) == len(SEGMENTS) and bool((judged["차이%p"] < 0).all())
    print(f"\n  판정 A 부호 {list(np.sign(ta['차이%p'].fillna(0)).astype(int))}"
          f" -> {'통과' if pass_a else '탈락'}  (캐리 가설대로면 전부 -1)")

    hot_all = fv[fv[f"엔{FAST}일변화%"] <= GATE_MAIN][f"코스피{FORWARD}일선행%"]
    if len(hot_all) >= 50:
        print("\n엔 강세 국면의 에피소드 단위 검정")
        for key, value in episode_bootstrap(hot_all).items():
            print(f"  {key}: {value}")

    # ---- 판정 B ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("판정 B. 돈치안이 실제로 사는 날에도 같은가")
    print("=" * 78)
    position, entry, exit_, warmup = donchian_position(kospi)
    tr = trades(position, kospi, panel[[f"엔{FAST}일변화%"]]).dropna(subset=[f"엔{FAST}일변화%"])
    tr["구간"] = segment_of(tr["신호일"]).values
    rows_b = []
    for seg in [f"{a}~{b}" for a, b in SEGMENTS]:
        sub = tr[tr["구간"] == seg]
        hot = sub[sub[f"엔{FAST}일변화%"] <= GATE_MAIN]
        rest = sub[sub[f"엔{FAST}일변화%"] > GATE_MAIN]
        rows_b.append({"구간": seg, "엔강세진입n": len(hot),
                       "엔강세수익%": round(hot["수익%"].mean(), 2),
                       "그밖진입n": len(rest), "그밖수익%": round(rest["수익%"].mean(), 2),
                       "차이%p": round(hot["수익%"].mean() - rest["수익%"].mean(), 2)})
    tb = pd.DataFrame(rows_b)
    print(tb.to_string(index=False))
    judged_b = tb[(tb["엔강세진입n"] >= VERDICT_MIN_TRADES) & (tb["그밖진입n"] >= VERDICT_MIN_TRADES)]
    pass_b = len(judged_b) == len(SEGMENTS) and bool((judged_b["차이%p"] < 0).all())
    print(f"\n  판정 B 부호 {list(np.sign(tb['차이%p'].fillna(0)).astype(int))}"
          f" -> {'통과' if pass_b else '탈락'}")

    # ---- 판정 C·D ----------------------------------------------------------
    print("\n" + "=" * 78)
    print("판정 C·D. 엔 강세 국면을 피하면 성과가 개선되는가")
    print("=" * 78)
    base_daily, base_held = daily_returns(position, kospi)
    rows = [stats(base_daily, base_held, "돈치안 단독")]
    for gate in GATE_GRID:
        allow = (fast > gate) | fast.isna()
        blocked = F.position_from_entries(entry & allow, exit_, warmup)
        d, h = daily_returns(blocked, kospi)
        rows.append(stats(d, h, f"엔 {gate:+.1f}% 이하 진입 차단"))
        d2, h2 = daily_returns(position, kospi,
                               pd.Series(np.where(allow, 1.0, 0.5), index=kospi.index))
        rows.append(stats(d2, h2, f"엔 {gate:+.1f}% 이하 비중 50%"))
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n구간별 기준선 대비 증감 (진입 차단)")
    deltas = []
    for gate in GATE_GRID:
        allow = (fast > gate) | fast.isna()
        blocked = F.position_from_entries(entry & allow, exit_, warmup)
        fd, fh = daily_returns(blocked, kospi)
        parts = []
        for a, b in SEGMENTS:
            mask = (kospi.index >= f"{a}-01-01") & (kospi.index <= f"{b}-12-31")
            bs = stats(base_daily[mask], base_held[mask], "")
            fs = stats(fd[mask], fh[mask], "")
            dc, dm = fs["CAGR%"] - bs["CAGR%"], fs["MDD%"] - bs["MDD%"]
            deltas.append({"임계%": gate, "구간": f"{a}~{b}", "기준CAGR%": bs["CAGR%"],
                           "필터CAGR%": fs["CAGR%"], "ΔCAGR%p": round(dc, 2),
                           "기준MDD%": bs["MDD%"], "필터MDD%": fs["MDD%"], "ΔMDD%p": round(dm, 2)})
            parts.append(f"{a}~{b} ΔCAGR {dc:+6.2f} ΔMDD {dm:+6.2f}")
        print(f"  {gate:+.1f}% | " + " | ".join(parts))

    dl = pd.DataFrame(deltas)
    main_rows = dl[dl["임계%"] == GATE_MAIN]
    pass_c = bool((main_rows["ΔCAGR%p"] >= VERDICT_TOL_CAGR).all()
                  and (main_rows["ΔMDD%p"] >= VERDICT_TOL_MDD).all())
    pass_d = bool((dl["ΔCAGR%p"] >= VERDICT_TOL_CAGR).all()
                  and (dl["ΔMDD%p"] >= VERDICT_TOL_MDD).all())

    print("\n" + "=" * 78)
    print("판정 (기준은 실행 전에 고정했다)")
    print("=" * 78)
    checks = [
        (f"A. 엔 강세 국면 {FORWARD}일 선행 수익이 세 구간 모두 더 낮다", pass_a),
        ("B. 돈치안 진입일 기준으로도 세 구간 모두 더 낮다", pass_b),
        (f"C. {GATE_MAIN:+.1f}% 차단의 ΔCAGR·ΔMDD가 세 구간 모두 악화 아님", pass_c),
        ("D. 임계 " + "·".join(f"{g:+.1f}" for g in GATE_GRID) + "% 전부에서 C 성립", pass_d),
    ]
    for name, ok in checks:
        print(f"  {'O' if ok else 'X'}  {name}")
    print("\n  결론: " + ("채택 후보 - 다중검정 보정 후 재판단"
                        if all(ok for _, ok in checks) else "폐기 - 구간별 부호가 일관되지 않는다"))

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    for frame, name in ((panel.reset_index(), "panel"), (ta, "forward"),
                        (tb, "trades"), (pd.DataFrame(rows), "strategy"), (dl, "segments")):
        path = outdir / f"yen_carry_{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        print(path)


if __name__ == "__main__":
    main()
