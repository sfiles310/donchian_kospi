# -*- coding: utf-8 -*-
"""밤사이 해외 충격으로 다음날 아침 매수를 조절할 수 있는지 검정한다.

발단은 2026-08-18이다. 그날 코스피는 장중 20일 상단을 뚫어 돈치안 BUY를 냈지만
시가 대비 -3.62%로 밀려 마감했고, 그날 밤 미국에서 반도체가 -4.98%, 미국 상장
한국 ETF(EWY)가 -8.13% 빠졌다. 다음날 코스피는 -4.96% 갭하락으로 열렸다.
규칙은 그 시가에 사라고 했다. 사는 것이 맞았는가.

세 가지를 차례로 묻는다.

1. 밤사이 해외 충격을 08:50에 알 수 있는가     → 자료 가용성과 룩어헤드
2. 그 충격이 그날 이후 수익을 예측하는가         → 예측력
3. 예측한다면 비중을 얼마로 줄여야 하는가        → 사이징

결론부터 적어 둔다. 2번의 답이 아니오였다. 해외 충격은 그날 시가 갭을 거의 다
설명하지만(상관 0.69) 그 시가에 사서 다음 시가에 파는 수익은 설명하지 못한다
(상관 -0.07). 오히려 충격이 컸던 날의 수익이 더 좋았다. 뉴스가 갭에서 이미
정산되고 그 뒤는 반등이기 때문이다.

대신 3번에서 다른 답이 나왔다. 밤사이 무슨 일이 있었는지가 아니라 코스피 자신의
실현변동성이 사이즈를 정하는 유일하게 안정적인 근거였다.

    python overnight_gate_study.py

로컬 전용 연구 스크립트다. 매일 돌리는 알림 경로에는 관여하지 않는다.
해외 지수는 KIS·KRX 관할이 아니므로 FinanceDataReader에서 받는다. 코스피도 같은
경로로 받아 비교 기준을 하나로 맞췄다. 실제 운용 수치는 ETF 시초가로 다시 재야 한다.
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
COST_BPS = 10.0          # 비중이 1 바뀔 때마다 부과. 왕복이면 20bp
VOL_WINDOW = 20
MIN_ENTRIES = 10         # 목표변동성 중앙값을 믿기 시작하는 최소 진입 횟수
SHOCK_BUCKETS = [-100, -5, -3, -1.5, 1.5, 100]
SHOCK_LABELS = ["-5 이하", "-5~-3", "-3~-1.5", "-1.5~+1.5", "+1.5 이상"]
EPISODE_GAP_DAYS = 30    # 이만큼 떨어지면 별개 사건으로 센다
BOOTSTRAP_DRAWS = 20000
SEED = 20260819

# 08:50에 값이 확정되어 있는 것만 남겼다. 무엇을 왜 뺐는지는 아래 주석에 적는다.
OVERSEAS = {
    "S&P500": "US500",
    "나스닥100": "^NDX",
    "반도체SOX": "^SOX",
    "한국ETF EWY": "EWY",
    "VIX변화": "VIX",
    "브렌트": "BZ=F",
    "달러지수": "^NYICDX",
}
# 뺀 것:
#   USD/KRW      당일 종가가 08:50에 없다. 캘린더도 주말 행이 섞여 어긋난다
#   FRED:*       공표가 늦다. DCOILBRENTEU는 6영업일까지 밀린다. 과거 연구용으로만
#   US10YT=X     FDR에서 404
#   DX=F         FDR에서 404. 달러지수는 ^NYICDX를 쓴다
#   ES/NQ 선물   현재가는 알아도 과거 08:50 스냅숏 이력이 없다


def load_kospi(start: str) -> pd.DataFrame:
    import FinanceDataReader as fdr

    frame = fdr.DataReader("KS11", start)
    frame.index = pd.to_datetime(frame.index)
    return frame.dropna(subset=["Open", "High", "Low", "Close"])


def strictly_prev(series: pd.Series, target: pd.DatetimeIndex) -> pd.Series:
    """한국 t일 08:50에 아는 해외 봉은 라벨이 t보다 엄격히 앞선 마지막 봉이다.

    날짜로 그냥 붙이면 안 된다. 미국 t일 세션은 한국 t일 밤 22:30에 시작하므로
    같은 라벨의 봉은 그 시점에 아직 존재하지 않는다. 그런데도 코스피 거래일의
    96.6%에는 같은 라벨의 미국 봉이 자료에 들어 있다. 그대로 merge하면 그만큼이
    전부 룩어헤드다.
    """
    pos = series.index.searchsorted(target, side="left") - 1
    out = pd.Series(np.nan, index=target)
    ok = pos >= 0
    out[ok] = series.values[pos[ok]]
    return out


def overnight_returns(target: pd.DatetimeIndex, start: str) -> pd.DataFrame:
    import FinanceDataReader as fdr

    columns = {}
    for name, code in OVERSEAS.items():
        try:
            frame = fdr.DataReader(code, start)
        except Exception as exc:  # 티커가 사라지는 일이 있다. 침묵시키지 않는다
            print(f"  {name}({code}) 수신 실패: {type(exc).__name__}")
            continue
        series = (frame["Close"] if "Close" in frame else frame.iloc[:, 0]).dropna()
        series.index = pd.to_datetime(series.index)
        columns[name] = strictly_prev(series.pct_change() * 100, target)
    return pd.DataFrame(columns, index=target)


def donchian_only(frame: pd.DataFrame) -> pd.Series:
    """진입은 돈치안 상단 돌파만 쓴다.

    FTD를 뺀 이유는 검정 결과에 있다. 1995~2007에서 FTD 결합은 MDD를 -36.6%에서
    -54.9%로 키웠고 2014년 이후에서만 도움이 됐다. 구간에 따라 부호가 뒤집히는
    규칙은 이 스크립트의 기준선으로 쓰지 않는다.
    """
    table = pd.DataFrame(
        {
            "date": frame.index,
            "index_open": frame["Open"], "index_high": frame["High"],
            "index_low": frame["Low"], "index_close": frame["Close"],
            "volume": frame["Volume"],
        }
    ).reset_index(drop=True)
    entry, exit_, warmup = F.donchian_parts(table)
    position = F.position_from_entries(entry, exit_, warmup)
    position.index = frame.index
    return position


def expanding_target_vol(vol: pd.Series, position: pd.Series) -> pd.Series:
    """목표변동성은 지금까지 일어난 진입일들의 직전 변동성 중앙값이다.

    전체 표본에서 한 번에 구하면 미래를 보고 정하는 것이 된다. 그 시점까지 실제로
    일어난 진입만 쓰고 오늘 판단에는 어제까지의 중앙값만 쓴다.
    """
    entry_day = (position == 1) & (position.shift(1) == 0)
    at_entry = vol.where(entry_day.shift(1).fillna(False))
    return at_entry.expanding(min_periods=MIN_ENTRIES).median().ffill().shift(1).ffill()


def measure(weight: pd.Series, held: pd.Series, ret: pd.Series,
            index: pd.DatetimeIndex, label: str) -> dict:
    """비중이 매일 바뀌므로 비용은 진입·청산이 아니라 절대 변화량마다 부과한다."""
    w = (held * weight.fillna(1.0).clip(0, 1)).fillna(0)
    daily = (w * ret).fillna(0) - w.diff().abs().fillna(0) * (COST_BPS / 10000)
    equity = (1 + daily).cumprod()
    years = (index[-1] - index[0]).days / 365.25
    live = daily[held.astype(bool)]
    return {
        "전략": label,
        "CAGR%": round((equity.iloc[-1] ** (1 / years) - 1) * 100, 2),
        "MDD%": round((equity / equity.cummax() - 1).min() * 100, 2),
        "CVaR5%": round(live[live <= live.quantile(0.05)].mean() * 100, 2),
        "샤프": round(daily.mean() / daily.std() * np.sqrt(252), 3),
        "평균노출": f"{w.mean() * 100:.1f}%",
        "최종배수": round(equity.iloc[-1], 1),
    }


def episode_bootstrap(shock_returns: pd.Series) -> dict:
    """일별 표본을 그대로 세면 안 된다. 충격은 위기마다 뭉쳐서 온다.

    30일 이상 떨어진 것만 별개 사건으로 세고 사건 평균들로 부트스트랩한다.
    """
    gaps = pd.Series(shock_returns.index).diff().dt.days.fillna(10 ** 6)
    episode = (gaps.values > EPISODE_GAP_DAYS).cumsum()
    means = shock_returns.groupby(episode).mean()
    rng = np.random.default_rng(SEED)
    draws = rng.choice(means.values, size=(BOOTSTRAP_DRAWS, len(means)), replace=True).mean(axis=1)
    return {
        "에피소드 수": len(means),
        "일수": len(shock_returns),
        "에피소드 평균의 평균%": round(float(means.mean()), 3),
        "양수 에피소드": f"{int((means > 0).sum())}/{len(means)}",
        "95% 하한%": round(float(np.percentile(draws, 2.5)), 3),
        "95% 상한%": round(float(np.percentile(draws, 97.5)), 3),
        "평균이 0보다 클 확률": round(float((draws > 0).mean()), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="밤사이 해외 충격의 사이징 가치 검정")
    parser.add_argument("--start", default=START)
    parser.add_argument("--outdir", default="output")
    args = parser.parse_args()

    kospi = load_kospi(args.start)
    print(f"코스피 {kospi.index[0].date()} ~ {kospi.index[-1].date()}, {len(kospi)}일\n")

    position = donchian_only(kospi)
    held = position.shift(1).fillna(0).astype(int)      # 오늘 시가에 이미 들고 있는가
    ret = kospi["Open"].shift(-1) / kospi["Open"] - 1   # 오늘 시가에서 다음 시가까지
    gap = (kospi["Open"] / kospi["Close"].shift(1) - 1) * 100
    vol = (kospi["Close"].pct_change() * 100).rolling(VOL_WINDOW).std().shift(1)

    print("해외 자료 수신")
    overseas = overnight_returns(kospi.index, "1990-01-01")

    panel = overseas.copy()
    panel["보유"] = held.astype(bool)
    panel["수익%"] = ret * 100
    panel["갭%"] = gap
    panel["직전20일변동성%"] = vol

    print("\n" + "=" * 78)
    print("질문 2. 밤사이 충격이 무엇을 설명하는가 (돈치안 보유일 한정)")
    print("=" * 78)
    live = panel[panel["보유"]].dropna(subset=["수익%"])
    names = [c for c in OVERSEAS if c in panel]
    compare = pd.DataFrame({
        "당일 갭 상관": panel[names + ["갭%"]].corr()["갭%"].drop("갭%"),
        "시가에서 익일시가 상관": live[names + ["수익%"]].corr()["수익%"].drop("수익%"),
    }).sort_values("당일 갭 상관", ascending=False)
    print(compare.round(3).to_string())
    print("\n갭은 잘 설명하고 그 다음 수익은 설명하지 못한다. 뉴스가 갭에서 정산된다.")

    for var in ("한국ETF EWY", "S&P500"):
        if var not in live:
            continue
        sub = live.dropna(subset=[var]).copy()
        sub["구간"] = pd.cut(sub[var], SHOCK_BUCKETS, labels=SHOCK_LABELS)
        table = sub.groupby("구간", observed=True).agg(
            일수=("수익%", "size"), 평균수익=("수익%", "mean"), 중앙=("수익%", "median"),
            CVaR5=("수익%", lambda x: x[x <= x.quantile(0.05)].mean() if len(x) >= 20 else np.nan),
            승률=("수익%", lambda x: (x > 0).mean() * 100), 평균갭=("갭%", "mean"),
        )
        print(f"\n[{var}] 구간별 시가에서 익일시가")
        print(table.round(2).to_string())

    if "한국ETF EWY" in live:
        shock = live[live["한국ETF EWY"] <= -5]["수익%"]
        if len(shock) >= 10:
            print("\n충격일(EWY -5% 이하)의 에피소드 단위 검정")
            for key, value in episode_bootstrap(shock).items():
                print(f"  {key}: {value}")
            print("  신뢰구간이 0을 포함하면 사이징 근거로 쓸 수 없다.")

    print("\n" + "=" * 78)
    print("질문 3. 비중을 무엇으로 정할 것인가")
    print("=" * 78)
    ones = pd.Series(1.0, index=kospi.index)
    shock_weight = pd.Series(1.0, index=kospi.index)
    if "한국ETF EWY" in panel:
        ewy = panel["한국ETF EWY"]
        shock_weight[ewy <= -3] = 0.75
        shock_weight[ewy <= -5] = 0.50
        shock_weight = shock_weight.where(ewy.notna(), 1.0)
    target = expanding_target_vol(vol, position)
    vol_weight = (target / vol).clip(0, 1)

    rows = [
        measure(ones, held, ret, kospi.index, "기준 100%"),
        measure(shock_weight, held, ret, kospi.index, "EWY 충격 축소"),
        measure(vol_weight, held, ret, kospi.index, "변동성타겟 (확장중앙값)"),
        measure(vol_weight * shock_weight, held, ret, kospi.index, "둘 다"),
    ]
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n구간별 안정성")
    for start_year, end_year in (("1995", "2007"), ("2008", "2013"), ("2014", "2026")):
        mask = (kospi.index >= f"{start_year}-01-01") & (kospi.index <= f"{end_year}-12-31")
        for weight, label in ((ones, "기준      "), (vol_weight, "변동성타겟")):
            w = (held * weight.fillna(1).clip(0, 1)).fillna(0)
            s = ((w * ret).fillna(0) - w.diff().abs().fillna(0) * (COST_BPS / 10000))[mask]
            eq = (1 + s).cumprod()
            print(f"  {start_year}~{end_year} {label}: "
                  f"CAGR {(eq.iloc[-1] ** (252 / len(s)) - 1) * 100:6.2f}%"
                  f"  MDD {(eq / eq.cummax() - 1).min() * 100:7.2f}%"
                  f"  샤프 {s.mean() / s.std() * np.sqrt(252):5.3f}")

    print("\n" + "=" * 78)
    print("오늘 적용값")
    print("=" * 78)
    print(f"  직전 {VOL_WINDOW}일 실현변동성 : {vol.iloc[-1]:.2f}%")
    print(f"  목표변동성 (확장중앙값)  : {target.iloc[-1]:.2f}%")
    print(f"  기준 포지션 대비 비중    : {min(1.0, target.iloc[-1] / vol.iloc[-1]) * 100:.1f}%")
    if "한국ETF EWY" in panel:
        print(f"  참고로 어젯밤 EWY는 {panel['한국ETF EWY'].iloc[-1]:.2f}%. 비중에는 쓰지 않는다")

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    panel.assign(비중=vol_weight).to_csv(outdir / "overnight_gate_panel.csv", encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(outdir / "overnight_gate_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\n{outdir / 'overnight_gate_panel.csv'}")
    print(f"{outdir / 'overnight_gate_summary.csv'}")


if __name__ == "__main__":
    main()
