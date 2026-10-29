# -*- coding: utf-8 -*-
"""돈치안·FTD 결합 규칙을 미국 지수(S&P500, 나스닥100)에서 검정한다.

코스피 표본은 12.6년에 FTD가 6~14회뿐이라 결론이 약하다. FTD는 원래 미국 지수에서
나온 규칙이므로 여기서 같은 규칙이 서는지 보는 것이 사실상 아웃오브샘플 검정이 된다.
1990년부터 36년치라 표본도 코스피의 세 배쯤 된다.

`ftd_study.py`와 판정 규칙·성과 계산을 그대로 공유한다. 다른 것은 자료원뿐이다.
코스피는 KIS 패널 DB에서, 여기서는 FinanceDataReader에서 받는다.

    python us_index_study.py                      # S&P500·나스닥100 둘 다
    python us_index_study.py --index US500 --etf SPY

주의: 나스닥100은 코드 `NDX`가 아니라 `^NDX`를 써야 한다. `NDX`는 거래량이 전부
0으로 와서 FTD 판정이 불가능하다.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from etf_replication_study import open_to_open, summarize
from gate_study import apply_costs
from ftd_study import (
    CORRECTION_DROP,
    CORRECTION_LOOKBACK,
    FIRST_TEST_DAY,
    LAST_TEST_DAY,
    VOLUME_MA,
    donchian_parts,
    entry_lag,
    evaluate_false_signals,
    ftd_signals,
    position_from_entries,
)

DEFAULT_COST_BPS = 10.0
DEFAULT_START = "1990-01-01"
DEFAULT_SPREAD_BPS = 150.0   # 기준금리에 얹는 조달 스프레드. 레버리지 ETF 총보수가 이 언저리다
FALLBACK_RATE = 3.0          # FRED를 못 받을 때 쓰는 연 조달 기준금리%
LEVERAGE_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]
# 신호는 지수로 내고 수익은 실제 매매 대상인 ETF로 잰다. ftd_study.py와 같은 방식이다.
DEFAULT_MARKETS = [("US500", "SPY", "S&P500"), ("^NDX", "QQQ", "나스닥100")]
THRESHOLDS = [1.0, 1.2, 1.5, 2.0, 2.5]


def load_market(index_code: str, etf_code: str, start: str) -> pd.DataFrame:
    """지수 OHLCV와 ETF 시가를 거래일 기준으로 맞춘다."""
    import FinanceDataReader as fdr

    index = fdr.DataReader(index_code, start)
    if index.empty:
        raise SystemExit(f"{index_code} 지수 자료를 받지 못했습니다.")
    if "Volume" not in index or (index["Volume"] > 0).sum() == 0:
        raise SystemExit(f"{index_code}는 거래량이 없어 FTD 판정을 할 수 없습니다.")

    frame = index[["Open", "High", "Low", "Close", "Volume"]].rename(
        columns={
            "Open": "index_open",
            "High": "index_high",
            "Low": "index_low",
            "Close": "index_close",
            "Volume": "volume",
        }
    )

    etf = fdr.DataReader(etf_code, start)
    if etf.empty:
        raise SystemExit(f"{etf_code} 가격 자료를 받지 못했습니다.")
    frame = frame.join(etf[["Open"]].rename(columns={"Open": "etf_open"}), how="inner")

    frame = frame[frame["volume"] > 0].dropna()
    frame = frame.reset_index()
    frame["date"] = pd.to_datetime(frame.iloc[:, 0]).dt.strftime("%Y-%m-%d")
    return frame[
        ["date", "index_open", "index_high", "index_low", "index_close", "volume", "etf_open"]
    ]


def load_policy_rate(dates: pd.Series) -> pd.Series:
    """실효 연방기금금리(연%)를 거래일에 맞춘다. 못 받으면 상수로 대체한다."""
    import FinanceDataReader as fdr

    index = pd.to_datetime(dates)
    try:
        rate = fdr.DataReader("FRED:DFF", dates.iloc[0])
        series = rate.iloc[:, 0].reindex(index, method="ffill")
        if series.notna().sum() >= len(index) * 0.9:
            return series.bfill().to_numpy()
    except Exception as exc:  # 네트워크·스키마 변경. 검정을 멈출 이유는 아니다
        print(f"  (FRED 금리를 못 받아 연 {FALLBACK_RATE}% 상수로 대체합니다: {exc})")
    return pd.Series(FALLBACK_RATE, index=index).to_numpy()


def leveraged_net(weight: pd.Series, returns: pd.Series, cost_bps: float,
                  leverage: float, policy_rate, spread_bps: float) -> pd.Series:
    """배율을 건 일간 순수익률(소수).

    1을 넘는 부분은 (기준금리 + 스프레드)로 빌리고, 현금으로 남은 부분은 기준금리를
    받는다. 계속 보유(비중 1)는 어느 쪽도 해당되지 않아 baseline이 왜곡되지 않는다.
    돈치안이 현금으로 앉아 있는 37%의 기간에 이자가 붙는 것도 실제로 있는 일이다.
    """
    exposure = weight * leverage
    gross = apply_costs(exposure, returns, cost_bps).fillna(0.0) / 100
    borrow = np.clip(exposure - 1.0, 0.0, None) * (policy_rate + spread_bps / 100) / 100 / 252
    cash = np.clip(1.0 - exposure, 0.0, None) * policy_rate / 100 / 252
    return gross - borrow + cash


def net_metrics(net: pd.Series) -> tuple[float, float, float]:
    """연수익%, MDD%, 샤프."""
    equity = (1 + net).cumprod()
    drawdown = equity / equity.cummax() - 1
    years = max(len(net) / 252, 1e-9)
    volatility = net.std()
    return (
        (equity.iloc[-1] ** (1 / years) - 1) * 100,
        drawdown.min() * 100,
        net.mean() / volatility * math.sqrt(252) if volatility else math.nan,
    )


def match_mdd(weight: pd.Series, returns: pd.Series, cost_bps: float,
              target_mdd: float, policy_rate, spread_bps: float) -> float:
    """계속 보유와 MDD가 같아지는 배율을 이분탐색으로 찾는다. 같은 고통, 다른 수익."""
    low, high = 1.0, 6.0
    if net_metrics(leveraged_net(weight, returns, cost_bps, high, policy_rate, spread_bps))[1] > target_mdd:
        return math.nan   # 6배를 걸어도 계속 보유만큼 안 빠진다
    for _ in range(40):
        mid = (low + high) / 2
        if net_metrics(leveraged_net(weight, returns, cost_bps, mid, policy_rate, spread_bps))[1] < target_mdd:
            high = mid
        else:
            low = mid
    return (low + high) / 2


def leverage_table(weight: pd.Series, returns: pd.Series, cost_bps: float,
                   policy_rate, spread_bps: float, bh_cagr: float,
                   bh_mdd: float) -> pd.DataFrame:
    rows = []
    for leverage in LEVERAGE_GRID:
        cagr, mdd, sharpe = net_metrics(
            leveraged_net(weight, returns, cost_bps, leverage, policy_rate, spread_bps))
        rows.append({"배율": f"x{leverage:.1f}", "연수익%": cagr, "MDD%": mdd,
                     "샤프": sharpe, "계속보유 대비 연수익%p": cagr - bh_cagr})
    matched = match_mdd(weight, returns, cost_bps, bh_mdd, policy_rate, spread_bps)
    if not math.isnan(matched):
        cagr, mdd, sharpe = net_metrics(
            leveraged_net(weight, returns, cost_bps, matched, policy_rate, spread_bps))
        rows.append({"배율": f"x{matched:.2f} (MDD 일치)", "연수익%": cagr, "MDD%": mdd,
                     "샤프": sharpe, "계속보유 대비 연수익%p": cagr - bh_cagr})
    return pd.DataFrame(rows)


def decade_counts(frame: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """표본이 특정 시기에 몰려 있지는 않은지 본다."""
    decade = frame["date"].str[:3] + "0년대"
    hits = table["ftd"].groupby(decade).sum()
    days = frame["date"].groupby(decade).count()
    return pd.DataFrame({"거래일": days, "FTD 횟수": hits.astype(int)}).reset_index(
        names="구간"
    )


def study(index_code: str, etf_code: str, label: str, start: str,
          cost_bps: float, spread_bps: float, output: Path) -> pd.DataFrame:
    frame = load_market(index_code, etf_code, start)
    entry_signal, exit_signal, warmup = donchian_parts(frame)
    etf_returns = open_to_open(frame["etf_open"])
    usable = etf_returns.notna()

    print("=" * 84)
    print(f"■ {label}  지수 {index_code} 신호 -> ETF {etf_code} 수익")
    print("=" * 84)
    print(f"{frame['date'].iloc[0]} ~ {frame['date'].iloc[-1]}  {len(frame)}일  "
          f"왕복 {cost_bps:.0f}bp")
    print(f"조정 정의: 최근 {CORRECTION_LOOKBACK}일 고점 대비 -{CORRECTION_DROP:.0%} 이하")
    print(f"판정 창: 랠리 {FIRST_TEST_DAY}~{LAST_TEST_DAY}일차, "
          f"거래량 {VOLUME_MA}일 이평 상회\n")

    tables = {t: ftd_signals(frame, t) for t in THRESHOLDS}

    print("1. FTD 발생 빈도와 위신호 비율")
    print("-" * 84)
    rows = []
    for threshold in THRESHOLDS:
        row = {"상승 기준%": threshold}
        row.update(evaluate_false_signals(frame, tables[threshold]))
        rows.append(row)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("  위신호 = FTD 이후 20일 안에 랠리 저점을 다시 깬 경우 (오닐 본인의 기준)")

    print("\n2. 표본이 몰려 있지는 않은가 (2.0% 기준)")
    print("-" * 84)
    print(decade_counts(frame, tables[2.0]).to_string(index=False))

    print("\n3. 저점 대비 진입 시점 비교")
    print("-" * 84)
    lag_rows = [entry_lag(frame, entry_signal & warmup, "돈치안 20일 돌파")]
    for threshold in [1.2, 2.0]:
        lag_rows.append(entry_lag(frame, tables[threshold]["ftd"], f"FTD {threshold}%"))
    print(pd.DataFrame(lag_rows).round(2).to_string(index=False))

    print("\n4. 전략 비교 (진입만 다르게, 청산은 모두 돈치안 20일 하단)")
    print("-" * 84)
    donchian_pos = position_from_entries(entry_signal, exit_signal, warmup)

    def weight(position: pd.Series) -> pd.Series:
        return (position.shift(1).fillna(0).astype(bool) & usable).astype(float)

    results = [
        summarize(etf_returns, pd.Series(1.0, index=frame.index),
                  f"{etf_code} 계속 보유", cost_bps),
        summarize(etf_returns, weight(donchian_pos), "돈치안 진입만", cost_bps),
    ]
    for threshold in [1.2, 2.0]:
        ftd = tables[threshold]["ftd"]
        results.append(summarize(
            etf_returns,
            weight(position_from_entries(ftd, exit_signal, warmup)),
            f"FTD {threshold}% 진입만", cost_bps))
        results.append(summarize(
            etf_returns,
            weight(position_from_entries(ftd | entry_signal, exit_signal, warmup)),
            f"돈치안 + FTD {threshold}% 결합", cost_bps))
    table_out = pd.DataFrame(results)
    print(table_out.round(3).to_string(index=False))

    print("\n5. 위험을 수익으로 환전하면 (결합 2.0%에 배율, 조달비용 반영)")
    print("-" * 84)
    policy_rate = load_policy_rate(frame["date"])
    combined_weight = weight(
        position_from_entries(tables[2.0]["ftd"] | entry_signal, exit_signal, warmup))
    bh_cagr, bh_mdd, bh_sharpe = net_metrics(
        apply_costs(pd.Series(1.0, index=frame.index), etf_returns, cost_bps).fillna(0.0) / 100)
    lev = leverage_table(combined_weight, etf_returns, cost_bps,
                         policy_rate, spread_bps, bh_cagr, bh_mdd)
    print(f"  조달금리 = 실효 FF금리 + {spread_bps:.0f}bp  "
          f"(표본 평균 연 {np.mean(policy_rate) + spread_bps / 100:.2f}%), "
          f"현금 구간은 FF금리 수취")
    print(f"  기준: {etf_code} 계속 보유  연{bh_cagr:.2f}%  MDD{bh_mdd:.1f}%  샤프{bh_sharpe:.3f}")
    print(lev.round(3).to_string(index=False))

    print("\n6. 지금 상태 (마지막 10거래일, 2.0% 기준)")
    print("-" * 84)
    recent = tables[2.0].tail(10).copy()
    recent["거래량/50일평균"] = (recent["volume"] / recent["volume_ma"]).round(3)
    print(recent[["date", "rally_day", "gain", "거래량/50일평균",
                  "in_window", "gain_ok", "volume_ok", "ftd"]].round(2).to_string(index=False))
    print()

    slug = index_code.lstrip("^").lower()
    output.mkdir(parents=True, exist_ok=True)
    table_out.insert(0, "시장", label)
    table_out.to_csv(output / f"us_{slug}_strategy.csv", index=False, encoding="utf-8-sig")
    lev.insert(0, "시장", label)
    lev.to_csv(output / f"us_{slug}_leverage.csv", index=False, encoding="utf-8-sig")
    tables[2.0].to_csv(output / f"us_{slug}_signals.csv", index=False, encoding="utf-8-sig")
    return table_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="미국 지수에서 돈치안·FTD 결합 규칙 검정")
    parser.add_argument("--index", default=None, help="지수 코드. 예: US500, ^NDX")
    parser.add_argument("--etf", default=None, help="수익을 재는 ETF. 예: SPY, QQQ")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument("--spread-bps", type=float, default=DEFAULT_SPREAD_BPS,
                        help="기준금리에 얹는 조달 스프레드. 개인 신용융자면 300~500이 현실적")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.index:
        markets = [(args.index, args.etf or "SPY", args.index)]
    else:
        markets = DEFAULT_MARKETS

    tables = [study(index_code, etf_code, label, args.start, args.cost_bps,
                    args.spread_bps, args.output)
              for index_code, etf_code, label in markets]

    if len(tables) > 1:
        print("=" * 84)
        print("■ 시장별 요약")
        print("=" * 84)
        merged = pd.concat(tables, ignore_index=True)
        print(merged[["시장", "전략", "연수익%", "MDD%", "샤프", "매매횟수"]]
              .round(3).to_string(index=False))
        merged.to_csv(args.output / "us_index_strategy.csv",
                      index=False, encoding="utf-8-sig")

    print(f"\n표는 {args.output.resolve()}에 저장했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
