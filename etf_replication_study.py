"""지수로 검증한 결과가 실제 매매 대상인 ETF에서도 나오는지 확인한다.

지금까지의 검정은 전부 코스피 지수(0001)로 했다. 그런데 지수는 살 수 없다.
실제로는 지수 신호를 받아 ETF를 산다. 둘 사이에는 세 가지 차이가 있다.

1. 코스피 지수는 배당을 빼고 계산한 가격지수다. ETF 수정주가에는 분배금이 들어 있다.
2. RISE 200은 코스피가 아니라 코스피200을 따라간다. 구성이 다르다.
3. 추적오차, 괴리율, 유동성이 붙는다.

그래서 네 가지를 같은 구간에서 비교한다.

- 지수 신호 -> 지수 수익 (지금까지 검정한 것. 실제로는 못 사는 수익)
- 지수 신호 -> ETF 수익 (저장소가 실제로 하는 운용 방식)
- ETF 신호 -> ETF 수익 (ETF 자체 가격으로 신호를 내는 방식)
- ETF 계속 보유
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from gate_study import apply_costs, cvar, donchian_position
from kis import PanelStore, get_dataset

DEFAULT_COST_BPS = 10.0


def load_pair(store: PanelStore, index_code: str, ticker: str) -> pd.DataFrame:
    """지수와 ETF를 같은 날짜로 맞춘다."""
    index = store.read(get_dataset("market_investor_flow_daily"), tickers=[index_code])
    if index.empty:
        raise SystemExit(f"{index_code} 지수 자료가 없습니다.")
    index = index[["date", "index_open", "index_high", "index_low", "index_close"]]

    etf = store.read(get_dataset("price_daily"), tickers=[ticker])
    if etf.empty:
        raise SystemExit(f"{ticker} 가격 자료가 없습니다.")
    etf = etf[["date", "open", "high", "low", "close", "volume"]]

    frame = index.merge(etf, on="date", how="inner").sort_values("date")
    return frame.reset_index(drop=True)


def open_to_open(open_: pd.Series) -> pd.Series:
    """오늘 시가에 사서 다음 거래일 시가에 파는 수익률(%)."""
    return (open_.shift(-1) / open_ - 1) * 100


def summarize(
    returns: pd.Series, weight: pd.Series, name: str, cost_bps: float
) -> dict:
    net = apply_costs(weight, returns, cost_bps).fillna(0.0) / 100
    equity = (1 + net).cumprod()
    drawdown = equity / equity.cummax() - 1
    years = max(len(net) / 252, 1e-9)

    underwater = longest = 0
    for value in drawdown:
        underwater = underwater + 1 if value < -1e-12 else 0
        longest = max(longest, underwater)

    volatility = net.std()
    trades = int((weight.diff().abs() > 1e-9).sum())
    return {
        "전략": name,
        "누적수익%": (equity.iloc[-1] - 1) * 100,
        "연수익%": (equity.iloc[-1] ** (1 / years) - 1) * 100,
        "MDD%": drawdown.min() * 100,
        "최장 손실구간(일)": longest,
        "CVaR5%": cvar(net * 100),
        "샤프": net.mean() / volatility * math.sqrt(252) if volatility else math.nan,
        "매매횟수": trades,
        "평균 노출": weight.mean(),
    }


def contribution_study(
    returns: pd.Series, weight: pd.Series, dates: pd.Series, cost_bps: float, amount: float
) -> dict:
    """적립식. 투입 원금 대비 얼마나 물려 있었는지가 실제로 느끼는 손실이다."""
    net = apply_costs(weight, returns, cost_bps).fillna(0.0) / 100
    month = dates.str[:7]
    contribute = month != month.shift(1)
    contribute.iloc[0] = True

    value = invested = 0.0
    values, investeds = [], []
    for i in range(len(net)):
        if contribute.iloc[i]:
            value += amount
            invested += amount
        value *= 1 + net.iloc[i]
        values.append(value)
        investeds.append(invested)
    value_series = pd.Series(values)
    invested_series = pd.Series(investeds)
    underwater = (invested_series - value_series) / invested_series
    return {
        "총 투입": invested_series.iloc[-1],
        "최종 평가액": value_series.iloc[-1],
        "수익률%": (value_series.iloc[-1] / invested_series.iloc[-1] - 1) * 100,
        "최대 원금손실%": underwater.max() * 100,
        "평가액 MDD%": (value_series / value_series.cummax() - 1).min() * 100,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="지수 검정 결과의 ETF 재현 확인")
    parser.add_argument("--index", default="0001")
    parser.add_argument("--ticker", default="148020")
    parser.add_argument("--db", type=Path, default=Path("data/kis_panel.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS,
                        help="왕복 거래비용 bp. ETF는 증권거래세가 없다")
    parser.add_argument("--contribution", type=float, default=1_000_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = PanelStore(args.db)
    frame = load_pair(store, args.index, args.ticker)

    index_position = donchian_position(frame, "index_high", "index_low")
    etf_position = donchian_position(frame, "high", "low")

    index_returns = open_to_open(frame["index_open"])
    etf_returns = open_to_open(frame["open"])

    usable = etf_returns.notna() & index_returns.notna()
    index_active = index_position.shift(1).fillna(0).astype(bool) & usable
    etf_active = etf_position.shift(1).fillna(0).astype(bool) & usable

    print(f"지수 {args.index} vs ETF {args.ticker}")
    print(f"{frame['date'].iloc[0]} ~ {frame['date'].iloc[-1]}  {len(frame)}일  "
          f"왕복비용 {args.cost_bps:.0f}bp\n")

    print("=" * 82)
    print("1. 지수와 ETF는 얼마나 같이 움직이나")
    print("=" * 82)
    common = index_returns.notna() & etf_returns.notna()
    correlation = index_returns[common].corr(etf_returns[common])
    index_total = (1 + index_returns[common] / 100).prod() - 1
    etf_total = (1 + etf_returns[common] / 100).prod() - 1
    years = len(frame) / 252
    print(f"  일간 수익률 상관 {correlation:.4f}")
    print(f"  누적 수익  지수 {index_total * 100:,.1f}%  ETF {etf_total * 100:,.1f}%")
    print(f"  연환산 차이 {(((1 + etf_total) / (1 + index_total)) ** (1 / years) - 1) * 100:+.2f}%p")
    print("  코스피는 배당을 뺀 가격지수이고 ETF 수정주가에는 분배금이 들어 있다.")
    tracking = (etf_returns[common] - index_returns[common]).std()
    print(f"  일간 추적오차 표준편차 {tracking:.3f}%p")

    agree = (index_position == etf_position).mean()
    print(f"\n  돈치안 포지션 일치율 {agree:.1%}")
    print(f"  지수 신호 진입 {int((index_position.diff() == 1).sum())}회  "
          f"ETF 신호 진입 {int((etf_position.diff() == 1).sum())}회")

    print("\n" + "=" * 82)
    print("2. 거치식 비교 (같은 구간, 같은 비용)")
    print("=" * 82)
    rows = [
        summarize(index_returns, pd.Series(1.0, index=frame.index),
                  "지수 계속 보유 (못 삼)", args.cost_bps),
        summarize(index_returns, index_active.astype(float),
                  "지수 신호 -> 지수 수익 (못 삼)", args.cost_bps),
        summarize(etf_returns, pd.Series(1.0, index=frame.index),
                  "ETF 계속 보유", args.cost_bps),
        summarize(etf_returns, index_active.astype(float),
                  "지수 신호 -> ETF (현재 운용 방식)", args.cost_bps),
        summarize(etf_returns, etf_active.astype(float),
                  "ETF 신호 -> ETF", args.cost_bps),
    ]
    table = pd.DataFrame(rows)
    print(table.round(3).to_string(index=False))

    print("\n" + "=" * 82)
    print("3. 적립식 비교 (매월 첫 거래일 정액)")
    print("=" * 82)
    rows = []
    for name, returns, weight in [
        ("ETF 계속 보유", etf_returns, pd.Series(1.0, index=frame.index)),
        ("지수 신호 -> ETF", etf_returns, index_active.astype(float)),
        ("ETF 신호 -> ETF", etf_returns, etf_active.astype(float)),
    ]:
        row = {"전략": name}
        row.update(
            contribution_study(returns, weight, frame["date"], args.cost_bps, args.contribution)
        )
        rows.append(row)
    dca = pd.DataFrame(rows)
    print(dca.round(3).to_string(index=False))

    print("\n" + "=" * 82)
    print("4. 비용 민감도 (지수 신호 -> ETF)")
    print("=" * 82)
    rows = []
    for cost in [5.0, 10.0, 25.0, 50.0]:
        row = summarize(etf_returns, index_active.astype(float), f"왕복 {cost:.0f}bp", cost)
        rows.append(row)
    cost_table = pd.DataFrame(rows)
    print(cost_table[["전략", "연수익%", "MDD%", "샤프", "매매횟수"]].round(3).to_string(index=False))

    print("\n" + "=" * 82)
    print("5. 연도별 (ETF 기준)")
    print("=" * 82)
    rows = []
    years_group = frame["date"].str[:4]
    for year, idx in frame.groupby(years_group).groups.items():
        piece = pd.Index(idx)
        if len(piece) < 40:
            continue
        row = {"연도": year, "일수": len(piece)}
        for label, weight in [
            ("보유", pd.Series(1.0, index=frame.index).loc[piece]),
            ("지수신호", index_active.astype(float).loc[piece]),
            ("ETF신호", etf_active.astype(float).loc[piece]),
        ]:
            net = apply_costs(weight, etf_returns.loc[piece], args.cost_bps).fillna(0.0) / 100
            equity = (1 + net).cumprod()
            row[f"{label} 수익%"] = (equity.iloc[-1] - 1) * 100
            row[f"{label} MDD%"] = (equity / equity.cummax() - 1).min() * 100
        rows.append(row)
    yearly = pd.DataFrame(rows)
    print(yearly.round(2).to_string(index=False))

    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "etf_replication_summary.csv", index=False, encoding="utf-8-sig")
    dca.to_csv(args.output / "etf_replication_dca.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(args.output / "etf_replication_yearly.csv", index=False, encoding="utf-8-sig")
    print(f"\n표는 {args.output.resolve()}에 저장했습니다.")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
