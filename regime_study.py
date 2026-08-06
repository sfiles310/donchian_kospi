# -*- coding: utf-8 -*-
"""변동성 국면별 전략 성과와 국면 예측 가능성을 검정한다.

두 가지 질문에 답한다.

1. 변동성이 낮은 국면에서 이 시스템은 여전히 값어치가 있는가
2. 그 국면을 미리 알 수 있는가

신호는 지수로 내고 수익은 실제 매매 대상 ETF로 잰다. 설명변수는 전부 그날까지의
자료만 쓰고 목표는 앞으로의 변동성이다.

    python regime_study.py                    # 전체
    python regime_study.py --year 2025        # 특정 해만
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import ftd_study as F
from etf_replication_study import open_to_open
from gate_study import apply_costs
from kis import PanelStore, get_dataset

COST_BPS = 10.0
VOL_WINDOW = 60
HORIZON = 60


def load(store: PanelStore, index_code: str, ticker: str) -> tuple[pd.DataFrame, dict]:
    frame = F.load_market(store, index_code, ticker)
    entry, exit_, warmup = F.donchian_parts(frame)
    ftd = F.ftd_signals(frame, 2.0)["ftd"]
    donchian = F.position_from_entries(entry, exit_, warmup)
    combo = F.position_from_entries(ftd | entry, exit_, warmup)

    returns = open_to_open(frame["etf_open"])
    usable = returns.notna()
    weights = {
        "ETF 계속 보유": pd.Series(1.0, index=frame.index),
        "돈치안": (donchian.shift(1).fillna(0).astype(bool) & usable).astype(float),
        "돈치안 + FTD": (combo.shift(1).fillna(0).astype(bool) & usable).astype(float),
    }
    frame["ret"] = returns
    return frame, weights


def performance(frame, weights, idx, cost_bps=COST_BPS) -> pd.DataFrame:
    rows = []
    for name, weight in weights.items():
        net = apply_costs(weight.loc[idx], frame["ret"].loc[idx], cost_bps).fillna(0.0) / 100
        equity = (1 + net).cumprod()
        years = max(len(net) / 252, 1e-9)
        volatility = net.std()
        rows.append({
            "전략": name,
            "수익률%": (equity.iloc[-1] - 1) * 100,
            "연환산%": (equity.iloc[-1] ** (1 / years) - 1) * 100,
            "MDD%": (equity / equity.cummax() - 1).min() * 100,
            "샤프": net.mean() / volatility * np.sqrt(252) if volatility else np.nan,
            "노출": weight.loc[idx].mean(),
        })
    return pd.DataFrame(rows)


def realized_vol(close: pd.Series, window: int = VOL_WINDOW) -> pd.Series:
    return close.pct_change().rolling(window).std() * np.sqrt(252) * 100


def forward_vol(close: pd.Series, horizon: int = HORIZON) -> pd.Series:
    """앞으로 horizon일의 변동성. 오늘 시점에는 알 수 없는 값이라 목표로만 쓴다."""
    ret = close.pct_change()
    return (ret[::-1].rolling(horizon).std()[::-1].shift(-1)) * np.sqrt(252) * 100


def build_features(store: PanelStore, index_code: str) -> pd.DataFrame:
    price = store.read(get_dataset("market_price_daily"), tickers=[index_code],
                       confirmed_only=False)
    flow = store.read(get_dataset("market_investor_flow_daily"), tickers=[index_code],
                      confirmed_only=False)
    loan = store.read(get_dataset("market_loan_trans_daily"), tickers=[index_code],
                      confirmed_only=False)
    d = (price[["date", "index_close", "volume"]]
         .merge(flow[["date", "foreign_net_value", "individual_net_value"]],
                on="date", how="left")
         .merge(loan[["date", "sbl_balance_qty"]], on="date", how="left")
         .sort_values("date").reset_index(drop=True))

    d["vol60"] = realized_vol(d["index_close"])
    d["vol20"] = realized_vol(d["index_close"], 20)
    d["future"] = forward_vol(d["index_close"])
    d["단기/장기 변동성비"] = d["vol20"] / d["vol60"]
    d["변동성의 변동성"] = d["vol60"].rolling(60).std()
    d["250일 고점대비%"] = (d["index_close"] / d["index_close"].rolling(250).max() - 1) * 100
    d["거래량/60일평균"] = d["volume"] / d["volume"].rolling(60).mean()
    d["대차잔고 20일증가"] = d["sbl_balance_qty"] / d["sbl_balance_qty"].shift(20) - 1
    d["외국인 수급규모"] = d["foreign_net_value"].abs().rolling(20).mean() / d["index_close"]
    d["개인 수급규모"] = d["individual_net_value"].abs().rolling(20).mean() / d["index_close"]
    return d


FEATURES = ["단기/장기 변동성비", "변동성의 변동성", "250일 고점대비%",
            "거래량/60일평균", "대차잔고 20일증가", "외국인 수급규모", "개인 수급규모"]


def _r2(table: pd.DataFrame, columns: list[str]) -> float:
    X = np.column_stack([np.ones(len(table))] + [table[c].to_numpy() for c in columns])
    y = table["future"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return 1 - (y - X @ beta).var() / y.var()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="변동성 국면과 예측 가능성 검정")
    parser.add_argument("--index", default="0001")
    parser.add_argument("--ticker", default="148020")
    parser.add_argument("--db", type=Path, default=Path("data/kis_panel.sqlite"))
    parser.add_argument("--year", default=None, help="이 해만 따로 본다")
    parser.add_argument("--start", default="2014-01-01")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = PanelStore(args.db)
    frame, weights = load(store, args.index, args.ticker)
    year = frame["date"].str[:4]

    if args.year:
        idx = frame.index[year == args.year]
        if not len(idx):
            raise SystemExit(f"{args.year}년 자료가 없습니다.")
        print(f"=== {args.year}년 ({len(idx)}거래일)")
        print(performance(frame, weights, idx).round(2).to_string(index=False))
        store.close()
        return 0

    print("=" * 76)
    print("1. 연도별 성과 (ETF 기준)")
    print("=" * 76)
    rows = []
    vol = realized_vol(frame["index_close"])
    for y, group in frame.groupby(year).groups.items():
        idx = pd.Index(group)
        if len(idx) < 100:
            continue
        table = performance(frame, weights, idx).set_index("전략")["수익률%"]
        rows.append({
            "연도": y, "평균 변동성%": vol.loc[idx].mean(),
            "보유%": table["ETF 계속 보유"], "돈치안%": table["돈치안"],
            "결합%": table["돈치안 + FTD"],
            "돈치안-보유%p": table["돈치안"] - table["ETF 계속 보유"],
        })
    by_year = pd.DataFrame(rows)
    print(by_year.round(2).to_string(index=False))
    corr = by_year["평균 변동성%"].corr(by_year["돈치안-보유%p"])
    print(f"\n변동성과 '돈치안 - 보유' 상관: {corr:+.3f}"
          "  (양수면 변동성이 클수록 돈치안이 유리)")

    print("\n" + "=" * 76)
    print("2. 변동성 국면별 성과 (60일 실현변동성 3등분)")
    print("=" * 76)
    valid = frame.index[vol.notna() & frame["ret"].notna()
                        & (frame["date"] >= args.start)]
    cuts = vol.loc[valid].quantile([1 / 3, 2 / 3]).tolist()
    buckets = {
        f"저변동 (~{cuts[0]:.0f}%)": valid[vol.loc[valid] <= cuts[0]],
        f"중간 ({cuts[0]:.0f}~{cuts[1]:.0f}%)":
            valid[(vol.loc[valid] > cuts[0]) & (vol.loc[valid] <= cuts[1])],
        f"고변동 ({cuts[1]:.0f}%~)": valid[vol.loc[valid] > cuts[1]],
    }
    for label, idx in buckets.items():
        print(f"\n[{label}] {len(idx)}일")
        print(performance(frame, weights, idx)[["전략", "연환산%", "MDD%", "노출"]]
              .round(2).to_string(index=False))

    print("\n" + "=" * 76)
    print("3. 변동성 국면을 미리 알 수 있나")
    print("=" * 76)
    d = build_features(store, args.index)
    use = d.dropna(subset=["vol60", "future"] + FEATURES).reset_index(drop=True)
    print(f"표본 {len(use)}일  {use['date'].min()} ~ {use['date'].max()}\n")

    print("  과거 변동성만으로 앞을 맞히는 정도")
    for horizon in [5, 20, 60, 120, 250]:
        fut = forward_vol(d["index_close"], horizon)
        pair = pd.concat([d["vol60"], fut], axis=1).dropna()
        c = pair.iloc[:, 0].corr(pair.iloc[:, 1])
        print(f"    앞으로 {horizon:>3}일   상관 {c:.3f}   R² {c ** 2:.3f}")

    cuts2 = use["vol60"].quantile([1 / 3, 2 / 3]).tolist()
    name = lambda v: "저변동" if v <= cuts2[0] else ("중간" if v <= cuts2[1] else "고변동")
    use["now"] = use["vol60"].map(name)
    use["then"] = use["future"].map(name)
    order = ["저변동", "중간", "고변동"]
    print(f"\n  {HORIZON}일 뒤 국면 전이 확률% (경계 {cuts2[0]:.1f} / {cuts2[1]:.1f})")
    trans = pd.crosstab(use["now"], use["then"], normalize="index") * 100
    print(trans.reindex(index=order, columns=order).round(1).to_string())

    print("\n  과거 변동성에 무엇을 더하면 나아지나")
    base = _r2(use, ["vol60"])
    print(f"    60일 변동성만            R² {base:.3f}")
    for col in FEATURES:
        value = _r2(use, ["vol60", col])
        print(f"      + {col:<16} R² {value:.3f}  ({value - base:+.3f})")
    print(f"    전부 넣으면              R² {_r2(use, ['vol60'] + FEATURES):.3f}")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
