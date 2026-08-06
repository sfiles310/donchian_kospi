# -*- coding: utf-8 -*-
"""외국인 수급의 예측력과 단타 적용 가능성을 비교 검증한다.

기존 돈치안 대시보드와 산출물은 수정하지 않는다. 외국인 수급은 장 마감 후
확정된 값만 사용하고, 모든 전략 수익률은 신호 다음 거래일에 반영한다.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from html.parser import HTMLParser
from pathlib import Path

import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import requests

from donchian_kospi_daily import compute_signals


SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = SCRIPT_DIR / "output"
CACHE_DIR = OUTPUT_DIR / "foreign_flow_cache"
TEMPLATE_PATH = SCRIPT_DIR / "foreign_flow_dashboard_template.html"

NAVER_FOREIGN_URL = "https://finance.naver.com/item/frgn.naver"
NAVER_MARKET_URL = "https://m.stock.naver.com/api/stocks/marketValue/KOSPI"
NAVER_INTEGRATION_URL = "https://m.stock.naver.com/api/stock/{ticker}/integration"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://finance.naver.com/",
}
HORIZONS = (1, 3, 5, 10, 20)


class _TableRowParser(HTMLParser):
    """네이버 표의 셀 텍스트만 추출하는 최소 HTML 파서."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell is not None:
            text = " ".join("".join(self._cell).split())
            if self._row is not None:
                self._row.append(text)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None


def _number(value: object) -> float:
    text = str(value).strip().replace(",", "").replace("+", "").replace("%", "")
    if text in {"", "-", "nan", "None"}:
        return math.nan
    return float(text)


def parse_naver_foreign_html(html: str) -> pd.DataFrame:
    """네이버 종목별 매매동향 HTML에서 확정 일별 수급을 읽는다."""
    parser = _TableRowParser()
    parser.feed(html)
    records = []
    for row in parser.rows:
        if len(row) < 9 or not re.fullmatch(r"\d{4}\.\d{2}\.\d{2}", row[0]):
            continue
        records.append({
            "date": pd.to_datetime(row[0], format="%Y.%m.%d"),
            "reported_close": _number(row[1]),
            "reported_change_pct": _number(row[3]),
            "reported_volume": _number(row[4]),
            "institution_net_qty": _number(row[5]),
            "foreign_net_qty": _number(row[6]),
            "foreign_holding_qty": _number(row[7]),
            "foreign_holding_pct": _number(row[8]),
        })
    if not records:
        return pd.DataFrame()
    return (
        pd.DataFrame(records)
        .drop_duplicates("date", keep="last")
        .set_index("date")
        .sort_index()
    )


def fetch_foreign_history(
    ticker: str,
    start: str,
    end: str | None = None,
    max_pages: int = 200,
    pause_seconds: float = 0.05,
) -> pd.DataFrame:
    """캐시와 네이버 일별 매매동향을 합쳐 지정 기간 수급을 반환한다."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{ticker}.csv"
    cached = pd.DataFrame()
    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["date"]).set_index("date").sort_index()

    start_at = pd.Timestamp(start)
    end_at = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    if cached.empty or cached.index.min() > start_at:
        stop_at = start_at
    else:
        stop_at = cached.index.max()

    session = requests.Session()
    downloaded = []
    previous_oldest = None
    for page in range(1, max_pages + 1):
        response = session.get(
            NAVER_FOREIGN_URL,
            params={"code": ticker, "page": page},
            headers=HEADERS,
            timeout=15,
        )
        response.raise_for_status()
        response.encoding = "euc-kr"
        frame = parse_naver_foreign_html(response.text)
        if frame.empty:
            break
        downloaded.append(frame)
        oldest = frame.index.min()
        if oldest <= stop_at or oldest == previous_oldest:
            break
        previous_oldest = oldest
        if pause_seconds:
            time.sleep(pause_seconds)
    else:
        raise RuntimeError(
            f"{ticker} 수급 데이터가 {start}까지 도달하지 못했습니다. "
            f"--max-pages 값을 늘리세요."
        )

    pieces = ([cached] if not cached.empty else []) + downloaded
    if not pieces:
        raise RuntimeError(f"{ticker} 외국인 수급 데이터를 가져오지 못했습니다.")
    merged = pd.concat(pieces).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    merged.reset_index().to_csv(cache_path, index=False, encoding="utf-8-sig")
    selected = merged.loc[(merged.index >= start_at) & (merged.index <= end_at)].copy()
    if selected.empty:
        raise RuntimeError(f"{ticker}의 {start} 이후 외국인 수급 데이터가 없습니다.")
    return selected


def fetch_price_history(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """돈치안 준비기간을 포함해 종목 OHLCV를 가져온다."""
    buffered_start = (pd.Timestamp(start) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    frame = fdr.DataReader(ticker, buffered_start, end)
    frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]
    required = ["open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{ticker} 가격 데이터 누락: {', '.join(missing)}")
    return frame[required].astype(float).dropna().sort_index()


def fetch_fx_history(start: str, end: str | None = None) -> pd.Series:
    buffered_start = (pd.Timestamp(start) - pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    frame = fdr.DataReader("USD/KRW", buffered_start, end)
    close = frame["Close"].astype(float).sort_index()
    close.name = "usdkrw_close"
    return close


def build_dataset(
    ticker: str,
    start: str,
    end: str | None = None,
    max_pages: int = 200,
    foreign: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
    fx_close: pd.Series | None = None,
) -> pd.DataFrame:
    """가격, 외국인 수급, 환율을 날짜 기준으로 결합하고 검증 변수를 만든다."""
    foreign = foreign if foreign is not None else fetch_foreign_history(
        ticker, start, end, max_pages=max_pages
    )
    prices = prices if prices is not None else fetch_price_history(ticker, start, end)
    fx_close = fx_close if fx_close is not None else fetch_fx_history(start, end)

    flow_columns = [
        "reported_close", "reported_volume", "institution_net_qty",
        "foreign_net_qty", "foreign_holding_qty", "foreign_holding_pct",
    ]
    data = prices.join(foreign[flow_columns], how="inner")
    data = data.loc[data.index >= pd.Timestamp(start)].copy()
    if data.empty:
        raise RuntimeError(f"{ticker} 가격과 수급 데이터의 공통 날짜가 없습니다.")

    fx_aligned = fx_close.reindex(data.index.union(fx_close.index)).sort_index().ffill()
    data["usdkrw_close"] = fx_aligned.reindex(data.index)

    volume = data["volume"].replace(0, np.nan)
    data["foreign_flow_1d"] = data["foreign_net_qty"] / volume
    data["foreign_flow_5d"] = (
        data["foreign_net_qty"].rolling(5).sum() / volume.rolling(5).sum()
    )
    data["foreign_flow_20d"] = (
        data["foreign_net_qty"].rolling(20).sum() / volume.rolling(20).sum()
    )
    data["institution_flow_5d"] = (
        data["institution_net_qty"].rolling(5).sum() / volume.rolling(5).sum()
    )
    data["foreign_positive_days_5d"] = (
        data["foreign_net_qty"].gt(0).rolling(5).sum()
    )
    data["foreign_holding_change_5d"] = data["foreign_holding_pct"].diff(5)
    data["price_return_1d"] = data["close"].pct_change()
    data["volume_ratio_20d"] = data["volume"] / data["volume"].rolling(20).mean().shift(1)
    data["usdkrw_return_1d"] = data["usdkrw_close"].pct_change()
    data["usdkrw_return_5d"] = data["usdkrw_close"].pct_change(5)

    donchian = compute_signals(data[["open", "high", "low", "close"]], 20, 20)
    data["donchian_position"] = donchian["position"]
    data["donchian_entry"] = donchian["long_entry"]
    data["donchian_exit"] = donchian["long_exit"]

    for horizon in HORIZONS:
        data[f"future_return_{horizon}d"] = (
            data["close"].shift(-horizon) / data["open"].shift(-1) - 1
        )

    data["foreign_signal"] = (
        data["foreign_flow_5d"].gt(0)
        & data["foreign_positive_days_5d"].ge(3)
    )
    data["foreign_fx_signal"] = data["foreign_signal"] & data["usdkrw_return_5d"].lt(0)
    data["placebo_foreign_signal"] = block_permute_signal(
        data["foreign_signal"], seed=sum(ord(char) for char in ticker)
    )
    data["combined_signal"] = data["foreign_signal"] & data["donchian_position"].eq(1)
    data["close_mismatch_pct"] = data["reported_close"] / data["close"] - 1
    data["volume_mismatch_pct"] = data["reported_volume"] / volume - 1
    data["ticker"] = ticker
    return data


def block_permute_signal(signal: pd.Series, block_size: int = 5, seed: int = 42) -> pd.Series:
    """연속성은 일부 보존하되 시간 순서를 섞은 재현 가능한 위약 신호."""
    values = signal.fillna(False).astype(bool).to_numpy()
    blocks = [values[i:i + block_size] for i in range(0, len(values), block_size)]
    order = np.arange(len(blocks))
    np.random.default_rng(seed).shuffle(order)
    shuffled = np.concatenate([blocks[i] for i in order]) if blocks else values
    return pd.Series(shuffled[:len(signal)], index=signal.index, dtype=bool)


def _strategy_returns(data: pd.DataFrame, signal: pd.Series, cost_bps: float) -> tuple[pd.Series, pd.Series]:
    """전일 확정 신호로 다음 날 시가 매수·종가 매도한 순수익률."""
    active = signal.astype(bool).shift(1, fill_value=False)
    intraday = data["close"] / data["open"] - 1
    returns = intraday.where(active, 0.0) - active.astype(float) * cost_bps / 10_000
    return returns.fillna(0.0), active


def _max_losing_streak(returns: pd.Series) -> int:
    longest = current = 0
    for losing in returns.lt(0):
        current = current + 1 if losing else 0
        longest = max(longest, current)
    return longest


def performance_stats(returns: pd.Series, active: pd.Series, name: str) -> dict:
    returns = returns.fillna(0.0)
    equity = (1 + returns).cumprod()
    active_returns = returns[active]
    years = max((returns.index[-1] - returns.index[0]).days / 365.25, 1 / 252)
    volatility = returns.std()
    tail_cut = active_returns.quantile(0.05) if len(active_returns) else np.nan
    tail = active_returns[active_returns <= tail_cut] if len(active_returns) else active_returns
    return {
        "strategy": name,
        "total_return": equity.iloc[-1] - 1,
        "cagr": equity.iloc[-1] ** (1 / years) - 1,
        "mdd": (equity / equity.cummax() - 1).min(),
        "sharpe": returns.mean() / volatility * np.sqrt(252) if volatility else np.nan,
        "active_days": int(active.sum()),
        "win_rate": active_returns.gt(0).mean() if len(active_returns) else np.nan,
        "avg_active_return": active_returns.mean() if len(active_returns) else np.nan,
        "cvar_95": tail.mean() if len(tail) else np.nan,
        "max_losing_streak": _max_losing_streak(active_returns),
    }


def compare_strategies(data_by_ticker: dict[str, pd.DataFrame], cost_bps: float) -> pd.DataFrame:
    rows = []
    portfolio_returns: dict[tuple[float, str], list[pd.Series]] = {}
    portfolio_active: dict[tuple[float, str], list[pd.Series]] = {}
    strategies = {
        "모든 거래일": lambda d: pd.Series(True, index=d.index),
        "외국인 당일 순매수": lambda d: d["foreign_net_qty"].gt(0),
        "외국인 5일 누적 양수": lambda d: d["foreign_flow_5d"].gt(0),
        "외국인 지속(3/5)": lambda d: d["foreign_signal"],
        "외국인+원달러 하락": lambda d: d["foreign_fx_signal"],
        "외국인 위약(블록섞기)": lambda d: d["placebo_foreign_signal"],
        "돈치안20만": lambda d: d["donchian_position"].eq(1),
        "돈치안20+외국인": lambda d: d["combined_signal"],
    }
    for multiplier in (1.0, 1.5, 2.0):
        stressed_cost = cost_bps * multiplier
        for ticker, data in data_by_ticker.items():
            for name, make_signal in strategies.items():
                returns, active = _strategy_returns(data, make_signal(data), stressed_cost)
                row = performance_stats(returns, active, name)
                row.update({"ticker": ticker, "cost_multiplier": multiplier})
                rows.append(row)
                portfolio_returns.setdefault((multiplier, name), []).append(returns.rename(ticker))
                portfolio_active.setdefault((multiplier, name), []).append(active.rename(ticker))

        for name in strategies:
            returns_frame = pd.concat(portfolio_returns[(multiplier, name)], axis=1).fillna(0.0)
            active_frame = pd.concat(portfolio_active[(multiplier, name)], axis=1).fillna(False)
            portfolio = returns_frame.mean(axis=1)
            active = active_frame.any(axis=1)
            row = performance_stats(portfolio, active, name)
            row.update({"ticker": "동일비중 포트폴리오", "cost_multiplier": multiplier})
            rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_conditional_excess(
    values: pd.Series,
    condition: pd.Series,
    repetitions: int,
    seed: int,
    block_size: int = 5,
) -> tuple[float, float, float, float]:
    """연속 의존성을 보존한 원형 블록 부트스트랩 신뢰구간."""
    valid = values.notna()
    sample = values.loc[valid].to_numpy(dtype=float)
    selected = condition.loc[valid].fillna(False).to_numpy(dtype=bool)
    if not len(sample) or not selected.any():
        return math.nan, math.nan, math.nan, math.nan
    observed = sample[selected].mean() - sample.mean()
    if repetitions <= 0 or len(sample) < block_size * 2:
        return observed, math.nan, math.nan, math.nan

    rng = np.random.default_rng(seed)
    draws = []
    blocks_needed = math.ceil(len(sample) / block_size)
    offsets = np.arange(block_size)
    for _ in range(repetitions):
        starts = rng.integers(0, len(sample), size=blocks_needed)
        indices = ((starts[:, None] + offsets) % len(sample)).ravel()[:len(sample)]
        boot_values = sample[indices]
        boot_selected = selected[indices]
        if boot_selected.any():
            draws.append(boot_values[boot_selected].mean() - boot_values.mean())
    if not draws:
        return observed, math.nan, math.nan, math.nan
    distribution = np.asarray(draws)
    low, high = np.quantile(distribution, [0.025, 0.975])
    p_value = min(
        1.0,
        2 * min(
            (np.count_nonzero(distribution <= 0) + 1) / (len(distribution) + 1),
            (np.count_nonzero(distribution >= 0) + 1) / (len(distribution) + 1),
        ),
    )
    return observed, low, high, p_value


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """여러 가설을 동시에 시험할 때 사용할 FDR 보정 q값."""
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().astype(float).sort_values()
    if valid.empty:
        return result
    count = len(valid)
    adjusted = valid * count / np.arange(1, count + 1)
    adjusted = adjusted.iloc[::-1].cummin().iloc[::-1].clip(upper=1.0)
    result.loc[adjusted.index] = adjusted
    return result


def build_condition_study(
    data_by_ticker: dict[str, pd.DataFrame],
    bootstrap_repetitions: int = 500,
) -> pd.DataFrame:
    conditions = {
        "전체": lambda d: pd.Series(True, index=d.index),
        "외국인 당일 순매수": lambda d: d["foreign_net_qty"].gt(0),
        "외국인 5일 누적 양수": lambda d: d["foreign_flow_5d"].gt(0),
        "외국인 5일 중 3일+ 순매수": lambda d: d["foreign_positive_days_5d"].ge(3),
        "외국인 보유율 5일 증가": lambda d: d["foreign_holding_change_5d"].gt(0),
        "원달러 5일 하락": lambda d: d["usdkrw_return_5d"].lt(0),
        "외국인 지속+원달러 하락": lambda d: d["foreign_signal"] & d["usdkrw_return_5d"].lt(0),
        "돈치안20 보유 국면": lambda d: d["donchian_position"].eq(1),
        "돈치안20+외국인": lambda d: d["combined_signal"],
    }
    rows = []
    for ticker, data in data_by_ticker.items():
        for condition_name, make_condition in conditions.items():
            condition = make_condition(data).fillna(False)
            for horizon in HORIZONS:
                target = data[f"future_return_{horizon}d"]
                values = target.loc[condition].dropna()
                seed = sum(ord(char) for char in f"{ticker}-{condition_name}-{horizon}")
                excess, ci_low, ci_high, p_value = _bootstrap_conditional_excess(
                    target, condition, bootstrap_repetitions, seed
                )
                rows.append({
                    "ticker": ticker,
                    "condition": condition_name,
                    "horizon_days": horizon,
                    "samples": len(values),
                    "average_return": values.mean(),
                    "median_return": values.median(),
                    "win_rate": values.gt(0).mean(),
                    "excess_vs_all": excess,
                    "ci_95_low": ci_low,
                    "ci_95_high": ci_high,
                    "bootstrap_p_value": p_value,
                })
    result = pd.DataFrame(rows)
    result["fdr_q_value"] = np.nan
    non_baseline = result["condition"].ne("전체")
    result.loc[non_baseline, "fdr_q_value"] = benjamini_hochberg(
        result.loc[non_baseline, "bootstrap_p_value"]
    )
    return result


def build_quantile_study(data_by_ticker: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for ticker, data in data_by_ticker.items():
        valid = data.dropna(subset=["foreign_flow_5d"]).copy()
        if valid["foreign_flow_5d"].nunique() < 5:
            continue
        valid["foreign_flow_quantile"] = pd.qcut(
            valid["foreign_flow_5d"], 5, labels=False, duplicates="drop"
        ) + 1
        for quantile, group in valid.groupby("foreign_flow_quantile"):
            for horizon in HORIZONS:
                values = group[f"future_return_{horizon}d"].dropna()
                rows.append({
                    "ticker": ticker,
                    "foreign_flow_quantile": int(quantile),
                    "horizon_days": horizon,
                    "samples": len(values),
                    "average_return": values.mean(),
                    "median_return": values.median(),
                    "win_rate": values.gt(0).mean(),
                })
    return pd.DataFrame(rows)


def build_correlations(data_by_ticker: dict[str, pd.DataFrame]) -> pd.DataFrame:
    features = [
        "foreign_flow_1d", "foreign_flow_5d", "foreign_flow_20d",
        "foreign_positive_days_5d", "foreign_holding_change_5d",
        "institution_flow_5d", "price_return_1d", "volume_ratio_20d",
        "usdkrw_return_1d", "usdkrw_return_5d",
    ]
    rows = []
    for ticker, data in data_by_ticker.items():
        for feature in features:
            for horizon in HORIZONS:
                target = f"future_return_{horizon}d"
                pair = data[[feature, target]].dropna()
                rows.append({
                    "ticker": ticker,
                    "feature": feature,
                    "horizon_days": horizon,
                    "samples": len(pair),
                    "pearson": pair[feature].corr(pair[target], method="pearson"),
                    "spearman": pair[feature].rank().corr(pair[target].rank()),
                })
    return pd.DataFrame(rows)


def _parse_deal_trends(payload: dict) -> list[dict]:
    records = []
    for row in payload.get("dealTrendInfos", []):
        try:
            records.append({
                "date": pd.to_datetime(str(row["bizdate"]), format="%Y%m%d"),
                "foreign_net_qty": _number(row.get("foreignerPureBuyQuant")),
                "foreign_holding_pct": _number(row.get("foreignerHoldRatio")),
                "institution_net_qty": _number(row.get("organPureBuyQuant")),
                "individual_net_qty": _number(row.get("individualPureBuyQuant")),
                "close": _number(row.get("closePrice")),
                "volume": _number(row.get("accumulatedTradingVolume")),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return records


def scan_foreign_candidates(top_n: int = 50, pause_seconds: float = 0.03) -> pd.DataFrame:
    """시총 상위 KOSPI 보통주에서 최근 외국인 매수강도를 검색한다."""
    if top_n <= 0:
        return pd.DataFrame()
    session = requests.Session()
    response = session.get(
        NAVER_MARKET_URL,
        params={"page": 1, "pageSize": max(top_n * 2, 50)},
        headers=HEADERS,
        timeout=15,
    )
    response.raise_for_status()
    stocks = [
        row for row in response.json().get("stocks", [])
        if row.get("stockEndType") == "stock"
    ][:top_n]

    rows = []
    for stock in stocks:
        ticker = stock.get("itemCode", "")
        try:
            detail = session.get(
                NAVER_INTEGRATION_URL.format(ticker=ticker),
                headers=HEADERS,
                timeout=15,
            )
            detail.raise_for_status()
            trends = _parse_deal_trends(detail.json())
            if not trends:
                continue
            trend = pd.DataFrame(trends).sort_values("date")
            total_volume = trend["volume"].sum()
            latest = trend.iloc[-1]
            rows.append({
                "date": latest["date"],
                "ticker": ticker,
                "name": stock.get("stockName", ticker),
                "close": latest["close"],
                "foreign_net_qty_1d": latest["foreign_net_qty"],
                "foreign_flow_1d": latest["foreign_net_qty"] / latest["volume"] if latest["volume"] else np.nan,
                "foreign_flow_5d": trend["foreign_net_qty"].sum() / total_volume if total_volume else np.nan,
                "foreign_positive_days_5d": int(trend["foreign_net_qty"].gt(0).sum()),
                "foreign_holding_pct": latest["foreign_holding_pct"],
                "foreign_holding_change_5d": latest["foreign_holding_pct"] - trend.iloc[0]["foreign_holding_pct"],
                "institution_flow_5d": trend["institution_net_qty"].sum() / total_volume if total_volume else np.nan,
                "individual_flow_5d": trend["individual_net_qty"].sum() / total_volume if total_volume else np.nan,
            })
        except (requests.RequestException, ValueError, KeyError):
            continue
        if pause_seconds:
            time.sleep(pause_seconds)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["foreign_flow_5d", "foreign_positive_days_5d"], ascending=False
    ).reset_index(drop=True)


def _records(frame: pd.DataFrame) -> list[dict]:
    safe = frame.replace([np.inf, -np.inf], np.nan)
    return json.loads(safe.to_json(orient="records", date_format="iso"))


def render_dashboard(
    data_by_ticker: dict[str, pd.DataFrame],
    strategies: pd.DataFrame,
    conditions: pd.DataFrame,
    quantiles: pd.DataFrame,
    correlations: pd.DataFrame,
    scanner: pd.DataFrame,
    cost_bps: float,
) -> Path:
    if not TEMPLATE_PATH.exists():
        raise RuntimeError(f"대시보드 템플릿이 없습니다: {TEMPLATE_PATH}")
    focus_ticker = next(iter(data_by_ticker))
    focus = data_by_ticker[focus_ticker].tail(120).reset_index(names="date")
    latest = []
    for ticker, data in data_by_ticker.items():
        row = data.iloc[-1]
        latest.append({
            "ticker": ticker,
            "date": data.index[-1],
            "close": row["close"],
            "foreign_flow_1d": row["foreign_flow_1d"],
            "foreign_flow_5d": row["foreign_flow_5d"],
            "foreign_flow_20d": row["foreign_flow_20d"],
            "foreign_positive_days_5d": row["foreign_positive_days_5d"],
            "foreign_holding_pct": row["foreign_holding_pct"],
            "foreign_holding_change_5d": row["foreign_holding_change_5d"],
            "usdkrw_close": row["usdkrw_close"],
            "usdkrw_return_5d": row["usdkrw_return_5d"],
            "donchian_position": row["donchian_position"],
            "foreign_signal": row["foreign_signal"],
        })

    payload = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "focus_ticker": focus_ticker,
        "cost_bps": cost_bps,
        "latest": _records(pd.DataFrame(latest)),
        "chart": _records(focus[[
            "date", "close", "foreign_flow_5d", "foreign_net_qty",
            "foreign_holding_pct", "usdkrw_close",
        ]]),
        "strategies": _records(strategies),
        "conditions": _records(conditions),
        "quantiles": _records(quantiles),
        "correlations": _records(correlations),
        "scanner": _records(scanner.head(30)) if not scanner.empty else [],
    }
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = template.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "foreign_flow_validation.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path


def save_results(
    data_by_ticker: dict[str, pd.DataFrame],
    strategies: pd.DataFrame,
    conditions: pd.DataFrame,
    quantiles: pd.DataFrame,
    correlations: pd.DataFrame,
    scanner: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    combined = pd.concat(data_by_ticker.values()).reset_index(names="date")
    outputs = {
        "foreign_flow_dataset.csv": combined,
        "foreign_flow_strategies.csv": strategies,
        "foreign_flow_conditions.csv": conditions,
        "foreign_flow_quantiles.csv": quantiles,
        "foreign_flow_correlations.csv": correlations,
        "foreign_flow_scan.csv": scanner,
    }
    for filename, frame in outputs.items():
        frame.to_csv(OUTPUT_DIR / filename, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="외국인 수급 비교 검증 대시보드 생성")
    parser.add_argument("--tickers", default="005930,000660", help="쉼표로 구분한 종목코드")
    parser.add_argument("--start", default="2024-01-01", help="검증 시작일")
    parser.add_argument("--end", default=None, help="검증 종료일")
    parser.add_argument("--round-trip-cost-bps", type=float, default=30.0,
                        help="당일 왕복 거래비용 가정(bp), 기본 30bp")
    parser.add_argument("--scan-top", type=int, default=50,
                        help="수급 후보를 검색할 KOSPI 시총 상위 종목 수, 0이면 생략")
    parser.add_argument("--max-pages", type=int, default=200,
                        help="종목별 수급 조회 최대 페이지")
    parser.add_argument("--bootstrap-reps", type=int, default=500,
                        help="조건부 초과수익 블록 부트스트랩 반복 수")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_tickers = (t.strip() for t in args.tickers.split(",") if t.strip())
    tickers = list(dict.fromkeys(
        ticker.zfill(6) if ticker.isdigit() and len(ticker) <= 6 else ticker
        for ticker in raw_tickers
    ))
    if not tickers:
        raise RuntimeError("검증할 종목코드가 없습니다.")
    if args.round_trip_cost_bps < 0:
        raise RuntimeError("거래비용은 0 이상이어야 합니다.")

    fx_close = fetch_fx_history(args.start, args.end)
    data_by_ticker = {}
    for ticker in tickers:
        print(f"[{ticker}] 가격·외국인 수급 결합 중...")
        data_by_ticker[ticker] = build_dataset(
            ticker, args.start, args.end, max_pages=args.max_pages, fx_close=fx_close
        )

    print("전략·조건·상관관계 비교 중...")
    strategies = compare_strategies(data_by_ticker, args.round_trip_cost_bps)
    conditions = build_condition_study(data_by_ticker, args.bootstrap_reps)
    quantiles = build_quantile_study(data_by_ticker)
    correlations = build_correlations(data_by_ticker)
    scanner = scan_foreign_candidates(args.scan_top)
    save_results(data_by_ticker, strategies, conditions, quantiles, correlations, scanner)
    dashboard = render_dashboard(
        data_by_ticker, strategies, conditions, quantiles, correlations,
        scanner, args.round_trip_cost_bps,
    )
    print(f"완료: {dashboard}")


if __name__ == "__main__":
    main()
