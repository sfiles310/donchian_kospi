"""KRX 전 종목 패널로 외국인 수급 가설을 검증하는 별도 연구 도구.

이 파일은 실거래 주문을 만들지 않는다. 공식 point-in-time 패널이 없거나
품질 게이트를 통과하지 못하면 성과 수치를 출력하지 않고 NOT_READY/FAIL로 끝난다.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"

REQUIRED_COLUMNS = {
    "date",
    "ticker",
    "open",
    "close",
    "volume",
    "trading_value",
    "foreign_net_value",
    "market_cap",
    "is_listed",
    "is_tradable",
}
OPTIONAL_COLUMNS = {
    "data_available_at",
    "asof",
    "sector_code",
    "exit_reason",
    "last_trading_date",
}


@dataclass
class GateResult:
    status: str
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    execution_lag_days: int = 1


def _parse_bool(series: pd.Series, column: str) -> pd.Series:
    mapping = {
        "1": True,
        "0": False,
        "true": True,
        "false": False,
        "y": True,
        "n": False,
        "yes": True,
        "no": False,
    }
    normalized = series.astype(str).str.strip().str.lower().map(mapping)
    if normalized.isna().any():
        bad = sorted(series[normalized.isna()].astype(str).unique())[:5]
        raise ValueError(f"{column} 값은 true/false 또는 1/0이어야 합니다: {bad}")
    return normalized.astype(bool)


def load_panel(path: Path) -> pd.DataFrame:
    """정규화된 UTF-8 CSV를 읽고 타입을 고정한다."""
    data = pd.read_csv(path, dtype={"ticker": str})
    missing = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing:
        raise ValueError(f"필수 열 누락: {', '.join(missing)}")

    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["ticker"] = data["ticker"].str.strip().str.zfill(6)
    for column in (
        "open",
        "close",
        "volume",
        "trading_value",
        "foreign_net_value",
        "market_cap",
    ):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in ("is_listed", "is_tradable"):
        data[column] = _parse_bool(data[column], column)
    if "data_available_at" in data:
        data["data_available_at"] = pd.to_datetime(
            data["data_available_at"], errors="coerce", utc=True
        )
    elif "asof" in data:
        data["data_available_at"] = pd.to_datetime(data["asof"], errors="coerce", utc=True)
    return data.sort_values(["date", "ticker"]).reset_index(drop=True)


def validate_panel(
    data: pd.DataFrame,
    research_end: str,
    validation_end: str,
    min_daily_names: int = 30,
    min_segment_days: int = 252,
    requested_lag_days: int = 1,
) -> GateResult:
    """수익 계산 전에 중단해야 할 데이터 오류를 검사한다."""
    reasons: list[str] = []
    warnings: list[str] = []
    missing = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing:
        return GateResult("FAIL", [f"필수 열 누락: {', '.join(missing)}"])

    if data.empty:
        return GateResult("NOT_READY", ["입력 패널이 비어 있습니다."])
    if data["date"].isna().any():
        reasons.append("해석할 수 없는 date가 있습니다.")
    if data["ticker"].isna().any() or data["ticker"].astype(str).str.strip().eq("").any():
        reasons.append("비어 있는 ticker가 있습니다.")
    if data.duplicated(["date", "ticker"]).any():
        reasons.append("date+ticker 중복 행이 있습니다.")

    numeric = ["open", "close", "volume", "trading_value", "foreign_net_value", "market_cap"]
    if data[numeric].isna().any().any():
        reasons.append("필수 숫자 열에 결측/비숫자 값이 있습니다.")
    if data[["volume", "trading_value", "market_cap"]].lt(0).any().any():
        reasons.append("거래량·거래대금·시가총액에 음수가 있습니다.")
    active = data["is_listed"] & data["is_tradable"]
    if data.loc[active, ["open", "close", "trading_value", "market_cap"]].le(0).any().any():
        reasons.append("거래가능 행의 가격·거래대금·시가총액은 0보다 커야 합니다.")
    impossible_flow = data["foreign_net_value"].abs() > data["trading_value"] * 1.001
    if impossible_flow.any():
        reasons.append("외국인 순매수대금 절댓값이 전체 거래대금을 넘는 행이 있습니다.")

    if requested_lag_days < 1:
        reasons.append("신호 지연은 최소 1거래일이어야 합니다.")
    lag_days = max(1, requested_lag_days)
    if "data_available_at" not in data or data["data_available_at"].isna().all():
        lag_days = max(2, lag_days)
        warnings.append(
            "수급 데이터의 이용 가능 시각이 없어 보수적으로 T+2 시가 진입을 적용합니다."
        )
    elif data["data_available_at"].isna().any():
        reasons.append("data_available_at가 일부 행에서만 비어 있습니다.")

    research_end_ts = pd.Timestamp(research_end)
    validation_end_ts = pd.Timestamp(validation_end)
    if research_end_ts >= validation_end_ts:
        reasons.append("research_end는 validation_end보다 빨라야 합니다.")
    dates = pd.Index(data["date"].dropna().unique()).sort_values()
    segments = {
        "research": dates[dates <= research_end_ts],
        "validation": dates[(dates > research_end_ts) & (dates <= validation_end_ts)],
        "holdout": dates[dates > validation_end_ts],
    }
    for name, segment_dates in segments.items():
        if len(segment_dates) < min_segment_days:
            reasons.append(
                f"{name} 구간 거래일이 {len(segment_dates)}일로 최소 {min_segment_days}일보다 적습니다."
            )

    eligible = active & data["trading_value"].gt(0)
    daily_names = data.loc[eligible].groupby("date")["ticker"].nunique()
    if daily_names.empty or int(daily_names.min()) < min_daily_names:
        observed = int(daily_names.min()) if not daily_names.empty else 0
        reasons.append(
            f"일별 거래가능 종목 수 최솟값이 {observed}개로 기준 {min_daily_names}개보다 적습니다."
        )

    return GateResult("FAIL" if reasons else "PASS", reasons, warnings, lag_days)


def _calendar_shift(dates: pd.Series, calendar: pd.DatetimeIndex, steps: int) -> pd.Series:
    positions = pd.Series(np.arange(len(calendar)), index=calendar)
    mapped = dates.map(positions)
    target = mapped + steps
    result = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    valid = target.notna() & target.lt(len(calendar))
    result.loc[valid] = calendar[target.loc[valid].astype(int)].to_numpy()
    return result


def build_trades(
    data: pd.DataFrame,
    horizon_days: int,
    execution_lag_days: int,
    min_trading_value: float,
    quantile: float = 0.10,
    sector_neutral: bool = False,
) -> tuple[pd.DataFrame, int]:
    """T일 횡단면 순위를 이후 거래일의 실제 가격 행에 연결한다."""
    if not 0 < quantile < 0.5:
        raise ValueError("quantile은 0과 0.5 사이여야 합니다.")
    if horizon_days < 1:
        raise ValueError("horizon_days는 1 이상이어야 합니다.")
    if sector_neutral and "sector_code" not in data:
        raise ValueError("섹터중립 검증에는 point-in-time sector_code가 필요합니다.")

    calendar = pd.DatetimeIndex(sorted(data["date"].dropna().unique()))
    eligible = data[
        data["is_listed"]
        & data["is_tradable"]
        & data["trading_value"].ge(min_trading_value)
        & data["open"].gt(0)
        & data["close"].gt(0)
    ].copy()
    eligible["flow_intensity"] = eligible["foreign_net_value"] / eligible["trading_value"]
    rank_groups = ["date", "sector_code"] if sector_neutral else ["date"]
    group_sizes = eligible.groupby(rank_groups)["ticker"].transform("size")
    eligible = eligible[group_sizes.ge(10)].copy()
    eligible["flow_rank"] = eligible.groupby(rank_groups)["flow_intensity"].rank(
        method="first", pct=True
    )
    eligible["bucket"] = "middle"
    eligible.loc[eligible["flow_rank"].le(quantile), "bucket"] = "bottom"
    eligible.loc[eligible["flow_rank"].gt(1 - quantile), "bucket"] = "top"
    eligible["signal_date"] = eligible["date"]
    eligible["entry_date"] = _calendar_shift(
        eligible["signal_date"], calendar, execution_lag_days
    )
    eligible["exit_date"] = _calendar_shift(
        eligible["entry_date"], calendar, horizon_days - 1
    )

    signal_columns = ["signal_date", "entry_date", "exit_date", "ticker", "bucket"]
    if sector_neutral:
        signal_columns.append("sector_code")
    signals = eligible[signal_columns].dropna(subset=["entry_date", "exit_date"])

    entry = data[["date", "ticker", "open", "is_listed", "is_tradable"]].rename(
        columns={
            "date": "entry_date",
            "open": "entry_open",
            "is_listed": "entry_listed",
            "is_tradable": "entry_tradable",
        }
    )
    exit_prices = data[["date", "ticker", "close", "is_tradable"]].rename(
        columns={
            "date": "exit_date",
            "close": "exit_close",
            "is_tradable": "exit_tradable",
        }
    )
    trades = signals.merge(entry, on=["entry_date", "ticker"], how="left")
    trades = trades.merge(exit_prices, on=["exit_date", "ticker"], how="left")
    selected = trades["bucket"].isin(["top", "bottom"])
    executable = (
        trades["entry_listed"].fillna(False)
        & trades["entry_tradable"].fillna(False)
        & trades["exit_tradable"].fillna(False)
        & trades["entry_open"].gt(0)
        & trades["exit_close"].gt(0)
    )
    missing_selected_exits = int((selected & ~executable).sum())
    trades = trades[executable].copy()
    trades["gross_return"] = trades["exit_close"] / trades["entry_open"] - 1
    return trades, missing_selected_exits


def daily_spreads(trades: pd.DataFrame, horizon_days: int, cost_bps: float) -> pd.DataFrame:
    """동일가중 상·하위 수익과 전체 유니버스 벤치마크를 일자별로 만든다."""
    grouped = trades.groupby(["entry_date", "bucket"])["gross_return"].mean().unstack()
    benchmark = trades.groupby("entry_date")["gross_return"].mean().rename("universe_gross")
    result = grouped.join(benchmark).dropna(subset=["top", "bottom"])
    if horizon_days > 1:
        result = result.iloc[::horizon_days]
    leg_cost = cost_bps / 10_000
    result["top_net"] = result["top"] - leg_cost
    result["universe_net"] = result["universe_gross"] - leg_cost
    result["top_excess_vs_universe"] = result["top"] - result["universe_gross"]
    result["spread_net"] = result["top"] - result["bottom"] - 2 * leg_cost
    return result


def _block_bootstrap_mean(
    values: pd.Series, repetitions: int, block_size: int, seed: int
) -> tuple[float, float, float, float]:
    sample = values.dropna().to_numpy(dtype=float)
    if not len(sample):
        return math.nan, math.nan, math.nan, math.nan
    observed = float(sample.mean())
    if repetitions <= 0 or len(sample) < 2:
        return observed, math.nan, math.nan, math.nan
    block_size = max(1, min(block_size, len(sample)))
    blocks_needed = math.ceil(len(sample) / block_size)
    offsets = np.arange(block_size)
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions)
    for index in range(repetitions):
        starts = rng.integers(0, len(sample), blocks_needed)
        indices = ((starts[:, None] + offsets) % len(sample)).ravel()[: len(sample)]
        means[index] = sample[indices].mean()
    low, high = np.quantile(means, [0.025, 0.975])
    one_sided_p = (np.count_nonzero(means <= 0) + 1) / (repetitions + 1)
    return observed, float(low), float(high), float(one_sided_p)


def _placebo_95(
    gross_spread: pd.Series, cost_bps: float, repetitions: int, block_size: int, seed: int
) -> float:
    """일별 상·하위 방향을 블록 단위로 뒤집은 위약 분포의 95% 경계."""
    sample = gross_spread.dropna().to_numpy(dtype=float)
    if not len(sample) or repetitions <= 0:
        return math.nan
    block_size = max(1, min(block_size, len(sample)))
    block_count = math.ceil(len(sample) / block_size)
    rng = np.random.default_rng(seed)
    placebo = np.empty(repetitions)
    for index in range(repetitions):
        signs = rng.choice([-1.0, 1.0], size=block_count)
        expanded = np.repeat(signs, block_size)[: len(sample)]
        placebo[index] = (sample * expanded).mean() - 2 * cost_bps / 10_000
    return float(np.quantile(placebo, 0.95))


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().astype(float).sort_values()
    if valid.empty:
        return result
    count = len(valid)
    adjusted = valid * count / np.arange(1, count + 1)
    adjusted = adjusted.iloc[::-1].cummin().iloc[::-1].clip(upper=1.0)
    result.loc[adjusted.index] = adjusted
    return result


def _risk_stats(values: pd.Series) -> dict[str, float | int]:
    sample = values.dropna().astype(float)
    if sample.empty:
        return {
            "total_return": math.nan,
            "mdd": math.nan,
            "win_rate": math.nan,
            "profit_factor": math.nan,
            "max_losing_streak": 0,
        }
    equity = (1 + sample).cumprod()
    equity_with_start = pd.concat([pd.Series([1.0]), equity.reset_index(drop=True)])
    losses = sample[sample < 0]
    gains = sample[sample > 0]
    losing_streak = longest = 0
    for losing in sample.lt(0):
        losing_streak = losing_streak + 1 if losing else 0
        longest = max(longest, losing_streak)
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "mdd": float((equity_with_start / equity_with_start.cummax() - 1).min()),
        "win_rate": float(sample.gt(0).mean()),
        "profit_factor": (
            float(gains.sum() / abs(losses.sum())) if losses.sum() < 0 else math.inf
        ),
        "max_losing_streak": longest,
    }


def evaluate_hypotheses(
    data: pd.DataFrame,
    gate: GateResult,
    research_end: str,
    validation_end: str,
    min_trading_value: float = 1_000_000_000,
    cost_bps: float = 30,
    bootstrap_repetitions: int = 500,
    min_observations: int = 60,
    max_allowed_mdd: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """사전 고정한 세 가설만 계산하고 검증/홀드아웃으로 판정한다."""
    hypotheses = [
        ("H1", "외국인 강도 상위-하위 익일 장중", 1, False),
        ("H2", "외국인 강도 상위-하위 5일 보유", 5, False),
    ]
    if "sector_code" in data and data["sector_code"].notna().all():
        hypotheses.append(("H3", "섹터중립 외국인 강도 상위-하위 익일 장중", 1, True))

    research_end_ts = pd.Timestamp(research_end)
    validation_end_ts = pd.Timestamp(validation_end)
    detail_rows: list[dict] = []
    missing_exits: dict[str, int] = {}
    for code, name, horizon, sector_neutral in hypotheses:
        trades, missing = build_trades(
            data,
            horizon,
            gate.execution_lag_days,
            min_trading_value,
            sector_neutral=sector_neutral,
        )
        missing_exits[code] = missing
        for multiplier in (1.0, 1.5, 2.0):
            spreads = daily_spreads(trades, horizon, cost_bps * multiplier)
            segments = {
                "research": spreads.loc[spreads.index <= research_end_ts],
                "validation": spreads.loc[
                    (spreads.index > research_end_ts) & (spreads.index <= validation_end_ts)
                ],
                "holdout": spreads.loc[spreads.index > validation_end_ts],
            }
            for segment_name, segment in segments.items():
                observed, low, high, p_value = _block_bootstrap_mean(
                    segment["spread_net"],
                    bootstrap_repetitions,
                    block_size=10,
                    seed=sum(ord(char) for char in f"{code}-{segment_name}-{multiplier}"),
                )
                top_observed, top_low, top_high, top_p_value = _block_bootstrap_mean(
                    segment["top_net"],
                    bootstrap_repetitions,
                    block_size=10,
                    seed=sum(ord(char) for char in f"top-{code}-{segment_name}-{multiplier}"),
                )
                placebo = _placebo_95(
                    segment["top"] - segment["bottom"],
                    cost_bps * multiplier,
                    bootstrap_repetitions,
                    block_size=10,
                    seed=sum(ord(char) for char in f"placebo-{code}-{segment_name}-{multiplier}"),
                )
                risk = _risk_stats(segment["top_net"])
                detail_rows.append(
                    {
                        "hypothesis": code,
                        "description": name,
                        "segment": segment_name,
                        "horizon_days": horizon,
                        "sector_neutral": sector_neutral,
                        "cost_multiplier": multiplier,
                        "observations": len(segment),
                        "spread_net_mean": observed,
                        "ci_95_low": low,
                        "ci_95_high": high,
                        "one_sided_p_value": p_value,
                        "placebo_95": placebo,
                        "top_net_mean": top_observed,
                        "top_net_ci_95_low": top_low,
                        "top_net_ci_95_high": top_high,
                        "top_net_one_sided_p_value": top_p_value,
                        "top_excess_vs_universe": segment["top_excess_vs_universe"].mean(),
                        "top_total_return": risk["total_return"],
                        "top_mdd": risk["mdd"],
                        "top_win_rate": risk["win_rate"],
                        "top_profit_factor": risk["profit_factor"],
                        "top_max_losing_streak": risk["max_losing_streak"],
                    }
                )

    details = pd.DataFrame(detail_rows)
    decisions: list[dict] = []
    details["spread_fdr_q_value"] = np.nan
    details["top_net_fdr_q_value"] = np.nan
    for multiplier in (1.0, 1.5, 2.0):
        holdout_mask = details["segment"].eq("holdout") & details[
            "cost_multiplier"
        ].eq(multiplier)
        details.loc[holdout_mask, "spread_fdr_q_value"] = benjamini_hochberg(
            details.loc[holdout_mask, "one_sided_p_value"]
        )
        details.loc[holdout_mask, "top_net_fdr_q_value"] = benjamini_hochberg(
            details.loc[holdout_mask, "top_net_one_sided_p_value"]
        )
    stress = details[details["cost_multiplier"].eq(2.0)].copy()
    for code, name, _, _ in hypotheses:
        validation = stress[(stress["hypothesis"] == code) & (stress["segment"] == "validation")]
        holdout = stress[(stress["hypothesis"] == code) & (stress["segment"] == "holdout")]
        reasons = []
        status = "FAIL"
        if validation.empty or holdout.empty:
            status = "NOT_READY"
            reasons.append("검증 또는 홀드아웃 결과가 없습니다.")
        else:
            val = validation.iloc[0]
            test = holdout.iloc[0]
            if val["observations"] < min_observations or test["observations"] < min_observations:
                status = "NOT_READY"
                reasons.append(f"표본 수가 구간별 최소 {min_observations}개보다 적습니다.")
            if missing_exits[code]:
                status = "NOT_READY"
                reasons.append(
                    f"선택 종목 중 {missing_exits[code]}건의 실제 진입/청산 가격 경로가 없습니다."
                )
            if max_allowed_mdd is None:
                status = "NOT_READY"
                reasons.append("최대 허용 MDD가 사전 등록되지 않았습니다.")
            if status != "NOT_READY":
                checks = {
                    "검증구간 절대수익 95% 하한이 0 초과": val["top_net_ci_95_low"] > 0,
                    "홀드아웃 절대수익 FDR q<0.05": test["top_net_fdr_q_value"] < 0.05,
                    "검증·홀드아웃 절대수익이 0 초과": (
                        val["top_net_mean"] > 0 and test["top_net_mean"] > 0
                    ),
                    "검증구간 신호우위 95% 하한이 0 초과": val["ci_95_low"] > 0,
                    "홀드아웃 신호우위 FDR q<0.05": test["spread_fdr_q_value"] < 0.05,
                    "홀드아웃 신호우위가 위약 95% 초과": (
                        test["spread_net_mean"] > test["placebo_95"]
                    ),
                    "검증·홀드아웃 시장평균 초과": (
                        val["top_excess_vs_universe"] > 0
                        and test["top_excess_vs_universe"] > 0
                    ),
                    "검증·홀드아웃 MDD 한도 준수": (
                        val["top_mdd"] >= -max_allowed_mdd
                        and test["top_mdd"] >= -max_allowed_mdd
                    ),
                }
                failed = [label for label, passed in checks.items() if not passed]
                if failed:
                    reasons.extend(failed)
                else:
                    status = "PASS"
        decisions.append(
            {
                "hypothesis": code,
                "description": name,
                "status": status,
                "reason": " / ".join(reasons) if reasons else "모든 사전 기준 충족",
            }
        )
    if "H3" not in {row["hypothesis"] for row in decisions}:
        decisions.append(
            {
                "hypothesis": "H3",
                "description": "섹터중립 외국인 강도 상위-하위 익일 장중",
                "status": "NOT_READY",
                "reason": "point-in-time sector_code가 없습니다.",
            }
        )
    return pd.DataFrame(decisions), details


def _write_summary(payload: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "krx_foreign_flow_panel_summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="공식 KRX 전 종목 패널 기반 외국인 수급 가설 검증기"
    )
    parser.add_argument("--input", type=Path, help="정규화된 UTF-8 CSV")
    parser.add_argument(
        "--data-kind",
        choices=("krx-real", "unverified", "synthetic"),
        default="unverified",
        help="krx-real을 명시한 실데이터만 성과 검증",
    )
    parser.add_argument("--research-end", help="연구구간 마지막 날짜 YYYY-MM-DD")
    parser.add_argument("--validation-end", help="검증구간 마지막 날짜 YYYY-MM-DD")
    parser.add_argument("--signal-lag-days", type=int, default=1)
    parser.add_argument("--min-daily-names", type=int, default=30)
    parser.add_argument("--min-segment-days", type=int, default=252)
    parser.add_argument("--min-observations", type=int, default=60)
    parser.add_argument("--min-trading-value", type=float, default=1_000_000_000)
    parser.add_argument("--round-trip-cost-bps", type=float, default=30)
    parser.add_argument("--bootstrap-reps", type=int, default=500)
    parser.add_argument(
        "--max-allowed-mdd",
        type=float,
        help="사전 등록할 최대 허용 낙폭(예: 0.10은 -10%%)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = {
        "tool": "KRX 외국인 수급 point-in-time 패널 검증기",
        "notice": "자동매매 신호나 수익 보장이 아닌 연구용 검증 결과입니다.",
    }
    if not args.input:
        payload = {
            **base,
            "status": "NOT_READY",
            "reasons": ["공식 KRX 전 종목 패널 CSV가 지정되지 않았습니다."],
        }
        path = _write_summary(payload)
        print(f"NOT_READY: {payload['reasons'][0]}\n요약: {path}")
        return
    if args.data_kind != "krx-real":
        payload = {
            **base,
            "status": "NOT_READY",
            "reasons": ["실데이터 출처가 확인되지 않아 성과 수치를 계산하지 않았습니다."],
        }
        path = _write_summary(payload)
        print(f"NOT_READY: {payload['reasons'][0]}\n요약: {path}")
        return
    if not args.research_end or not args.validation_end:
        payload = {
            **base,
            "status": "NOT_READY",
            "reasons": ["research-end와 validation-end를 사전에 지정해야 합니다."],
        }
        path = _write_summary(payload)
        print(f"NOT_READY: {payload['reasons'][0]}\n요약: {path}")
        return
    if args.max_allowed_mdd is None or not 0 < args.max_allowed_mdd < 1:
        payload = {
            **base,
            "status": "NOT_READY",
            "reasons": ["0과 1 사이의 max-allowed-mdd를 실행 전에 지정해야 합니다."],
        }
        path = _write_summary(payload)
        print(f"NOT_READY: {payload['reasons'][0]}\n요약: {path}")
        return

    try:
        data = load_panel(args.input)
        gate = validate_panel(
            data,
            args.research_end,
            args.validation_end,
            args.min_daily_names,
            args.min_segment_days,
            args.signal_lag_days,
        )
    except (OSError, ValueError) as error:
        gate = GateResult("FAIL", [str(error)])
        data = pd.DataFrame()

    if gate.status != "PASS":
        payload = {
            **base,
            "status": gate.status,
            "reasons": gate.reasons,
            "warnings": gate.warnings,
        }
        path = _write_summary(payload)
        print(f"{gate.status}: {' / '.join(gate.reasons)}\n요약: {path}")
        return

    decisions, details = evaluate_hypotheses(
        data,
        gate,
        args.research_end,
        args.validation_end,
        args.min_trading_value,
        args.round_trip_cost_bps,
        args.bootstrap_reps,
        args.min_observations,
        args.max_allowed_mdd,
    )
    if decisions["status"].eq("PASS").any():
        overall = "PASS"
    elif decisions["status"].eq("FAIL").any():
        overall = "FAIL"
    else:
        overall = "NOT_READY"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    decisions_path = OUTPUT_DIR / "krx_foreign_flow_panel_decisions.csv"
    details_path = OUTPUT_DIR / "krx_foreign_flow_panel_details.csv"
    decisions.to_csv(decisions_path, index=False, encoding="utf-8-sig")
    details.to_csv(details_path, index=False, encoding="utf-8-sig")
    payload = {
        **base,
        "status": overall,
        "data_gate": "PASS",
        "execution_lag_days": gate.execution_lag_days,
        "warnings": gate.warnings,
        "hypotheses": decisions.to_dict(orient="records"),
        "decision_rule": (
            "2배 비용에서 절대수익과 신호우위가 모두 검증 CI>0, "
            "홀드아웃 FDR q<0.05, 위약 95% 초과, 시장평균 초과, MDD 한도 준수"
        ),
        "outputs": [str(decisions_path), str(details_path)],
    }
    summary_path = _write_summary(payload)
    print(decisions.to_string(index=False))
    print(f"\n전체 판정: {overall}\n요약: {summary_path}")


if __name__ == "__main__":
    main()
