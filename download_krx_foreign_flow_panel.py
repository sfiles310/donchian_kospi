"""Download a resumable KOSPI daily price/foreign-flow panel from KRX."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = [
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
    "data_available_at",
    "sector_code",
    "exit_reason",
    "last_trading_date",
]


def normalize_day(price: pd.DataFrame, flow: pd.DataFrame, date: str) -> pd.DataFrame:
    """Convert one KRX trading day to the validation panel schema."""
    price_columns = {"시가", "종가", "거래량", "거래대금", "시가총액"}
    missing = sorted(price_columns - set(price.columns))
    if missing:
        raise ValueError(f"가격 데이터 필수 열 누락: {', '.join(missing)}")
    if price.empty:
        raise ValueError(f"{date} 가격 데이터가 비어 있습니다.")
    if flow.empty or "순매수거래대금" not in flow.columns:
        raise ValueError(f"{date} 외국인 수급 데이터가 비어 있거나 열이 누락됐습니다.")

    price = price.copy()
    flow = flow.copy()
    price.index = price.index.astype(str).str.strip().str.zfill(6)
    flow.index = flow.index.astype(str).str.strip().str.zfill(6)
    foreign_net = pd.to_numeric(flow["순매수거래대금"], errors="raise")

    panel = pd.DataFrame(index=price.index)
    panel["date"] = pd.Timestamp(date).strftime("%Y-%m-%d")
    panel["ticker"] = panel.index
    panel["open"] = pd.to_numeric(price["시가"], errors="raise").astype("int64")
    panel["close"] = pd.to_numeric(price["종가"], errors="raise").astype("int64")
    panel["volume"] = pd.to_numeric(price["거래량"], errors="raise").astype("int64")
    panel["trading_value"] = pd.to_numeric(
        price["거래대금"], errors="raise"
    ).astype("int64")
    panel["foreign_net_value"] = foreign_net.reindex(panel.index).fillna(0).astype("int64")
    panel["market_cap"] = pd.to_numeric(price["시가총액"], errors="raise").astype(
        "int64"
    )
    panel["is_listed"] = True
    panel["is_tradable"] = (
        panel["open"].gt(0)
        & panel["close"].gt(0)
        & panel["volume"].gt(0)
        & panel["trading_value"].gt(0)
    )
    panel["data_available_at"] = f"{panel['date'].iat[0]}T18:00:00+09:00"
    panel["sector_code"] = ""
    panel["exit_reason"] = ""
    panel["last_trading_date"] = ""

    impossible = panel["foreign_net_value"].abs() > panel["trading_value"] * 1.001
    if impossible.any():
        raise ValueError(f"{date} 외국인 순매수대금이 거래대금을 넘는 행이 있습니다.")
    return panel[OUTPUT_COLUMNS].reset_index(drop=True)


def _existing_dates(output: Path) -> set[str]:
    if not output.exists() or output.stat().st_size == 0:
        return set()
    dates = pd.read_csv(output, usecols=["date"])["date"]
    return set(pd.to_datetime(dates, errors="raise").dt.strftime("%Y-%m-%d"))


def _sort_output(output: Path) -> None:
    data = pd.read_csv(output, dtype={"ticker": str})
    data = data.drop_duplicates(["date", "ticker"], keep="last")
    data = data.sort_values(["date", "ticker"])
    temporary = output.with_suffix(".tmp")
    data.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    temporary.replace(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KRX KOSPI 일별 외국인 수급 패널 수집기")
    parser.add_argument("--start", required=True, help="시작일 YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="종료일 YYYY-MM-DD")
    parser.add_argument(
        "--output", type=Path, default=Path("data/krx_kospi_panel.csv")
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--max-retries", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.getenv("KRX_ID") or not os.getenv("KRX_PW"):
        raise SystemExit("KRX_ID와 KRX_PW 환경변수를 설정한 뒤 다시 실행하세요.")

    from pykrx import stock

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if start > end:
        raise SystemExit("start는 end보다 늦을 수 없습니다.")

    calendar = stock.get_index_ohlcv_by_date(
        start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), "1001"
    )
    if calendar.empty:
        raise SystemExit("KRX 거래일을 조회하지 못했습니다.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = _existing_dates(args.output)
    pending = [date for date in calendar.index if date.strftime("%Y-%m-%d") not in completed]
    print(f"수집 대상 {len(pending)}일 (완료 {len(completed)}일)")

    for position, date in enumerate(pending, start=1):
        compact = date.strftime("%Y%m%d")
        iso_date = date.strftime("%Y-%m-%d")
        for attempt in range(1, max(1, args.max_retries) + 1):
            try:
                price = stock.get_market_ohlcv_by_ticker(compact, market="KOSPI")
                flow = stock.get_market_net_purchases_of_equities_by_ticker(
                    compact, compact, "KOSPI", "외국인"
                )
                daily = normalize_day(price, flow, iso_date)
                break
            except (KeyError, TypeError, ValueError) as error:
                if attempt >= max(1, args.max_retries):
                    raise
                wait_seconds = min(30, 2**attempt)
                print(
                    f"{iso_date} 조회 실패 ({attempt}/{args.max_retries}): "
                    f"{error}; {wait_seconds}초 후 재시도"
                )
                time.sleep(wait_seconds)
        daily.to_csv(
            args.output,
            mode="a",
            header=not args.output.exists() or args.output.stat().st_size == 0,
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )
        print(f"[{position}/{len(pending)}] {iso_date}: {len(daily)}행")
        time.sleep(max(0.0, args.sleep_seconds))

    _sort_output(args.output)
    print(f"완료: {args.output.resolve()}")


if __name__ == "__main__":
    main()
