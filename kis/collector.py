"""데이터셋별 조회 계획. 엔드포인트마다 기간을 다루는 방식이 달라서 여기서 흡수한다.

- range: 시작일과 종료일을 함께 받는다. 앞에서 뒤로 잘라서 요청한다.
- anchor_back: 기준일 하나만 받고 그 이전 며칠을 돌려준다. 뒤에서 앞으로 거슬러 간다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import pandas as pd

from .client import KisClient
from .datasets import Dataset, get_dataset
from .endpoints import (
    INDEX_KOSPI,
    INDEX_SEGMENT_KOSPI,
    MARKET_INDEX,
    MARKET_KRX,
    get_endpoint,
)
from .normalize import normalize

RANGE = "range"
ANCHOR_BACK = "anchor_back"

ParamBuilder = Callable[[str, str, str], Mapping[str, object]]


@dataclass(frozen=True)
class Plan:
    mode: str
    build: ParamBuilder
    chunk_days: int = 100
    # 한 페이지에 30건만 주는 엔드포인트가 있다. 10년치면 100회를 넘기므로 여유를 둔다.
    max_calls: int = 500


def _compact(value: str) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


# 대차거래 엔드포인트의 시장 구분값. 지수 코드를 이 값으로 옮긴다.
INDEX_TO_LOAN_MARKET = {INDEX_KOSPI: "1", "1001": "2"}


PLANS: dict[str, Plan] = {
    "investor_flow_daily": Plan(
        ANCHOR_BACK,
        lambda ticker, _start, anchor: {
            "FID_COND_MRKT_DIV_CODE": MARKET_KRX,
            "FID_INPUT_ISCD": ticker,
            "FID_INPUT_DATE_1": _compact(anchor),
        },
    ),
    # 2026-08-06 실호출로 확인: FID_INPUT_DATE_2는 종료일이 아니다. DATE_1을 기준일로
    # 삼아 그 이전 300거래일을 돌려준다. 범위로 다루면 최근 구간이 통째로 빠진다.
    "market_investor_flow_daily": Plan(
        ANCHOR_BACK,
        lambda index_code, _start, anchor: {
            "FID_COND_MRKT_DIV_CODE": MARKET_INDEX,
            "FID_INPUT_ISCD": index_code,
            "FID_INPUT_DATE_1": _compact(anchor),
            "FID_INPUT_DATE_2": _compact(anchor),
            "FID_INPUT_ISCD_1": INDEX_SEGMENT_KOSPI,
            "FID_INPUT_ISCD_2": index_code,
        },
    ),
    "market_price_daily": Plan(
        RANGE,
        lambda index_code, start, end: {
            "FID_COND_MRKT_DIV_CODE": MARKET_INDEX,
            "FID_INPUT_ISCD": index_code,
            "FID_INPUT_DATE_1": _compact(start),
            "FID_INPUT_DATE_2": _compact(end),
            "FID_PERIOD_DIV_CODE": "D",
        },
        # 이 엔드포인트는 한 번에 50건만 준다. 요청 범위가 그보다 길면 오래된 쪽을
        # 조용히 잘라낸다(2026-08-06 확인). 60일이면 거래일 42일 정도라 안전하다.
        chunk_days=60,
    ),
    "price_daily": Plan(
        RANGE,
        lambda ticker, start, end: {
            "FID_COND_MRKT_DIV_CODE": MARKET_KRX,
            "FID_INPUT_ISCD": ticker,
            "FID_INPUT_DATE_1": _compact(start),
            "FID_INPUT_DATE_2": _compact(end),
            "FID_PERIOD_DIV_CODE": "D",
            "FID_ORG_ADJ_PRC": "0",
        },
        chunk_days=100,
    ),
    "short_sale_daily": Plan(
        RANGE,
        lambda ticker, start, end: {
            "FID_COND_MRKT_DIV_CODE": MARKET_KRX,
            "FID_INPUT_ISCD": ticker,
            "FID_INPUT_DATE_1": _compact(start),
            "FID_INPUT_DATE_2": _compact(end),
        },
        chunk_days=60,
    ),
    "credit_balance_daily": Plan(
        ANCHOR_BACK,
        lambda ticker, _start, anchor: {
            "FID_COND_MRKT_DIV_CODE": MARKET_KRX,
            "FID_COND_SCR_DIV_CODE": "20476",
            "FID_INPUT_ISCD": ticker,
            "FID_INPUT_DATE_1": _compact(anchor),
        },
    ),
    # MRKT_DIV_CLS_CODE=3이라야 종목별이다. 공식 예제가 쓰는 1은 코스피 시장 전체이고
    # 그 경우 MKSC_SHRN_ISCD가 조용히 무시된다. 2026-08-06 실호출로 확인.
    "loan_trans_daily": Plan(
        RANGE,
        lambda ticker, start, end: {
            "MRKT_DIV_CLS_CODE": "3",
            "MKSC_SHRN_ISCD": ticker,
            "START_DATE": _compact(start),
            "END_DATE": _compact(end),
        },
        chunk_days=60,
    ),
    "market_loan_trans_daily": Plan(
        RANGE,
        lambda index_code, start, end: {
            "MRKT_DIV_CLS_CODE": INDEX_TO_LOAN_MARKET[index_code],
            "MKSC_SHRN_ISCD": index_code,
            "START_DATE": _compact(start),
            "END_DATE": _compact(end),
        },
        chunk_days=60,
    ),
    "program_trade_daily": Plan(
        ANCHOR_BACK,
        lambda ticker, _start, anchor: {
            "FID_COND_MRKT_DIV_CODE": MARKET_KRX,
            "FID_INPUT_ISCD": ticker,
            "FID_INPUT_DATE_1": _compact(anchor),
        },
    ),
}


def get_plan(dataset: Dataset | str) -> Plan:
    name = dataset if isinstance(dataset, str) else dataset.name
    try:
        return PLANS[name]
    except KeyError:
        raise KeyError(f"{name}의 조회 계획이 정의되지 않았습니다.") from None


def collect(
    client: KisClient,
    dataset: Dataset | str,
    ticker: str,
    start: str,
    end: str,
    *,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """한 종목·한 데이터셋을 [start, end] 구간까지 모아 정규화해 돌려준다."""
    dataset = dataset if isinstance(dataset, Dataset) else get_dataset(dataset)
    plan = get_plan(dataset)
    endpoint = get_endpoint(dataset.endpoint)
    ticker = str(ticker).strip()
    if dataset.pads_ticker:
        ticker = ticker.zfill(6)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts > end_ts:
        raise ValueError("start는 end보다 늦을 수 없습니다.")

    records: list[dict[str, Any]] = []
    truncated = 0
    if plan.mode == RANGE:
        cursor = start_ts
        calls = 0
        while cursor <= end_ts and calls < plan.max_calls:
            chunk_end = min(cursor + pd.Timedelta(days=plan.chunk_days - 1), end_ts)
            values = plan.build(ticker, cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"))
            page = client.fetch(endpoint, values)[dataset.output]
            records.extend(page)
            calls += 1
            # 건수 상한이 있는 엔드포인트는 오래된 쪽을 조용히 잘라낸다. 요청한
            # 시작일보다 늦은 날짜만 돌아오면 잘렸다고 보고 알린다.
            oldest = _oldest_date(dataset, page)
            if oldest and pd.Timestamp(oldest) > cursor + pd.Timedelta(days=7):
                truncated += 1
            if progress:
                progress(
                    f"{dataset.name} {ticker} "
                    f"{cursor:%Y-%m-%d}~{chunk_end:%Y-%m-%d} {len(page)}건"
                    + (f"  [잘림? 가장 이른 {oldest}]" if oldest and truncated else "")
                )
            cursor = chunk_end + pd.Timedelta(days=1)
        if truncated and progress:
            progress(
                f"경고: {dataset.name} {ticker} — {truncated}개 구간에서 응답이 잘린 것으로 "
                f"보입니다. chunk_days를 줄이세요."
            )
    else:
        anchor = end_ts
        calls = 0
        while anchor >= start_ts and calls < plan.max_calls:
            values = plan.build(ticker, start, anchor.strftime("%Y-%m-%d"))
            page = client.fetch(endpoint, values)[dataset.output]
            calls += 1
            if not page:
                break
            records.extend(page)
            oldest = _oldest_date(dataset, page)
            if progress:
                progress(
                    f"{dataset.name} {ticker} ~{anchor:%Y-%m-%d} {len(page)}건 "
                    f"(가장 이른 {oldest or '없음'})"
                )
            if oldest is None:
                break
            oldest_ts = pd.Timestamp(oldest)
            if oldest_ts <= start_ts or oldest_ts >= anchor:
                break
            anchor = oldest_ts - pd.Timedelta(days=1)

    frame = normalize(dataset, records, ticker)
    if frame.empty:
        return frame
    keep = (frame["date"] >= start_ts.strftime("%Y-%m-%d")) & (
        frame["date"] <= end_ts.strftime("%Y-%m-%d")
    )
    return frame.loc[keep].reset_index(drop=True)


def _oldest_date(dataset: Dataset, records: list[Mapping[str, Any]]) -> str | None:
    from .normalize import parse_date

    raw_key = dataset.date_field.raw
    dates = [parse_date(record.get(raw_key)) for record in records]
    valid = sorted(value for value in dates if value)
    return valid[0] if valid else None
