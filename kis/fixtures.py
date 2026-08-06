"""모의 응답 생성기.

실제 호출 전에 정규화·검사·저장 경로를 전부 돌려보기 위한 것이다. 스키마에서
필드 이름을 직접 읽어 만들기 때문에, 데이터셋을 새로 추가하면 모의 응답도 따라온다.
여기서 나온 값은 어떤 경우에도 검증이나 판단에 쓰지 않는다.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import pandas as pd

from .datasets import DATASETS, DATE, TEXT, Dataset
from .endpoints import ENDPOINTS
from .transport import HttpResponse

MOCK_SOURCE = "mock"

_BY_PATH = {endpoint.path: endpoint for endpoint in ENDPOINTS.values()}
_BY_ENDPOINT = {dataset.endpoint: dataset for dataset in DATASETS.values()}


def _unit(*parts: str) -> float:
    """부분 문자열들로부터 0~1 사이 값을 결정적으로 만든다."""
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / 0xFFFFFFFF


def _row(dataset: Dataset, ticker: str, date: pd.Timestamp) -> dict[str, Any]:
    key = date.strftime("%Y%m%d")
    base = 60000 + round(_unit(ticker, key, "base") * 20000)
    low = base - round(_unit(ticker, key, "low") * 1500)
    high = base + round(_unit(ticker, key, "high") * 1500)
    close = low + round((high - low) * _unit(ticker, key, "close"))
    open_ = low + round((high - low) * _unit(ticker, key, "open"))
    volume = 2_000_000 + round(_unit(ticker, key, "vol") * 8_000_000)
    trading_value = volume * close

    record: dict[str, Any] = {}
    for field in dataset.fields:
        column = field.column
        if field.kind == DATE:
            value: Any = key if column == "date" else (date + pd.Timedelta(days=2)).strftime("%Y%m%d")
        elif field.kind == TEXT:
            value = "00" if column == "rights_code" else "N"
        elif column == "open":
            value = open_
        elif column == "high":
            value = high
        elif column == "low":
            value = low
        elif column in {"close", "short_avg_price"}:
            value = close
        elif column.startswith("index_"):
            index_base = 4000 + _unit(ticker, key, "index") * 400
            if column == "index_change_pct":
                value = round((_unit(ticker, key, "chg") - 0.5) * 4, 2)
            elif column == "index_change":
                value = round((_unit(ticker, key, "chg") - 0.5) * 80, 2)
            else:
                value = round(index_base + _unit(ticker, key, column) * 30, 2)
        elif column == "volume":
            value = volume
        elif column == "trading_value":
            value = trading_value
        elif column.endswith("ratio"):
            value = round(_unit(ticker, key, column) * 20, 2)
        elif column.endswith("_qty") or column.endswith("_change"):
            signed = (_unit(ticker, key, column) - 0.5) * 2
            value = round(signed * volume * 0.15)
        elif column.endswith("_value"):
            signed = (_unit(ticker, key, column) - 0.5) * 2
            value = round(signed * volume * 0.15) * close
        elif column == "split_rate":
            value = 1.0
        else:
            value = round(_unit(ticker, key, column) * 1000)
        record[field.raw] = str(value)
    return record


class MockTransport:
    """KisClient에 그대로 끼울 수 있는 가짜 전송기."""

    def __init__(self, lookback_days: int = 30) -> None:
        self.lookback_days = lookback_days
        self.calls: list[tuple[str, Mapping[str, Any]]] = []

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
    ) -> HttpResponse:
        if url.endswith("/oauth2/tokenP"):
            return HttpResponse(
                200, {}, {"access_token": "MOCK-TOKEN", "expires_in": 86400}
            )
        if url.endswith("/oauth2/revokeP"):
            return HttpResponse(200, {}, {"code": 200, "message": "폐기 완료"})

        path = url.split(":9443", 1)[-1].split(":29443", 1)[-1]
        endpoint = _BY_PATH.get(path)
        if endpoint is None:
            return HttpResponse(404, {}, {"rt_cd": "1", "msg_cd": "MOCK404", "msg1": path})
        dataset = _BY_ENDPOINT[endpoint.name]
        self.calls.append((endpoint.name, dict(payload)))

        ticker = str(
            payload.get("FID_INPUT_ISCD") or payload.get("MKSC_SHRN_ISCD") or "000000"
        )
        dates = self._dates(payload)
        records = [_row(dataset, ticker, date) for date in dates]
        body = {
            "rt_cd": "0",
            "msg_cd": "MCA00000",
            "msg1": "모의 응답",
            dataset.output: records,
        }
        return HttpResponse(200, {"tr_cont": "D"}, body)

    def _dates(self, payload: Mapping[str, Any]) -> list[pd.Timestamp]:
        start = payload.get("FID_INPUT_DATE_1") or payload.get("START_DATE") or ""
        end = payload.get("FID_INPUT_DATE_2") or payload.get("END_DATE") or ""
        if start and end:
            first, last = pd.Timestamp(str(start)), pd.Timestamp(str(end))
        elif start:
            last = pd.Timestamp(str(start))
            first = last - pd.Timedelta(days=self.lookback_days * 2)
        else:
            last = pd.Timestamp.today().normalize()
            first = last - pd.Timedelta(days=self.lookback_days * 2)
        days = pd.bdate_range(first, last)
        return list(days[-self.lookback_days :])
