"""원시 KIS 응답을 표준 패널로 바꾼다. 통신도 파일 접근도 하지 않는다."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping

import pandas as pd

from .datasets import DATE, FLOAT, INT, TEXT, Dataset

KST = timezone(timedelta(hours=9))

# 모든 데이터셋이 공유하는 계보 열.
META_COLUMNS = ("ticker", "source", "data_available_at", "collected_at", "is_provisional")


def now_kst() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(tz=KST))


def parse_number(value: object) -> float | None:
    """KIS는 숫자를 문자열로 준다. 빈 값과 하이픈은 결측으로 본다."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip().replace(",", "").replace("%", "")
    if text in {"", "-", "--", "N/A"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_date(value: object) -> str | None:
    """YYYYMMDD 또는 YYYY-MM-DD를 ISO 날짜 문자열로 맞춘다."""
    if value is None:
        return None
    text = str(value).strip().replace("-", "").replace("/", "")
    if len(text) != 8 or not text.isdigit():
        return None
    try:
        return datetime.strptime(text, "%Y%m%d").strftime("%Y-%m-%d")
    except ValueError:
        return None


def available_at(dataset: Dataset, trade_date: str) -> pd.Timestamp:
    """해당 기준일 자료를 실제로 볼 수 있게 된 시각."""
    day = pd.Timestamp(trade_date) + pd.Timedelta(days=dataset.available_after_days)
    return pd.Timestamp(
        datetime(day.year, day.month, day.day, dataset.available_at_hour, tzinfo=KST)
    )


def _availability_basis(dataset: Dataset, frame: pd.DataFrame) -> pd.Series:
    """공개 시점을 재는 기준 날짜. 결제 기준 자료는 결제일에서 잰다."""
    basis = frame["date"]
    column = dataset.available_basis_column
    if column and column in frame.columns:
        # 결제일이 비어 있으면 매매일로 물러선다. 앞당겨 보는 쪽으로 기울지 않게
        # 둘 중 늦은 날짜를 쓴다.
        fallback = basis.where(frame[column].isna(), frame[column])
        basis = fallback.combine(basis, lambda a, b: max(a, b))
    return basis


def normalize(
    dataset: Dataset,
    records: Iterable[Mapping[str, Any]],
    ticker: str,
    *,
    collected_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """레코드 목록을 (date, ticker) 기준 패널로 정규화한다.

    응답에 없는 열은 조용히 결측으로 남긴다. KIS는 종목·시장에 따라 일부 필드를
    빼고 주기 때문에, 열 하나가 없다고 수집 전체를 중단시키지 않는다.
    """
    ticker = str(ticker).strip()
    if dataset.pads_ticker:
        ticker = ticker.zfill(6)
    collected_at = collected_at if collected_at is not None else now_kst()

    rows: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {}
        for field in dataset.fields:
            raw = record.get(field.raw)
            if field.kind == DATE:
                row[field.column] = parse_date(raw)
            elif field.kind == TEXT:
                row[field.column] = None if raw is None else str(raw).strip() or None
            else:
                row[field.column] = parse_number(raw)
        if row.get("date"):
            rows.append(row)

    frame = pd.DataFrame(rows, columns=list(dataset.columns))
    if frame.empty:
        return _empty(dataset)

    # 상장 전 날짜에도 API가 빈 행을 채워 보내는 경우가 있다(0167A0에서 779일 확인).
    # 거래된 날이라면 종가가 반드시 있으므로, 종가가 없는 행은 관측이 아니다.
    anchor = next(
        (name for name in ("close", "index_close") if name in frame.columns), None
    )
    if anchor is not None:
        frame = frame[frame[anchor].notna()]
        if frame.empty:
            return _empty(dataset)

    for field in dataset.fields:
        if field.kind == INT:
            numeric = pd.to_numeric(frame[field.column], errors="coerce")
            frame[field.column] = numeric.round().astype("Int64")
        elif field.kind == FLOAT:
            frame[field.column] = pd.to_numeric(frame[field.column], errors="coerce")

    frame["ticker"] = ticker
    frame["source"] = f"kis:{dataset.endpoint}"
    basis = _availability_basis(dataset, frame)
    stamps = [available_at(dataset, value) for value in basis]
    frame["data_available_at"] = [stamp.isoformat() for stamp in stamps]
    frame["collected_at"] = collected_at.isoformat()
    frame["is_provisional"] = [collected_at < stamp for stamp in stamps]

    frame = frame.drop_duplicates(subset=["date"], keep="last")
    frame = frame.sort_values("date").reset_index(drop=True)
    return frame[list(dataset.columns) + list(META_COLUMNS)]


def _empty(dataset: Dataset) -> pd.DataFrame:
    frame = pd.DataFrame(columns=list(dataset.columns) + list(META_COLUMNS))
    for field in dataset.fields:
        if field.kind == INT:
            frame[field.column] = frame[field.column].astype("Int64")
        elif field.kind == FLOAT:
            frame[field.column] = frame[field.column].astype("float64")
    frame["is_provisional"] = frame["is_provisional"].astype("bool")
    return frame


def check_consistency(dataset: Dataset, frame: pd.DataFrame) -> list[str]:
    """저장 전에 잡을 수 있는 모순만 경고로 모은다. 예외로 막지는 않는다."""
    warnings: list[str] = []
    if frame.empty:
        return warnings

    # 순매수 항등식은 네 주체로 닫힌다. 기타단체는 그 안에 든 세부 항목이고
    # 2019년부터는 0으로만 온다. 다섯을 더하면 2018년 이전 자료가 전부 깨진다.
    identity = [
        "foreign_net_qty",
        "individual_net_qty",
        "institution_net_qty",
        "etc_corp_net_qty",
    ]
    if set(identity) <= set(frame.columns):
        total = frame[identity].sum(axis=1).abs()
        broken = frame.loc[total > 1, "date"].tolist()
        if broken:
            warnings.append(
                f"{dataset.name}: 4주체 순매수 합이 0이 아닌 날 {len(broken)}건 "
                f"(예: {broken[0]})"
            )

    if {"trading_value", "foreign_net_value"} <= set(frame.columns):
        over = frame["foreign_net_value"].abs() > frame["trading_value"] * 1.001
        bad = frame.loc[over.fillna(False), "date"].tolist()
        if bad:
            warnings.append(
                f"{dataset.name}: 외국인 순매수대금이 거래대금을 넘는 날 {len(bad)}건 "
                f"(예: {bad[0]})"
            )

    if {"short_qty", "volume"} <= set(frame.columns):
        over = frame["short_qty"] > frame["volume"]
        bad = frame.loc[over.fillna(False), "date"].tolist()
        if bad:
            warnings.append(
                f"{dataset.name}: 공매도 수량이 거래량을 넘는 날 {len(bad)}건 (예: {bad[0]})"
            )

    if {"high", "low"} <= set(frame.columns):
        over = frame["high"] < frame["low"]
        bad = frame.loc[over.fillna(False), "date"].tolist()
        if bad:
            warnings.append(f"{dataset.name}: 고가가 저가보다 낮은 날 {len(bad)}건")

    provisional = int(frame["is_provisional"].sum())
    if provisional:
        warnings.append(
            f"{dataset.name}: 확정 전 잠정치 {provisional}건. 검증에서 제외하세요."
        )
    return warnings
