"""데이터셋을 SQLite 한 파일에 모으는 point-in-time 저장소.

여러 CSV로 흩어놓으면 출처가 늘어날수록 결합이 깨진다. 표 하나당 (date, ticker)
기본키를 두고, 열이 늘어나면 표를 그 자리에서 확장한다.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd

from .datasets import DATE, FLOAT, INT, TEXT, Dataset

_SQL_TYPES = {INT: "INTEGER", FLOAT: "REAL", DATE: "TEXT", TEXT: "TEXT"}
# 최근 이 기간의 종가 차이로 판정한다. 배당 보정은 최근일수록 0에 가깝다.
RECENT_WINDOW = 60
# 최근 구간에서 이 비율 넘게 벌어지면 보정이 아니라 다른 대상을 받아온 것으로 본다.
RECENT_MISMATCH_LIMIT = 0.02
_META_TYPES = {
    "ticker": "TEXT",
    "source": "TEXT",
    "data_available_at": "TEXT",
    "collected_at": "TEXT",
    "is_provisional": "INTEGER",
}


def _key(dataset: Dataset | str, code: object) -> str:
    """종목코드는 6자리로 맞추고 지수 코드는 원형을 지킨다."""
    text = str(code).strip()
    if isinstance(dataset, Dataset) and not dataset.pads_ticker:
        return text
    return text.zfill(6)


class PanelStore:
    """수집기와 검증기가 공유하는 유일한 저장 경로."""

    def __init__(self, path: Path | str = Path("data/kis_panel.sqlite")) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute("PRAGMA journal_mode=WAL")

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "PanelStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ 쓰기

    def ensure_table(self, dataset: Dataset) -> None:
        columns = [f'"{field.column}" {_SQL_TYPES[field.kind]}' for field in dataset.fields]
        columns += [f'"{name}" {sql_type}' for name, sql_type in _META_TYPES.items()]
        self._connection.execute(
            f'CREATE TABLE IF NOT EXISTS "{dataset.name}" '
            f'({", ".join(columns)}, PRIMARY KEY (date, ticker))'
        )
        existing = self._columns(dataset.name)
        for definition, name in zip(columns, list(dataset.columns) + list(_META_TYPES)):
            if name not in existing:
                self._connection.execute(
                    f'ALTER TABLE "{dataset.name}" ADD COLUMN {definition}'
                )
        self._connection.commit()

    def upsert(self, dataset: Dataset, frame: pd.DataFrame) -> int:
        """같은 (date, ticker)는 나중 값으로 덮어쓴다. 재실행이 항상 안전해야 한다."""
        if frame.empty:
            return 0
        self.ensure_table(dataset)

        columns = [name for name in frame.columns if name in self._columns(dataset.name)]
        payload = frame[columns].copy()
        if "is_provisional" in payload:
            payload["is_provisional"] = payload["is_provisional"].astype("int64")
        payload = payload.astype(object).where(pd.notna(payload), None)

        placeholders = ", ".join("?" for _ in columns)
        quoted = ", ".join(f'"{name}"' for name in columns)
        self._connection.executemany(
            f'INSERT OR REPLACE INTO "{dataset.name}" ({quoted}) VALUES ({placeholders})',
            [tuple(row) for row in payload.itertuples(index=False, name=None)],
        )
        self._connection.commit()
        return len(payload)

    def delete(self, dataset: Dataset | str, *, tickers: Iterable[str] | None = None) -> int:
        """잘못 수집한 구간을 지운다. 파라미터를 틀리게 넣으면 조용히 다른 대상의
        값이 저장되므로, 덮어쓰기만으로는 남는 행이 생긴다."""
        name = dataset if isinstance(dataset, str) else dataset.name
        if name not in self.tables():
            return 0
        if tickers is None:
            cursor = self._connection.execute(f'DELETE FROM "{name}"')
        else:
            codes = [_key(dataset, code) for code in tickers]
            if not codes:
                return 0
            cursor = self._connection.execute(
                f'DELETE FROM "{name}" WHERE ticker IN ({", ".join("?" for _ in codes)})',
                codes,
            )
        self._connection.commit()
        return cursor.rowcount

    # ------------------------------------------------------------------ 읽기

    def tables(self) -> list[str]:
        rows = self._connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        return [row[0] for row in rows]

    def read(
        self,
        dataset: Dataset | str,
        *,
        tickers: Iterable[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        confirmed_only: bool = True,
    ) -> pd.DataFrame:
        """검증에는 기본적으로 확정치만 넘긴다."""
        name = dataset if isinstance(dataset, str) else dataset.name
        if name not in self.tables():
            return pd.DataFrame()

        clauses: list[str] = []
        params: list[object] = []
        if tickers is not None:
            codes = [_key(dataset, code) for code in tickers]
            if not codes:
                return pd.DataFrame()
            clauses.append(f"ticker IN ({', '.join('?' for _ in codes)})")
            params.extend(codes)
        if start:
            clauses.append("date >= ?")
            params.append(start)
        if end:
            clauses.append("date <= ?")
            params.append(end)
        if confirmed_only:
            clauses.append("is_provisional = 0")

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        frame = pd.read_sql_query(
            f'SELECT * FROM "{name}"{where} ORDER BY date, ticker',
            self._connection,
            params=params,
        )
        if "is_provisional" in frame.columns:
            frame["is_provisional"] = frame["is_provisional"].astype(bool)
        return frame

    def coverage(self, dataset: Dataset | str) -> pd.DataFrame:
        """종목별로 어디까지 모였는지. 이어받기와 상태 보고의 근거."""
        name = dataset if isinstance(dataset, str) else dataset.name
        if name not in self.tables():
            return pd.DataFrame(columns=["ticker", "rows", "first_date", "last_date"])
        return pd.read_sql_query(
            f'SELECT ticker, COUNT(*) AS rows, MIN(date) AS first_date, '
            f'MAX(date) AS last_date FROM "{name}" GROUP BY ticker ORDER BY ticker',
            self._connection,
        )

    def earliest_date(self, dataset: Dataset | str, ticker: str) -> str | None:
        name = dataset if isinstance(dataset, str) else dataset.name
        if name not in self.tables():
            return None
        row = self._connection.execute(
            f'SELECT MIN(date) FROM "{name}" WHERE ticker = ?',
            (_key(dataset, ticker),),
        ).fetchone()
        return row[0] if row and row[0] else None

    def latest_date(self, dataset: Dataset | str, ticker: str) -> str | None:
        name = dataset if isinstance(dataset, str) else dataset.name
        if name not in self.tables():
            return None
        row = self._connection.execute(
            f'SELECT MAX(date) FROM "{name}" WHERE ticker = ?',
            (_key(dataset, ticker),),
        ).fetchone()
        return row[0] if row and row[0] else None

    def cross_check_close(self, ticker: str, *, tolerance: int = 1) -> pd.DataFrame:
        """모든 종목 데이터셋의 종가가 price_daily와 맞는지 대조한다.

        파라미터를 잘못 넣으면 API가 오류 대신 엉뚱한 대상의 값을 돌려주는 경우가 있다.
        그때 데이터셋 안쪽 항등식은 멀쩡히 성립하므로 내부 검사로는 잡히지 않는다.
        종가는 거의 모든 데이터셋에 들어 있어 대조 기준으로 쓸 수 있다.
        """
        from .datasets import DATASETS

        price = self.read("price_daily", tickers=[ticker], confirmed_only=False)
        if price.empty:
            return pd.DataFrame(columns=["dataset", "compared", "mismatched", "verdict"])

        rows = []
        for dataset in DATASETS.values():
            if not dataset.pads_ticker or dataset.name == "price_daily":
                continue
            if "close" not in dataset.columns:
                continue
            other = self.read(dataset, tickers=[ticker], confirmed_only=False)
            if other.empty:
                continue
            merged = price[["date", "close"]].merge(
                other[["date", "close"]], on="date", suffixes=("_p", "_o")
            )
            if merged.empty:
                continue
            gap = (merged["close_p"] - merged["close_o"]).abs()
            mismatched = int((gap > tolerance).sum())
            relative = (gap / merged["close_p"].abs().replace(0, pd.NA)).dropna()
            median_gap = float(relative.median()) if len(relative) else 0.0
            # 배당·분배 보정은 누적이라 과거로 갈수록 벌어지고 최근에는 0에 수렴한다.
            # 전 구간 중앙값으로 보면 12년치에서 오탐하므로 최근 구간을 기준으로 삼는다.
            recent = float(relative.tail(RECENT_WINDOW).median()) if len(relative) else 0.0
            if mismatched == 0:
                verdict = "OK"
            elif recent < RECENT_MISMATCH_LIMIT:
                verdict = "수정주가 차이로 보임"
            else:
                verdict = "다른 대상의 값일 수 있음"
            rows.append(
                {
                    "dataset": dataset.name,
                    "compared": len(merged),
                    "mismatched": mismatched,
                    "median_gap": round(median_gap, 4),
                    "recent_gap": round(recent, 5),
                    "verdict": verdict,
                }
            )
        return pd.DataFrame(rows)

    def _columns(self, table: str) -> set[str]:
        rows = self._connection.execute(f'PRAGMA table_info("{table}")').fetchall()
        return {row[1] for row in rows}
