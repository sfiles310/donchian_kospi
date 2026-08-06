"""KIS 수급·시세 수집기 진입점.

기본은 모의 응답으로 도는 예행 모드다. 실제 호출은 `--live`를 붙이고 확인을
거쳐야 시작한다. 주문 API는 호출하지 않는다.

    python collect_kis_flow.py                       # 모의 응답으로 전체 경로 점검
    python collect_kis_flow.py --status              # 지금까지 모인 범위 확인
    python collect_kis_flow.py --live --tickers 005930 --start 2024-01-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from kis import (
    DATASETS,
    Credentials,
    KisApiError,
    KisClient,
    PanelStore,
    check_consistency,
    collect,
    get_dataset,
)
from kis.endpoints import INDEX_KOSPI
from kis.fixtures import MockTransport
from kis.transport import requests_transport

DEFAULT_DB = Path("data/kis_panel.sqlite")
DEFAULT_DATASETS = (
    "market_investor_flow_daily",
    "investor_flow_daily",
    "price_daily",
    "short_sale_daily",
    "program_trade_daily",
    "credit_balance_daily",
    "loan_trans_daily",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KIS 국내주식 수급 패널 수집기")
    parser.add_argument("--tickers", default="005930", help="쉼표로 구분한 6자리 종목코드")
    parser.add_argument(
        "--indices",
        default=INDEX_KOSPI,
        help="지수 단위 데이터셋에 쓸 지수 코드. 기본 0001(코스피)",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help=f"수집할 데이터셋. 사용 가능: {', '.join(sorted(DATASETS))}",
    )
    parser.add_argument("--start", default=None, help="시작일 YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="종료일 YYYY-MM-DD")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--live", action="store_true", help="실제 KIS API를 호출한다 (기본은 모의 응답)"
    )
    parser.add_argument("--demo", action="store_true", help="모의투자 도메인 사용")
    parser.add_argument("--yes", action="store_true", help="실호출 확인 절차를 건너뛴다")
    parser.add_argument("--status", action="store_true", help="저장 현황만 출력하고 끝낸다")
    parser.add_argument("--per-second", type=float, default=None, help="초당 호출 상한")
    return parser.parse_args(argv)


def show_status(store: PanelStore) -> None:
    tables = [name for name in store.tables() if name in DATASETS]
    if not tables:
        print(f"{store.path}에 아직 수집된 데이터셋이 없습니다.")
        return
    for name in tables:
        coverage = store.coverage(name)
        print(f"\n[{name}] {DATASETS[name].description}")
        if coverage.empty:
            print("  (비어 있음)")
            continue
        for row in coverage.itertuples(index=False):
            print(
                f"  {row.ticker}  {row.first_date} ~ {row.last_date}  {row.rows}행"
            )


def confirm_live(
    tickers: list[str], indices: list[str], names: list[str], start: str, end: str
) -> bool:
    print("실제 KIS API를 호출합니다. 조회 전용이며 주문은 보내지 않습니다.")
    print(f"  종목    : {', '.join(tickers) or '없음'}")
    print(f"  지수    : {', '.join(indices) or '없음'}")
    print(f"  데이터셋: {', '.join(names)}")
    print(f"  기간    : {start} ~ {end}")
    answer = input("진행하려면 yes를 입력하세요: ").strip().lower()
    return answer == "yes"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    store = PanelStore(args.db)

    if args.status:
        show_status(store)
        store.close()
        return 0

    tickers = [code.strip().zfill(6) for code in args.tickers.split(",") if code.strip()]
    indices = [code.strip() for code in args.indices.split(",") if code.strip()]
    names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    for name in names:
        get_dataset(name)
    if not names or not (tickers or indices):
        raise SystemExit("데이터셋과 대상 코드를 각각 하나 이상 지정하세요.")

    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    start = args.start or (pd.Timestamp(end) - pd.Timedelta(days=120)).strftime("%Y-%m-%d")

    if args.live:
        if not args.yes and not confirm_live(tickers, indices, names, start, end):
            print("취소했습니다.")
            store.close()
            return 1
        credentials = Credentials.from_env()
        transport = requests_transport
    else:
        print("모의 응답 모드입니다. 저장되는 값은 검증에 쓸 수 없는 가짜입니다.")
        credentials = Credentials(app_key="mock", app_secret="mock")
        transport = MockTransport()

    warnings: list[str] = []
    total = 0
    with KisClient(
        credentials,
        demo=args.demo,
        transport=transport,
        per_second=args.per_second,
    ) as client:
        for name in names:
            dataset = get_dataset(name)
            # 지수 단위 데이터셋은 지수 코드로, 나머지는 종목코드로 돈다.
            for ticker in (tickers if dataset.pads_ticker else indices):
                try:
                    frame = collect(
                        client, dataset, ticker, start, end, progress=lambda text: print(f"  {text}")
                    )
                except KisApiError as error:
                    print(f"  ! {name} {ticker} 건너뜀: {error}", file=sys.stderr)
                    continue
                warnings.extend(check_consistency(dataset, frame))
                saved = store.upsert(dataset, frame)
                total += saved
                print(f"{name} {ticker}: {saved}행 저장")

    if warnings:
        print("\n확인이 필요한 항목")
        for text in warnings:
            print(f"  - {text}")

    print(f"\n합계 {total}행 -> {store.path.resolve()}")
    show_status(store)
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
