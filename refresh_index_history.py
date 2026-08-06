# -*- coding: utf-8 -*-
"""커밋된 지수 거래량 이력을 KIS Open API로 갱신한다.

일일 실행에는 필요 없다. 네이버가 최근 110일을 대주므로 평소에는 이 파일을 건드릴
일이 없고, 몇 달에 한 번 과거 구간을 채워 넣을 때만 로컬에서 돌린다.

이렇게 나눠 두는 이유는 주문 권한이 있는 KIS 키를 CI에 두지 않기 위해서다.
GitHub Actions는 커밋된 CSV와 네이버만 쓴다.

    python refresh_index_history.py            # 마지막 날짜 이후만 이어받기
    python refresh_index_history.py --full     # 2013년부터 다시 받기
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from kis import Credentials, KisClient, collect, get_dataset

OUTPUT = Path('data/kospi_index_history.csv')
COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume']
DEFAULT_START = '2013-01-02'
INDEX_CODE = '0001'


def load_existing() -> pd.DataFrame:
    if not OUTPUT.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(OUTPUT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='지수 거래량 이력 갱신')
    parser.add_argument('--start', default=None, help='시작일 YYYY-MM-DD')
    parser.add_argument('--end', default=None, help='종료일 YYYY-MM-DD')
    parser.add_argument('--full', action='store_true', help=f'{DEFAULT_START}부터 다시 받기')
    parser.add_argument('--index', default=INDEX_CODE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    existing = load_existing()
    end = args.end or pd.Timestamp.today().strftime('%Y-%m-%d')

    if args.start:
        start = args.start
    elif args.full or existing.empty:
        start = DEFAULT_START
    else:
        # 마지막 날 하루 전부터 받아 경계가 비지 않게 한다.
        last = pd.Timestamp(existing['date'].max())
        start = (last - pd.Timedelta(days=5)).strftime('%Y-%m-%d')

    print(f'수집 {start} ~ {end}  (기존 {len(existing)}행)')
    dataset = get_dataset('market_price_daily')
    with KisClient(Credentials.from_env()) as client:
        frame = collect(client, dataset, args.index, start, end,
                        progress=lambda text: print(f'  {text}'))

    if frame.empty:
        print('새로 받은 자료가 없습니다.')
        return 1

    # 장 마감 전 값은 커밋하지 않는다. 확정 전 행은 is_provisional로 표시돼 있다.
    provisional = frame[frame['is_provisional']]
    if not provisional.empty:
        print(f'  확정 전 {len(provisional)}행 제외: '
              f'{", ".join(provisional["date"].tolist())}')
        frame = frame[~frame['is_provisional']]
    if frame.empty:
        print('확정된 새 자료가 없습니다.')
        return 1

    fresh = frame[['date', 'index_open', 'index_high', 'index_low',
                   'index_close', 'volume']].copy()
    fresh.columns = COLUMNS
    merged = pd.concat([existing, fresh], ignore_index=True)
    merged = merged.drop_duplicates(subset=['date'], keep='last').sort_values('date')
    merged['volume'] = merged['volume'].astype('int64')

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT, index=False, encoding='utf-8',
                  lineterminator='\n', float_format='%.2f')
    added = len(merged) - len(existing)
    print(f'\n{OUTPUT} 저장: {len(merged)}행 (신규 {added}행)')
    print(f'  {merged["date"].min()} ~ {merged["date"].max()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
