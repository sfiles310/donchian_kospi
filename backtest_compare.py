# -*- coding: utf-8 -*-
"""돈치안20 전략 vs 코스피 Buy&Hold 수익률 비교 백테스트"""
import sys
from datetime import datetime
import numpy as np
import pandas as pd

from donchian_kospi_daily import compute_signals

START = sys.argv[1] if len(sys.argv) > 1 else '2005-01-01'
FEE   = 0.0003   # 편도 거래비용 0.03% (ETF 기준, 세금 없음)


def fetch(start: str) -> pd.DataFrame:
    end = datetime.now().strftime('%Y%m%d')
    s = start.replace('-', '')
    try:
        from pykrx import stock
        df = stock.get_index_ohlcv(s, end, '1001')
        cm = {}
        for c in df.columns:
            if '시' in c: cm[c] = 'open'
            elif '고' in c: cm[c] = 'high'
            elif '저' in c: cm[c] = 'low'
            elif '종' in c: cm[c] = 'close'
        df = df.rename(columns=cm)[['open', 'high', 'low', 'close']].astype(float)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except Exception as e:
        print(f'pykrx 실패({e}) → FDR', file=sys.stderr)
        import FinanceDataReader as fdr
        df = fdr.DataReader('KS11', start)
        df.columns = [c.lower() for c in df.columns]
        return df[['open', 'high', 'low', 'close']].astype(float).dropna().sort_index()


def stats(eq: pd.Series, name: str, ntrade=None) -> dict:
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    ret = eq.iloc[-1] / eq.iloc[0]
    cagr = ret ** (1 / yrs) - 1
    dd = eq / eq.cummax() - 1
    r = eq.pct_change().dropna()
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() else np.nan
    return {
        '전략': name,
        '총수익률': f'{(ret - 1) * 100:,.1f}%',
        'CAGR': f'{cagr * 100:.2f}%',
        'MDD': f'{dd.min() * 100:.1f}%',
        'Sharpe': f'{sharpe:.2f}',
        '변동성': f'{r.std() * np.sqrt(252) * 100:.1f}%',
        '매매횟수': ntrade if ntrade is not None else '-',
    }


def main():
    df = fetch(START)
    d = compute_signals(df, 20, 20)
    d = d.dropna(subset=['dc_high', 'dc_low']).copy()

    # 신호 다음날 시가 체결 가정 → 포지션 1일 지연
    d['pos'] = d['position'].shift(1).fillna(0)
    d['mkt'] = d['close'].pct_change().fillna(0)
    d['strat'] = d['mkt'] * d['pos']

    # 포지션 변경일에 거래비용 차감
    trades = (d['pos'].diff().abs() > 0)
    d.loc[trades, 'strat'] -= FEE
    ntrade = int(trades.sum())

    eq_s = (1 + d['strat']).cumprod()
    eq_m = (1 + d['mkt']).cumprod()

    print(f"\n기간: {d.index[0].date()} ~ {d.index[-1].date()} "
          f"({(d.index[-1]-d.index[0]).days/365.25:.1f}년)  거래비용 편도 {FEE*100:.2f}%\n")
    res = pd.DataFrame([
        stats(eq_s, '돈치안20', ntrade),
        stats(eq_m, '코스피 Buy&Hold'),
    ])
    print(res.to_string(index=False))

    # 연도별 비교
    yr = pd.DataFrame({
        '돈치안20': (1 + d['strat']).groupby(d.index.year).prod() - 1,
        'B&H':      (1 + d['mkt']).groupby(d.index.year).prod() - 1,
    })
    yr['차이'] = yr['돈치안20'] - yr['B&H']
    print('\n[연도별 수익률]')
    print((yr * 100).round(1).to_string())

    out = pd.DataFrame({'donchian': eq_s, 'buyhold': eq_m})
    out.to_csv('output/backtest_equity.csv', encoding='utf-8-sig')
    print('\n→ output/backtest_equity.csv 저장')


if __name__ == '__main__':
    main()
