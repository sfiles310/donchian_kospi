# -*- coding: utf-8 -*-
"""가정별 민감도: 체결시점 / 기간 / 비용에 따른 돈치안20 vs B&H"""
import numpy as np, pandas as pd, FinanceDataReader as fdr
from donchian_kospi_daily import compute_signals

df = fdr.DataReader('KS11', '1995-01-01')
df.columns = [c.lower() for c in df.columns]
df = df[['open', 'high', 'low', 'close']].astype(float).dropna().sort_index()
d = compute_signals(df, 20, 20).dropna(subset=['dc_high', 'dc_low']).copy()
d['mkt'] = d['close'].pct_change().fillna(0)


def run(sub, lag, fee):
    pos = sub['position'].shift(lag).fillna(0) if lag else sub['position']
    r = sub['mkt'] * pos
    r = r - (pos.diff().abs().fillna(0) > 0) * fee
    eq = (1 + r).cumprod()
    eqm = (1 + sub['mkt']).cumprod()
    yrs = (sub.index[-1] - sub.index[0]).days / 365.25
    f = lambda e: (e.iloc[-1] ** (1 / yrs) - 1) * 100
    mdd = lambda e: (e / e.cummax() - 1).min() * 100
    return f(eq), f(eqm), mdd(eq), mdd(eqm), eq.iloc[-1], eqm.iloc[-1]


print('[1] 체결시점 가정 (2005~현재, 비용 0.03%)')
sub = d[d.index >= '2005-01-01']
for lag, name in [(0, '신호당일 종가체결(낙관)'), (1, '다음날 체결(현실)')]:
    a, b, ma, mb, ea, eb = run(sub, lag, 0.0003)
    print(f'  {name:22s} 돈치안 CAGR {a:5.2f}% / B&H {b:5.2f}%   배수 {ea:6.1f}x vs {eb:5.1f}x')

print('\n[2] 시작연도별 (다음날 체결, 비용 0.03%)')
for s in ['1996', '2000', '2005', '2010', '2015', '2020']:
    sub = d[d.index >= f'{s}-01-01']
    a, b, ma, mb, ea, eb = run(sub, 1, 0.0003)
    print(f'  {s}~  돈치안 {a:6.2f}% (MDD{ma:6.1f}%) | B&H {b:6.2f}% (MDD{mb:6.1f}%) | 차이 {a-b:+.2f}%p')

print('\n[3] 2025년 대세상승 제외 (2005~2024말)')
sub = d[(d.index >= '2005-01-01') & (d.index <= '2024-12-31')]
a, b, ma, mb, ea, eb = run(sub, 1, 0.0003)
print(f'  돈치안 {a:.2f}% (MDD{ma:.1f}%) | B&H {b:.2f}% (MDD{mb:.1f}%) | 차이 {a-b:+.2f}%p')

print('\n[4] 배당 반영 B&H (연 1.8% 가산, 2005~현재)')
sub = d[d.index >= '2005-01-01'].copy()
a, b, ma, mb, ea, eb = run(sub, 1, 0.0003)
print(f'  돈치안 {a:.2f}% | B&H+배당 {b+1.8:.2f}%')
