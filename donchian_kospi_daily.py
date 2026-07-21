# -*- coding: utf-8 -*-
"""
KOSPI 돈치안 20일 채널 전략 — 매일 자동 신호 생성기
─────────────────────────────────────────────────────
실행 시 동작:
  1) pykrx로 코스피 지수 OHLCV 최신 데이터 수집
  2) 돈치안 20일 채널 신호 계산
  3) 신호 변화 발생 시(BUY/SELL) 텔레그램 푸시
  4) HTML 대시보드 자동 갱신 (output/dashboard.html)
  5) 신호 이력 CSV 저장 (output/signals_log.csv)

작성일: 2026-05-09
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

# ─────────────────────────────────────────────────────────────────
# 설정 (사용자가 수정해야 할 부분)
# ─────────────────────────────────────────────────────────────────
CONFIG = {
    # 텔레그램 (BotFather → 봇 생성 후 토큰, /getUpdates로 chat_id 확인)
    'TELEGRAM_BOT_TOKEN': os.environ.get('TG_BOT_TOKEN', ''),   # 환경변수 권장
    'TELEGRAM_CHAT_ID':   os.environ.get('TG_CHAT_ID',   ''),

    # Supabase 발행 (앱 strategy_signal 테이블 upsert). 미설정 시 발행 생략.
    'SUPABASE_URL':         os.environ.get('SUPABASE_URL', ''),          # 예: https://xxxx.supabase.co
    'SUPABASE_SERVICE_KEY': os.environ.get('SUPABASE_SERVICE_KEY', ''),  # service_role 키(쓰기 권한)

    # 전략 파라미터
    'DC_WINDOW_ENTRY': 20,
    'DC_WINDOW_EXIT':  20,
    'FETCH_DAYS':      400,   # 채널 계산용 충분한 과거 데이터 (영업일 기준 약 1.5년)

    # 출력 경로 (스크립트와 같은 폴더 기준)
    'OUTPUT_DIR':      'output',

    # 항상 알림 vs 신호 발생일에만 알림
    'NOTIFY_ALWAYS':   True,   # True면 매일 현재 상태도 푸시

    # 지수 신호로 관리하는 종목 (코스피 상관 0.9 이상)
    'MANAGED': [
        ('148020', 'RISE 200'),
        ('0167A0', 'SOL AI반도체TOP2+'),
        ('292150', 'TIGER 코리아TOP10'),
    ],
    # 지수 신호 미적용 — 참고 표시만 (상관 0.71, 표본 부족)
    'REFERENCE': [
        ('0190C0', 'RISE 현대차그룹피지컬AI', '상관 0.71 · 재판정 2026-11'),
    ],
}

# ─────────────────────────────────────────────────────────────────
# 로깅
# ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
OUT_DIR = SCRIPT_DIR / CONFIG['OUTPUT_DIR']
OUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(OUT_DIR / 'donchian_daily.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# 데이터 수집 (pykrx 우선, 실패 시 FinanceDataReader 백업)
# ─────────────────────────────────────────────────────────────────
def fetch_kospi_data(days: int = 400) -> pd.DataFrame:
    """코스피 지수 OHLCV 가져오기. pykrx → fdr 순으로 시도."""
    end = datetime.now()
    start = end - timedelta(days=days * 1.6)  # 주말·휴일 고려해 넉넉히
    start_str = start.strftime('%Y%m%d')
    end_str   = end.strftime('%Y%m%d')

    # 1차: pykrx
    try:
        from pykrx import stock
        log.info(f"pykrx로 코스피 데이터 수집: {start_str} ~ {end_str}")
        df = stock.get_index_ohlcv(start_str, end_str, '1001')  # 1001=코스피
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={
            '시가':'open', '고가':'high', '저가':'low', '종가':'close', '거래량':'volume'
        })
        # pykrx는 한글 컬럼이 다를 수 있어 매핑 재정렬
        col_map = {}
        for c in df.columns:
            if '시' in c or 'open' in c.lower(): col_map[c] = 'open'
            elif '고' in c or 'high' in c.lower(): col_map[c] = 'high'
            elif '저' in c or 'low' in c.lower():  col_map[c] = 'low'
            elif '종' in c or 'close' in c.lower(): col_map[c] = 'close'
            elif '거래량' in c or 'volume' in c.lower(): col_map[c] = 'volume'
        df = df.rename(columns=col_map)
        df = df[['open','high','low','close']].astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        log.info(f"pykrx 수집 완료: {len(df)}행 ({df.index[0].date()} ~ {df.index[-1].date()})")
        return df
    except Exception as e:
        log.warning(f"pykrx 실패: {e}. FinanceDataReader로 fallback 시도")

    # 2차: FinanceDataReader 백업
    try:
        import FinanceDataReader as fdr
        log.info("FinanceDataReader로 코스피 데이터 수집")
        df = fdr.DataReader('KS11',
                            start.strftime('%Y-%m-%d'),
                            end.strftime('%Y-%m-%d'))
        df.columns = [c.lower() for c in df.columns]
        df = df[['open','high','low','close']].astype(float)
        df = df.dropna().sort_index()
        log.info(f"FDR 수집 완료: {len(df)}행")
        return df
    except Exception as e:
        log.error(f"모든 데이터 소스 실패: {e}")
        raise RuntimeError("코스피 데이터를 가져올 수 없습니다.")

# ─────────────────────────────────────────────────────────────────
# 돈치안 신호 계산
# ─────────────────────────────────────────────────────────────────
def compute_signals(df: pd.DataFrame,
                    n_entry: int = 20,
                    n_exit: int = 20) -> pd.DataFrame:
    """돈치안 20일 채널 신호 + 포지션 시뮬레이션"""
    d = df.copy()
    # shift(1): 룩어헤드 방지 — 어제까지의 채널로 오늘 신호 판단
    d['dc_high'] = d['high'].rolling(n_entry).max().shift(1)
    d['dc_low']  = d['low'].rolling(n_exit).min().shift(1)
    d['long_entry'] = d['high'] > d['dc_high']
    d['long_exit']  = d['low']  < d['dc_low']

    # 포지션 시뮬레이션
    pos = 0
    positions = []
    for i in range(len(d)):
        if pd.isna(d['dc_high'].iloc[i]):
            positions.append(0)
            continue
        if pos == 0 and d['long_entry'].iloc[i]:
            pos = 1
        elif pos == 1 and d['long_exit'].iloc[i]:
            pos = 0
        positions.append(pos)
    d['position'] = positions
    return d

# ─────────────────────────────────────────────────────────────────
# 신호 이력 관리
# ─────────────────────────────────────────────────────────────────
def update_signal_log(d: pd.DataFrame) -> pd.DataFrame:
    """포지션 변화 = 매매 신호. 이력 CSV 갱신"""
    log_path = OUT_DIR / 'signals_log.csv'

    new_signals = []
    for i in range(1, len(d)):
        p_prev = d['position'].iloc[i-1]
        p_curr = d['position'].iloc[i]
        if p_prev == 0 and p_curr == 1:
            new_signals.append({
                'date': d.index[i].strftime('%Y-%m-%d'),
                'type': 'BUY',
                'price': float(d['close'].iloc[i]),
                'dc_high': float(d['dc_high'].iloc[i]),
                'dc_low':  float(d['dc_low'].iloc[i]),
            })
        elif p_prev == 1 and p_curr == 0:
            new_signals.append({
                'date': d.index[i].strftime('%Y-%m-%d'),
                'type': 'SELL',
                'price': float(d['close'].iloc[i]),
                'dc_high': float(d['dc_high'].iloc[i]),
                'dc_low':  float(d['dc_low'].iloc[i]),
            })

    new_df = pd.DataFrame(new_signals)
    new_df.to_csv(log_path, index=False, encoding='utf-8-sig')
    log.info(f"신호 이력 저장: {log_path} ({len(new_df)}건)")
    return new_df

# ─────────────────────────────────────────────────────────────────
# Supabase 발행 — 앱(strategy_signal)이 읽는 최신 스냅샷 upsert
# ─────────────────────────────────────────────────────────────────
def publish_supabase(d: pd.DataFrame, signals: pd.DataFrame) -> bool:
    """최근 60일 차트 + 현재 국면을 앱 스키마로 변환해 strategy_signal에 upsert."""
    url = CONFIG['SUPABASE_URL'].rstrip('/')
    key = CONFIG['SUPABASE_SERVICE_KEY']
    if not url or not key:
        log.warning("SUPABASE_URL/SERVICE_KEY 미설정. Supabase 발행 생략")
        return False

    plot_df = d.dropna(subset=['dc_high', 'dc_low']).tail(60)
    chart = [{
        'date':     idx.strftime('%Y-%m-%d'),
        'close':    round(float(row['close']),   2),
        'dc_high':  round(float(row['dc_high']), 2),
        'dc_low':   round(float(row['dc_low']),  2),
        'position': int(row['position']),
    } for idx, row in plot_df.iterrows()]

    last = d.iloc[-1]
    last_sig = signals.iloc[-1] if not signals.empty else None
    payload = {
        'strategy':         'donchian_kospi',
        'as_of_date':       d.index[-1].strftime('%Y-%m-%d'),
        'close':            round(float(last['close']),   2),
        'dc_high':          round(float(last['dc_high']), 2),
        'dc_low':           round(float(last['dc_low']),  2),
        'position':         int(last['position']),
        'last_signal':      (last_sig['type'] if last_sig is not None else None),
        'last_signal_date': (last_sig['date'] if last_sig is not None else None),
        'chart':            chart,
        'updated_at':       datetime.now().isoformat(),
    }

    endpoint = f"{url}/rest/v1/strategy_signal?on_conflict=strategy"
    headers = {
        'apikey': key,
        'Authorization': f'Bearer {key}',
        'Content-Type': 'application/json',
        'Prefer': 'resolution=merge-duplicates',  # upsert
    }
    try:
        r = requests.post(endpoint, headers=headers, json=payload, timeout=15)
        r.raise_for_status()
        log.info(f"Supabase 발행 성공: {payload['as_of_date']} (pos={payload['position']})")
        return True
    except Exception as e:
        detail = getattr(getattr(e, 'response', None), 'text', '')
        log.error(f"Supabase 발행 실패: {e} {detail}")
        return False

# ─────────────────────────────────────────────────────────────────
# 텔레그램 푸시
# ─────────────────────────────────────────────────────────────────
def send_telegram(text: str) -> bool:
    """텔레그램 메시지 발송"""
    token = CONFIG['TELEGRAM_BOT_TOKEN']
    chat_id = CONFIG['TELEGRAM_CHAT_ID']

    if not token or not chat_id:
        log.warning("텔레그램 토큰/chat_id 미설정. 푸시 생략")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': text,
        'parse_mode': 'Markdown',
        'disable_web_page_preview': True,
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        log.info("텔레그램 푸시 성공")
        return True
    except Exception as e:
        log.error(f"텔레그램 푸시 실패: {e}")
        return False

# ─────────────────────────────────────────────────────────────────
# 보조 정보: 종목 현재가 / 채널 전망
# ─────────────────────────────────────────────────────────────────
def fetch_prices(items) -> dict:
    """종목별 최신 종가. 실패해도 신호 알림은 나가야 하므로 예외를 삼킨다."""
    out = {}
    for entry in items:
        code, name = entry[0], entry[1]
        try:
            import FinanceDataReader as fdr
            start = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
            e = fdr.DataReader(code, start)
            e.columns = [c.lower() for c in e.columns]
            px = e['close'].astype(float).dropna()
            out[code] = (float(px.iloc[-1]), float(px.pct_change().iloc[-1]) * 100)
        except Exception as ex:
            log.warning(f"{name}({code}) 가격 조회 실패: {ex}")
            out[code] = None
    return out


def project_dc_high(d: pd.DataFrame, n: int = 20, steps=(4, 8, 12, 16, 20)):
    """신고가 없이 현재가 수준 횡보 시 20일 상단이 내려오는 경로."""
    highs = list(d['high'].values[-n:])
    cur = float(d['close'].iloc[-1])
    path = []
    for k in steps:
        fut = (highs[k:] if k < n else []) + [cur] * k
        lvl = max(fut)
        eta = (d.index[-1] + pd.tseries.offsets.BDay(k)).strftime('%m/%d')
        path.append((k, eta, lvl, (lvl / cur - 1) * 100))
    return path


def _fmt_holdings(prices: dict) -> str:
    lines = []
    for code, name in CONFIG['MANAGED']:
        p = prices.get(code)
        if p:
            lines.append(f" · {name} `{p[0]:,.0f}` ({p[1]:+.2f}%)")
        else:
            lines.append(f" · {name} (가격 조회 실패)")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# 알림 메시지 작성
# ─────────────────────────────────────────────────────────────────
def build_message(d: pd.DataFrame, signal_today: str, prices: dict = None) -> str:
    """조건 충족 여부 기준으로 매일 알림. 포지션 전환일에만 알리던 방식에서 변경."""
    prices = prices or {}
    last = d.iloc[-1]
    date_str = d.index[-1].strftime('%Y-%m-%d')
    close, high, low = last['close'], last['high'], last['low']
    dch, dcl = last['dc_high'], last['dc_low']
    pos = int(last['position'])
    entry_hit = bool(last['long_entry'])
    exit_hit  = bool(last['long_exit'])

    # 국면이 유지된 거래일 수
    chg = d['position'].diff()
    since = d.index[chg != 0]
    since = since[since <= d.index[-1]]
    last_chg = since[-1] if len(since) else d.index[0]
    ndays = int((d.index > last_chg).sum())

    # ── 머리말: 조건 충족 여부가 기준 ──────────────────────────
    if exit_hit:
        head = ("[매도 조건 충족]\n\n"
                f"저가 `{low:,.2f}` < 20일 하단 `{dcl:,.2f}`\n"
                "다음 거래일 시가 매도 대상:\n" + _fmt_holdings(prices))
    elif entry_hit:
        head = ("[매수 조건 충족]\n\n"
                f"고가 `{high:,.2f}` > 20일 상단 `{dch:,.2f}`\n"
                "다음 거래일 시가 매수 대상:\n" + _fmt_holdings(prices))
    else:
        head = ("[조건 미충족] 오늘은 매수·매도 조건 모두 해당 없음\n\n"
                "보유 종목:\n" + _fmt_holdings(prices))

    # ── 국면 (실제 보유가 아니라 전략상 상태임을 명시) ──────────
    state = "보유" if pos == 1 else "현금"
    phase = (f"\n\n_전략 국면_ (실제 보유와 무관)\n"
             f"현재 국면: *{state}* · {last_chg.strftime('%Y-%m-%d')} 전환 후 {ndays}거래일째")
    if signal_today:
        phase += f"\n오늘 국면 전환 발생: *{signal_today}*"

    # ── 조건 검증 ────────────────────────────────────────────
    detail = (f"\n\n_조건 검증_ (기준일 `{date_str}`, 종가 `{close:,.2f}`)\n"
              f"매수: 고가 {high:,.2f} > 상단 {dch:,.2f} → {'TRUE' if entry_hit else 'FALSE'}\n"
              f"매도: 저가 {low:,.2f} < 하단 {dcl:,.2f} → {'TRUE' if exit_hit else 'FALSE'}")

    # ── 재진입선 + 횡보 시 하락 경로 ──────────────────────────
    proj = ""
    if pos == 0:
        proj = f"\n\n_재진입선_ `{dch:,.2f}` (현재가 {(dch/close-1)*100:+.1f}%)\n"
        proj += "횡보 시 하락 예상:\n"
        for k, eta, lvl, gap in project_dc_high(d):
            proj += f"  {k:2d}거래일 뒤({eta})  `{lvl:,.0f}`  {gap:+.1f}%\n"
        proj = proj.rstrip()

    # ── 참고 종목 ────────────────────────────────────────────
    ref = ""
    if CONFIG['REFERENCE']:
        ref = "\n\n─────────────\n_지수신호 미적용_"
        for code, name, note in CONFIG['REFERENCE']:
            p = prices.get(code)
            px = f"`{p[0]:,.0f}` ({p[1]:+.2f}%)" if p else "(조회 실패)"
            ref += f"\n · {name} {px}\n   {note}"

    return head + phase + detail + proj + ref

# ─────────────────────────────────────────────────────────────────
# HTML 대시보드 생성
# ─────────────────────────────────────────────────────────────────
def render_dashboard(d: pd.DataFrame, signals: pd.DataFrame) -> Path:
    """최근 60일 데이터로 정적 HTML 대시보드 생성"""
    plot_df = d.dropna(subset=['dc_high','dc_low']).tail(60)
    chart_data = []
    for idx, row in plot_df.iterrows():
        chart_data.append({
            'date': idx.strftime('%Y-%m-%d'),
            'close': round(float(row['close']), 2),
            'high':  round(float(row['high']),  2),
            'low':   round(float(row['low']),   2),
            'dc_high': round(float(row['dc_high']), 2),
            'dc_low':  round(float(row['dc_low']),  2),
            'position': int(row['position']),
            'entry':    bool(row['long_entry']),
            'exit_sig': bool(row['long_exit']),
        })

    last = d.iloc[-1]
    summary = {
        'last_date': d.index[-1].strftime('%Y-%m-%d'),
        'close':     round(float(last['close']), 2),
        'high':      round(float(last['high']),  2),
        'low':       round(float(last['low']),   2),
        'dc_high20': round(float(last['dc_high']), 2),
        'dc_low20':  round(float(last['dc_low']),  2),
        'buy_signal':  bool(last['long_entry']),
        'sell_signal': bool(last['long_exit']),
        'current_position': int(last['position']),
        'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    sig_list = signals.tail(10).to_dict(orient='records') if not signals.empty else []

    payload = json.dumps({
        'summary': summary, 'chart': chart_data, 'signals': sig_list
    }, ensure_ascii=False)

    html = HTML_TEMPLATE.replace('__PAYLOAD__', payload)
    out_path = OUT_DIR / 'dashboard.html'
    out_path.write_text(html, encoding='utf-8')
    log.info(f"대시보드 저장: {out_path}")
    return out_path

# HTML 템플릿 (별도 파일에서 로드)
HTML_TEMPLATE_PATH = SCRIPT_DIR / 'dashboard_template.html'
if HTML_TEMPLATE_PATH.exists():
    HTML_TEMPLATE = HTML_TEMPLATE_PATH.read_text(encoding='utf-8')
else:
    HTML_TEMPLATE = ''  # main에서 체크

# ─────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("KOSPI Donchian 20 Daily Signal — START")
    log.info("=" * 60)

    if not HTML_TEMPLATE:
        log.error(f"dashboard_template.html이 없습니다. 같은 폴더에 두세요: {HTML_TEMPLATE_PATH}")
        sys.exit(1)

    # 1. 데이터 수집
    df = fetch_kospi_data(days=CONFIG['FETCH_DAYS'])

    # 2. 신호 계산
    d = compute_signals(df,
                        n_entry=CONFIG['DC_WINDOW_ENTRY'],
                        n_exit=CONFIG['DC_WINDOW_EXIT'])

    # 3. 신호 이력 갱신
    signals = update_signal_log(d)

    # 4. 오늘 신호 판정
    last_date = d.index[-1].strftime('%Y-%m-%d')
    signal_today = None
    if not signals.empty and signals.iloc[-1]['date'] == last_date:
        signal_today = signals.iloc[-1]['type']
    log.info(f"오늘({last_date}) 신호: {signal_today or '없음'}")

    # 5. HTML 대시보드 갱신
    dashboard_path = render_dashboard(d, signals)

    # 5-2. Supabase 발행 (앱 카드 데이터 갱신)
    publish_supabase(d, signals)

    # 6. 텔레그램 푸시 — 조건 충족일 또는 NOTIFY_ALWAYS=True
    prices = fetch_prices(CONFIG['MANAGED'] + CONFIG['REFERENCE'])
    msg = build_message(d, signal_today, prices)
    # 콘솔 인코딩(cp949)이 못 찍는 문자가 있어도 알림은 나가야 한다
    print("\n" + "=" * 60)
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode(sys.stdout.encoding or 'utf-8', 'replace')
                 .decode(sys.stdout.encoding or 'utf-8', 'replace'))
    print("=" * 60)

    last = d.iloc[-1]
    cond_hit = bool(last['long_entry']) or bool(last['long_exit'])
    if cond_hit:
        log.info(f"조건 충족일 — {'매수' if last['long_entry'] else '매도'} 조건 TRUE")

    if signal_today or cond_hit or CONFIG['NOTIFY_ALWAYS']:
        send_telegram(msg)
    else:
        log.info("신호 변화 없음 → 텔레그램 푸시 생략 (NOTIFY_ALWAYS=False)")

    log.info(f"대시보드: file:///{dashboard_path.as_posix()}")
    log.info("DONE")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.exception(f"실행 중 오류: {e}")
        # 오류도 텔레그램으로 알림
        send_telegram(f"[ERROR] *KOSPI 돈치안 스크립트 오류*\n\n```\n{e}\n```")
        sys.exit(1)
