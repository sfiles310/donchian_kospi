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

    # 전략 검증 기준일. 각 기준일마다 '신호대로 실행' vs '계속 보유'를 비교한다.
    # 금액은 넣지 않는다 — 공개 저장소라 수량이 드러나면 자산 규모가 계산된다.
    'TRACK': [
        ('2026-07-03', '미실행'),   # 놓친 매도 신호. 이미 치른 비용
        ('2026-07-21', '검증'),     # 이제부터 신호대로 할 경우의 검증
    ],

    # 대시보드 주소 (알림 하단 링크)
    'PAGES_URL': 'https://sfiles310.github.io/donchian_kospi/',
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
# 데이터 수집 (pykrx/FDR 이력 + 네이버 확정 시세 검증)
# ─────────────────────────────────────────────────────────────────
NAVER_CHART_URL = "https://api.stock.naver.com/chart/domestic/{kind}/{code}"
NAVER_INDEX_REALTIME_URL = "https://polling.finance.naver.com/api/realtime/domestic/index/KOSPI"


def _validate_ohlc(df: pd.DataFrame, source: str) -> None:
    """가격 기본 관계가 깨진 데이터는 알림·차트에 사용하지 않는다."""
    if df.empty or df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise RuntimeError(f"{source} 데이터의 날짜축이 올바르지 않습니다.")
    if df[['open', 'high', 'low', 'close']].isna().any().any():
        raise RuntimeError(f"{source} 데이터에 결측값이 있습니다.")
    invalid = (
        (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1)
        | (df['high'] < df[['open', 'close']].max(axis=1))
        | (df['low'] > df[['open', 'close']].min(axis=1))
        | (df['high'] < df['low'])
    )
    if invalid.any():
        dates = ', '.join(df.index[invalid].strftime('%Y-%m-%d')[:3])
        raise RuntimeError(f"{source} OHLC 관계 오류: {dates}")


def _fetch_naver_chart(code: str, kind: str = 'index') -> pd.DataFrame:
    """네이버 일봉 API에서 지수 또는 종목의 확정 OHLC를 가져온다."""
    url = NAVER_CHART_URL.format(kind=kind, code=code)
    r = requests.get(
        url,
        params={'periodType': 'dayCandle'},
        headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://finance.naver.com/'},
        timeout=15,
    )
    r.raise_for_status()
    rows = r.json().get('priceInfos', [])
    df = pd.DataFrame([{
        'date': x['localDate'],
        'open': x['openPrice'],
        'high': x['highPrice'],
        'low': x['lowPrice'],
        'close': x['closePrice'],
    } for x in rows])
    if df.empty:
        raise RuntimeError(f"네이버 {code} 일봉 데이터가 비어 있습니다.")
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').astype(float).sort_index()
    _validate_ohlc(df, f"네이버 {code}")
    return df


def _fetch_closed_kospi_data() -> pd.DataFrame:
    """서로 다른 네이버 엔드포인트가 동일한 장 마감값을 줄 때만 반환한다."""
    chart = _fetch_naver_chart('KOSPI')
    r = requests.get(
        NAVER_INDEX_REALTIME_URL,
        headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://finance.naver.com/'},
        timeout=15,
    )
    r.raise_for_status()
    rows = r.json().get('datas', [])
    if not rows:
        raise RuntimeError("코스피 장 상태 데이터가 비어 있습니다.")
    market = rows[0]
    if market.get('marketStatus') != 'CLOSE':
        raise RuntimeError(f"코스피 장 마감 전입니다: {market.get('marketStatus', 'UNKNOWN')}")

    traded_at = str(market.get('localTradedAt', ''))[:10]
    last_date = chart.index[-1].strftime('%Y-%m-%d')
    if traded_at != last_date:
        raise RuntimeError(f"코스피 확정일 불일치: 일봉 {last_date}, 실시간 {traded_at}")

    last = chart.iloc[-1]
    realtime = {
        'open': float(market['openPriceRaw']),
        'high': float(market['highPriceRaw']),
        'low': float(market['lowPriceRaw']),
        'close': float(market['closePriceRaw']),
    }
    mismatch = [k for k, v in realtime.items() if abs(float(last[k]) - v) > 0.001]
    if mismatch:
        raise RuntimeError(f"코스피 확정 OHLC 교차검증 실패: {', '.join(mismatch)}")
    return chart


def _merge_verified_history(history: pd.DataFrame, verified: pd.DataFrame) -> pd.DataFrame:
    """최근 구간은 교차검증된 시세로 덮어써 차트의 오래된 캐시를 제거한다."""
    if len(verified) < 60:
        raise RuntimeError(f"검증 시세가 60거래일보다 적습니다: {len(verified)}행")
    overlap = history.index.intersection(verified.index)
    if len(overlap) < 60:
        raise RuntimeError(f"기본 시세와 검증 시세의 공통 구간이 부족합니다: {len(overlap)}행")

    changed = (history.loc[overlap] - verified.loc[overlap]).abs().gt(0.001).any(axis=1)
    if changed.any():
        dates = ', '.join(overlap[changed].strftime('%Y-%m-%d'))
        log.warning(f"기본 시세 불일치 {int(changed.sum())}건을 검증 시세로 교정: {dates}")

    merged = pd.concat([history.loc[history.index < verified.index[0]], verified]).sort_index()
    _validate_ohlc(merged, "병합 코스피")
    return merged


def fetch_kospi_data(days: int = 400) -> pd.DataFrame:
    """코스피 이력을 가져온 뒤 최근 구간을 장 마감 확정 시세로 검증·교정한다."""
    end = datetime.now()
    start = end - timedelta(days=days * 1.6)  # 주말·휴일 고려해 넉넉히
    start_str = start.strftime('%Y%m%d')
    end_str   = end.strftime('%Y%m%d')
    df = None

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
    except Exception as e:
        df = None
        log.warning(f"pykrx 실패: {e}. FinanceDataReader로 fallback 시도")

    # 2차: FinanceDataReader 백업
    if df is None:
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
        except Exception as e:
            log.error(f"모든 데이터 소스 실패: {e}")
            raise RuntimeError("코스피 데이터를 가져올 수 없습니다.")

    verified = _fetch_closed_kospi_data()
    df = _merge_verified_history(df, verified)
    log.info(f"코스피 확정 시세 검증 완료: {df.index[-1].date()} 종가 {df['close'].iloc[-1]:,.2f}")
    return df

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
def notification_completed(as_of_date: str) -> bool:
    """Supabase에 같은 지수 기준일이 발행됐으면 중복 알림을 막는다."""
    url = CONFIG['SUPABASE_URL'].rstrip('/')
    key = CONFIG['SUPABASE_SERVICE_KEY']
    if not url or not key:
        return False

    endpoint = f"{url}/rest/v1/strategy_signal"
    headers = {
        'apikey': key,
        'Authorization': f'Bearer {key}',
    }
    params = {
        'strategy': 'eq.donchian_kospi',
        'select': 'as_of_date',
        'limit': '1',
    }
    try:
        r = requests.get(endpoint, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        rows = r.json()
        published_date = rows[0].get('as_of_date', '') if rows else ''
        return str(published_date) == as_of_date
    except Exception as e:
        log.warning(f"알림 완료 여부 확인 실패. 누락 방지를 위해 발송 시도: {_redact(str(e))}")
        return False


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
def _redact(msg: str) -> str:
    """예외 메시지에 담긴 봇 토큰을 가린다.

    requests 예외는 실패한 URL을 통째로 담는데, 텔레그램 URL에는 토큰이 들어 있다.
    그대로 로그에 남기면 로그 파일·CI 콘솔에 토큰이 평문으로 노출된다.
    """
    import re
    msg = re.sub(r'bot\d{6,}:[A-Za-z0-9_-]{20,}', 'bot<REDACTED>', msg)
    token = CONFIG['TELEGRAM_BOT_TOKEN']
    if token:
        msg = msg.replace(token, '<REDACTED>')
    return msg


def send_telegram(text: str) -> bool:
    """텔레그램 메시지 발송. parse_mode는 HTML을 쓴다.

    레거시 Markdown은 '[', '_', '*'가 본문에 섞이면 파싱에 실패해 400을 낸다.
    HTML은 &, <, > 세 글자만 이스케이프하면 되므로 훨씬 덜 깨진다.
    """
    token = CONFIG['TELEGRAM_BOT_TOKEN']
    chat_id = CONFIG['TELEGRAM_CHAT_ID']

    if not token or not chat_id:
        log.warning("텔레그램 토큰/chat_id 미설정. 푸시 생략")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': text,
        'parse_mode': 'HTML',
        'disable_web_page_preview': True,
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        if not r.ok:
            # 텔레그램은 실패 사유를 본문 description에 담아준다
            try:
                desc = r.json().get('description', '')
            except Exception:
                desc = r.text[:200]
            log.error(f"텔레그램 푸시 실패: HTTP {r.status_code} — {_redact(desc)}")
            return False
        log.info("텔레그램 푸시 성공")
        return True
    except Exception as e:
        log.error(f"텔레그램 푸시 실패: {_redact(str(e))}")
        return False

# ─────────────────────────────────────────────────────────────────
# 보조 정보: 종목 현재가 / 채널 전망
# ─────────────────────────────────────────────────────────────────
def fetch_prices(items, as_of_date: str) -> dict:
    """지수 기준일과 같은 날짜의 종목 종가만 반환한다."""
    out = {}
    target = pd.Timestamp(as_of_date)
    for entry in items:
        code, name = entry[0], entry[1]
        try:
            e = _fetch_naver_chart(code, kind='item').loc[:target]
            if e.empty or e.index[-1] != target:
                actual = e.index[-1].strftime('%Y-%m-%d') if not e.empty else '없음'
                raise RuntimeError(f"기준일 {as_of_date} 종가 없음 (최신 {actual})")
            px = e['close']
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


def track_scenarios(d: pd.DataFrame, start: str) -> list:
    """기준일(start) 이후 두 경로를 비교.

      A. 신호 준수  — 지수 신호대로 매도/매수. 체결은 신호 다음 거래일 시가.
      B. 계속 보유  — 팔지 않고 그대로 들고 감.

    두 경로 모두 시장 가격만으로 계산된다. 실제 매매 여부와 무관하게 값이 나오고,
    사용자는 자기가 어느 선 위에 있는지만 알면 된다.

    상태 파일을 쓰지 않고 매일 가격에서 재계산하므로 실행이 빠진 날이 있어도
    누적값이 어긋나지 않는다.
    """
    rows = []
    end = d.index[-1]
    for code, name in CONFIG['MANAGED']:
        try:
            e = _fetch_naver_chart(code, kind='item').loc[pd.Timestamp(start):end, ['open', 'close']]
            if len(e) < 2:
                continue
            if e.index[-1] != end:
                raise RuntimeError(
                    f"지수 기준일 {end.strftime('%Y-%m-%d')}과 종목 기준일 "
                    f"{e.index[-1].strftime('%Y-%m-%d')} 불일치"
                )

            # 지수 신호를 ETF 날짜축에 정렬. 신호(종가) -> 다음 거래일 시가 체결이므로
            # [시가(t-1) -> 시가(t)] 구간의 보유 여부는 종가(t-2) 시점 신호가 결정한다.
            pos = d['position'].reindex(e.index, method='ffill').fillna(0)
            held = pos.shift(2).fillna(0)

            oo = e['open'].pct_change().fillna(0)
            strat_eq = float((1 + oo * held).prod())

            # 미실행: 체결 예정이던 시가(= 신호 다음 거래일 시가)부터 현재 종가까지 계속 보유
            entry = float(e['open'].iloc[1])
            hold_eq = float(e['close'].iloc[-1]) / entry

            # 신호 준수 경로도 마지막 구간은 종가까지 반영
            if held.iloc[-1] == 1:
                strat_eq *= float(e['close'].iloc[-1]) / float(e['open'].iloc[-1])

            rows.append({
                'name': name,
                'entry': entry,
                'now': float(e['close'].iloc[-1]),
                'strat': (strat_eq - 1) * 100,
                'hold': (hold_eq - 1) * 100,
                'gap': (hold_eq - strat_eq) * 100,
            })
        except Exception as ex:
            log.warning(f"{name}({code}) 추적 계산 실패: {ex}")
    return rows


def _dwidth(s: str) -> int:
    """한글·전각 문자는 화면상 2칸을 차지한다. len()으로는 정렬이 깨진다."""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in s)


def _dpad(s: str, width: int) -> str:
    """표시폭 기준 좌측 정렬 패딩. 넘치면 잘라낸다."""
    while _dwidth(s) > width:
        s = s[:-1]
    return s + ' ' * (width - _dwidth(s))


def _fmt_tracking(rows: list, ndays: int = None) -> str:
    """한눈에 읽히도록 고정폭 블록으로. 결론 한 줄이 먼저 오게 배치한다."""
    if not rows:
        return ""
    avg_s = sum(r['strat'] for r in rows) / len(rows)
    avg_h = sum(r['hold'] for r in rows) / len(rows)
    gap = avg_h - avg_s
    money = abs(gap) * 10000  # 100만원당 원

    since = f"{CONFIG['TRACK_FROM']} 매도 신호"
    if ndays is not None:
        since += f" · {ndays}거래일 경과"

    out = f"\n\n─────────────\n*안 팔아서 생긴 차이  {gap:+.1f}%p*\n_{since}_\n"
    W = 20
    out += "```\n"
    out += _dpad('팔았다면', W)    + f"{avg_s:>7.1f}%\n"
    out += _dpad('안 팔았다면', W) + f"{avg_h:>7.1f}%\n"
    out += '-' * (W + 8) + "\n"
    out += _dpad('100만원당', W)   + f"{-money:>7,.0f}원\n\n"
    for r in rows:
        out += _dpad(r['name'], W) + f"{r['hold']:>7.1f}%\n"
    out += "```"
    out += "\n'안 팔았다면'이 오늘 매도 시 확정되는 값입니다."
    return out


def _fmt_holdings(prices: dict) -> str:
    """종목명은 표시폭으로 좌측 정렬, 가격은 우측 정렬해 세로줄을 맞춘다."""
    import html as _html
    names = [n for _, n in CONFIG['MANAGED']]
    w = max(_dwidth(n) for n in names) + 2
    lines = ["<pre>"]
    for code, name in CONFIG['MANAGED']:
        p = prices.get(code)
        val = f"{p[0]:>9,.0f}" if p else "     조회실패"
        lines.append(_html.escape(_dpad(name, w) + val))
    lines.append("</pre>")
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

    md = d.index[-1].strftime('%m/%d')

    # ── 머리말: 오늘 할 행동 ──────────────────────────────────
    if exit_hit:
        head = (f"<b>[매도 조건 충족] {md}</b>\n"
                f"저가 {low:,.0f} &lt; 20일 하단 {dcl:,.0f}\n\n"
                "다음 거래일 시가 매도\n" + _fmt_holdings(prices))
    elif entry_hit:
        head = (f"<b>[매수 조건 충족] {md}</b>\n"
                f"고가 {high:,.0f} &gt; 20일 상단 {dch:,.0f}\n\n"
                "다음 거래일 시가 매수\n" + _fmt_holdings(prices))
    else:
        head = (f"<b>[조건 미충족] {md}</b>\n"
                f"종가 {close:,.0f} · 상단 {dch:,.0f} / 하단 {dcl:,.0f}\n"
                + _fmt_holdings(prices))

    # ── 국면 + 재진입선 ──────────────────────────────────────
    state = "보유" if pos == 1 else "현금"
    # %-m/%-d 는 리눅스 전용이라 Windows에서 죽는다. 직접 조립한다.
    body = f"\n국면: {state} ({last_chg.month}/{last_chg.day} 이후 {ndays}거래일째)"
    if signal_today:
        body += f"\n오늘 국면 전환: <b>{signal_today}</b>"
    if pos == 0:
        path = project_dc_high(d, steps=(20,))
        _, eta, lvl, _g = path[0]
        body += (f"\n재진입선 {dch:,.0f} ({(dch/close-1)*100:+.1f}%)"
                 f"\n  → 횡보 시 {eta}경 {lvl:,.0f}")

    # ── 기준일별 검증 (수익률만. 금액은 자산 규모가 드러나므로 넣지 않는다) ──
    track = ""
    for start, label in CONFIG['TRACK']:
        rows = track_scenarios(d, start)
        if not rows:
            continue
        avg_h = sum(r['hold'] for r in rows) / len(rows)
        avg_s = sum(r['strat'] for r in rows) / len(rows)
        nd = int((d.index > pd.Timestamp(start)).sum())
        track += (f"\n{label} {start[5:]} 기준 {nd}일"
                  f" · 신호 {avg_s:+.1f}% / 보유 {avg_h:+.1f}%"
                  f" · 차이 {avg_s - avg_h:+.1f}%p")
    if track:
        track = "\n" + track

    # ── 참고 종목 (한 줄) ────────────────────────────────────
    ref = ""
    for code, name, _note in CONFIG['REFERENCE']:
        p = prices.get(code)
        if p:
            ref += f"\n[미적용] {name} {p[0]:,.0f}"

    link = f"\n\n상세 ▸ {CONFIG['PAGES_URL']}"
    return head + body + track + ref + link

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
    # 이력도 텔레그램·차트와 같은 기준으로: 포지션 전환일이 아니라 '조건 충족일'.
    # 전환일만 담으면 7/3 이후 조건이 5번 더 참이었던 사실이 이력에서 사라진다.
    turn = d['position'].diff().fillna(0)
    cond = d[d['long_entry'] | d['long_exit']].tail(12)
    sig_list = [{
        'date':  idx.strftime('%Y-%m-%d'),
        'type':  'BUY' if row['long_entry'] else 'SELL',
        'price': round(float(row['close']), 2),
        'turn':  bool(turn.get(idx, 0) != 0),
    } for idx, row in cond.iterrows()]
    sig_list.reverse()   # 최신이 위로

    summary['managed'] = [name for _, name in CONFIG['MANAGED']]

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

    # 6. 텔레그램 푸시 — 조건 충족일 또는 NOTIFY_ALWAYS=True
    prices = fetch_prices(CONFIG['MANAGED'] + CONFIG['REFERENCE'], last_date)
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

    notify_due = bool(signal_today or cond_hit or CONFIG['NOTIFY_ALWAYS'])
    notification_complete = not notify_due
    if notify_due:
        if notification_completed(last_date):
            log.info("오늘 알림은 이미 발송됨. 중복 푸시 생략")
            notification_complete = True
        else:
            notification_complete = send_telegram(msg)
            # 토큰/chat_id를 설정했는데도 전달이 실패했다면 CI를 빨간불로 만든다.
            # 그러지 않으면 "초록불인데 알림은 안 옴" 상태를 알아챌 방법이 없다.
            if not notification_complete:
                raise RuntimeError("텔레그램 전달 실패 (위 로그의 사유 확인)")
    else:
        log.info("신호 변화 없음 → 텔레그램 푸시 생략 (NOTIFY_ALWAYS=False)")

    # 알림 성공 뒤에 완료 시각을 기록한다. 실패한 실행은 다음 실행이 재시도한다.
    publish_supabase(d, signals)

    log.info(f"대시보드: file:///{dashboard_path.as_posix()}")
    log.info("DONE")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.exception(f"실행 중 오류: {e}")
        # 오류도 텔레그램으로 알림
        import html as _html
        send_telegram("<b>[ERROR] KOSPI 돈치안 스크립트 오류</b>\n\n"
                      f"<pre>{_html.escape(_redact(str(e)))}</pre>")
        sys.exit(1)
