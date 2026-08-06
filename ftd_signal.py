# -*- coding: utf-8 -*-
"""오닐 FTD(추격매수일) 판정. 일일 알림과 검증 스크립트가 같은 규칙을 쓴다.

규칙은 `ftd_study.py`에서 12.6년 표본으로 검정한 것과 같다. 요약하면

1. 조정 국면: 종가가 최근 250일 고점 대비 -10% 이하
2. 그 국면에서 저점이 갱신되면 관망일을 1로 되돌린다. 저점을 사후에 알고 세는 것이
   아니라 실시간으로 계산하기 위한 장치다.
3. 랠리 4~10일차에 (전일 대비 상승률 >= 2.0%) 그리고 (거래량 > 50일 이동평균)이면
   FTD 확정 = 상승 추세 전환 신호

검정에서 확인한 것과 한계는 SESSION_HANDOFF.md에 적어 두었다. 표본이 12.6년에
6~14회뿐이라 단독 규칙으로 쓰지 않고 돈치안 진입에 조기 진입만 더하는 용도다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

CORRECTION_LOOKBACK = 250
CORRECTION_DROP = 0.10
GAIN_THRESHOLD = 2.0
VOLUME_MA = 50
FIRST_TEST_DAY = 4
LAST_TEST_DAY = 10


@dataclass(frozen=True)
class FtdState:
    """오늘 시점의 FTD 판정 결과."""

    has_volume: bool
    in_correction: bool
    rally_day: int
    rally_low: float
    gain: float
    volume_ratio: float
    in_window: bool
    gain_ok: bool
    volume_ok: bool
    confirmed: bool
    days_left: int

    @property
    def blocked_by(self) -> str:
        """판정 창 안에서 무엇 때문에 막혔는지 한 마디로."""
        if not self.in_window:
            return "판정 창 밖"
        if self.gain_ok and not self.volume_ok:
            return "거래량 미달"
        if not self.gain_ok and self.volume_ok:
            return "상승률 미달"
        if not self.gain_ok and not self.volume_ok:
            return "상승률·거래량 미달"
        return "없음"


def _rally_counter(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """조정 국면에서 저점이 갱신되면 1로 되돌리는 관망일 카운터."""
    peak = frame['close'].rolling(CORRECTION_LOOKBACK, min_periods=60).max()
    in_correction = frame['close'] < peak * (1 - CORRECTION_DROP)

    counter = [0] * len(frame)
    low_mark = [math.nan] * len(frame)
    running_low = math.inf
    day = 0
    for i in range(len(frame)):
        if not bool(in_correction.iloc[i]):
            running_low, day, counter[i] = math.inf, 0, 0
            continue
        low = float(frame['low'].iloc[i])
        if low <= running_low:
            running_low, day = low, 1
        else:
            day += 1
        counter[i] = day
        low_mark[i] = running_low
    return (pd.Series(counter, index=frame.index),
            pd.Series(low_mark, index=frame.index))


def compute_ftd(frame: pd.DataFrame) -> pd.DataFrame:
    """일자별 FTD 판정 요소를 붙인 표를 돌려준다.

    frame은 close, low, volume 열을 가진 일봉이어야 한다. volume이 없으면
    거래량 조건을 판정할 수 없으므로 관련 열을 결측으로 남긴다.
    """
    result = pd.DataFrame(index=frame.index)
    day, low_mark = _rally_counter(frame)
    peak = frame['close'].rolling(CORRECTION_LOOKBACK, min_periods=60).max()

    result['in_correction'] = frame['close'] < peak * (1 - CORRECTION_DROP)
    result['rally_day'] = day
    result['rally_low'] = low_mark
    result['gain'] = frame['close'].pct_change() * 100
    result['in_window'] = (day >= FIRST_TEST_DAY) & (day <= LAST_TEST_DAY)
    result['gain_ok'] = result['gain'] >= GAIN_THRESHOLD

    has_volume = 'volume' in frame.columns and frame['volume'].notna().sum() >= VOLUME_MA
    if has_volume:
        volume_ma = frame['volume'].rolling(VOLUME_MA).mean()
        result['volume_ma'] = volume_ma
        result['volume_ratio'] = frame['volume'] / volume_ma
        result['volume_ok'] = frame['volume'] > volume_ma
    else:
        result['volume_ma'] = float('nan')
        result['volume_ratio'] = float('nan')
        result['volume_ok'] = False
    result['ftd'] = result['in_window'] & result['gain_ok'] & result['volume_ok']
    return result


def current_state(frame: pd.DataFrame) -> FtdState:
    """마지막 거래일 기준 판정 결과."""
    table = compute_ftd(frame)
    row = table.iloc[-1]
    has_volume = bool(pd.notna(row['volume_ratio']))
    rally_day = int(row['rally_day'])
    return FtdState(
        has_volume=has_volume,
        in_correction=bool(row['in_correction']),
        rally_day=rally_day,
        rally_low=float(row['rally_low']) if pd.notna(row['rally_low']) else float('nan'),
        gain=float(row['gain']) if pd.notna(row['gain']) else float('nan'),
        volume_ratio=float(row['volume_ratio']) if has_volume else float('nan'),
        in_window=bool(row['in_window']),
        gain_ok=bool(row['gain_ok']),
        volume_ok=bool(row['volume_ok']),
        confirmed=bool(row['ftd']),
        days_left=max(0, LAST_TEST_DAY - rally_day) if rally_day else 0,
    )


def format_alert_block(state: FtdState, dc_high: float, close: float) -> str:
    """현금 보유 중일 때 알림에 붙일 FTD 상태 블록."""
    if not state.in_correction:
        return ""
    gap = (dc_high / close - 1) * 100 if close else float('nan')
    lines = [f"\n\n<b>FTD 재진입 점검</b> (랠리 {state.rally_day}일차)"]

    if not state.has_volume:
        lines.append("거래량 자료가 없어 판정 보류")
        return "\n".join(lines)

    if state.confirmed:
        lines.append(f"✅ FTD 확정 — 상승 {state.gain:+.2f}% · "
                     f"거래량 {state.volume_ratio:.2f}배")
        lines.append("돈치안 상단 전이라도 조기 재진입 검토")
        return "\n".join(lines)

    # cp949 콘솔에서도 깨지지 않는 기호만 쓴다. 알림은 UTF-8이지만 로그는 아니다.
    mark = lambda ok: "○" if ok else "×"
    lines.append(f"{mark(state.gain_ok)} 상승 {state.gain:+.2f}% (기준 {GAIN_THRESHOLD:.1f}%)")
    lines.append(f"{mark(state.volume_ok)} 거래량 {state.volume_ratio:.2f}배 (기준 1.00배)")
    if state.in_window:
        lines.append(f"미확정 · {state.blocked_by} · 판정창 {state.days_left}일 남음")
    elif state.rally_day:
        lines.append(f"관망기 ({FIRST_TEST_DAY}일차부터 판정)")
    lines.append(f"돈치안 재진입선 {dc_high:,.0f} ({gap:+.1f}%)")
    return "\n".join(lines)
