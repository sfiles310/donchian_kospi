# -*- coding: utf-8 -*-
"""FTD 판정 규칙. 저점을 사후에 알고 세지 않는지, 거래량이 없을 때 어떻게 되는지."""

import unittest

import numpy as np
import pandas as pd

import ftd_signal


def falling_then_rally(drop_days: int = 80, rally: list[float] | None = None,
                       volumes: list[float] | None = None) -> pd.DataFrame:
    """고점에서 충분히 떨어진 뒤 반등하는 최소 표본을 만든다."""
    rally = rally or []
    dates = pd.bdate_range('2025-01-01', periods=drop_days + len(rally))
    closes = list(np.linspace(3000, 2000, drop_days)) + list(rally)
    frame = pd.DataFrame({'close': closes}, index=dates[:len(closes)])
    frame['low'] = frame['close'] * 0.995
    frame['high'] = frame['close'] * 1.005
    frame['open'] = frame['close']
    base = [1000.0] * drop_days + (volumes or [1000.0] * len(rally))
    frame['volume'] = base[:len(frame)]
    return frame


class RallyCounterTest(unittest.TestCase):
    def test_new_low_resets_the_count(self) -> None:
        # 반등하다 저점을 다시 깨면 관망일이 1로 돌아가야 한다.
        frame = falling_then_rally(rally=[2010, 2020, 2030, 1990, 2000])
        table = ftd_signal.compute_ftd(frame)
        days = table['rally_day'].tolist()
        self.assertEqual(days[-2], 1, "저점 갱신일은 1일차")
        self.assertEqual(days[-1], 2)

    def test_count_grows_while_holding_above_the_low(self) -> None:
        frame = falling_then_rally(rally=[2010, 2020, 2030, 2040, 2050])
        table = ftd_signal.compute_ftd(frame)
        self.assertEqual(table['rally_day'].iloc[-1], 6)

    def test_no_correction_means_no_count(self) -> None:
        dates = pd.bdate_range('2025-01-01', periods=80)
        frame = pd.DataFrame({'close': np.linspace(2000, 3000, 80)}, index=dates)
        frame['low'] = frame['close'] * 0.99
        frame['high'] = frame['close'] * 1.01
        frame['open'] = frame['close']
        frame['volume'] = 1000.0
        table = ftd_signal.compute_ftd(frame)
        self.assertFalse(table['in_correction'].iloc[-1])
        self.assertEqual(table['rally_day'].iloc[-1], 0)


class WindowTest(unittest.TestCase):
    def test_three_day_wait_before_judging(self) -> None:
        # 저점을 찍은 날이 1일차다. 그 뒤 이틀은 아직 관망 구간이다.
        table = ftd_signal.compute_ftd(falling_then_rally(rally=[2010, 2020]))
        self.assertEqual(table['rally_day'].iloc[-1], 3)
        self.assertFalse(table['in_window'].iloc[-1])

    def test_window_opens_on_day_four(self) -> None:
        table = ftd_signal.compute_ftd(falling_then_rally(rally=[2010, 2020, 2030]))
        self.assertEqual(table['rally_day'].iloc[-1], ftd_signal.FIRST_TEST_DAY)
        self.assertTrue(table['in_window'].iloc[-1])

    def test_window_closes_after_ten_days(self) -> None:
        frame = falling_then_rally(rally=[2000 + i * 5 for i in range(1, 12)])
        table = ftd_signal.compute_ftd(frame)
        self.assertGreater(table['rally_day'].iloc[-1], ftd_signal.LAST_TEST_DAY)
        self.assertFalse(table['in_window'].iloc[-1])


class ConfirmTest(unittest.TestCase):
    def test_needs_both_gain_and_volume(self) -> None:
        rally = [2010, 2020, 2030, 2100]     # 마지막 날 +2.3%
        frame = falling_then_rally(rally=rally, volumes=[900, 900, 900, 3000])
        table = ftd_signal.compute_ftd(frame)
        row = table.iloc[-1]
        self.assertTrue(row['gain_ok'] and row['volume_ok'])
        self.assertTrue(row['ftd'])

    def test_gain_without_volume_is_not_ftd(self) -> None:
        frame = falling_then_rally(rally=[2010, 2020, 2030, 2100],
                                   volumes=[900, 900, 900, 500])
        row = ftd_signal.compute_ftd(frame).iloc[-1]
        self.assertTrue(row['gain_ok'])
        self.assertFalse(row['volume_ok'])
        self.assertFalse(row['ftd'])

    def test_volume_without_gain_is_not_ftd(self) -> None:
        frame = falling_then_rally(rally=[2010, 2020, 2030, 2035],
                                   volumes=[900, 900, 900, 3000])
        row = ftd_signal.compute_ftd(frame).iloc[-1]
        self.assertFalse(row['gain_ok'])
        self.assertFalse(row['ftd'])


class MissingVolumeTest(unittest.TestCase):
    def test_no_volume_column_never_confirms(self) -> None:
        frame = falling_then_rally(rally=[2010, 2020, 2030, 2100]).drop(columns=['volume'])
        table = ftd_signal.compute_ftd(frame)
        self.assertFalse(table['ftd'].any())
        self.assertTrue(table['volume_ratio'].isna().all())

    def test_state_reports_missing_volume(self) -> None:
        frame = falling_then_rally(rally=[2010, 2020, 2030, 2100]).drop(columns=['volume'])
        state = ftd_signal.current_state(frame)
        self.assertFalse(state.has_volume)
        block = ftd_signal.format_alert_block(state, dc_high=2500, close=2100)
        self.assertIn('판정 보류', block)


class AlertBlockTest(unittest.TestCase):
    def test_no_block_outside_correction(self) -> None:
        dates = pd.bdate_range('2025-01-01', periods=80)
        frame = pd.DataFrame({'close': np.linspace(2000, 3000, 80)}, index=dates)
        frame['low'] = frame['close'] * 0.99
        frame['high'] = frame['close'] * 1.01
        frame['open'] = frame['close']
        frame['volume'] = 1000.0
        state = ftd_signal.current_state(frame)
        self.assertEqual(ftd_signal.format_alert_block(state, 3100, 3000), '')

    def test_confirmed_block_says_so(self) -> None:
        frame = falling_then_rally(rally=[2010, 2020, 2030, 2100],
                                   volumes=[900, 900, 900, 3000])
        state = ftd_signal.current_state(frame)
        self.assertTrue(state.confirmed)
        self.assertIn('FTD 확정', ftd_signal.format_alert_block(state, 2500, 2100))

    def test_pending_block_names_the_blocker(self) -> None:
        frame = falling_then_rally(rally=[2010, 2020, 2030, 2100],
                                   volumes=[900, 900, 900, 500])
        state = ftd_signal.current_state(frame)
        self.assertEqual(state.blocked_by, '거래량 미달')
        block = ftd_signal.format_alert_block(state, 2500, 2100)
        self.assertIn('거래량 미달', block)
        self.assertIn('판정창', block)

    def test_block_uses_cp949_safe_marks(self) -> None:
        # 윈도우 콘솔 로그가 인코딩 오류로 죽지 않아야 한다.
        frame = falling_then_rally(rally=[2010, 2020, 2030, 2100],
                                   volumes=[900, 900, 900, 500])
        block = ftd_signal.format_alert_block(
            ftd_signal.current_state(frame), 2500, 2100)
        block.encode('cp949')


if __name__ == '__main__':
    unittest.main()
