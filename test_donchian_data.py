import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import donchian_kospi_daily as app


def sample_ohlc(periods=61):
    index = pd.bdate_range('2026-04-01', periods=periods)
    return pd.DataFrame({
        'open': range(100, 100 + periods),
        'high': range(102, 102 + periods),
        'low': range(99, 99 + periods),
        'close': range(101, 101 + periods),
    }, index=index, dtype=float)


class MarketDataValidationTest(unittest.TestCase):
    def test_verified_history_replaces_stale_recent_value(self):
        verified = sample_ohlc()
        history = verified.copy()
        history.iloc[-1, history.columns.get_loc('close')] += 1

        merged = app._merge_verified_history(history, verified)

        self.assertEqual(merged.iloc[-1]['close'], verified.iloc[-1]['close'])

    def test_invalid_ohlc_is_rejected(self):
        data = sample_ohlc()
        data.iloc[-1, data.columns.get_loc('low')] = data.iloc[-1]['high'] + 1

        with self.assertRaisesRegex(RuntimeError, 'OHLC 관계 오류'):
            app._validate_ohlc(data, 'test')

    @patch.object(app, '_fetch_naver_chart')
    @patch.object(app.requests, 'get')
    def test_market_open_is_rejected(self, get, chart):
        chart.return_value = sample_ohlc()
        response = Mock()
        response.json.return_value = {'datas': [{'marketStatus': 'OPEN'}]}
        response.raise_for_status.return_value = None
        get.return_value = response

        with self.assertRaisesRegex(RuntimeError, '장 마감 전'):
            app._fetch_closed_kospi_data()


class DashboardNotificationTest(unittest.TestCase):
    def test_test_notification_title_keeps_korean_text(self):
        message = app.build_test_message('body')

        self.assertEqual(
            message,
            '<b>[반영 확인 테스트]</b>\nbody',
        )

    @patch.object(app, 'track_scenarios', return_value=[])
    def test_dashboard_link_is_versioned_with_data_date(self, _track):
        data = app.compute_signals(sample_ohlc())
        date = data.index[-1].strftime('%Y-%m-%d')

        message = app.build_message(data, signal_today=None)

        expected = f"{app.CONFIG['PAGES_URL']}{app.CONFIG['FTD_PAGE']}?as_of={date}"
        self.assertIn(expected, message)

    @patch.object(app, 'track_scenarios', return_value=[])
    def test_no_ftd_block_while_holding(self, _track):
        """보유 중이면 더 살 게 없으므로 FTD 판정을 붙이지 않는다."""
        data = app.compute_signals(sample_ohlc())
        data.loc[data.index[-1], 'position'] = 1

        self.assertNotIn('FTD 재진입 점검', app.build_message(data, signal_today=None))

    @patch.object(app.requests, 'get')
    def test_dashboard_date_must_match_notification_date(self, get):
        response = Mock()
        response.text = '"last_date": "2026-07-28"'
        response.raise_for_status.return_value = None
        get.return_value = response

        self.assertFalse(app.dashboard_is_current('2026-07-29'))

    def test_pending_run_uses_same_message_and_payload_after_deploy(self):
        data = app.compute_signals(sample_ohlc())
        signals = pd.DataFrame(columns=['date', 'type'])

        with TemporaryDirectory() as tmp:
            pending_path = Path(tmp) / 'pending_run.json'
            with patch.object(app, 'PENDING_RUN_PATH', pending_path):
                app.save_pending_run(
                    data, signals, 'same-run-message', notify=True
                )
                with patch.object(app, 'send_telegram', return_value=True) as send, \
                        patch.object(app, '_publish_supabase_payload',
                                     return_value=True) as publish:
                    app.finalize_pending_run()

        send.assert_called_once_with('same-run-message')
        self.assertEqual(
            publish.call_args.args[0]['as_of_date'],
            data.index[-1].strftime('%Y-%m-%d'),
        )


if __name__ == '__main__':
    unittest.main()


class IndexHistoryNoteTest(unittest.TestCase):
    """지수 거래량 이력 갱신 안내. 평소에는 조용하고 필요할 때만 나와야 한다."""

    def frame(self, volume=1000.0, rows=250):
        index = pd.bdate_range('2026-01-01', periods=rows)
        data = pd.DataFrame({'close': 2500.0, 'high': 2510.0, 'low': 2490.0,
                             'open': 2500.0}, index=index)
        if volume is not None:
            data['volume'] = volume
        return data

    def test_quiet_when_history_is_fresh(self):
        with patch.object(app, '_load_index_history') as load:
            load.return_value = self.frame().tail(5)
            self.assertEqual(app.index_history_note(self.frame()), '')

    def test_warns_when_volume_column_missing(self):
        note = app.index_history_note(self.frame(volume=None))
        self.assertIn('거래량 자료가 없습니다', note)
        self.assertIn('refresh_index_history.py', note)

    def test_warns_when_volume_has_a_hole(self):
        data = self.frame()
        data.iloc[-100:-60, data.columns.get_loc('volume')] = np.nan
        note = app.index_history_note(data)
        self.assertIn('끊겼습니다', note)
        self.assertIn('40일 결측', note)

    def test_warns_before_the_hole_opens(self):
        stale = self.frame()
        stale.index = pd.bdate_range('2025-01-01', periods=len(stale))
        with patch.object(app, '_load_index_history', return_value=stale):
            note = app.index_history_note(self.frame())
        self.assertIn('갱신 필요', note)
        self.assertIn('refresh_index_history.py', note)
