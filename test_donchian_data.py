import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

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

        self.assertIn(f"{app.CONFIG['PAGES_URL']}?as_of={date}", message)

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
