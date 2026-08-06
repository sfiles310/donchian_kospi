import unittest

import numpy as np
import pandas as pd

import krx_foreign_flow_panel as panel


def sample_panel(days=36, names=40, with_timestamp=False):
    dates = pd.bdate_range("2024-01-02", periods=days)
    rows = []
    for day_index, date in enumerate(dates):
        for ticker_index in range(names):
            flow = (ticker_index - names / 2) * 1_000_000
            rows.append(
                {
                    "date": date,
                    "ticker": f"{ticker_index:06d}",
                    "open": 100.0,
                    "close": 100.0 + (ticker_index - names / 2) / 100,
                    "volume": 100_000,
                    "trading_value": 10_000_000_000,
                    "foreign_net_value": flow,
                    "market_cap": 100_000_000_000,
                    "is_listed": True,
                    "is_tradable": True,
                    "sector_code": f"S{ticker_index % 2}",
                }
            )
    data = pd.DataFrame(rows)
    if with_timestamp:
        data["data_available_at"] = pd.to_datetime(data["date"], utc=True) + pd.Timedelta(
            hours=8
        )
    return data, dates


class DataGateTest(unittest.TestCase):
    def test_missing_availability_timestamp_forces_t_plus_two(self):
        data, dates = sample_panel()

        result = panel.validate_panel(
            data,
            str(dates[11].date()),
            str(dates[23].date()),
            min_daily_names=30,
            min_segment_days=10,
            requested_lag_days=1,
        )

        self.assertEqual(result.status, "PASS")
        self.assertEqual(result.execution_lag_days, 2)
        self.assertTrue(result.warnings)

    def test_timestamp_allows_t_plus_one_but_never_same_day(self):
        data, dates = sample_panel(with_timestamp=True)

        result = panel.validate_panel(
            data,
            str(dates[11].date()),
            str(dates[23].date()),
            min_daily_names=30,
            min_segment_days=10,
            requested_lag_days=0,
        )

        self.assertEqual(result.status, "FAIL")
        self.assertIn("최소 1거래일", " ".join(result.reasons))

    def test_partial_availability_timestamp_is_a_hard_failure(self):
        data, dates = sample_panel(with_timestamp=True)
        data.loc[0, "data_available_at"] = pd.NaT

        result = panel.validate_panel(
            data,
            str(dates[11].date()),
            str(dates[23].date()),
            min_daily_names=30,
            min_segment_days=10,
        )

        self.assertEqual(result.status, "FAIL")
        self.assertIn("일부 행", " ".join(result.reasons))

    def test_impossible_foreign_flow_fails_before_backtest(self):
        data, dates = sample_panel()
        data.loc[0, "foreign_net_value"] = data.loc[0, "trading_value"] * 2

        result = panel.validate_panel(
            data,
            str(dates[11].date()),
            str(dates[23].date()),
            min_daily_names=30,
            min_segment_days=10,
        )

        self.assertEqual(result.status, "FAIL")
        self.assertIn("전체 거래대금", " ".join(result.reasons))

    def test_duplicate_date_ticker_fails(self):
        data, dates = sample_panel()
        data = pd.concat([data, data.iloc[[0]]], ignore_index=True)

        result = panel.validate_panel(
            data,
            str(dates[11].date()),
            str(dates[23].date()),
            min_daily_names=30,
            min_segment_days=10,
        )

        self.assertEqual(result.status, "FAIL")
        self.assertIn("중복", " ".join(result.reasons))


class TimingAndCostTest(unittest.TestCase):
    def test_signal_without_timestamp_enters_on_second_later_market_date(self):
        data, dates = sample_panel()

        trades, missing = panel.build_trades(
            data,
            horizon_days=1,
            execution_lag_days=2,
            min_trading_value=0,
        )

        first = trades[trades["signal_date"].eq(dates[0])]
        self.assertTrue(first["entry_date"].eq(dates[2]).all())
        self.assertEqual(missing, 0)

    def test_cost_stress_reduces_spread_exactly(self):
        data, _ = sample_panel()
        trades, _ = panel.build_trades(data, 1, 1, min_trading_value=0)

        base = panel.daily_spreads(trades, 1, cost_bps=30)
        stress = panel.daily_spreads(trades, 1, cost_bps=60)

        expected = 2 * (60 - 30) / 10_000
        difference = base["spread_net"] - stress["spread_net"]
        self.assertTrue(np.allclose(difference, expected))

    def test_missing_selected_exit_is_not_silently_dropped(self):
        data, dates = sample_panel()
        top_ticker = f"{39:06d}"
        mask = data["date"].eq(dates[2]) & data["ticker"].eq(top_ticker)
        data.loc[mask, "is_tradable"] = False

        _, missing = panel.build_trades(data, 1, 2, min_trading_value=0)

        self.assertGreater(missing, 0)


class DecisionTest(unittest.TestCase):
    def test_mdd_includes_loss_on_first_trade(self):
        stats = panel._risk_stats(pd.Series([-0.10, 0.05]))

        self.assertAlmostEqual(stats["mdd"], -0.10)

    def test_profitability_cannot_pass_without_preregistered_mdd(self):
        data, dates = sample_panel(days=45)
        gate = panel.validate_panel(
            data,
            str(dates[14].date()),
            str(dates[29].date()),
            min_daily_names=30,
            min_segment_days=10,
        )

        decisions, _ = panel.evaluate_hypotheses(
            data,
            gate,
            str(dates[14].date()),
            str(dates[29].date()),
            min_trading_value=0,
            bootstrap_repetitions=20,
            min_observations=2,
        )

        h1 = decisions[decisions["hypothesis"].eq("H1")].iloc[0]
        self.assertEqual(h1["status"], "NOT_READY")
        self.assertIn("MDD", h1["reason"])

    def test_positive_spread_with_negative_absolute_return_never_passes(self):
        data, dates = sample_panel(days=45)
        data["close"] = data["close"] - 2.0
        gate = panel.validate_panel(
            data,
            str(dates[14].date()),
            str(dates[29].date()),
            min_daily_names=30,
            min_segment_days=10,
        )

        decisions, details = panel.evaluate_hypotheses(
            data,
            gate,
            str(dates[14].date()),
            str(dates[29].date()),
            min_trading_value=0,
            bootstrap_repetitions=50,
            min_observations=2,
            max_allowed_mdd=0.99,
        )

        h1 = decisions[decisions["hypothesis"].eq("H1")].iloc[0]
        holdout = details[
            details["hypothesis"].eq("H1")
            & details["segment"].eq("holdout")
            & details["cost_multiplier"].eq(2.0)
        ].iloc[0]
        self.assertEqual(h1["status"], "FAIL")
        self.assertLess(holdout["top_net_mean"], 0)
        self.assertIn("절대수익", h1["reason"])

    def test_missing_sector_data_makes_h3_not_ready(self):
        data, dates = sample_panel(days=45)
        data = data.drop(columns="sector_code")
        gate = panel.validate_panel(
            data,
            str(dates[14].date()),
            str(dates[29].date()),
            min_daily_names=30,
            min_segment_days=10,
        )

        decisions, details = panel.evaluate_hypotheses(
            data,
            gate,
            str(dates[14].date()),
            str(dates[29].date()),
            min_trading_value=0,
            bootstrap_repetitions=20,
            min_observations=2,
        )

        h3 = decisions[decisions["hypothesis"].eq("H3")].iloc[0]
        self.assertEqual(h3["status"], "NOT_READY")
        self.assertFalse(details["hypothesis"].eq("H3").any())


if __name__ == "__main__":
    unittest.main()
