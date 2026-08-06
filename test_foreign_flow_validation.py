import unittest

import numpy as np
import pandas as pd

import foreign_flow_validation as app


def sample_inputs(periods=50):
    index = pd.bdate_range("2026-01-02", periods=periods)
    base = np.arange(periods, dtype=float) + 100
    prices = pd.DataFrame({
        "open": base,
        "high": base + 3,
        "low": base - 1,
        "close": base + 2,
        "volume": np.full(periods, 1_000.0),
    }, index=index)
    foreign = pd.DataFrame({
        "reported_close": prices["close"],
        "reported_volume": prices["volume"],
        "institution_net_qty": np.zeros(periods),
        "foreign_net_qty": np.full(periods, 100.0),
        "foreign_holding_qty": np.arange(periods) + 1_000,
        "foreign_holding_pct": 40 + np.arange(periods) / 100,
    }, index=index)
    fx = pd.Series(1_400 - np.arange(periods), index=index, name="usdkrw_close")
    return prices, foreign, fx


class ForeignFlowParserTest(unittest.TestCase):
    def test_parses_confirmed_investor_row(self):
        html = """
        <table><tr>
          <td>2026.07.31</td><td>262,500</td><td>상승 55,500</td>
          <td>+26.81%</td><td>58,478,873</td><td>+3,618,959</td>
          <td><span>+8,359,011</span></td><td>2,730,295,646</td><td>46.70%</td>
        </tr></table>
        """

        result = app.parse_naver_foreign_html(html)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["foreign_net_qty"], 8_359_011)
        self.assertEqual(result.iloc[0]["foreign_holding_pct"], 46.70)

    def test_ignores_unrelated_tables(self):
        html = "<table><tr><td>삼성전자</td><td>123</td></tr></table>"

        self.assertTrue(app.parse_naver_foreign_html(html).empty)


class ForeignFlowTimingTest(unittest.TestCase):
    def test_future_return_enters_at_next_open(self):
        prices, foreign, fx = sample_inputs()
        data = app.build_dataset(
            "TEST", "2026-01-02", foreign=foreign, prices=prices, fx_close=fx
        )
        day = data.index[25]
        next_day = data.index[26]

        expected = data.loc[next_day, "close"] / data.loc[next_day, "open"] - 1

        self.assertAlmostEqual(data.loc[day, "future_return_1d"], expected)

    def test_strategy_uses_previous_day_signal(self):
        prices, foreign, fx = sample_inputs()
        data = app.build_dataset(
            "TEST", "2026-01-02", foreign=foreign, prices=prices, fx_close=fx
        )
        signal = pd.Series(False, index=data.index)
        signal.iloc[10] = True

        returns, active = app._strategy_returns(data, signal, cost_bps=30)

        self.assertFalse(active.iloc[10])
        self.assertTrue(active.iloc[11])
        expected = data.iloc[11]["close"] / data.iloc[11]["open"] - 1 - 0.003
        self.assertAlmostEqual(returns.iloc[11], expected)

    def test_cost_stress_never_improves_same_strategy(self):
        prices, foreign, fx = sample_inputs()
        data = app.build_dataset(
            "TEST", "2026-01-02", foreign=foreign, prices=prices, fx_close=fx
        )

        result = app.compare_strategies({"TEST": data}, cost_bps=30)
        rows = result[
            (result["ticker"] == "TEST")
            & (result["strategy"] == "외국인 지속(3/5)")
        ].sort_values("cost_multiplier")

        self.assertEqual(list(rows["cost_multiplier"]), [1.0, 1.5, 2.0])
        self.assertTrue(rows["total_return"].is_monotonic_decreasing)

    def test_placebo_preserves_signal_count_but_changes_timing(self):
        signal = pd.Series(
            [True] * 5 + [False] * 5 + [True] * 5 + [False] * 10,
            index=pd.bdate_range("2026-01-02", periods=25),
        )

        placebo = app.block_permute_signal(signal, block_size=5, seed=7)

        self.assertEqual(int(placebo.sum()), int(signal.sum()))
        self.assertFalse(placebo.equals(signal))


class ScannerPayloadTest(unittest.TestCase):
    def test_fdr_adjustment_is_monotonic_and_bounded(self):
        p_values = pd.Series([0.01, 0.04, 0.03, 0.80])

        adjusted = app.benjamini_hochberg(p_values)

        ordered = adjusted.loc[p_values.sort_values().index]
        self.assertTrue(ordered.is_monotonic_increasing)
        self.assertTrue(adjusted.between(0, 1).all())

    def test_parses_naver_deal_trends(self):
        payload = {"dealTrendInfos": [{
            "bizdate": "20260731",
            "foreignerPureBuyQuant": "+8,359,011",
            "foreignerHoldRatio": "46.70%",
            "organPureBuyQuant": "+3,618,959",
            "individualPureBuyQuant": "-11,681,307",
            "closePrice": "262,500",
            "accumulatedTradingVolume": "58,478,873",
        }]}

        result = app._parse_deal_trends(payload)

        self.assertEqual(result[0]["foreign_net_qty"], 8_359_011)
        self.assertEqual(result[0]["individual_net_qty"], -11_681_307)


if __name__ == "__main__":
    unittest.main()
