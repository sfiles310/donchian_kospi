import unittest

import pandas as pd

from download_krx_foreign_flow_panel import normalize_day


class NormalizeDayTests(unittest.TestCase):
    def test_normalizes_and_fills_missing_foreign_flow_with_zero(self):
        price = pd.DataFrame(
            {
                "시가": [100, 0],
                "종가": [110, 0],
                "거래량": [10, 0],
                "거래대금": [1_050, 0],
                "시가총액": [11_000, 5_000],
            },
            index=["005930", "000300"],
        )
        flow = pd.DataFrame({"순매수거래대금": [200]}, index=["005930"])

        result = normalize_day(price, flow, "2026-08-03")

        self.assertEqual(result["ticker"].tolist(), ["005930", "000300"])
        self.assertEqual(result["foreign_net_value"].tolist(), [200, 0])
        self.assertEqual(result["is_tradable"].tolist(), [True, False])
        self.assertEqual(
            result["data_available_at"].iat[0], "2026-08-03T18:00:00+09:00"
        )

    def test_rejects_impossible_foreign_flow(self):
        price = pd.DataFrame(
            {
                "시가": [100],
                "종가": [100],
                "거래량": [10],
                "거래대금": [1_000],
                "시가총액": [10_000],
            },
            index=["005930"],
        )
        flow = pd.DataFrame({"순매수거래대금": [1_001_001]}, index=["005930"])

        with self.assertRaisesRegex(ValueError, "거래대금을 넘는"):
            normalize_day(price, flow, "2026-08-03")

if __name__ == "__main__":
    unittest.main()
