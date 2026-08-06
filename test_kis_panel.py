"""정규화, point-in-time 표시, 저장소, 수집 계획을 검사한다."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from kis import check_consistency, collect, get_dataset, normalize
from kis.client import KisClient
from kis.auth import Credentials
from kis.datasets import DATASETS
from kis.fixtures import MockTransport
from kis.normalize import KST, available_at, parse_date, parse_number
from kis.store import PanelStore

INVESTOR = get_dataset("investor_flow_daily")


def sample_investor_record(date: str = "20260805") -> dict[str, str]:
    return {
        "stck_bsop_date": date,
        "stck_oprc": "70,100",
        "stck_hgpr": "71,500",
        "stck_lwpr": "69,800",
        "stck_clpr": "71,000",
        "acml_vol": "12,345,678",
        "acml_tr_pbmn": "876,543,210,000",
        "frgn_ntby_qty": "-2,298,569",
        "prsn_ntby_qty": "1,100,000",
        "orgn_ntby_qty": "1,198,569",
        "fund_ntby_qty": "300,000",
        "pe_fund_ntby_vol": "-50,000",
        "frgn_ntby_tr_pbmn": "-163,198,399,000",
    }


class ParsingTest(unittest.TestCase):
    def test_parse_number_handles_kis_formats(self) -> None:
        self.assertEqual(parse_number("-2,298,569"), -2298569.0)
        self.assertEqual(parse_number("12.34%"), 12.34)
        self.assertIsNone(parse_number(""))
        self.assertIsNone(parse_number("-"))
        self.assertIsNone(parse_number(None))
        self.assertIsNone(parse_number("알수없음"))

    def test_parse_date_accepts_both_shapes(self) -> None:
        self.assertEqual(parse_date("20260805"), "2026-08-05")
        self.assertEqual(parse_date("2026-08-05"), "2026-08-05")
        self.assertIsNone(parse_date("202608"))
        self.assertIsNone(parse_date("20261345"))


class NormalizeTest(unittest.TestCase):
    def test_maps_fields_and_pads_ticker(self) -> None:
        frame = normalize(INVESTOR, [sample_investor_record()], "5930")
        self.assertEqual(len(frame), 1)
        row = frame.iloc[0]
        self.assertEqual(row["ticker"], "005930")
        self.assertEqual(row["date"], "2026-08-05")
        self.assertEqual(row["foreign_net_qty"], -2298569)
        self.assertEqual(row["pension_net_qty"], 300000)
        self.assertEqual(row["private_fund_net_qty"], -50000)
        self.assertEqual(row["source"], "kis:investor_trade_by_stock_daily")

    def test_missing_fields_become_null_not_zero(self) -> None:
        frame = normalize(INVESTOR, [sample_investor_record()], "005930")
        # 응답에 없던 열을 0으로 채우면 "수급 없음"과 구분이 사라진다.
        self.assertTrue(pd.isna(frame.iloc[0]["bank_net_qty"]))

    def test_rows_without_close_are_dropped(self) -> None:
        # 상장 전 날짜에 API가 채워 보내는 빈 행. 종가가 없으면 관측이 아니다.
        empty_day = {"stck_bsop_date": "20230102", "frgn_ntby_qty": "0", "prsn_ntby_qty": "0"}
        frame = normalize(INVESTOR, [empty_day, sample_investor_record()], "005930")
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["date"], "2026-08-05")

    def test_all_empty_days_produce_empty_frame_with_schema(self) -> None:
        empty_day = {"stck_bsop_date": "20230102", "frgn_ntby_qty": "0"}
        frame = normalize(INVESTOR, [empty_day], "005930")
        self.assertTrue(frame.empty)
        self.assertIn("foreign_net_value", frame.columns)

    def test_rows_without_date_are_dropped(self) -> None:
        bad = sample_investor_record()
        bad["stck_bsop_date"] = ""
        frame = normalize(INVESTOR, [bad, sample_investor_record()], "005930")
        self.assertEqual(len(frame), 1)

    def test_empty_input_keeps_schema(self) -> None:
        frame = normalize(INVESTOR, [], "005930")
        self.assertTrue(frame.empty)
        self.assertIn("foreign_net_value", frame.columns)
        self.assertIn("data_available_at", frame.columns)

    def test_duplicate_dates_keep_last(self) -> None:
        first = sample_investor_record()
        second = sample_investor_record()
        second["frgn_ntby_qty"] = "999"
        frame = normalize(INVESTOR, [first, second], "005930")
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["foreign_net_qty"], 999)


class PointInTimeTest(unittest.TestCase):
    def test_available_at_uses_dataset_offset(self) -> None:
        stamp = available_at(INVESTOR, "2026-08-05")
        self.assertEqual(stamp.hour, 18)
        self.assertEqual(stamp.strftime("%Y-%m-%d"), "2026-08-05")

        credit = available_at(get_dataset("credit_balance_daily"), "2026-08-05")
        self.assertEqual(credit.strftime("%Y-%m-%d"), "2026-08-06")
        self.assertEqual(credit.hour, 9)

    def test_credit_balance_availability_follows_settlement_date(self) -> None:
        # 잔고는 결제일 기준이라 매매일 다음날에는 아직 알 수 없다.
        credit = get_dataset("credit_balance_daily")
        record = {
            "deal_date": "20260803",
            "stlm_date": "20260805",
            "stck_prpr": "246000",
            "whol_loan_rmnd_stcn": "22674026",
        }
        frame = normalize(
            credit, [record], "005930", collected_at=pd.Timestamp("2026-08-04 09:00", tz=KST)
        )
        self.assertTrue(bool(frame.iloc[0]["is_provisional"]))
        self.assertTrue(frame.iloc[0]["data_available_at"].startswith("2026-08-06"))

        later = normalize(
            credit, [record], "005930", collected_at=pd.Timestamp("2026-08-06 10:00", tz=KST)
        )
        self.assertFalse(bool(later.iloc[0]["is_provisional"]))

    def test_missing_settlement_date_falls_back_to_trade_date(self) -> None:
        credit = get_dataset("credit_balance_daily")
        record = {
            "deal_date": "20260803",
            "stlm_date": "",
            "stck_prpr": "246000",
            "whol_loan_rmnd_stcn": "1",
        }
        frame = normalize(
            credit, [record], "005930", collected_at=pd.Timestamp("2026-08-10 09:00", tz=KST)
        )
        self.assertTrue(frame.iloc[0]["data_available_at"].startswith("2026-08-04"))

    def test_provisional_when_collected_before_publication(self) -> None:
        early = pd.Timestamp("2026-08-05 15:00", tz=KST)
        frame = normalize(INVESTOR, [sample_investor_record()], "005930", collected_at=early)
        self.assertTrue(bool(frame.iloc[0]["is_provisional"]))

        late = pd.Timestamp("2026-08-05 19:00", tz=KST)
        frame = normalize(INVESTOR, [sample_investor_record()], "005930", collected_at=late)
        self.assertFalse(bool(frame.iloc[0]["is_provisional"]))


class ConsistencyTest(unittest.TestCase):
    def test_identity_uses_four_parties_not_five(self) -> None:
        # 기타단체는 2019년부터 0으로만 온다. 다섯을 더하면 그 이전 자료가 전부 깨진다.
        record = sample_investor_record()
        record.update(
            {
                "frgn_ntby_qty": "-100",
                "prsn_ntby_qty": "60",
                "orgn_ntby_qty": "30",
                "etc_corp_ntby_vol": "10",
                "etc_orgt_ntby_vol": "-7",
            }
        )
        late = pd.Timestamp("2026-08-06 09:00", tz=KST)
        frame = normalize(INVESTOR, [record], "005930", collected_at=late)
        self.assertEqual(check_consistency(INVESTOR, frame), [])

    def test_broken_four_party_identity_is_flagged(self) -> None:
        record = sample_investor_record()
        record.update(
            {
                "frgn_ntby_qty": "-100",
                "prsn_ntby_qty": "60",
                "orgn_ntby_qty": "30",
                "etc_corp_ntby_vol": "999",
            }
        )
        late = pd.Timestamp("2026-08-06 09:00", tz=KST)
        frame = normalize(INVESTOR, [record], "005930", collected_at=late)
        warnings = check_consistency(INVESTOR, frame)
        self.assertTrue(any("4주체" in text for text in warnings))

    def test_flags_flow_larger_than_turnover(self) -> None:
        record = sample_investor_record()
        record["frgn_ntby_tr_pbmn"] = "999,999,999,999,999"
        frame = normalize(INVESTOR, [record], "005930")
        warnings = check_consistency(INVESTOR, frame)
        self.assertTrue(any("거래대금을 넘는" in text for text in warnings))

    def test_clean_confirmed_row_has_no_warning(self) -> None:
        late = pd.Timestamp("2026-08-06 09:00", tz=KST)
        frame = normalize(INVESTOR, [sample_investor_record()], "005930", collected_at=late)
        self.assertEqual(check_consistency(INVESTOR, frame), [])


class StoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.store = PanelStore(Path(self.directory.name) / "panel.sqlite")

    def tearDown(self) -> None:
        self.store.close()
        self.directory.cleanup()

    def frame(self, date: str, net: str, collected: str = "2026-08-10 09:00") -> pd.DataFrame:
        record = sample_investor_record(date)
        record["frgn_ntby_qty"] = net
        return normalize(
            INVESTOR, [record], "005930", collected_at=pd.Timestamp(collected, tz=KST)
        )

    def test_upsert_is_idempotent(self) -> None:
        self.store.upsert(INVESTOR, self.frame("20260805", "-100"))
        self.store.upsert(INVESTOR, self.frame("20260805", "-100"))
        self.assertEqual(len(self.store.read(INVESTOR)), 1)

    def test_rerun_overwrites_with_latest_value(self) -> None:
        self.store.upsert(INVESTOR, self.frame("20260805", "-100"))
        self.store.upsert(INVESTOR, self.frame("20260805", "-200"))
        stored = self.store.read(INVESTOR)
        self.assertEqual(stored.iloc[0]["foreign_net_qty"], -200)

    def test_read_hides_provisional_by_default(self) -> None:
        self.store.upsert(INVESTOR, self.frame("20260805", "-100", collected="2026-08-05 15:00"))
        self.assertTrue(self.store.read(INVESTOR).empty)
        self.assertEqual(len(self.store.read(INVESTOR, confirmed_only=False)), 1)

    def test_coverage_reports_range(self) -> None:
        self.store.upsert(INVESTOR, self.frame("20260803", "-100"))
        self.store.upsert(INVESTOR, self.frame("20260805", "-100"))
        coverage = self.store.coverage(INVESTOR)
        self.assertEqual(coverage.iloc[0]["rows"], 2)
        self.assertEqual(coverage.iloc[0]["first_date"], "2026-08-03")
        self.assertEqual(coverage.iloc[0]["last_date"], "2026-08-05")

    def test_missing_table_reads_empty(self) -> None:
        self.assertTrue(self.store.read("price_daily").empty)
        self.assertIsNone(self.store.latest_date("price_daily", "005930"))

    def test_null_stays_null_through_roundtrip(self) -> None:
        self.store.upsert(INVESTOR, self.frame("20260805", "-100"))
        stored = self.store.read(INVESTOR)
        self.assertTrue(pd.isna(stored.iloc[0]["bank_net_qty"]))


class IndexKeyTest(unittest.TestCase):
    """지수 코드 0001이 종목코드 규칙에 휩쓸려 000001이 되면 안 된다."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.store = PanelStore(Path(self.directory.name) / "panel.sqlite")
        self.market = get_dataset("market_investor_flow_daily")

    def tearDown(self) -> None:
        self.store.close()
        self.directory.cleanup()

    def market_frame(self) -> pd.DataFrame:
        record = {
            "stck_bsop_date": "20260805",
            "bstp_nmix_prpr": "4,120.55",
            "bstp_nmix_oprc": "4,090.10",
            "bstp_nmix_hgpr": "4,131.00",
            "bstp_nmix_lwpr": "4,088.20",
            "bstp_nmix_prdy_ctrt": "0.85",
            "frgn_ntby_tr_pbmn": "812,400,000,000",
            "prsn_ntby_tr_pbmn": "-511,200,000,000",
            "orgn_ntby_tr_pbmn": "-301,200,000,000",
        }
        return normalize(
            self.market, [record], "0001", collected_at=pd.Timestamp("2026-08-06 09:00", tz=KST)
        )

    def test_index_code_is_not_padded(self) -> None:
        frame = self.market_frame()
        self.assertEqual(frame.iloc[0]["ticker"], "0001")

    def test_index_values_keep_decimals(self) -> None:
        frame = self.market_frame()
        self.assertAlmostEqual(frame.iloc[0]["index_close"], 4120.55)
        self.assertAlmostEqual(frame.iloc[0]["index_change_pct"], 0.85)

    def test_store_roundtrip_keeps_index_code(self) -> None:
        self.store.upsert(self.market, self.market_frame())
        stored = self.store.read(self.market, tickers=["0001"])
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored.iloc[0]["ticker"], "0001")

    def test_stock_and_index_flows_share_column_names(self) -> None:
        # 같은 이름이어야 종목 수급과 지수 수급을 같은 코드로 다룰 수 있다.
        shared = {"foreign_net_value", "individual_net_value", "pension_net_value"}
        self.assertTrue(shared <= set(INVESTOR.columns))
        self.assertTrue(shared <= set(self.market.columns))


class CrossCheckTest(unittest.TestCase):
    """엉뚱한 대상의 값이 저장돼도 데이터셋 안쪽 항등식은 성립한다. 종가로 대조한다."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.store = PanelStore(Path(self.directory.name) / "panel.sqlite")
        self.late = pd.Timestamp("2026-08-10 09:00", tz=KST)

    def tearDown(self) -> None:
        self.store.close()
        self.directory.cleanup()

    def seed(self, loan_close: str) -> None:
        price = normalize(
            get_dataset("price_daily"),
            [{"stck_bsop_date": "20260805", "stck_clpr": "246000", "acml_vol": "1"}],
            "005930",
            collected_at=self.late,
        )
        self.store.upsert(get_dataset("price_daily"), price)
        loan = normalize(
            get_dataset("loan_trans_daily"),
            [{"bsop_date": "20260805", "stck_prpr": loan_close, "rmnd_stcn": "10"}],
            "005930",
            collected_at=self.late,
        )
        self.store.upsert(get_dataset("loan_trans_daily"), loan)

    def test_matching_close_passes(self) -> None:
        self.seed("246000")
        result = self.store.cross_check_close("005930")
        row = result[result["dataset"] == "loan_trans_daily"].iloc[0]
        self.assertEqual(row["mismatched"], 0)
        self.assertEqual(row["verdict"], "OK")

    def test_wrong_target_is_flagged(self) -> None:
        # 시장 전체를 받아오면 종가 자리에 지수가 들어온다.
        self.seed("6598")
        result = self.store.cross_check_close("005930")
        row = result[result["dataset"] == "loan_trans_daily"].iloc[0]
        self.assertEqual(row["mismatched"], 1)
        self.assertEqual(row["verdict"], "다른 대상의 값일 수 있음")

    def test_dividend_adjustment_is_not_flagged_as_wrong_target(self) -> None:
        # 원주가로 오는 데이터셋은 배당·분배락만큼 어긋난다. 오탐하면 안 된다.
        self.seed("244000")
        result = self.store.cross_check_close("005930")
        row = result[result["dataset"] == "loan_trans_daily"].iloc[0]
        self.assertEqual(row["mismatched"], 1)
        self.assertEqual(row["verdict"], "수정주가 차이로 보임")

    def test_delete_removes_only_named_ticker(self) -> None:
        self.seed("246000")
        self.assertEqual(self.store.delete(get_dataset("loan_trans_daily"), tickers=["000660"]), 0)
        self.assertEqual(len(self.store.read("loan_trans_daily", confirmed_only=False)), 1)
        self.store.delete(get_dataset("loan_trans_daily"), tickers=["005930"])
        self.assertTrue(self.store.read("loan_trans_daily", confirmed_only=False).empty)


class CollectTest(unittest.TestCase):
    """모의 전송기로 수집 계획 전체를 돌린다. 통신은 일어나지 않는다."""

    def client(self) -> KisClient:
        return KisClient(
            Credentials(app_key="mock", app_secret="mock"),
            transport=MockTransport(lookback_days=20),
            per_second=1000.0,
        )

    def test_every_dataset_has_a_plan_and_collects(self) -> None:
        with self.client() as client:
            for name in DATASETS:
                dataset = get_dataset(name)
                code = "005930" if dataset.pads_ticker else "0001"
                frame = collect(client, dataset, code, "2026-06-01", "2026-08-05")
                self.assertFalse(frame.empty, name)
                self.assertEqual(set(frame["ticker"]), {code}, name)

    def test_results_stay_inside_requested_window(self) -> None:
        with self.client() as client:
            frame = collect(client, INVESTOR, "005930", "2026-07-01", "2026-07-31")
        self.assertGreaterEqual(frame["date"].min(), "2026-07-01")
        self.assertLessEqual(frame["date"].max(), "2026-07-31")

    def test_dates_are_unique_and_sorted(self) -> None:
        with self.client() as client:
            frame = collect(client, INVESTOR, "005930", "2026-05-01", "2026-08-05")
        self.assertEqual(frame["date"].is_monotonic_increasing, True)
        self.assertFalse(frame["date"].duplicated().any())

    def test_rejects_inverted_range(self) -> None:
        with self.client() as client:
            with self.assertRaises(ValueError):
                collect(client, INVESTOR, "005930", "2026-08-05", "2026-06-01")

    def test_anchor_walk_back_terminates(self) -> None:
        transport = MockTransport(lookback_days=20)
        client = KisClient(
            Credentials(app_key="mock", app_secret="mock"),
            transport=transport,
            per_second=1000.0,
        )
        collect(client, INVESTOR, "005930", "2026-01-02", "2026-08-05")
        # 거슬러 올라가는 호출이 유한 횟수 안에 끝나야 한다.
        self.assertLess(len(transport.calls), 40)


if __name__ == "__main__":
    unittest.main()
