"""KIS 클라이언트와 인증 계층. 실제 통신 없이 전송기만 바꿔 끼워 검사한다."""

from __future__ import annotations

import unittest

from kis.auth import Credentials, TokenError, TokenManager
from kis.client import KisApiError, KisClient, RateLimiter
from kis.endpoints import Endpoint, get_endpoint
from kis.transport import HttpResponse

CREDENTIALS = Credentials(app_key="key-abc", app_secret="secret-xyz")


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def token_response(expires_in: int = 86400) -> HttpResponse:
    return HttpResponse(200, {}, {"access_token": "T-1", "expires_in": expires_in})


class CredentialsTest(unittest.TestCase):
    def test_repr_hides_secret(self) -> None:
        text = f"{CREDENTIALS!r} {CREDENTIALS}"
        self.assertNotIn("key-abc", text)
        self.assertNotIn("secret-xyz", text)

    def test_from_env_requires_both(self) -> None:
        with self.assertRaises(SystemExit):
            Credentials.from_env("NO_SUCH_KEY_ENV", "NO_SUCH_SECRET_ENV")


class TokenManagerTest(unittest.TestCase):
    def test_reuses_token_until_renew_margin(self) -> None:
        clock = FakeClock()
        calls = []

        def transport(method, url, headers, payload):
            calls.append(url)
            return token_response()

        manager = TokenManager(
            CREDENTIALS, "https://x", transport, clock=clock, sleeper=clock.sleep
        )
        self.assertEqual(manager.token(), "T-1")
        clock.now += 3600
        self.assertEqual(manager.token(), "T-1")
        self.assertEqual(len(calls), 1)

    def test_reissue_waits_for_one_minute_window(self) -> None:
        clock = FakeClock()

        def transport(method, url, headers, payload):
            return token_response()

        manager = TokenManager(
            CREDENTIALS, "https://x", transport, clock=clock, sleeper=clock.sleep
        )
        manager.token()
        manager.invalidate()
        manager.token()
        # 재발급 제한 60초를 스스로 기다렸어야 한다.
        self.assertGreaterEqual(clock.now, 60.0)

    def test_failure_message_omits_credentials(self) -> None:
        def transport(method, url, headers, payload):
            return HttpResponse(403, {}, {"error_code": "EGW00133", "error_description": "거부"})

        manager = TokenManager(CREDENTIALS, "https://x", transport)
        with self.assertRaises(TokenError) as caught:
            manager.token()
        self.assertNotIn("secret-xyz", str(caught.exception))
        self.assertIn("EGW00133", str(caught.exception))


class RateLimiterTest(unittest.TestCase):
    def test_spaces_calls_evenly(self) -> None:
        clock = FakeClock()
        limiter = RateLimiter(4.0, clock=clock, sleeper=clock.sleep)
        for _ in range(3):
            limiter.acquire()
        self.assertAlmostEqual(clock.now, 0.5)


class ClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.endpoint = get_endpoint("program_trade_by_stock_daily")

    def build(self, responses):
        self.sent: list[dict] = []
        queue = list(responses)

        def transport(method, url, headers, payload):
            if url.endswith("/oauth2/tokenP"):
                return token_response()
            self.sent.append({"headers": dict(headers), "payload": dict(payload)})
            return queue.pop(0)

        return KisClient(
            CREDENTIALS,
            transport=transport,
            clock=self.clock,
            sleeper=self.clock.sleep,
        )

    def test_sends_tr_id_and_bearer_token(self) -> None:
        client = self.build([HttpResponse(200, {}, {"rt_cd": "0", "output": []})])
        client.fetch(self.endpoint, {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": "005930"})
        headers = self.sent[0]["headers"]
        self.assertEqual(headers["tr_id"], "FHPPG04650201")
        self.assertEqual(headers["authorization"], "Bearer T-1")
        self.assertEqual(headers["custtype"], "P")

    def test_object_output_becomes_single_record(self) -> None:
        client = self.build([HttpResponse(200, {}, {"rt_cd": "0", "output": {"a": "1"}})])
        result = client.fetch(
            self.endpoint, {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": "005930"}
        )
        self.assertEqual(result["output"], [{"a": "1"}])

    def test_retries_on_rate_limit_code(self) -> None:
        client = self.build(
            [
                HttpResponse(200, {}, {"rt_cd": "1", "msg_cd": "EGW00201", "msg1": "초과"}),
                HttpResponse(200, {}, {"rt_cd": "0", "output": [{"a": "1"}]}),
            ]
        )
        result = client.fetch(
            self.endpoint, {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": "005930"}
        )
        self.assertEqual(len(result["output"]), 1)
        self.assertEqual(len(self.sent), 2)

    def test_raises_on_business_error(self) -> None:
        client = self.build(
            [HttpResponse(200, {}, {"rt_cd": "1", "msg_cd": "40580000", "msg1": "종목 오류"})]
        )
        with self.assertRaises(KisApiError) as caught:
            client.fetch(
                self.endpoint, {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": "999999"}
            )
        self.assertIn("종목 오류", str(caught.exception))

    def test_follows_tr_cont_pages(self) -> None:
        endpoint = get_endpoint("daily_credit_balance")
        client = self.build(
            [
                HttpResponse(200, {"tr_cont": "M"}, {"rt_cd": "0", "output": [{"a": "1"}]}),
                HttpResponse(200, {"tr_cont": "D"}, {"rt_cd": "0", "output": [{"a": "2"}]}),
            ]
        )
        result = client.fetch(
            endpoint,
            {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": "005930",
                "FID_INPUT_DATE_1": "20260805",
            },
        )
        self.assertEqual(len(result["output"]), 2)
        self.assertEqual(self.sent[1]["headers"]["tr_cont"], "N")


class EndpointTest(unittest.TestCase):
    def test_rejects_unknown_parameter(self) -> None:
        endpoint = get_endpoint("program_trade_by_stock_daily")
        with self.assertRaises(ValueError):
            endpoint.build_params({"FID_COND_MRKT_DIV_CODE": "J", "OOPS": "1"})

    def test_reports_missing_required_parameter(self) -> None:
        endpoint = get_endpoint("program_trade_by_stock_daily")
        with self.assertRaises(ValueError) as caught:
            endpoint.build_params({"FID_COND_MRKT_DIV_CODE": "J"})
        self.assertIn("FID_INPUT_ISCD", str(caught.exception))

    def test_defaults_are_applied(self) -> None:
        endpoint = get_endpoint("investor_trade_by_stock_daily")
        params = endpoint.build_params(
            {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": "005930",
                "FID_INPUT_DATE_1": "20260805",
            }
        )
        self.assertEqual(params["FID_ORG_ADJ_PRC"], "")
        self.assertEqual(params["FID_ETC_CLS_CODE"], "")

    def test_no_order_endpoints_are_registered(self) -> None:
        from kis.endpoints import ENDPOINTS

        for endpoint in ENDPOINTS.values():
            self.assertIn("/quotations/", endpoint.path)
            self.assertNotIn("order", endpoint.path)

    def test_endpoint_is_hashable_and_frozen(self) -> None:
        endpoint = Endpoint(name="x", path="/p", tr_id="T", outputs=("output",))
        with self.assertRaises(Exception):
            endpoint.tr_id = "Y"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
