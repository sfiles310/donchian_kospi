"""KIS Open API 호출기. 유량 제한, 재시도, 연속조회를 한곳에서 처리한다."""

from __future__ import annotations

import time
from typing import Any, Callable, Iterator, Mapping

from .auth import Credentials, TokenManager
from .endpoints import Endpoint
from .transport import HttpResponse, Transport, requests_transport

DOMAIN_REAL = "https://openapi.koreainvestment.com:9443"
DOMAIN_DEMO = "https://openapivts.koreainvestment.com:29443"

# 실전은 초당 20건이지만 여유를 두고 절반 이하로 쓴다. 모의는 초당 2건이 한도다.
RATE_LIMIT_REAL = 8.0
RATE_LIMIT_DEMO = 2.0

# 초당 거래건수 초과. 잠시 쉬면 회복된다.
RATE_LIMIT_CODES = frozenset({"EGW00201"})
# 토큰 만료·불일치. 재발급 후 한 번 더 시도한다.
TOKEN_CODES = frozenset({"EGW00121", "EGW00123", "EGW00133"})
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

CONTINUATION_FLAGS = frozenset({"M", "F"})


class KisApiError(RuntimeError):
    """rt_cd가 0이 아닌 응답. 메시지에 인증 정보를 담지 않는다."""

    def __init__(self, endpoint: str, msg_cd: str, msg1: str, status_code: int) -> None:
        super().__init__(f"{endpoint} 실패 [{status_code}/{msg_cd}] {msg1}")
        self.endpoint = endpoint
        self.msg_cd = msg_cd
        self.msg1 = msg1
        self.status_code = status_code


class RateLimiter:
    """초당 호출 수를 균등 간격으로 제한한다."""

    def __init__(
        self,
        per_second: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if per_second <= 0:
            raise ValueError("per_second는 0보다 커야 합니다.")
        self._interval = 1.0 / per_second
        self._clock = clock
        self._sleeper = sleeper
        self._next_allowed = 0.0

    def acquire(self) -> None:
        now = self._clock()
        wait = self._next_allowed - now
        if wait > 0:
            self._sleeper(wait)
            now = self._next_allowed
        self._next_allowed = now + self._interval


class KisClient:
    """엔드포인트 하나를 호출해 output 목록을 돌려주는 최소 단위."""

    def __init__(
        self,
        credentials: Credentials,
        *,
        demo: bool = False,
        transport: Transport = requests_transport,
        per_second: float | None = None,
        max_retries: int = 4,
        max_pages: int = 40,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        cust_type: str = "P",
    ) -> None:
        self.base_url = DOMAIN_DEMO if demo else DOMAIN_REAL
        self.demo = demo
        self.max_retries = max(1, max_retries)
        self.max_pages = max(1, max_pages)
        self._transport = transport
        self._sleeper = sleeper
        self._cust_type = cust_type
        self._tokens = TokenManager(
            credentials, self.base_url, transport, clock=clock, sleeper=sleeper
        )
        self._credentials = credentials
        self._limiter = RateLimiter(
            per_second if per_second else (RATE_LIMIT_DEMO if demo else RATE_LIMIT_REAL),
            clock=clock,
            sleeper=sleeper,
        )

    def close(self) -> None:
        self._tokens.revoke()

    def __enter__(self) -> "KisClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def fetch(
        self, endpoint: Endpoint, values: Mapping[str, object]
    ) -> dict[str, list[dict[str, Any]]]:
        """연속조회까지 마친 결과를 output 키별 레코드 목록으로 모은다."""
        collected: dict[str, list[dict[str, Any]]] = {key: [] for key in endpoint.outputs}
        for body in self._pages(endpoint, values):
            for key in endpoint.outputs:
                collected[key].extend(_as_records(body.get(key)))
        return collected

    def _pages(
        self, endpoint: Endpoint, values: Mapping[str, object]
    ) -> Iterator[Mapping[str, Any]]:
        params = endpoint.build_params(values)
        tr_cont = ""
        for _ in range(self.max_pages if endpoint.paginated else 1):
            response = self._request(endpoint, params, tr_cont)
            yield response.body
            if not endpoint.paginated:
                return
            flag = response.header("tr_cont").strip()
            if flag not in CONTINUATION_FLAGS:
                return
            tr_cont = "N"

    def _request(
        self, endpoint: Endpoint, params: Mapping[str, str], tr_cont: str
    ) -> HttpResponse:
        url = f"{self.base_url}{endpoint.path}"
        token_retried = False

        for attempt in range(1, self.max_retries + 1):
            self._limiter.acquire()
            headers = self._headers(endpoint, tr_cont)
            response = self._transport("GET", url, headers, params)
            body = response.body if isinstance(response.body, Mapping) else {}
            msg_cd = str(body.get("msg_cd", "")).strip()
            msg1 = str(body.get("msg1", "")).strip()
            rt_cd = str(body.get("rt_cd", "")).strip()

            if response.status_code == 200 and rt_cd == "0":
                return response

            if msg_cd in TOKEN_CODES and not token_retried:
                token_retried = True
                self._tokens.invalidate()
                continue

            retryable = response.status_code in RETRYABLE_STATUS or msg_cd in RATE_LIMIT_CODES
            if retryable and attempt < self.max_retries:
                self._sleeper(min(30.0, 2.0**attempt))
                continue

            raise KisApiError(
                endpoint.name,
                msg_cd or "-",
                msg1 or "응답 본문에 사유가 없습니다.",
                response.status_code,
            )

        raise KisApiError(endpoint.name, "-", "재시도 한도를 넘었습니다.", 0)

    def _headers(self, endpoint: Endpoint, tr_cont: str) -> dict[str, str]:
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self._tokens.token()}",
            "appkey": self._credentials.app_key,
            "appsecret": self._credentials.app_secret,
            "tr_id": endpoint.tr_id,
            "tr_cont": tr_cont,
            "custtype": self._cust_type,
        }


def _as_records(output: object) -> list[dict[str, Any]]:
    """output은 배열일 수도 객체 하나일 수도 있다."""
    if output is None:
        return []
    if isinstance(output, Mapping):
        return [dict(output)]
    if isinstance(output, list):
        return [dict(item) for item in output if isinstance(item, Mapping)]
    return []
