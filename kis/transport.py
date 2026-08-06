"""HTTP 계층 분리. 테스트는 실제 통신 없이 이 함수만 바꿔 끼운다."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

TIMEOUT_SECONDS = 20.0


@dataclass(frozen=True)
class HttpResponse:
    status_code: int
    headers: Mapping[str, str] = field(default_factory=dict)
    body: Mapping[str, Any] = field(default_factory=dict)

    def header(self, name: str, default: str = "") -> str:
        for key, value in self.headers.items():
            if key.lower() == name.lower():
                return value
        return default


# (method, url, headers, payload) -> HttpResponse
# payload는 GET이면 질의 문자열, POST면 JSON 본문으로 쓴다.
Transport = Callable[[str, str, Mapping[str, str], Mapping[str, Any]], HttpResponse]


def requests_transport(
    method: str,
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
) -> HttpResponse:
    """운영용 전송 구현. requests는 이 함수 안에서만 쓴다."""
    import requests

    if method.upper() == "GET":
        response = requests.get(
            url, headers=dict(headers), params=dict(payload), timeout=TIMEOUT_SECONDS
        )
    else:
        response = requests.post(
            url, headers=dict(headers), data=json.dumps(payload), timeout=TIMEOUT_SECONDS
        )

    try:
        body = response.json()
    except ValueError:
        body = {"msg1": "응답을 JSON으로 해석하지 못했습니다."}
    if not isinstance(body, dict):
        body = {"output": body}
    return HttpResponse(
        status_code=response.status_code, headers=dict(response.headers), body=body
    )
