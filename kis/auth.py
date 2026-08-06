"""KIS Open API 인증. 키와 토큰은 메모리에만 두고 어디에도 기록하지 않는다."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable

from .transport import HttpResponse, Transport

TOKEN_PATH = "/oauth2/tokenP"
REVOKE_PATH = "/oauth2/revokeP"

# KIS는 접근토큰 재발급을 1분에 1회로 제한한다.
REISSUE_INTERVAL_SECONDS = 60.0
# 만료 직전 호출이 실패하지 않도록 미리 갱신하는 여유 시간.
RENEW_MARGIN_SECONDS = 600.0


@dataclass(frozen=True)
class Credentials:
    """앱키와 시크릿. 로그와 예외 메시지에 원문이 새지 않도록 표현을 가린다."""

    app_key: str
    app_secret: str

    @classmethod
    def from_env(
        cls,
        key_name: str = "KIS_APP_KEY",
        secret_name: str = "KIS_APP_SECRET",
    ) -> "Credentials":
        app_key = os.getenv(key_name, "").strip()
        app_secret = os.getenv(secret_name, "").strip()
        if not app_key or not app_secret:
            raise SystemExit(
                f"환경변수 {key_name}, {secret_name}를 설정한 뒤 다시 실행하세요."
            )
        return cls(app_key=app_key, app_secret=app_secret)

    def __repr__(self) -> str:  # pragma: no cover - 표현 전용
        return "Credentials(app_key='***', app_secret='***')"

    __str__ = __repr__


class TokenError(RuntimeError):
    """토큰 발급 실패. 메시지에 키나 토큰 값을 담지 않는다."""


class TokenManager:
    """접근토큰의 발급·재사용·폐기를 담당한다.

    토큰을 파일로 저장하지 않는다는 저장소 규칙을 지키기 위해 메모리에만 보관한다.
    그 대신 수집기는 한 프로세스 안에서 이어 달리도록 설계해야 한다.
    """

    def __init__(
        self,
        credentials: Credentials,
        base_url: str,
        transport: Transport,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._credentials = credentials
        self._base_url = base_url.rstrip("/")
        self._transport = transport
        self._clock = clock
        self._sleeper = sleeper
        self._token: str | None = None
        self._expires_at = 0.0
        self._issued_at: float | None = None

    @property
    def has_token(self) -> bool:
        return self._token is not None

    def token(self) -> str:
        """유효한 토큰을 돌려주고, 없거나 만료가 가까우면 새로 발급받는다."""
        now = self._clock()
        if self._token is not None and now < self._expires_at - RENEW_MARGIN_SECONDS:
            return self._token
        return self._issue()

    def invalidate(self) -> None:
        """서버가 토큰 만료를 알려온 경우 다음 호출에서 재발급하도록 표시한다."""
        self._token = None
        self._expires_at = 0.0

    def _issue(self) -> str:
        self._wait_for_reissue_window()
        response = self._transport(
            "POST",
            f"{self._base_url}{TOKEN_PATH}",
            {"content-type": "application/json; charset=utf-8"},
            {
                "grant_type": "client_credentials",
                "appkey": self._credentials.app_key,
                "appsecret": self._credentials.app_secret,
            },
        )
        token = self._read_token(response)
        expires_in = self._read_expires_in(response)
        self._token = token
        self._issued_at = self._clock()
        self._expires_at = self._issued_at + expires_in
        return token

    def _wait_for_reissue_window(self) -> None:
        if self._issued_at is None:
            return
        elapsed = self._clock() - self._issued_at
        remaining = REISSUE_INTERVAL_SECONDS - elapsed
        if remaining > 0:
            self._sleeper(remaining)

    @staticmethod
    def _read_token(response: HttpResponse) -> str:
        body = response.body
        token = body.get("access_token")
        if response.status_code != 200 or not token:
            code = body.get("error_code") or body.get("msg_cd") or response.status_code
            message = body.get("error_description") or body.get("msg1") or "사유 미상"
            raise TokenError(f"접근토큰 발급 실패 ({code}): {message}")
        return str(token)

    @staticmethod
    def _read_expires_in(response: HttpResponse) -> float:
        try:
            expires_in = float(response.body.get("expires_in", 0))
        except (TypeError, ValueError):
            expires_in = 0.0
        # 값이 비정상이면 짧게 잡아 다음 호출에서 다시 확인하게 만든다.
        return expires_in if expires_in > 0 else RENEW_MARGIN_SECONDS + 60.0

    def revoke(self) -> None:
        """수집이 끝나면 토큰을 폐기한다. 실패해도 흐름을 막지 않는다."""
        if self._token is None:
            return
        try:
            self._transport(
                "POST",
                f"{self._base_url}{REVOKE_PATH}",
                {"content-type": "application/json; charset=utf-8"},
                {
                    "appkey": self._credentials.app_key,
                    "appsecret": self._credentials.app_secret,
                    "token": self._token,
                },
            )
        except Exception:  # noqa: BLE001 - 폐기 실패는 수집 결과에 영향이 없다
            pass
        finally:
            self.invalidate()
            self._issued_at = None
