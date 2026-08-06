"""KIS Open API 국내주식 엔드포인트 정의.

경로와 TR ID는 한국투자증권 공식 예제 저장소에서 확인한 값이다.
https://github.com/koreainvestment/open-trading-api/tree/main/examples_llm/domestic_stock

새 엔드포인트는 이 파일에만 추가하고 수집기 코드는 건드리지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class Endpoint:
    """호출 한 건의 계약. 파라미터 화이트리스트까지 여기서 강제한다."""

    name: str
    path: str
    tr_id: str
    outputs: tuple[str, ...]
    required: tuple[str, ...] = ()
    defaults: Mapping[str, str] = field(default_factory=dict)
    paginated: bool = False
    description: str = ""

    def build_params(self, values: Mapping[str, object]) -> dict[str, str]:
        """기본값과 호출값을 합치고 필수/미허용 파라미터를 검사한다."""
        allowed = set(self.required) | set(self.defaults)
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(
                f"{self.name}: 정의되지 않은 파라미터 {', '.join(unknown)}"
            )

        params = {key: str(value) for key, value in self.defaults.items()}
        for key, value in values.items():
            params[key] = "" if value is None else str(value)

        missing = [key for key in self.required if not params.get(key)]
        if missing:
            raise ValueError(f"{self.name}: 필수 파라미터 누락 {', '.join(missing)}")
        return params


MARKET_KRX = "J"
MARKET_UNIFIED = "UN"
MARKET_INDEX = "U"

# 지수 코드. 공식 예제가 쓰는 값이며 실호출로 한 번 대조해야 한다.
INDEX_KOSPI = "0001"
INDEX_SEGMENT_KOSPI = "KSP"


ENDPOINTS: dict[str, Endpoint] = {
    endpoint.name: endpoint
    for endpoint in (
        Endpoint(
            name="investor_trade_by_stock_daily",
            path="/uapi/domestic-stock/v1/quotations/investor-trade-by-stock-daily",
            tr_id="FHPTJ04160001",
            outputs=("output2",),
            required=("FID_COND_MRKT_DIV_CODE", "FID_INPUT_ISCD", "FID_INPUT_DATE_1"),
            defaults={"FID_ORG_ADJ_PRC": "", "FID_ETC_CLS_CODE": ""},
            paginated=True,
            description="종목별 투자자매매동향(일별). 11개 주체 순매수 수량·대금",
        ),
        Endpoint(
            name="inquire_investor_daily_by_market",
            path="/uapi/domestic-stock/v1/quotations/inquire-investor-daily-by-market",
            tr_id="FHPTJ04040000",
            outputs=("output",),
            required=(
                "FID_COND_MRKT_DIV_CODE",
                "FID_INPUT_ISCD",
                "FID_INPUT_DATE_1",
                "FID_INPUT_DATE_2",
                "FID_INPUT_ISCD_1",
                "FID_INPUT_ISCD_2",
            ),
            description=(
                "시장별 투자자매매동향(일별). 지수 자체의 수급. "
                "DATE_1이 기준일이고 그 이전 300거래일을 돌려준다 (DATE_2는 종료일이 아님)"
            ),
        ),
        Endpoint(
            name="inquire_daily_indexchartprice",
            path="/uapi/domestic-stock/v1/quotations/inquire-daily-indexchartprice",
            tr_id="FHKUP03500100",
            outputs=("output2",),
            required=(
                "FID_COND_MRKT_DIV_CODE",
                "FID_INPUT_ISCD",
                "FID_INPUT_DATE_1",
                "FID_INPUT_DATE_2",
                "FID_PERIOD_DIV_CODE",
            ),
            description="업종/지수 기간별 시세. 지수 거래량과 거래대금이 여기에만 있다",
        ),
        Endpoint(
            name="inquire_daily_itemchartprice",
            path="/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
            tr_id="FHKST03010100",
            outputs=("output2",),
            required=(
                "FID_COND_MRKT_DIV_CODE",
                "FID_INPUT_ISCD",
                "FID_INPUT_DATE_1",
                "FID_INPUT_DATE_2",
                "FID_PERIOD_DIV_CODE",
            ),
            defaults={"FID_ORG_ADJ_PRC": "0"},
            description="국내주식 기간별 시세. FID_ORG_ADJ_PRC=0이 수정주가",
        ),
        Endpoint(
            name="daily_short_sale",
            path="/uapi/domestic-stock/v1/quotations/daily-short-sale",
            tr_id="FHPST04830000",
            outputs=("output2",),
            required=("FID_COND_MRKT_DIV_CODE", "FID_INPUT_ISCD"),
            defaults={"FID_INPUT_DATE_1": "", "FID_INPUT_DATE_2": ""},
            description="종목별 일별 공매도 체결량·비중",
        ),
        Endpoint(
            name="daily_credit_balance",
            path="/uapi/domestic-stock/v1/quotations/daily-credit-balance",
            tr_id="FHPST04760000",
            outputs=("output",),
            required=(
                "FID_COND_MRKT_DIV_CODE",
                "FID_COND_SCR_DIV_CODE",
                "FID_INPUT_ISCD",
                "FID_INPUT_DATE_1",
            ),
            defaults={"FID_COND_SCR_DIV_CODE": "20476"},
            paginated=True,
            description="국내주식 신용잔고 일별추이(융자·대주)",
        ),
        Endpoint(
            name="daily_loan_trans",
            path="/uapi/domestic-stock/v1/quotations/daily-loan-trans",
            tr_id="HHPST074500C0",
            outputs=("output1",),
            required=("MRKT_DIV_CLS_CODE", "MKSC_SHRN_ISCD", "START_DATE", "END_DATE"),
            defaults={"CTS": ""},
            description=(
                "일별 대차거래 체결·상환·잔고. "
                "MRKT_DIV_CLS_CODE는 1=코스피 시장 전체, 2=코스닥 시장 전체, 3=종목별. "
                "1·2에서는 MKSC_SHRN_ISCD가 무시되므로 종목 조회는 반드시 3을 쓴다"
            ),
        ),
        Endpoint(
            name="program_trade_by_stock_daily",
            path="/uapi/domestic-stock/v1/quotations/program-trade-by-stock-daily",
            tr_id="FHPPG04650201",
            outputs=("output",),
            required=("FID_COND_MRKT_DIV_CODE", "FID_INPUT_ISCD"),
            defaults={"FID_INPUT_DATE_1": ""},
            description="종목별 프로그램매매 일별추이",
        ),
    )
}


def get_endpoint(name: str) -> Endpoint:
    try:
        return ENDPOINTS[name]
    except KeyError:
        known = ", ".join(sorted(ENDPOINTS))
        raise KeyError(f"알 수 없는 엔드포인트 {name}. 사용 가능: {known}") from None
