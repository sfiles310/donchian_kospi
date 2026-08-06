"""KIS 원시 응답을 표준 패널 스키마로 옮기는 규칙.

원칙 세 가지.
1. 필드 이름은 영문 snake_case로 통일해 나중에 다른 출처를 붙여도 충돌하지 않게 한다.
2. 모든 행은 그 값을 실제로 볼 수 있었던 시각(`data_available_at`)을 함께 저장한다.
   미래 정보로 과거를 판단하는 실수를 스키마 차원에서 막기 위한 것이다.
3. 아직 확정 전이면 `is_provisional`로 표시하고 검증에서 제외할 수 있게 한다.
"""

from __future__ import annotations

from dataclasses import dataclass

INT = "int"
FLOAT = "float"
DATE = "date"
TEXT = "text"


@dataclass(frozen=True)
class Field:
    raw: str
    column: str
    kind: str = INT


@dataclass(frozen=True)
class Dataset:
    """수집 단위 하나. 어떤 엔드포인트의 어떤 output을 어떻게 읽을지 담는다."""

    name: str
    endpoint: str
    output: str
    fields: tuple[Field, ...]
    # 자료가 공개되는 시점. (기준일로부터 며칠 뒤, 그날 몇 시 KST)
    available_after_days: int = 0
    available_at_hour: int = 18
    description: str = ""
    # 종목코드는 6자리로 채우지만 지수 코드는 원형을 유지해야 한다.
    pads_ticker: bool = True
    # 공개 시점을 date가 아닌 다른 날짜 열에서 잰다. 결제 기준 자료에 쓴다.
    available_basis_column: str | None = None

    @property
    def date_field(self) -> Field:
        for field in self.fields:
            if field.column == "date":
                return field
        raise ValueError(f"{self.name}: date 열이 정의되지 않았습니다.")

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(field.column for field in self.fields)


# 종목 수급과 시장 수급이 같은 필드명을 쓰므로 한 벌만 정의해 둘 다에 쓴다.
# 기관을 하나로 합치지 않는 것이 요점이다. 연기금과 사모펀드는 성격이 반대다.
_NET_QTY_FIELDS = (
    Field("frgn_ntby_qty", "foreign_net_qty"),
    Field("frgn_reg_ntby_qty", "foreign_reg_net_qty"),
    Field("frgn_nreg_ntby_qty", "foreign_nreg_net_qty"),
    Field("prsn_ntby_qty", "individual_net_qty"),
    Field("orgn_ntby_qty", "institution_net_qty"),
    Field("scrt_ntby_qty", "securities_net_qty"),
    Field("ivtr_ntby_qty", "invtrust_net_qty"),
    Field("pe_fund_ntby_vol", "private_fund_net_qty"),
    Field("bank_ntby_qty", "bank_net_qty"),
    Field("insu_ntby_qty", "insurance_net_qty"),
    Field("mrbn_ntby_qty", "merchant_bank_net_qty"),
    Field("fund_ntby_qty", "pension_net_qty"),
    Field("etc_ntby_qty", "etc_net_qty"),
    Field("etc_corp_ntby_vol", "etc_corp_net_qty"),
    Field("etc_orgt_ntby_vol", "etc_org_net_qty"),
)

_NET_VALUE_FIELDS = (
    Field("frgn_ntby_tr_pbmn", "foreign_net_value"),
    Field("frgn_reg_ntby_pbmn", "foreign_reg_net_value"),
    Field("frgn_nreg_ntby_pbmn", "foreign_nreg_net_value"),
    Field("prsn_ntby_tr_pbmn", "individual_net_value"),
    Field("orgn_ntby_tr_pbmn", "institution_net_value"),
    Field("scrt_ntby_tr_pbmn", "securities_net_value"),
    Field("ivtr_ntby_tr_pbmn", "invtrust_net_value"),
    Field("pe_fund_ntby_tr_pbmn", "private_fund_net_value"),
    Field("bank_ntby_tr_pbmn", "bank_net_value"),
    Field("insu_ntby_tr_pbmn", "insurance_net_value"),
    Field("mrbn_ntby_tr_pbmn", "merchant_bank_net_value"),
    Field("fund_ntby_tr_pbmn", "pension_net_value"),
    Field("etc_ntby_tr_pbmn", "etc_net_value"),
    Field("etc_corp_ntby_tr_pbmn", "etc_corp_net_value"),
    Field("etc_orgt_ntby_tr_pbmn", "etc_org_net_value"),
)


INVESTOR_FLOW = Dataset(
    name="investor_flow_daily",
    endpoint="investor_trade_by_stock_daily",
    output="output2",
    available_at_hour=18,
    description="종목별 투자자 11주체 일별 순매수 수량·대금",
    fields=(
        Field("stck_bsop_date", "date", DATE),
        Field("stck_oprc", "open"),
        Field("stck_hgpr", "high"),
        Field("stck_lwpr", "low"),
        Field("stck_clpr", "close"),
        Field("acml_vol", "volume"),
        Field("acml_tr_pbmn", "trading_value"),
    )
    + _NET_QTY_FIELDS
    + _NET_VALUE_FIELDS
    + (
        # 외국인 매도·매수 총량. 순매수만으로는 회전이 안 보인다.
        Field("frgn_seln_vol", "foreign_sell_qty"),
        Field("frgn_shnu_vol", "foreign_buy_qty"),
        Field("frgn_seln_tr_pbmn", "foreign_sell_value"),
        Field("frgn_shnu_tr_pbmn", "foreign_buy_value"),
    ),
)


# 2026-08-06 실호출 검증: 5주체(외국인·개인·기관계·기타법인·기타단체) 순매수 대금의
# 합이 정확히 0으로 떨어졌고, 기관 세부 7항목의 합도 기관계와 반올림 오차 안에서 일치했다.
# 대금 단위는 백만원이다.
MARKET_INVESTOR_FLOW = Dataset(
    name="market_investor_flow_daily",
    endpoint="inquire_investor_daily_by_market",
    output="output",
    available_at_hour=18,
    pads_ticker=False,
    description="지수(코스피) 자체의 투자자별 일별 순매수. 지수 신호용. 대금 단위 백만원",
    fields=(
        Field("stck_bsop_date", "date", DATE),
        Field("bstp_nmix_oprc", "index_open", FLOAT),
        Field("bstp_nmix_hgpr", "index_high", FLOAT),
        Field("bstp_nmix_lwpr", "index_low", FLOAT),
        Field("bstp_nmix_prpr", "index_close", FLOAT),
        Field("stck_prdy_clpr", "index_prev_close", FLOAT),
        Field("bstp_nmix_prdy_vrss", "index_change", FLOAT),
        Field("bstp_nmix_prdy_ctrt", "index_change_pct", FLOAT),
    )
    + _NET_QTY_FIELDS
    + _NET_VALUE_FIELDS,
)


# 지수 거래량은 market_investor_flow_daily에 없다. FTD처럼 거래량을 쓰는 규칙에는
# 이 데이터셋이 필요하다.
MARKET_PRICE_DAILY = Dataset(
    name="market_price_daily",
    endpoint="inquire_daily_indexchartprice",
    output="output2",
    available_at_hour=16,
    pads_ticker=False,
    description="지수 일봉과 거래량·거래대금",
    fields=(
        Field("stck_bsop_date", "date", DATE),
        Field("bstp_nmix_oprc", "index_open", FLOAT),
        Field("bstp_nmix_hgpr", "index_high", FLOAT),
        Field("bstp_nmix_lwpr", "index_low", FLOAT),
        Field("bstp_nmix_prpr", "index_close", FLOAT),
        Field("acml_vol", "volume"),
        Field("acml_tr_pbmn", "trading_value"),
        Field("mod_yn", "is_adjusted", TEXT),
    ),
)


PRICE_DAILY = Dataset(
    name="price_daily",
    endpoint="inquire_daily_itemchartprice",
    output="output2",
    available_at_hour=16,
    description="수정주가 일봉",
    fields=(
        Field("stck_bsop_date", "date", DATE),
        Field("stck_oprc", "open"),
        Field("stck_hgpr", "high"),
        Field("stck_lwpr", "low"),
        Field("stck_clpr", "close"),
        Field("acml_vol", "volume"),
        Field("acml_tr_pbmn", "trading_value"),
        Field("flng_cls_code", "rights_code", TEXT),
        Field("prtt_rate", "split_rate", FLOAT),
        Field("mod_yn", "is_adjusted", TEXT),
        Field("revl_issu_reas", "reval_reason", TEXT),
    ),
)


SHORT_SALE = Dataset(
    name="short_sale_daily",
    endpoint="daily_short_sale",
    output="output2",
    available_at_hour=18,
    description="일별 공매도 체결량과 비중",
    fields=(
        Field("stck_bsop_date", "date", DATE),
        Field("stck_clpr", "close"),
        Field("acml_vol", "volume"),
        Field("acml_tr_pbmn", "trading_value"),
        Field("ssts_cntg_qty", "short_qty"),
        Field("ssts_vol_rlim", "short_qty_ratio", FLOAT),
        Field("ssts_tr_pbmn", "short_value"),
        Field("ssts_tr_pbmn_rlim", "short_value_ratio", FLOAT),
        Field("acml_ssts_cntg_qty", "short_qty_cum"),
        Field("acml_ssts_cntg_qty_rlim", "short_qty_cum_ratio", FLOAT),
        Field("acml_ssts_tr_pbmn", "short_value_cum"),
        Field("acml_ssts_tr_pbmn_rlim", "short_value_cum_ratio", FLOAT),
        Field("avrg_prc", "short_avg_price", FLOAT),
    ),
)


# 2026-08-06 실호출로 확인: 행은 매매일(deal_date)로 매겨지지만 잔고는 결제일 기준이고,
# 결제일은 매매일 +2영업일이었다(124/135일). 그래서 이 행을 실제로 알 수 있는 시점은
# 매매일 다음날이 아니라 결제일 다음날이다. date 기준으로 재면 하루가 아니라 사흘을 앞당겨
# 보게 되므로 백테스트가 실제보다 좋게 나온다.
CREDIT_BALANCE = Dataset(
    name="credit_balance_daily",
    endpoint="daily_credit_balance",
    output="output",
    available_basis_column="settlement_date",
    available_after_days=1,
    available_at_hour=9,
    description="신용 융자·대주 잔고. 매매일로 매겨지고 잔고는 결제일(+2영업일) 기준",
    fields=(
        Field("deal_date", "date", DATE),
        Field("stlm_date", "settlement_date", DATE),
        Field("stck_prpr", "close"),
        Field("acml_vol", "volume"),
        Field("whol_loan_new_stcn", "margin_new_qty"),
        Field("whol_loan_rdmp_stcn", "margin_redeem_qty"),
        Field("whol_loan_rmnd_stcn", "margin_balance_qty"),
        Field("whol_loan_new_amt", "margin_new_value"),
        Field("whol_loan_rdmp_amt", "margin_redeem_value"),
        Field("whol_loan_rmnd_amt", "margin_balance_value"),
        Field("whol_loan_rmnd_rate", "margin_balance_ratio", FLOAT),
        Field("whol_loan_gvrt", "margin_offer_ratio", FLOAT),
        Field("whol_stln_new_stcn", "stock_loan_new_qty"),
        Field("whol_stln_rdmp_stcn", "stock_loan_redeem_qty"),
        Field("whol_stln_rmnd_stcn", "stock_loan_balance_qty"),
        Field("whol_stln_new_amt", "stock_loan_new_value"),
        Field("whol_stln_rdmp_amt", "stock_loan_redeem_value"),
        Field("whol_stln_rmnd_amt", "stock_loan_balance_value"),
        Field("whol_stln_rmnd_rate", "stock_loan_balance_ratio", FLOAT),
        Field("whol_stln_gvrt", "stock_loan_offer_ratio", FLOAT),
    ),
)


_LOAN_TRANS_FIELDS = (
    Field("bsop_date", "date", DATE),
    Field("stck_prpr", "close"),
    Field("acml_vol", "volume"),
    Field("new_stcn", "sbl_new_qty"),
    Field("rdmp_stcn", "sbl_redeem_qty"),
    Field("prdy_rmnd_vrss", "sbl_balance_change"),
    Field("rmnd_stcn", "sbl_balance_qty"),
    Field("rmnd_amt", "sbl_balance_value"),
)


# 2026-08-06 실호출로 확인: 이 데이터셋만 원주가로 온다. 다른 데이터셋의 종가는
# price_daily(수정주가)와 876일 전부 일치하지만, 여기 close는 배당·분배락 보정이
# 없어서 락 이후 며칠만 일치한다. 종가가 필요하면 price_daily를 쓴다.
LOAN_TRANS = Dataset(
    name="loan_trans_daily",
    endpoint="daily_loan_trans",
    output="output1",
    available_after_days=1,
    available_at_hour=9,
    description="종목별 대차거래 체결·상환·잔고. 공매도 선행 지표. close는 원주가",
    fields=_LOAN_TRANS_FIELDS,
)


# 시장 전체 대차잔고는 지수 레벨 위험 신호로 따로 쓸 값이 있어 데이터셋을 나눴다.
MARKET_LOAN_TRANS = Dataset(
    name="market_loan_trans_daily",
    endpoint="daily_loan_trans",
    output="output1",
    available_after_days=1,
    available_at_hour=9,
    pads_ticker=False,
    description="시장 전체 대차거래 잔고. close 열에는 종가가 아니라 지수가 들어온다",
    fields=_LOAN_TRANS_FIELDS,
)


PROGRAM_TRADE = Dataset(
    name="program_trade_daily",
    endpoint="program_trade_by_stock_daily",
    output="output",
    available_at_hour=18,
    description="프로그램매매 일별추이",
    fields=(
        Field("stck_bsop_date", "date", DATE),
        Field("stck_clpr", "close"),
        Field("acml_vol", "volume"),
        Field("acml_tr_pbmn", "trading_value"),
        Field("whol_smtn_seln_vol", "program_sell_qty"),
        Field("whol_smtn_shnu_vol", "program_buy_qty"),
        Field("whol_smtn_ntby_qty", "program_net_qty"),
        Field("whol_smtn_seln_tr_pbmn", "program_sell_value"),
        Field("whol_smtn_shnu_tr_pbmn", "program_buy_value"),
        Field("whol_smtn_ntby_tr_pbmn", "program_net_value"),
        # 주의: 아래 증감 두 열은 페이지 첫 행에서 전일 값을 못 찾아 순매수 자체를
        # 그대로 돌려준다(2026-08-06 확인, 144일 중 4일). 필요하면 저장된 순매수
        # 시계열에서 직접 차분해 쓰고 이 열은 믿지 않는다.
        Field("whol_ntby_vol_icdc", "program_net_qty_change"),
        Field("whol_ntby_tr_pbmn_icdc2", "program_net_value_change"),
    ),
)


DATASETS: dict[str, Dataset] = {
    dataset.name: dataset
    for dataset in (
        INVESTOR_FLOW,
        MARKET_INVESTOR_FLOW,
        MARKET_PRICE_DAILY,
        PRICE_DAILY,
        SHORT_SALE,
        CREDIT_BALANCE,
        LOAN_TRANS,
        MARKET_LOAN_TRANS,
        PROGRAM_TRADE,
    )
}


def get_dataset(name: str) -> Dataset:
    try:
        return DATASETS[name]
    except KeyError:
        known = ", ".join(sorted(DATASETS))
        raise KeyError(f"알 수 없는 데이터셋 {name}. 사용 가능: {known}") from None
