# -*- coding: utf-8 -*-
"""실제로 살 수 있는 물건으로 다시 잰다.

지금까지의 모든 수치는 KS11로 쟀다. KS11은 매매할 수 없다. 그리고 **배당이 빠진
가격지수**다. 핸드오프의 '남은 결함' 첫 줄이 이것이다. 이 스크립트는 세 가지를
분리해서 잰다.

1. 배당    — 가격형 ETF(069500)와 배당 재투자형(278530 TR)의 차이
2. 추적    — 지수와 ETF의 차이 (보수·추적오차)
3. 체결    — 지수 시가와 ETF 시가의 차이 (동시호가에 실제로 체결되는 값)

앞의 여덟 검정은 전부 "규칙을 얹어 수익을 올리려는" 시도였고 전부 실패했다.
이것은 다른 종류의 물음이다. **이미 벌고 있는데 장부에 안 적힌 것이 있는가.**

자료 제약. FDR의 ETF는 3,000행이 상한이라 2014-05-28부터다. TR형은 2017-11-21부터다.
31년 표본에는 못 미치지만 이 물음은 31년이 필요하지 않다. 배당수익률과 추적오차는
구조적 상수에 가깝기 때문이다.

    python etf_execution_study.py

로컬 전용 연구 스크립트다.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

COST_BPS = 10.0
ENTRY_WINDOW = EXIT_WINDOW = 20
CASH_RATE = 0.035

TICKERS = {
    "KS11 지수(현행 기준)": "KS11",
    "KS200 지수": "KS200",
    "KODEX200 가격형": "069500",
    "KODEX200 TR(배당재투자)": "278530",
    "KODEX코스피 가격형": "226490",
}


def load(code: str) -> pd.DataFrame:
    import FinanceDataReader as fdr

    frame = fdr.DataReader(code, "2000-01-01")
    frame.index = pd.to_datetime(frame.index)
    return frame.dropna(subset=["Open", "High", "Low", "Close"])


def donchian(high: pd.Series, low: pd.Series) -> pd.Series:
    ch_high = high.rolling(ENTRY_WINDOW).max().shift(1)
    ch_low = low.rolling(EXIT_WINDOW).min().shift(1)
    e = (high > ch_high).to_numpy()
    x = (low < ch_low).to_numpy()
    warm = ch_high.notna().to_numpy() & ch_low.notna().to_numpy()
    pos, state = np.zeros(len(high), dtype=np.int8), 0
    for i in range(len(high)):
        if not warm[i]:
            continue
        if state == 0 and e[i]:
            state = 1
        elif state == 1 and x[i]:
            state = 0
        pos[i] = state
    return pd.Series(pos, index=high.index, dtype=int)


def run(position: pd.Series, open_: pd.Series, cash_rate: float = 0.0):
    """신호는 어디서 오든 체결은 이 시가로 한다."""
    held = position.shift(1).fillna(0).astype(float)
    ret = (open_.shift(-1) / open_ - 1).fillna(0)
    cash = (1 + cash_rate) ** (1 / 252) - 1
    cost = held.diff().abs().fillna(0) * (COST_BPS / 10000)
    return held * ret + (1 - held) * cash - cost, held


def stats(daily: pd.Series, held: pd.Series, label: str) -> dict:
    eq = (1 + daily).cumprod()
    return {
        "구성": label,
        "CAGR%": round((eq.iloc[-1] ** (252 / len(daily)) - 1) * 100, 2),
        "MDD%": round((eq / eq.cummax() - 1).min() * 100, 2),
        "샤프": round(daily.mean() / daily.std() * np.sqrt(252), 3),
        "노출%": round(held.mean() * 100, 0),
        "배수": round(eq.iloc[-1], 2),
    }


def annualized(series: pd.Series) -> float:
    years = (series.index[-1] - series.index[0]).days / 365.25
    return ((series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1) * 100


def main() -> None:
    parser = argparse.ArgumentParser(description="ETF 실측으로 다시 재기")
    parser.add_argument("--outdir", default="output")
    args = parser.parse_args()

    data = {}
    for name, code in TICKERS.items():
        try:
            data[name] = load(code)
            d = data[name]
            print(f"  {name:24s} {d.index[0].date()}~{d.index[-1].date()} {len(d)}일")
        except Exception as exc:
            print(f"  {name} 수신 실패: {type(exc).__name__}")

    # ---- 1. 배당이 얼마인가 ------------------------------------------------
    print("\n" + "=" * 78)
    print("1. 배당 — 가격형과 배당재투자형(TR)의 차이")
    print("=" * 78)
    price = data["KODEX200 가격형"]["Close"]
    tr = data["KODEX200 TR(배당재투자)"]["Close"]
    common = price.index.intersection(tr.index)
    p, t = price.loc[common], tr.loc[common]
    print(f"  공통 구간 {common[0].date()} ~ {common[-1].date()} ({len(common)}일)")
    print(f"  가격형 069500 연환산 {annualized(p):+.2f}%")
    print(f"  TR형   278530 연환산 {annualized(t):+.2f}%")
    div = annualized(t) - annualized(p)
    print(f"  차이 = 배당 재투자 효과 **연 {div:+.2f}%p**")
    print("\n  가격형을 들고 있으면 이 배당은 사라지는 것이 아니라 분배금으로 현금에")
    print("  들어온다. 즉 실제 수익은 069500 가격수익 + 분배금이고, 지금까지의")
    print("  백테스트는 그 분배금을 통째로 빼고 계산해 왔다.")

    # ---- 1b. 한 종목의 자료 문제가 아닌지 교차검증 --------------------------
    # 069500 하나만 보고 결론을 내면 그 종목의 수정주가 처리 방식에 속을 수 있다.
    # 같은 지수를 추종하는 다른 운용사 ETF들과, 지수 선택이 섞이지 않도록
    # 코스피 자체를 추종하는 ETF까지 같이 본다.
    print("\n" + "-" * 78)
    print("교차검증 — 한 종목의 자료 처리 문제가 아닌가")
    print("-" * 78)
    ks200_c = data["KS200 지수"]["Close"]
    cross = []
    for code, label in (("069500", "KODEX200"), ("102110", "TIGER200"),
                        ("148020", "KBSTAR200"), ("069660", "KOSEF200")):
        try:
            e = load(code)["Close"]
        except Exception:
            continue
        idx = e.index.intersection(ks200_c.index)
        cross.append({"ETF": label, "ETF 연환산%": round(annualized(e.loc[idx]), 2),
                      "KS200 연환산%": round(annualized(ks200_c.loc[idx]), 2),
                      "차이%p": round(annualized(e.loc[idx]) - annualized(ks200_c.loc[idx]), 2)})
    ks11_c = data["KS11 지수(현행 기준)"]["Close"]
    kospi_etf = data["KODEX코스피 가격형"]["Close"]
    idx = kospi_etf.index.intersection(ks11_c.index)
    cross.append({"ETF": "KODEX코스피", "ETF 연환산%": round(annualized(kospi_etf.loc[idx]), 2),
                  "KS200 연환산%": round(annualized(ks11_c.loc[idx]), 2),
                  "차이%p": round(annualized(kospi_etf.loc[idx]) - annualized(ks11_c.loc[idx]), 2)})
    print(pd.DataFrame(cross).to_string(index=False))
    print("  마지막 줄은 코스피200이 아니라 코스피 자체를 추종한다. 지수 선택 차이가")
    print("  섞이지 않은 순수 배당 몫이고 위 네 줄과 크기가 같다.")

    e = data["KODEX200 가격형"]["Close"]
    idx = e.index.intersection(ks200_c.index)
    yr = (((1 + e.loc[idx].pct_change()).groupby(idx.year).prod() - 1) * 100
          - ((1 + ks200_c.loc[idx].pct_change()).groupby(idx.year).prod() - 1) * 100)
    print("\n  연도별 (KODEX200 - KS200)")
    line = [f"{y}:{v:+.2f}" for y, v in yr.items()]
    for i in range(0, len(line), 7):
        print("    " + "  ".join(line[i:i + 7]))
    print(f"  {len(yr)}년 중 양수 {int((yr > 0).sum())}년. 중앙 {yr.median():+.2f}%p.")
    print("  매년 같은 방향으로 같은 크기다. 우연이나 자료 오류의 모양이 아니다.")

    # ---- 2. 지수와 ETF의 차이 ----------------------------------------------
    print("\n" + "=" * 78)
    print("2. 추적 — 지수와 ETF가 얼마나 다른가")
    print("=" * 78)
    ks200 = data["KS200 지수"]["Close"]
    idx = price.index.intersection(ks200.index)
    print(f"  공통 구간 {idx[0].date()} ~ {idx[-1].date()}")
    print(f"  KS200 지수  연환산 {annualized(ks200.loc[idx]):+.2f}%")
    print(f"  069500 ETF 연환산 {annualized(price.loc[idx]):+.2f}%")
    print(f"  차이 {annualized(price.loc[idx]) - annualized(ks200.loc[idx]):+.2f}%p"
          "  (보수·추적오차·분배 유출)")
    r_i = ks200.loc[idx].pct_change()
    r_e = price.loc[idx].pct_change()
    print(f"  일간수익 상관 {r_i.corr(r_e):.4f}, "
          f"추적오차(차이 표준편차 연환산) {(r_e - r_i).std() * np.sqrt(252) * 100:.2f}%")

    # ---- 3. 체결 — 시가가 실제로 다른가 ------------------------------------
    print("\n" + "=" * 78)
    print("3. 체결 — 지수 시가와 ETF 시가")
    print("=" * 78)
    gap_i = (data["KS200 지수"]["Open"] / data["KS200 지수"]["Close"].shift(1) - 1) * 100
    gap_e = (data["KODEX200 가격형"]["Open"]
             / data["KODEX200 가격형"]["Close"].shift(1) - 1) * 100
    both = gap_i.index.intersection(gap_e.index)
    diff = (gap_e.loc[both] - gap_i.loc[both]).dropna()
    print(f"  ETF 시가갭 - 지수 시가갭: 평균 {diff.mean():+.3f}%p, "
          f"표준편차 {diff.std():.3f}%p")
    print(f"  절대값 중앙 {diff.abs().median():.3f}%p, 95분위 {diff.abs().quantile(.95):.3f}%p")
    print("  동시호가에서 ETF가 지수와 어긋나는 폭이다. 왕복비용 20bp와 견줄 크기다.")

    # ---- 4. 전략을 다시 돌린다 ---------------------------------------------
    print("\n" + "=" * 78)
    print("4. 전략을 실제 물건으로 다시 돌린다 (공통 구간)")
    print("=" * 78)
    etf = data["KODEX200 가격형"]
    ks11 = data["KS11 지수(현행 기준)"]
    span = etf.index.intersection(ks11.index).intersection(data["KS200 지수"].index)
    print(f"  구간 {span[0].date()} ~ {span[-1].date()} ({len(span)}일)\n")

    # 먼저 매수후보유로 세 물건의 차이를 분해한다. 전략 수치의 차이가 어디서
    # 오는지 모르면 개선으로 착각한다
    print("  매수후보유 연환산 (같은 구간)")
    for name in ("KS11 지수(현행 기준)", "KS200 지수", "KODEX200 가격형"):
        print(f"    {name:22s} {annualized(data[name]['Close'].loc[span]):+6.2f}%")
    bh_ks11 = annualized(ks11["Close"].loc[span])
    bh_ks200 = annualized(data["KS200 지수"]["Close"].loc[span])
    bh_etf = annualized(etf["Close"].loc[span])
    print(f"    KS200 - KS11 = {bh_ks200 - bh_ks11:+.2f}%p  (지수 선택 차이, 공짜 아님)")
    print(f"    ETF - KS200  = {bh_etf - bh_ks200:+.2f}%p  (배당이 ETF 가격에 남는다)\n")

    rows = []
    # (a) 신호도 체결도 KS11 — 지금까지 해 온 것
    pos = donchian(ks11["High"], ks11["Low"]).loc[span]
    d, h = run(pos, ks11["Open"].loc[span])
    rows.append(stats(d, h, "(a) KS11 신호 / KS11 체결  [현행 기준]"))
    # (a2) 체결만 KS200으로 — 지수 선택 차이만 분리한다
    d, h = run(pos, data["KS200 지수"]["Open"].loc[span])
    rows.append(stats(d, h, "(a2) KS11 신호 / KS200 체결  [지수만 교체]"))
    # (b) 신호는 KS11, 체결은 ETF — 지수를 보고 ETF를 산다. 실제 운용 모습이다
    d, h = run(pos, etf["Open"].loc[span])
    rows.append(stats(d, h, "(b) KS11 신호 / ETF 체결  [배당 포함]"))
    # (c) 신호도 체결도 ETF — ETF만 보고 판단한다
    pos_e = donchian(etf["High"], etf["Low"]).loc[span]
    d, h = run(pos_e, etf["Open"].loc[span])
    rows.append(stats(d, h, "(c) ETF 신호 / ETF 체결"))
    # (d) (b)에 분배금을 더한다
    d, h = run(pos, etf["Open"].loc[span])
    d_div = d + h * ((1 + div / 100) ** (1 / 252) - 1)
    rows.append(stats(d_div, h, f"(d) (b) + 분배금 연 {div:.2f}%"))
    # (e) (d)에 현금 파킹까지
    d, h = run(pos, etf["Open"].loc[span], CASH_RATE)
    d_all = d + h * ((1 + div / 100) ** (1 / 252) - 1)
    rows.append(stats(d_all, h, f"(e) (d) + 현금 {CASH_RATE:.1%} 파킹"))
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    # ---- 4b. 지수 선택을 배제한 깨끗한 배당 효과 ----------------------------
    # 위 표에는 코스피->코스피200 교체가 섞여 있다. 코스피 자체를 추종하는
    # ETF로 다시 재면 배당 몫만 남는다. 같은 지수, 같은 신호, 배당만 다르다.
    print("\n" + "-" * 78)
    print("지수 교체를 배제한 배당 효과 (코스피 추종 ETF)")
    print("-" * 78)
    kospi_etf_f = data["KODEX코스피 가격형"]
    span2 = kospi_etf_f.index.intersection(ks11.index)
    pos2 = donchian(ks11["High"], ks11["Low"]).loc[span2]
    clean = pd.DataFrame([
        stats(*run(pos2, ks11["Open"].loc[span2]), "KS11 신호 / KS11 체결"),
        stats(*run(pos2, kospi_etf_f["Open"].loc[span2]), "KS11 신호 / KODEX코스피 체결"),
    ])
    print(f"  구간 {span2[0].date()} ~ {span2[-1].date()} ({len(span2)}일)")
    print(clean.to_string(index=False))
    print(f"  ΔCAGR {clean.iloc[1]['CAGR%'] - clean.iloc[0]['CAGR%']:+.2f}%p, "
          f"ΔMDD {clean.iloc[1]['MDD%'] - clean.iloc[0]['MDD%']:+.2f}%p"
          "  — 규칙은 한 글자도 안 바뀌었다")

    base = table.iloc[0]
    final = table.iloc[-1]
    print(f"\n  현행 기준 대비 최종: CAGR {final['CAGR%'] - base['CAGR%']:+.2f}%p, "
          f"MDD {final['MDD%'] - base['MDD%']:+.2f}%p, "
          f"배수 {base['배수']} -> {final['배수']}")
    print("\n  (a)->(b)는 실제 물건으로 바꾼 대가, (b)->(d)와 (d)->(e)는")
    print("  규칙을 하나도 바꾸지 않고 장부에 안 적혀 있던 수익을 적은 것이다.")

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    table.to_csv(outdir / "etf_execution.csv", index=False, encoding="utf-8-sig")
    print(f"\n{outdir / 'etf_execution.csv'}")


if __name__ == "__main__":
    main()
