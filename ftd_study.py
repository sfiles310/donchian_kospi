"""윌리엄 오닐의 FTD(추격매수일) 규칙을 코스피에서 검정한다.

돈치안의 강점은 청산이고 약점은 재진입이다. 12.6년 표본에서 저점 이후 재진입까지
중앙 11일이 걸렸고 그동안 저점 대비 중앙 5.1%가 지나갔다. 2020년 코로나 때는
16일이 걸려 29.3%를 놓쳤다. FTD는 저점 후 4~10일에 판정하므로 이 구멍을 겨냥한다.

원문(네이버 카페 글, 2026-08-05)의 규칙은 이렇다.

1. 지수가 저점을 찍음 -> 랠리 시도 시작
2. 3일 관망, 4일차부터 판정
3. 조건 (1) 전일 대비 상승률이 기준 이상. 원래 1.2%이나 글쓴이는 2.0%를 썼다.
   조건 (2) 거래량이 50일 이동평균보다 많을 것
4. 4~10일 안에 충족하면 FTD 확정 = 상승 추세 전환

그대로 쓰면 안 되는 곳이 둘 있다.

- "저점"은 사후에만 안다. 실시간으로는 오늘이 저점인지 모른다. 그래서 여기서는
  조정 국면에서 저점이 갱신되면 관망일을 0으로 되돌리는 방식으로 인과적으로 다시 짰다.
  원문과 판정일이 다를 수 있다.
- 임계값 2.0%는 근거 없이 조정된 값이다. 1.2%와 2.0%를 모두 검정하고 민감도를 본다.

신호는 지수로 내고 수익은 실제 매매 대상인 ETF로 잰다.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from etf_replication_study import open_to_open, summarize
from gate_study import apply_costs, donchian_position
from kis import PanelStore, get_dataset

CORRECTION_LOOKBACK = 250   # 고점을 재는 기간
CORRECTION_DROP = 0.10      # 고점 대비 이만큼 빠지면 조정 국면으로 본다
VOLUME_MA = 50
WAIT_DAYS = 3               # 오닐의 관망 기간
FIRST_TEST_DAY = 4
LAST_TEST_DAY = 10
DEFAULT_COST_BPS = 10.0


def load_market(store: PanelStore, index_code: str, ticker: str) -> pd.DataFrame:
    price = store.read(get_dataset("market_price_daily"), tickers=[index_code])
    if price.empty:
        raise SystemExit("지수 일봉·거래량이 없습니다. market_price_daily를 먼저 수집하세요.")
    price = price[
        ["date", "index_open", "index_high", "index_low", "index_close", "volume"]
    ].sort_values("date")

    etf = store.read(get_dataset("price_daily"), tickers=[ticker])
    if etf.empty:
        raise SystemExit(f"{ticker} 가격 자료가 없습니다.")
    etf = etf[["date", "open"]].rename(columns={"open": "etf_open"})
    return price.merge(etf, on="date", how="inner").reset_index(drop=True)


def rally_day_counter(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """조정 국면에서 저점이 갱신되면 0으로 되돌리는 관망일 카운터.

    오늘까지의 정보만 쓴다. 저점이 나중에 깨지면 그날 카운터가 다시 0이 되므로,
    사후에 저점을 알고 세는 것과 달리 실시간으로 계산할 수 있다.
    """
    peak = frame["index_close"].rolling(CORRECTION_LOOKBACK, min_periods=60).max()
    in_correction = frame["index_close"] < peak * (1 - CORRECTION_DROP)

    counter = np.zeros(len(frame), dtype=int)
    low_mark = np.full(len(frame), np.nan)
    running_low = math.inf
    day = 0
    for i in range(len(frame)):
        if not bool(in_correction.iloc[i]):
            running_low = math.inf
            day = 0
            counter[i] = 0
            continue
        low = float(frame["index_low"].iloc[i])
        if low <= running_low:
            # 저점 갱신. 랠리 시도가 무산된 것으로 보고 처음부터 다시 센다.
            running_low = low
            day = 1
        else:
            day += 1
        counter[i] = day
        low_mark[i] = running_low
    return pd.Series(counter, index=frame.index), pd.Series(low_mark, index=frame.index)


def ftd_signals(frame: pd.DataFrame, gain_threshold: float) -> pd.DataFrame:
    """FTD 확정일과 그 판단 근거를 표로 만든다."""
    day, low_mark = rally_day_counter(frame)
    gain = frame["index_close"].pct_change() * 100
    volume_ma = frame["volume"].rolling(VOLUME_MA).mean()

    in_window = (day >= FIRST_TEST_DAY) & (day <= LAST_TEST_DAY)
    gain_ok = gain >= gain_threshold
    volume_ok = frame["volume"] > volume_ma

    return pd.DataFrame(
        {
            "date": frame["date"],
            "rally_day": day,
            "rally_low": low_mark,
            "gain": gain,
            "volume": frame["volume"],
            "volume_ma": volume_ma,
            "in_window": in_window,
            "gain_ok": gain_ok,
            "volume_ok": volume_ok,
            "ftd": in_window & gain_ok & volume_ok,
        }
    )


def position_from_entries(
    entries: pd.Series, exits: pd.Series, warmup: pd.Series
) -> pd.Series:
    """진입 신호와 청산 신호로 보유 상태를 만든다."""
    position = 0
    values = []
    for i in range(len(entries)):
        if not bool(warmup.iloc[i]):
            values.append(0)
            continue
        if position == 0 and bool(entries.iloc[i]):
            position = 1
        elif position == 1 and bool(exits.iloc[i]):
            position = 0
        values.append(position)
    return pd.Series(values, index=entries.index, dtype=int)


def donchian_parts(frame: pd.DataFrame, window: int = 20) -> tuple[pd.Series, pd.Series, pd.Series]:
    high, low = frame["index_high"], frame["index_low"]
    channel_high = high.rolling(window).max().shift(1)
    channel_low = low.rolling(window).min().shift(1)
    return high > channel_high, low < channel_low, channel_high.notna()


def evaluate_false_signals(frame: pd.DataFrame, table: pd.DataFrame, horizon: int = 20) -> dict:
    """FTD 이후 랠리 저점이 깨지면 실패로 본다. 오닐 본인의 판정 기준이다."""
    hits = table.index[table["ftd"]]
    failures = 0
    forward = []
    for i in hits:
        low = table["rally_low"].iloc[i]
        window = frame["index_low"].iloc[i + 1 : i + 1 + horizon]
        if len(window) and window.min() < low:
            failures += 1
        ahead = frame["index_close"].iloc[i + 1 : i + 1 + horizon]
        if len(ahead):
            forward.append((ahead.iloc[-1] / frame["index_close"].iloc[i] - 1) * 100)
    return {
        "FTD 횟수": len(hits),
        "위신호 비율": failures / len(hits) if len(hits) else math.nan,
        f"{horizon}일 후 평균 수익%": float(np.mean(forward)) if forward else math.nan,
        f"{horizon}일 후 승률": float(np.mean([x > 0 for x in forward])) if forward else math.nan,
    }


def entry_lag(frame: pd.DataFrame, entries: pd.Series, label: str) -> dict:
    """저점 대비 얼마나 늦게, 얼마나 비싸게 들어가는지."""
    day, low_mark = rally_day_counter(frame)
    lags, premiums = [], []
    for i in np.where(entries.to_numpy())[0]:
        if math.isnan(low_mark.iloc[i]):
            continue
        lags.append(int(day.iloc[i]))
        premiums.append((frame["index_close"].iloc[i] / low_mark.iloc[i] - 1) * 100)
    return {
        "진입 방식": label,
        "조정 국면 진입 횟수": len(lags),
        "저점 후 지연 중앙": float(np.median(lags)) if lags else math.nan,
        "저점 대비 진입가 중앙%": float(np.median(premiums)) if premiums else math.nan,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="오닐 FTD 규칙 검정")
    parser.add_argument("--index", default="0001")
    parser.add_argument("--ticker", default="148020")
    parser.add_argument("--db", type=Path, default=Path("data/kis_panel.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = PanelStore(args.db)
    frame = load_market(store, args.index, args.ticker)

    entry_signal, exit_signal, warmup = donchian_parts(frame)
    etf_returns = open_to_open(frame["etf_open"])
    usable = etf_returns.notna()

    print(f"지수 {args.index} 신호 -> ETF {args.ticker} 수익")
    print(f"{frame['date'].iloc[0]} ~ {frame['date'].iloc[-1]}  {len(frame)}일  "
          f"왕복 {args.cost_bps:.0f}bp")
    print(f"조정 정의: 최근 {CORRECTION_LOOKBACK}일 고점 대비 -{CORRECTION_DROP:.0%} 이하")
    print(f"판정 창: 랠리 {FIRST_TEST_DAY}~{LAST_TEST_DAY}일차, 거래량 {VOLUME_MA}일 이평 상회\n")

    print("=" * 84)
    print("1. FTD 발생 빈도와 위신호 비율")
    print("=" * 84)
    rows = []
    tables = {}
    for threshold in [1.0, 1.2, 1.5, 2.0, 2.5]:
        table = ftd_signals(frame, threshold)
        tables[threshold] = table
        row = {"상승 기준%": threshold}
        row.update(evaluate_false_signals(frame, table))
        rows.append(row)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("  위신호 = FTD 이후 20일 안에 랠리 저점을 다시 깬 경우 (오닐 본인의 기준)")

    print("\n" + "=" * 84)
    print("2. 조건별로 몇 번 걸렀나 (상승 2.0% 기준)")
    print("=" * 84)
    table = tables[2.0]
    window = table[table["in_window"]]
    print(f"  판정 창에 들어온 날 {len(window)}일")
    print(f"    상승 조건만 통과 {int(window['gain_ok'].sum())}일")
    print(f"    거래량 조건만 통과 {int(window['volume_ok'].sum())}일")
    print(f"    둘 다 통과 (FTD) {int(table['ftd'].sum())}일")
    both = window["gain_ok"] & ~window["volume_ok"]
    print(f"    상승은 됐는데 거래량에서 탈락 {int(both.sum())}일  "
          f"<- 2026-08-05가 이 경우다")

    print("\n" + "=" * 84)
    print("3. 저점 대비 진입 시점 비교")
    print("=" * 84)
    lag_rows = [entry_lag(frame, entry_signal & warmup, "돈치안 20일 돌파")]
    for threshold in [1.2, 2.0]:
        lag_rows.append(
            entry_lag(frame, tables[threshold]["ftd"], f"FTD {threshold}%")
        )
    print(pd.DataFrame(lag_rows).round(2).to_string(index=False))

    print("\n" + "=" * 84)
    print("4. 전략 비교 (진입만 다르게, 청산은 모두 돈치안 20일 하단)")
    print("=" * 84)
    donchian_position_series = position_from_entries(entry_signal, exit_signal, warmup)
    results = [
        summarize(etf_returns, pd.Series(1.0, index=frame.index),
                  f"ETF {args.ticker} 계속 보유", args.cost_bps),
        summarize(
            etf_returns,
            (donchian_position_series.shift(1).fillna(0).astype(bool) & usable).astype(float),
            "돈치안 진입 (현재 방식)",
            args.cost_bps,
        ),
    ]
    for threshold in [1.2, 2.0]:
        ftd = tables[threshold]["ftd"]
        only = position_from_entries(ftd, exit_signal, warmup)
        combined = position_from_entries(ftd | entry_signal, exit_signal, warmup)
        results.append(
            summarize(
                etf_returns,
                (only.shift(1).fillna(0).astype(bool) & usable).astype(float),
                f"FTD {threshold}% 진입만",
                args.cost_bps,
            )
        )
        results.append(
            summarize(
                etf_returns,
                (combined.shift(1).fillna(0).astype(bool) & usable).astype(float),
                f"돈치안 + FTD {threshold}% 조기진입",
                args.cost_bps,
            )
        )
    table_out = pd.DataFrame(results)
    print(table_out.round(3).to_string(index=False))

    print("\n" + "=" * 84)
    print("5. 최근 FTD 판정 이력 (마지막 10건, 2.0% 기준)")
    print("=" * 84)
    hits = tables[2.0][tables[2.0]["ftd"]].tail(10)
    print(hits[["date", "rally_day", "gain", "volume", "volume_ma"]].round(2).to_string(index=False))

    print("\n" + "=" * 84)
    print("6. 지금 상태 (마지막 15거래일)")
    print("=" * 84)
    recent = tables[2.0].tail(15).copy()
    recent["거래량/50일평균"] = (recent["volume"] / recent["volume_ma"]).round(3)
    print(recent[["date", "rally_day", "gain", "거래량/50일평균",
                  "in_window", "gain_ok", "volume_ok", "ftd"]].round(2).to_string(index=False))

    args.output.mkdir(parents=True, exist_ok=True)
    table_out.to_csv(args.output / "ftd_strategy.csv", index=False, encoding="utf-8-sig")
    tables[2.0].to_csv(args.output / "ftd_signals.csv", index=False, encoding="utf-8-sig")
    print(f"\n표는 {args.output.resolve()}에 저장했습니다.")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
