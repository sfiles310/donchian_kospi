"""위험 게이트를 워크포워드로 검증하고 돈치안과의 조합 방식을 비교한다.

`gate_study.py`는 전 구간을 한 번에 봤다. 게이트를 고른 눈과 성과를 잰 눈이 같아서
실전에서 그대로 나온다는 보장이 없다. 여기서는 학습 구간에서 게이트를 고르고,
그 뒤 구간에서만 성과를 잰다. 고르는 절차 자체를 검증하는 것이다.

같이 답하는 질문: 게이트를 돈치안에 붙일 것인가, 따로 쓸 것인가.
세 가지를 같은 구간에서 비교한다.

- 돈치안 단독
- 게이트 단독 (돈치안을 무시하고 위험 신호가 적을 때만 보유)
- 돈치안 + 게이트

기준은 수익이 아니라 MDD와 CVaR다. 수익을 얼마나 내주고 손실을 얼마나 줄이는지 본다.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from gate_study import (
    apply_costs,
    build_gates,
    build_returns,
    bootstrap_difference,
    _cvar,
    cvar,
    donchian_position,
    load_index,
)
from kis import PanelStore

WARMUP = 250          # 게이트의 250일 롤링 분위가 자리 잡는 데 필요한 기간
MIN_TRAIN = 250       # 게이트를 고르기 위한 최소 학습 일수
TEST_WINDOW = 84      # 검증 창. 약 4개월
SELECT_P = 0.10       # 학습 구간에서 게이트를 채택할 p 기준
SELECT_DRAWS = 600    # 학습 단계 부트스트랩은 가볍게
ROUND_TRIP_COST_BPS = 25.0
RISKY_WEIGHT = 0.5


def make_folds(length: int) -> list[tuple[int, int, int]]:
    """(학습 시작, 학습 끝=검증 시작, 검증 끝). 학습 구간은 앞으로 늘려 간다."""
    folds = []
    start_test = WARMUP + MIN_TRAIN
    while start_test < length:
        end_test = min(start_test + TEST_WINDOW, length)
        if end_test - start_test < TEST_WINDOW // 2:
            break
        folds.append((WARMUP, start_test, end_test))
        start_test = end_test
    return folds


def select_gates(
    returns: pd.Series, gates: pd.DataFrame, active: pd.Series, window: slice
) -> list[str]:
    """학습 구간에서 꼬리 위험을 실제로 키우는 게이트만 고른다."""
    held = returns[window][active[window]]
    chosen = []
    for name in gates.columns:
        flag = gates[name][window][active[window]]
        if flag.sum() < 10 or (~flag).sum() < 10:
            continue
        observed, _, _, p_value = bootstrap_difference(
            held, flag, _cvar, draws=SELECT_DRAWS
        )
        if not math.isnan(observed) and observed < 0 and p_value < SELECT_P:
            chosen.append(name)
    return chosen


def gate_weight(
    gates: pd.DataFrame, chosen: list[str], base: pd.Series, risky_weight: float
) -> pd.Series:
    """채택된 게이트가 충분히 켜지면 비중을 줄인다."""
    if not chosen:
        return base
    count = gates[chosen].sum(axis=1)
    threshold = min(2, len(chosen))
    return base.where(count < threshold, base * risky_weight)


def summarize(returns: pd.Series, weight: pd.Series, name: str, cost_bps: float) -> dict:
    net = apply_costs(weight, returns, cost_bps).fillna(0.0) / 100
    equity = (1 + net).cumprod()
    drawdown = equity / equity.cummax() - 1
    years = max(len(net) / 252, 1e-9)

    underwater = longest = 0
    for value in drawdown:
        underwater = underwater + 1 if value < -1e-12 else 0
        longest = max(longest, underwater)

    volatility = net.std()
    return {
        "전략": name,
        "누적수익%": (equity.iloc[-1] - 1) * 100,
        "연수익%": (equity.iloc[-1] ** (1 / years) - 1) * 100,
        "MDD%": drawdown.min() * 100,
        "최장 손실구간(일)": longest,
        "CVaR5%": cvar(net * 100),
        "손실일 비율": (net < 0).mean(),
        "샤프": net.mean() / volatility * math.sqrt(252) if volatility else math.nan,
        "평균 노출": weight.mean(),
    }


def diagnose_selection(
    frame: pd.DataFrame,
    returns: pd.Series,
    gates: pd.DataFrame,
    active: pd.Series,
    folds: list[tuple[int, int, int]],
) -> pd.DataFrame:
    """왜 채택되지 않았는지 보여준다. 부호가 뒤집히는지, 표본이 모자란지 구분한다."""
    rows = []
    for number, (train_start, train_end, _) in enumerate(folds, start=1):
        window = slice(train_start, train_end)
        held = returns[window][active[window]]
        for name in gates.columns:
            flag = gates[name][window][active[window]]
            observed, _, _, p_value = bootstrap_difference(
                held, flag, _cvar, draws=SELECT_DRAWS
            )
            rows.append(
                {
                    "구간": number,
                    "학습 보유일": len(held),
                    "게이트": name,
                    "위험일": int(flag.sum()),
                    "CVaR 하위5% 실표본": max(1, int(flag.sum() * 0.05)),
                    "CVaR차": observed,
                    "p": p_value,
                }
            )
    return pd.DataFrame(rows)


def fixed_rule_weight(
    gates: pd.DataFrame, names: list[str], base: pd.Series, risky_weight: float, threshold: int
) -> pd.Series:
    """게이트를 고르지 않고 사전 지정한 것을 그대로 쓴다. 학습이 필요 없다."""
    count = gates[names].sum(axis=1)
    return base.where(count < threshold, base * risky_weight)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="위험 게이트 워크포워드 검증")
    parser.add_argument("--index", default="0001")
    parser.add_argument("--db", type=Path, default=Path("data/kis_panel.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--round-trip-cost-bps", type=float, default=ROUND_TRIP_COST_BPS)
    parser.add_argument("--risky-weight", type=float, default=RISKY_WEIGHT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = PanelStore(args.db)
    frame = load_index(store, args.index, None)

    position = donchian_position(frame)
    gates = build_gates(frame)
    returns = build_returns(frame)

    active = position.shift(1).fillna(0).astype(bool) & returns.notna()
    gates_lagged = gates.shift(1).astype("boolean").fillna(False).astype(bool)

    folds = make_folds(len(frame))
    if not folds:
        raise SystemExit("표본이 짧아 워크포워드 구간을 만들 수 없습니다.")

    print(f"지수 {args.index}  {frame['date'].min()} ~ {frame['date'].max()}  {len(frame)}일")
    print(f"준비 {WARMUP}일 + 최소 학습 {MIN_TRAIN}일, 검증 창 {TEST_WINDOW}일, {len(folds)}개 구간\n")

    print("=" * 84)
    print("1. 구간별 게이트 채택 결과 (학습 구간에서 고르고 다음 구간에서만 씀)")
    print("=" * 84)
    selections = []
    oos_index: list[int] = []
    oos_chosen: dict[int, list[str]] = {}
    for number, (train_start, train_end, test_end) in enumerate(folds, start=1):
        train = slice(train_start, train_end)
        chosen = select_gates(returns, gates_lagged, active, train)
        selections.append(
            {
                "구간": number,
                "학습": f"{frame['date'].iloc[train_start]} ~ {frame['date'].iloc[train_end - 1]}",
                "검증": f"{frame['date'].iloc[train_end]} ~ {frame['date'].iloc[test_end - 1]}",
                "채택 게이트": ", ".join(chosen) if chosen else "없음",
            }
        )
        for i in range(train_end, test_end):
            oos_index.append(i)
            oos_chosen[i] = chosen
    print(pd.DataFrame(selections).to_string(index=False))

    diagnosis = diagnose_selection(frame, returns, gates_lagged, active, folds)
    if not any(row["채택 게이트"] != "없음" for row in selections):
        print("\n  채택된 게이트가 없다. 아래에서 이유를 나눠 본다.")
        print(diagnosis.round(4).to_string(index=False))
        signs = diagnosis.groupby("게이트")["CVaR차"].apply(
            lambda s: f"{int((s < 0).sum())}/{len(s)}구간에서 음수"
        )
        print("\n  부호 안정성 (음수라야 게이트로 쓸 수 있다)")
        print(signs.to_string())
        print(
            "\n  CVaR5는 하위 5%만 쓰므로 위험일이 22일이면 실제로는 한두 날로 판정한다.\n"
            "  학습 구간이 짧아 검정력이 없고, 부호도 구간마다 뒤집힌다."
        )

    # 검증 구간만 이어 붙인다.
    oos = pd.Index(oos_index)
    oos_returns = returns.iloc[oos]
    oos_active = active.iloc[oos]
    oos_gates = gates_lagged.iloc[oos]

    # 구간마다 채택 게이트가 다르므로 비중을 하루씩 계산한다.
    combined = []
    gate_only = []
    for i in oos:
        chosen = oos_chosen[i]
        risky = (
            sum(bool(gates_lagged[name].iloc[i]) for name in chosen) >= min(2, len(chosen))
            if chosen
            else False
        )
        base = 1.0 if active.iloc[i] else 0.0
        combined.append(base * (args.risky_weight if risky else 1.0))
        gate_only.append(args.risky_weight if risky else 1.0)
    combined_weight = pd.Series(combined, index=oos)
    gate_only_weight = pd.Series(gate_only, index=oos)
    donchian_weight = oos_active.astype(float)
    hold_weight = pd.Series(1.0, index=oos)

    print("\n" + "=" * 84)
    print("2. 검증 구간 성과 (앞 구간에서 고른 게이트만 사용, 표본 밖)")
    print("=" * 84)
    print(f"검증 일수 {len(oos)}일  {frame['date'].iloc[oos[0]]} ~ {frame['date'].iloc[oos[-1]]}")
    rows = [
        summarize(oos_returns, hold_weight, "지수 보유 (기준)", args.round_trip_cost_bps),
        summarize(oos_returns, donchian_weight, "돈치안 단독", args.round_trip_cost_bps),
        summarize(oos_returns, gate_only_weight, "게이트 단독", args.round_trip_cost_bps),
        summarize(oos_returns, combined_weight, "돈치안 + 게이트", args.round_trip_cost_bps),
    ]
    result = pd.DataFrame(rows)
    print(result.round(3).to_string(index=False))

    print("\n" + "=" * 84)
    print("3. 돈치안 대비 교환 관계")
    print("=" * 84)
    base = result[result["전략"] == "돈치안 단독"].iloc[0]
    trade = []
    for _, row in result.iterrows():
        if row["전략"] == "돈치안 단독":
            continue
        trade.append(
            {
                "전략": row["전략"],
                "연수익 차%p": row["연수익%"] - base["연수익%"],
                "MDD 개선%p": row["MDD%"] - base["MDD%"],
                "MDD 개선율": (row["MDD%"] - base["MDD%"]) / abs(base["MDD%"]),
                "CVaR 개선율": (row["CVaR5%"] - base["CVaR5%"]) / abs(base["CVaR5%"]),
                "샤프 차": row["샤프"] - base["샤프"],
                "노출 차": row["평균 노출"] - base["평균 노출"],
            }
        )
    print(pd.DataFrame(trade).round(3).to_string(index=False))
    print("  MDD 개선%p가 양수면 낙폭이 얕아진 것이다.")

    print("\n" + "=" * 84)
    print("4. 구간별 검증 성과 (안정적인지 확인)")
    print("=" * 84)
    per_fold = []
    cursor = 0
    for number, (_, train_end, test_end) in enumerate(folds, start=1):
        size = test_end - train_end
        piece = oos[cursor : cursor + size]
        cursor += size
        if len(piece) < 10:
            continue
        segment_returns = returns.iloc[piece]
        row = {
            "구간": number,
            "기간": f"{frame['date'].iloc[piece[0]]} ~ {frame['date'].iloc[piece[-1]]}",
        }
        for label, weight in [
            ("돈치안", active.iloc[piece].astype(float)),
            ("게이트단독", gate_only_weight.loc[piece]),
            ("결합", combined_weight.loc[piece]),
        ]:
            net = apply_costs(weight, segment_returns, args.round_trip_cost_bps).fillna(0.0) / 100
            equity = (1 + net).cumprod()
            row[f"{label} 수익%"] = (equity.iloc[-1] - 1) * 100
            row[f"{label} MDD%"] = (equity / equity.cummax() - 1).min() * 100
        per_fold.append(row)
    folds_table = pd.DataFrame(per_fold)
    print(folds_table.round(2).to_string(index=False))

    print("\n" + "=" * 84)
    print("5. 위험일 비중을 바꾸면 (결합 전략)")
    print("=" * 84)
    sensitivity = []
    for weight_value in [0.0, 0.25, 0.5, 0.75, 1.0]:
        values = []
        for i in oos:
            chosen = oos_chosen[i]
            risky = (
                sum(bool(gates_lagged[name].iloc[i]) for name in chosen) >= min(2, len(chosen))
                if chosen
                else False
            )
            base_weight = 1.0 if active.iloc[i] else 0.0
            values.append(base_weight * (weight_value if risky else 1.0))
        sensitivity.append(
            summarize(
                oos_returns,
                pd.Series(values, index=oos),
                f"위험일 비중 {weight_value:.0%}",
                args.round_trip_cost_bps,
            )
        )
    print(pd.DataFrame(sensitivity).round(3).to_string(index=False))

    print("\n" + "=" * 84)
    print("6. 고정 규칙 비교 (게이트를 고르지 않고 사전 지정한 것을 그대로 사용)")
    print("=" * 84)
    print("  게이트 선택에는 검정력이 없지만, 규칙 자체는 학습이 필요 없다.")
    print("  임계값이 모두 후행 250일 분위라서 준비기간 이후는 계산상 표본 밖이다.\n")
    span = slice(WARMUP, len(frame))
    span_returns = returns[span]
    span_active = active[span]
    span_gates = gates_lagged[span]
    base_weight = span_active.astype(float)
    all_four = list(gates.columns)
    three = [name for name in all_four if name != "대차잔고 급증"]

    print(f"기간 {frame['date'].iloc[WARMUP]} ~ {frame['date'].iloc[-1]}  {len(span_returns)}일")
    fixed_rows = [
        summarize(span_returns, pd.Series(1.0, index=span_returns.index),
                  "지수 보유 (기준)", args.round_trip_cost_bps),
        summarize(span_returns, base_weight, "돈치안 단독", args.round_trip_cost_bps),
        summarize(
            span_returns,
            fixed_rule_weight(span_gates, three, pd.Series(1.0, index=span_returns.index),
                              args.risky_weight, 2),
            "게이트 단독 (돈치안 미사용)",
            args.round_trip_cost_bps,
        ),
        summarize(
            span_returns,
            fixed_rule_weight(span_gates, all_four, base_weight, args.risky_weight, 2),
            "돈치안 + 게이트 4개",
            args.round_trip_cost_bps,
        ),
        summarize(
            span_returns,
            fixed_rule_weight(span_gates, three, base_weight, args.risky_weight, 2),
            "돈치안 + 게이트 3개",
            args.round_trip_cost_bps,
        ),
    ]
    fixed_table = pd.DataFrame(fixed_rows)
    print(fixed_table.round(3).to_string(index=False))

    print("\n  연도별 안정성 (돈치안 단독 대비 결합 3개)")
    yearly = []
    combined_fixed = fixed_rule_weight(span_gates, three, base_weight, args.risky_weight, 2)
    dates = frame["date"][span]
    for year, idx in dates.groupby(dates.str[:4]).groups.items():
        piece = pd.Index(idx)
        if len(piece) < 40:
            continue
        row = {"연도": year, "일수": len(piece)}
        for label, weight in [("돈치안", base_weight.loc[piece]),
                              ("결합", combined_fixed.loc[piece])]:
            net = apply_costs(weight, returns.loc[piece], args.round_trip_cost_bps).fillna(0.0) / 100
            equity = (1 + net).cumprod()
            row[f"{label} 수익%"] = (equity.iloc[-1] - 1) * 100
            row[f"{label} MDD%"] = (equity / equity.cummax() - 1).min() * 100
        row["수익 차%p"] = row["결합 수익%"] - row["돈치안 수익%"]
        row["MDD 차%p"] = row["결합 MDD%"] - row["돈치안 MDD%"]
        yearly.append(row)
    yearly_table = pd.DataFrame(yearly)
    print(yearly_table.round(2).to_string(index=False))

    args.output.mkdir(parents=True, exist_ok=True)
    fixed_table.to_csv(
        args.output / "gate_walkforward_fixed.csv", index=False, encoding="utf-8-sig"
    )
    yearly_table.to_csv(
        args.output / "gate_walkforward_yearly.csv", index=False, encoding="utf-8-sig"
    )
    diagnosis.to_csv(
        args.output / "gate_walkforward_diagnosis.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(selections).to_csv(
        args.output / "gate_walkforward_selection.csv", index=False, encoding="utf-8-sig"
    )
    result.to_csv(
        args.output / "gate_walkforward_summary.csv", index=False, encoding="utf-8-sig"
    )
    folds_table.to_csv(
        args.output / "gate_walkforward_folds.csv", index=False, encoding="utf-8-sig"
    )
    print(f"\n표는 {args.output.resolve()}에 저장했습니다.")
    print(f"검증 구간이 {len(oos)}일뿐이라 결론을 확정으로 다루면 안 됩니다.")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
