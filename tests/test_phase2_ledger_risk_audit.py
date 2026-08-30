import numpy as np
import pandas as pd
import pytest

from yenibot.automation.phase2_ledger_risk_audit import parse_mark_minute
from yenibot.phase2.contracts import DEFAULT_PHASE2_CONTRACT, CostScenario
from yenibot.phase2.ledger_risk_audit import (
    funding_price_scenarios,
    replay_frozen_ledger,
    validate_funding_grid,
    validate_frozen_ledger,
)

ZERO = CostScenario("zero", 0, 0, 0, 0, 0)


def fixture():
    times = pd.date_range("2023-01-01", periods=12, freq="h", tz="UTC")
    bars = pd.DataFrame(
        {
            "bar_open_time": times,
            "bar_close_time": times + pd.Timedelta(hours=1),
            "open": 100.0,
            "high": 102.0,
            "low": 97.0,
            "close": 100.0,
            "atr_14": 1.0,
            "fold": 2,
        }
    )
    bars.loc[1, "close"] = 98.0
    bars.loc[2, "close"] = 101.0
    ledger = pd.DataFrame(
        [
            {
                "decision_time": times[1],
                "entry_time": times[1],
                "exit_time": times[3],
                "entry_price": 100.0,
                "exit_price": 101.0,
                "atr": 1.0,
                "holding_hours": 2.0,
                "net_return": 0.01,
                "initial_stop_price": 95.0,
                "evaluation_fold": 2,
                "trade_status": "completed",
                "cost_scenario": "old",
            }
        ]
    )
    return bars, ledger


def replay(bars, ledger, **kwargs):
    return replay_frozen_ledger(
        bars,
        ledger,
        contract=DEFAULT_PHASE2_CONTRACT,
        scenario=kwargs.pop("scenario", ZERO),
        **kwargs,
    )


def test_hourly_marks_reveal_recovered_loss_without_changing_fills():
    bars, ledger = fixture()
    summary, curve, trades = replay(bars, ledger)
    assert summary["completed_trade_return"] == pytest.approx(0.01)
    assert summary["chained_independent_fold_return"] == pytest.approx(0.01)
    assert summary["hourly_contract_close_max_drawdown"] == pytest.approx(-0.02)
    assert summary["occupied_fraction_of_observed_hours"] == pytest.approx(2 / 12)
    assert trades.cost_scenario.iloc[0] == "zero"
    assert trades.entry_time.equals(ledger.entry_time)
    assert trades.exit_time.equals(ledger.exit_time)
    assert curve.exposure.iloc[-1] == 0


def test_terminal_mark_and_hypothetical_liquidation_are_explicit_and_different():
    bars, ledger = fixture()
    ledger.loc[0, "exit_time"] = bars.bar_close_time.iloc[-1]
    ledger.loc[0, "exit_price"] = 100.0
    ledger.loc[0, "holding_hours"] = 11.0
    ledger.loc[0, "trade_status"] = "censored"
    scenario = CostScenario("fees", 4, 4, 2, 2, 0)
    mark, _, mt = replay(bars, ledger, scenario=scenario)
    cash, _, ct = replay(bars, ledger, scenario=scenario, liquidate_terminal=True)
    assert mark["completed_trades"] == cash["completed_trades"] == 0
    assert mark["original_censored_positions"] == 1
    assert (
        cash["chained_independent_fold_return"]
        < mark["chained_independent_fold_return"]
    )
    assert mt.exit_fee_return.iloc[0] == 0
    assert ct.exit_fee_return.iloc[0] > 0
    assert ct.trade_status.iloc[0] == "hypothetical_terminal_liquidation"
    assert ledger.trade_status.iloc[0] == "censored"


def test_independent_fold_chain_and_no_trade_folds_are_not_dropped():
    bars, ledger = fixture()
    second = bars.copy()
    second[["bar_open_time", "bar_close_time"]] += pd.Timedelta(days=31)
    second["fold"] = 3
    third = second.copy()
    third[["bar_open_time", "bar_close_time"]] += pd.Timedelta(days=31)
    third["fold"] = 4
    trade = ledger.copy()
    trade[["decision_time", "entry_time", "exit_time"]] += pd.Timedelta(days=31)
    trade["evaluation_fold"] = 3
    summary, curve, _ = replay(
        pd.concat([bars, second, third], ignore_index=True),
        pd.concat([ledger, trade], ignore_index=True),
    )
    assert summary["chained_independent_fold_return"] == pytest.approx(1.01**2 - 1)
    assert summary["observed_fold_hours"] == 36
    assert curve.fold.unique().tolist() == [2, 3, 4]
    assert curve.loc[curve.fold.eq(4), "equity"].eq(1.01**2).all()


@pytest.mark.parametrize(
    "failure",
    ["duplicate", "overlap", "entry_price", "duration", "gap", "censor_before_end"],
)
def test_invalid_or_changed_frozen_ledgers_fail_closed(failure):
    bars, ledger = fixture()
    if failure == "duplicate":
        ledger = pd.concat([ledger, ledger], ignore_index=True)
    elif failure == "overlap":
        other = ledger.copy()
        other.loc[0, "decision_time"] += pd.Timedelta(hours=1)
        other.loc[0, "entry_time"] += pd.Timedelta(hours=1)
        other.loc[0, "holding_hours"] = 1.0
        ledger = pd.concat([ledger, other], ignore_index=True)
    elif failure == "entry_price":
        ledger.loc[0, "entry_price"] = 99.0
    elif failure == "duration":
        ledger.loc[0, "holding_hours"] = 12.0
    elif failure == "gap":
        bars = bars.drop(index=2)
    else:
        ledger.loc[0, "trade_status"] = "censored"
    with pytest.raises(ValueError):
        validate_frozen_ledger(ledger, bars)


def test_funding_grid_rejects_missing_rates_and_preserves_millisecond_timestamps():
    times = pd.date_range("2023-01-01", periods=4, freq="8h", tz="UTC")
    events = pd.DataFrame(
        {
            "timestamp": times + pd.Timedelta(milliseconds=5),
            "funding_rate": 0.0001,
            "mark_price": np.nan,
        }
    )
    checked = validate_funding_grid(events, start=times[0], end=times[-1])
    assert checked.timestamp.equals(events.timestamp)
    for bad in (
        events.drop(index=1),
        pd.concat([events, events.iloc[:1]]),
        events.assign(funding_rate=np.nan),
    ):
        with pytest.raises(ValueError, match="funding grid"):
            validate_funding_grid(bad, start=times[0], end=times[-1])


def test_funding_price_bounds_respect_sign_and_do_not_replace_known_marks():
    times = pd.date_range("2023-01-01", periods=3, freq="8h", tz="UTC")
    f = pd.DataFrame(
        {
            "timestamp": times,
            "funding_rate": [0.01, -0.01, 0.01],
            "mark_price": [np.nan, np.nan, 130.0],
        }
    )
    marks = {t: {"low": 90.0, "high": 110.0} for t in times[:2]}
    scenarios = funding_price_scenarios(f, marks)
    lo, hi = scenarios["favorable_charge"], scenarios["adverse_charge"]
    assert lo.mark_price.tolist() == [90.0, 110.0, 130.0]
    assert hi.mark_price.tolist() == [110.0, 90.0, 130.0]
    assert (lo.funding_rate * lo.mark_price <= hi.funding_rate * hi.mark_price).all()
    assert lo.price_basis.iloc[-1] == "reported_settlement_mark"
    with pytest.raises(ValueError, match="Missing funding mark interval"):
        funding_price_scenarios(f, {})


def test_event_funding_is_not_zero_filled_when_a_held_mark_is_missing():
    bars, ledger = fixture()
    ledger.loc[0, "decision_time"] = bars.bar_open_time.iloc[7]
    ledger.loc[0, "entry_time"] = bars.bar_open_time.iloc[7]
    ledger.loc[0, "exit_time"] = bars.bar_open_time.iloc[10]
    ledger.loc[0, "holding_hours"] = 3.0
    ledger.loc[0, "exit_price"] = 100.0
    events = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2023-01-01T00:00Z", "2023-01-01T08:00Z"]),
            "funding_rate": [0.001, 0.001],
            "mark_price": [np.nan, 120.0],
        }
    )
    summary, _, trades = replay(bars, ledger, funding_events=events)
    assert summary["chained_independent_fold_return"] == pytest.approx(-0.0012)
    assert trades.funding_return.iloc[0] == pytest.approx(0.0012)
    with pytest.raises(ValueError, match="finite"):
        replay(bars, ledger, funding_events=events.assign(mark_price=np.nan))
    with pytest.raises(ValueError, match="funding grid"):
        replay(bars, ledger, funding_events=events.iloc[:1])


def test_mark_minute_parser_enforces_exact_requested_interval_and_geometry():
    time = pd.Timestamp("2023-01-01T08:00Z")
    start = int(time.timestamp() * 1000)
    payload = [[start, "100", "110", "90", "101", "0", start + 59999]]
    assert parse_mark_minute(payload, time)["low"] == 90.0
    for bad in (
        [],
        payload * 2,
        [[start, "100", "99", "90", "101", "0", start + 59999]],
        [[start + 1, "100", "110", "90", "101", "0", start + 60000]],
    ):
        with pytest.raises(ValueError):
            parse_mark_minute(bad, time)
