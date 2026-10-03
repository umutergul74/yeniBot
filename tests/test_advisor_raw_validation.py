import numpy as np
import pandas as pd
import pytest

from yenibot.data.advisor_validation import validate_advisor_klines


def raw():
    return pd.DataFrame({"timestamp": pd.date_range("2022-01-01", periods=5, freq="h", tz="UTC"),
                         "open": 100., "high": 101., "low": 99., "close": 100.,
                         "volume": 10., "quote_volume": 1000., "num_trades": 20,
                         "taker_buy_base_vol": 5., "taker_buy_quote_vol": 500.})


def empty_bar():
    frame = raw()
    frame.loc[2, ["open", "high", "low", "close"]] = 100.
    frame.loc[2, ["volume", "quote_volume", "num_trades", "taker_buy_base_vol", "taker_buy_quote_vol"]] = 0
    return frame


def test_preserve_verified_empty_keeps_clock_and_source_prices():
    original = empty_bar()
    clean, audit = validate_advisor_klines(original, "1h", zero_activity_policy="preserve_verified_empty")
    pd.testing.assert_frame_equal(clean, original)
    assert audit["verified_empty_rows"] == 1
    assert audit["dropped_rows"] == audit["filled_rows"] == 0
    assert clean.timestamp.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all()
    with pytest.raises(ValueError, match="invalid raw bars"):
        validate_advisor_klines(original, "1h")


@pytest.mark.parametrize("column,value", [("close",0), ("volume",-1), ("volume",1),
                                          ("num_trades",1), ("high",101),
                                          ("taker_buy_base_vol",1), ("quote_volume",1),
                                          ("close",np.nan)])
def test_malformed_bar_never_passes_as_empty(column, value):
    frame = empty_bar()
    frame.loc[2, column] = value
    with pytest.raises(ValueError, match="examples="):
        validate_advisor_klines(frame, "1h", zero_activity_policy="preserve_verified_empty")


def test_empty_bar_must_agree_with_previous_close():
    frame = empty_bar()
    frame.loc[2, ["open", "high", "low", "close"]] = 102.
    with pytest.raises(ValueError, match="invalid raw bars"):
        validate_advisor_klines(frame, "1h", zero_activity_policy="preserve_verified_empty")


def test_missing_hour_not_hidden_by_data_cleaning():
    with pytest.raises(ValueError, match="contiguous"):
        validate_advisor_klines(raw().drop(index=2), "1h", zero_activity_policy="preserve_verified_empty")
