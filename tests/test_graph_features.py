import pytest

from itgnn.graph import FEATURE_COLS_B, user_features_extended
from test_smoke import _fake_logs


def test_extended_features():
    logon, device, file = _fake_logs()
    f = user_features_extended(logon, device, file, ["U1", "U2"])
    assert list(f.columns) == FEATURE_COLS_B
    assert f.loc["U1", "active_days"] == 1  # both logons on 2010-01-04
    assert f.loc["U2", "active_days"] == 1  # the Logoff row is ignored
    assert f.loc["U1", "n_logon_per_day"] == pytest.approx(2)
    assert f.loc["U1", "n_device_per_day"] == pytest.approx(1)
