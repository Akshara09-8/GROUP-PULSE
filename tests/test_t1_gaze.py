"""T1.5 logic and T1.8: gaze direction -> target with the seat layout."""

import pytest

from grouppulse.perception import angle_diff, bin8_name, gaze_target, load_layout, wrap180
from grouppulse.perception.geometry import direction_to


@pytest.fixture
def room():
    return load_layout("edinburgh")


def test_wrap_and_bins():
    assert wrap180(190) == -170
    assert wrap180(-180) == 180
    assert angle_diff(170, -170) == pytest.approx(20)
    assert bin8_name(0) == "front" and bin8_name(91) == "right" and bin8_name(-179) == "back"


def test_facing_closeup_means_looking_across_the_table(room):
    # P1's camera sits across the table; facing it = looking toward P3
    d = room.world_gaze("Closeup1", 0.0)
    assert gaze_target(room, "P1", d, 0.0).target == "P3"


def test_turning_toward_neighbour(room):
    want = direction_to(room.seats["P1"], room.seats["P2"])
    yaw = wrap180(want - (room.camera_dir["Closeup1"] + 180))
    d = room.world_gaze("Closeup1", yaw)
    res = gaze_target(room, "P1", d, 0.0)
    assert res.target == "P2" and res.off_angle == pytest.approx(0, abs=1e-6)


def test_screen_table_away_unknown(room):
    to_screen = direction_to(room.seats["P2"], room.objects["screen"])
    assert gaze_target(room, "P2", to_screen, 0.0).target == "screen"
    assert gaze_target(room, "P2", to_screen, -35.0).target == "table"
    assert gaze_target(room, "P1", -90.0, 0.0).target == "away"       # facing the wall behind them
    assert gaze_target(room, "P1", None, None).target == "unknown"    # no face found (T1.8)


def test_nearest_target_wins(room):
    # From P1, P3 is straight across (90 deg) and P4 is diagonal (~53 deg); 80 deg is closer to P3
    assert gaze_target(room, "P1", 80.0, 0.0, cone_deg=30).target == "P3"
