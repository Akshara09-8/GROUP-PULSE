"""crossval.relabel: new targets per person, in time order, smoothed; other columns untouched."""

from grouppulse.perception.crossval import relabel
from grouppulse.perception.io import TrackRow
from grouppulse.perception.vfoa import Sample, fit


def test_relabel_uses_model_and_smooths():
    model = fit([Sample("Closeup1", -30 + d, 0, "P2") for d in range(-3, 4)] * 5
                + [Sample("Closeup1", 30 + d, 0, "P3") for d in range(-3, 4)] * 5, "idiap", min_count=5)
    yaws = [-30] * 6 + [30] + [-30] * 6 + [30] * 6          # one-frame blip, then a real switch
    rows = [TrackRow("IS0000a", k * 5, k / 5, "P1", True, y, 0.0, None, "away", 3.0, None, 0.0, False)
            for k, y in enumerate(yaws)]
    out = relabel(list(reversed(rows)), model)              # order of input rows must not matter
    by_t = {r.t: r for r in out}
    labels = [by_t[k / 5].gaze_target for k in range(len(yaws))]
    assert labels[6] == "P2"                                # blip smoothed
    assert labels[-1] == "P3"
    assert all(r.target_off is None and r.head_yaw is not None for r in out)
