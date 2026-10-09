"""Task C: the leak-free threshold must depend only on validation-split controls."""
import numpy as np
import pandas as pd

from src.revision.run_task_c import leak_free_calls, threshold


def _frames(test_scale=1.0):
    rng = np.random.default_rng(0)
    ids = [str(i) for i in range(20)]
    test = pd.DataFrame({"repeat": [1] * 10 + [1] * 10, "fold": [1] * 10 + [2] * 10, "record_id": ids,
                         "y_true": [0, 1] * 10, "y_proba": rng.random(20) * test_scale})
    val = pd.DataFrame({"repeat": 1, "fold": [1] * 30 + [2] * 30, "record_id": [f"v{i}" for i in range(60)],
                        "y_true": [0, 0, 1] * 20, "y_proba": np.linspace(0, 1, 60)})
    # repeats 2 and 3 identical copies so every record has a call per repeat
    test = pd.concat([test.assign(repeat=r) for r in (1, 2, 3)])
    val = pd.concat([val.assign(repeat=r) for r in (1, 2, 3)])
    return ids, test, val


def test_threshold_is_quantile_of_validation_controls():
    ids, test, val = _frames()
    _, _, thr = leak_free_calls(test, val, ids, 0.10)
    v = val[(val.repeat == 1) & (val.fold == 1) & (val.y_true == 0)]["y_proba"].to_numpy()
    assert thr.loc[(thr.repeat == 1) & (thr.fold == 1), "threshold"].item() == threshold(v, 0.10)


def test_threshold_unchanged_by_test_scores_or_validation_cases():
    ids, test, val = _frames()
    _, _, a = leak_free_calls(test, val, ids, 0.20)
    ids, test2, val2 = _frames(test_scale=50.0)
    val2.loc[val2.y_true == 1, "y_proba"] = 999.0
    _, _, b = leak_free_calls(test2, val2, ids, 0.20)
    assert np.array_equal(a["threshold"].to_numpy(), b["threshold"].to_numpy())
