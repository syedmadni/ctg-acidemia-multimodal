"""A3b: post-birth fields must never enter a predictor matrix."""
import pytest

from src.revision.predictors import FORBIDDEN_PREDICTORS, INTRAPARTUM_EXTRA, MATERNAL_8, assert_allowed


def test_allowed_sets_have_no_forbidden_fields():
    assert not (set(MATERNAL_8) | set(INTRAPARTUM_EXTRA)) & FORBIDDEN_PREDICTORS


@pytest.mark.parametrize("field", sorted(FORBIDDEN_PREDICTORS))
def test_forbidden_field_rejected(field):
    with pytest.raises(ValueError):
        assert_allowed(MATERNAL_8 + [field])


def test_dummy_of_forbidden_rejected():
    with pytest.raises(ValueError):
        assert_allowed(["deliv_type__2"])


def test_every_revision_predictor_list_is_allowed():
    from src.revision import run_task_a
    for cols in run_task_a.PREDICTOR_BLOCKS.values():
        assert_allowed(cols)
