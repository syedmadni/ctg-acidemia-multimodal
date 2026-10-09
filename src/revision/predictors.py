"""Which header fields may enter a predictor matrix in the EJOG revision.

Frozen in results/revision/DEVIATIONS.md (A3b). Everything recorded at or after birth, and
the outcome itself, is FORBIDDEN as a predictor. Sex is known only after birth in many
settings, so it is used descriptively (Table 1) and never as a predictor.
"""
from __future__ import annotations

from typing import Iterable

# The 8 pre-specified maternal variables (configs/baseline.yaml), in the published order.
MATERNAL_8 = ["age", "gravidity", "parity", "gest_weeks", "diabetes", "hypertension",
              "preeclampsia", "induced"]

# Pre-declared extended intrapartum block (A3b): maternal 8 + these, all observable
# during labour. `presentation` is categorical and dummy-coded (reference = code 1).
INTRAPARTUM_EXTRA = ["pyrexia", "meconium", "presentation", "liq_praecox"]

ALLOWED_PREDICTORS = set(MATERNAL_8) | set(INTRAPARTUM_EXTRA)

FORBIDDEN_PREDICTORS = {
    "ph", "bdecf", "pco2", "be", "apgar1", "apgar5",
    "nicu_days", "seizures", "hie", "intubation", "main_diag", "other_diag",
    "deliv_type", "ck_kp", "no_progress", "i_stage", "ii_stage", "weight_g", "sex",
    "pos_ii_st", "sig2birth", "db_id", "rec_type",
    "label", "y",
}


def assert_allowed(columns: Iterable[str]) -> None:
    """Raise if any column is forbidden or not on the allow-list.

    Dummy columns are named `<field>__<level>` and are checked by their base field.
    """
    for col in columns:
        base = str(col).split("__")[0]
        if base in FORBIDDEN_PREDICTORS:
            raise ValueError(f"Forbidden (post-birth or outcome) field used as predictor: {col}")
        if base not in ALLOWED_PREDICTORS:
            raise ValueError(f"Field not on the predictor allow-list: {col}")
