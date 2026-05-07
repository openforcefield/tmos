"""Integration tests for sanitize_complex against CCD benchmark structures.

Each entry in ccd_tmbenchmarks.json defines a CCD-derived transition metal
complex with reference values for:

- ``charge``           — net molecular charge of the complex
- ``oxidation_state``  — metal oxidation state
- ``n_x_lig``          — number of X-type ligand connections (CBC)
- ``n_l_lig``          — number of L-type ligand connections (CBC)
- ``positions``        — 3-D atomic coordinates
- ``symbols``          — element symbols in matching order

Tests are grouped by property:
- Charge consistency and X/L connector counts are regular assertions.
- Oxidation state is strict only for entries tagged with
    ``oxidation_state_confidence = \"high\"`` in the benchmark JSON.
    Non-high confidence oxidation-state checks are marked ``xfail``.

Note on scientific validity: the ``n_x_lig`` and ``n_l_lig`` reference values
are believed to be mostly correct but may warrant review for individual
entries.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tmos import build_rdmol
from tmos import tmos as tmos_module

# ---------------------------------------------------------------------------
# Load benchmark data at module level
# ---------------------------------------------------------------------------

_BENCHMARKS_PATH = Path(__file__).parent / "ccd_tmbenchmarks.json"
with _BENCHMARKS_PATH.open() as _fh:
    BENCHMARKS: dict[str, dict] = json.load(_fh)

_CCD_IDS = list(BENCHMARKS.keys())


# ---------------------------------------------------------------------------
# Module-scoped fixture: build mol and run sanitize_complex once per CCD entry
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", params=_CCD_IDS)
def ccd_result(request):
    """Return (ccd_id, expected_dict, results_list) for one benchmark entry.

    The fixture is module-scoped so ``sanitize_complex`` is called at most
    once per CCD ID across all tests in this module.
    """
    ccd_id: str = request.param
    entry: dict = BENCHMARKS[ccd_id]
    mol = build_rdmol.xyz_to_rdkit(
        entry["symbols"],
        np.array(entry["positions"]),
        ignore_scale=True,
        distance_tolerance=entry.get("distance_tolerance", 0.2),
    )
    results = tmos_module.sanitize_complex(
        mol,
        target_charge=entry["charge"],
        target_electron_count=entry.get("target_electron_count", 18),
        score_cutoff=None,
        n_results=5,
    )
    return ccd_id, entry, results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCCDSelectConsistency:
    """Consistency checks for CCD benchmark entries."""

    def test_complex_charge(self, ccd_result):
        """The best-scoring state's net complex charge matches the reference charge."""
        ccd_id, expected, results = ccd_result
        assert results, f"{ccd_id}: sanitize_complex returned no states"
        best = results[0]
        assert best.complex.charge == expected["charge"], (
            f"{ccd_id}: expected charge={expected['charge']}, "
            f"got charge={best.complex.charge}. "
            f"score={best.score}, {best.score_components.summary}"
        )

    def test_n_x_lig(self, ccd_result):
        """The best-scoring state's X-type connector count matches the reference."""
        ccd_id, expected, results = ccd_result
        assert results, f"{ccd_id}: sanitize_complex returned no states"
        best = results[0]
        assert best.ligands.number_Xtype_connectors == expected["n_x_lig"], (
            f"{ccd_id}: expected n_x_lig={expected['n_x_lig']}, "
            f"got number_Xtype_connectors={best.ligands.number_Xtype_connectors}. "
            f"score={best.score}, {best.ligands.summary}"
        )

    def test_n_l_lig(self, ccd_result):
        """The best-scoring state's L-type connector count matches the reference."""
        ccd_id, expected, results = ccd_result
        assert results, f"{ccd_id}: sanitize_complex returned no states"
        best = results[0]
        assert best.ligands.number_Ltype_connectors == expected["n_l_lig"], (
            f"{ccd_id}: expected n_l_lig={expected['n_l_lig']}, "
            f"got number_Ltype_connectors={best.ligands.number_Ltype_connectors}. "
            f"score={best.score}, {best.ligands.summary}"
        )

    def test_oxidation_state(self, ccd_result):
        """Assert oxidation state for high-confidence references only.

        Entries with non-high confidence are marked xfail so this test module
        can focus strict failures on CCD-grounded oxidation-state labels.
        """
        ccd_id, expected, results = ccd_result
        assert results, f"{ccd_id}: sanitize_complex returned no states"
        best = results[0]

        confidence = expected.get("oxidation_state_confidence", "high")
        if confidence != "high":
            basis = expected.get("oxidation_state_basis", "inferred reference value")
            pytest.xfail(
                f"{ccd_id}: oxidation_state_confidence={confidence}; basis={basis}"
            )

        assert best.metal.oxidation_state == expected["oxidation_state"], (
            f"{ccd_id}: expected oxidation_state={expected['oxidation_state']}, "
            f"got oxidation_state={best.metal.oxidation_state}. "
            f"score={best.score}, {best.score_components.summary}"
        )
