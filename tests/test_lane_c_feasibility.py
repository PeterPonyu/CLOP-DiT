"""Structural tests for Lane C feasibility / data plan and blocker-to-action synthesis.

Scope (task-2, team clop-dit-revision-next-step-ex):
- Verify that the Lane C feasibility note codifies the priority-ordered data
  additions, the five-slice evaluation contract, and the retrain-budget
  blocker called out in `revision/README.md` and `revision/REVISION_LOG.md`.
- Verify that the blocker-to-action synthesis maps each identified blocker
  to a concrete next action with a decision gate.

Tests skip when the artifact file is absent (so the suite stays green
while the implementing worker is still drafting), but fail loudly if an
artifact exists but does not meet the documented contract.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
REVISION_DIR = REPO_ROOT / "revision"
LANE_C_DIR = REVISION_DIR / "experiments" / "lane_c_data"

LANE_C_README = LANE_C_DIR / "README.md"
LANE_C_RESULTS = LANE_C_DIR / "results.md"
GO_NO_GO_DECISION = LANE_C_DIR / "go_no_go_decision.md"
STAFFING_CHECKLIST = LANE_C_DIR / "conditional_go_staffing_checklist.md"
FEASIBILITY_CANDIDATES = [
    LANE_C_DIR / "feasibility_and_blockers.md",
    LANE_C_DIR / "feasibility_and_data_plan.md",
    LANE_C_DIR / "feasibility.md",
    LANE_C_DIR / "data_plan.md",
]
SYNTHESIS_CANDIDATES = [
    LANE_C_DIR / "feasibility_and_blockers.md",
    LANE_C_DIR / "blocker_to_action.md",
    LANE_C_DIR / "blockers_to_actions.md",
    REVISION_DIR / "REVIEWER_RESPONSE_SYNTHESIS.md",
    REVISION_DIR / "REVISION_LOG.md",
]

# Five reviewer-facing evaluation slices per revision/README.md and
# revision/experiments/lane_c_data/README.md. Each slice is expressed as a
# regex so we tolerate the small phrasing variations already present in the
# tracked Lane C docs ("rare-classification" vs "rare-cell classification").
FIVE_SLICES = [
    ("overall", r"overall"),
    ("low-abundance", r"low[- ]?abundance"),
    ("mouse", r"mouse"),
    ("strict-OOD", r"strict[- ]?ood"),
    ("rare-classification", r"rare[- ](cell[- ])?classification"),
]

# Three priority tiers per the Lane C status checklist.
PRIORITY_TIERS = [
    ("priority-1", r"mouse"),
    ("priority-2", r"(ood|out[- ]?of[- ]?distribution|held[- ]?out)"),
    ("priority-3", r"(heterogeneity|real_intra_cos|transitional|rare)"),
]


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Existing artifact sanity
# ---------------------------------------------------------------------------


class TestLaneCREADME:
    """The pre-existing Lane C README must describe the canonical scope."""

    @pytest.fixture
    def text(self) -> str:
        if not LANE_C_README.exists():
            pytest.skip(f"missing {LANE_C_README}")
        return _read(LANE_C_README).lower()

    def test_mentions_reviewer_comments(self, text: str) -> None:
        # R3.1 strict-OOD, R3.2 species stratification, R2.7 low abundance.
        # Accept either the fully qualified tag ("r3.1") or the bare
        # subsection number ("3.1") as used in the current README.
        for tag in ("3.1", "3.2"):
            assert tag in text, f"Lane C README missing reviewer tag {tag}"

    def test_mentions_five_slices(self, text: str) -> None:
        for name, pattern in FIVE_SLICES:
            assert re.search(pattern, text), (
                f"Lane C README missing required slice '{name}'"
            )

    def test_gate_on_a3(self, text: str) -> None:
        assert "a3" in text, "Lane C README must cite the A3 gate dependency"


class TestLaneCResults:
    """Results.md must at minimum flag Lane C as pending and document blockers."""

    @pytest.fixture
    def text(self) -> str:
        if not LANE_C_RESULTS.exists():
            pytest.skip(f"missing {LANE_C_RESULTS}")
        return _read(LANE_C_RESULTS).lower()

    def test_status_checklist_present(self, text: str) -> None:
        assert "[x] a3 gate cleared" in text or "a3 gate" in text

    def test_five_slices_referenced(self, text: str) -> None:
        # results.md enumerates the five slices in prose; accept either
        # the hyphenated form or the spelled-out phrasing.
        for name, pattern in FIVE_SLICES:
            assert re.search(pattern, text), (
                f"results.md missing required slice '{name}'"
            )

    def test_current_gate_text_is_post_a3(self, text: str) -> None:
        assert "current gates" in text, "results.md should describe the post-A3 gate state"
        assert "current gates:** a3 is already complete" in text, (
            "results.md should explicitly state that A3 is already complete"
        )
        assert "confirms the long-tail gap is material" not in text, (
            "results.md must not retain the stale pre-A3 gate wording"
        )

    def test_r27_is_descope_but_slice_retained(self, text: str) -> None:
        assert "r2.7 is already discharged" in text, (
            "results.md should say that R2.7 is already discharged outside Lane C"
        )
        assert "reporting slice" in text, (
            "results.md should clarify that low-abundance remains a reporting slice"
        )

    def test_b7_and_b6_are_committed(self, text: str) -> None:
        assert "b-7 decision committed" in text, (
            "results.md should record the committed B-7 decision state"
        )
        assert "regression stop-rule decision committed (b-6)" in text, (
            "results.md should record the committed B-6 state"
        )


class TestLaneCDecisionMemo:
    """The B-7/B-6 decision memo must convert planning into an executable gate."""

    @pytest.fixture
    def text(self) -> str:
        if not GO_NO_GO_DECISION.exists():
            pytest.skip(f"missing {GO_NO_GO_DECISION}")
        return _read(GO_NO_GO_DECISION).lower()

    def test_full_lane_is_no_go(self, text: str) -> None:
        assert "no-go for the full lane c plan" in text

    def test_priority_2_only_is_conditional_go(self, text: str) -> None:
        assert "conditional go" in text
        assert "priority-2" in text or "priority 2" in text
        assert "<= 18 engineer-days" in text

    def test_stop_rule_is_explicit(self, text: str) -> None:
        assert "> 0.01" in text and "> 0.05" in text, (
            "go/no-go memo must encode the B-6 numeric stop-rule thresholds"
        )

    def test_next_step_is_staffing_not_retrain(self, text: str) -> None:
        assert "do not start curation or retraining yet" in text
        assert "b-2" in text and "b-3" in text


class TestLaneCStaffingChecklist:
    @pytest.fixture
    def text(self) -> str:
        if not STAFFING_CHECKLIST.exists():
            pytest.skip(f"missing {STAFFING_CHECKLIST}")
        return _read(STAFFING_CHECKLIST).lower()

    def test_contains_engineer_day_gate(self, text: str) -> None:
        assert "<= 18 engineer-days" in text

    def test_contains_b2_b3_owner_requirements(self, text: str) -> None:
        assert "b-2 owner assigned" in text
        assert "b-3 owner assigned" in text

    def test_contains_binary_outcome(self, text: str) -> None:
        assert "conditional go approved" in text
        assert "no-go retained" in text


# ---------------------------------------------------------------------------
# Feasibility / data plan artifact (produced by worker-1)
# ---------------------------------------------------------------------------


class TestLaneCFeasibilityPlan:
    """Validate the feasibility / data plan document once worker-1 commits it."""

    @pytest.fixture
    def plan(self) -> tuple[Path, str]:
        p = _first_existing(FEASIBILITY_CANDIDATES)
        if p is None:
            pytest.skip(
                "No Lane C feasibility/data plan artifact found yet. "
                "Expected one of: "
                + ", ".join(str(c.relative_to(REPO_ROOT)) for c in FEASIBILITY_CANDIDATES)
            )
        return p, _read(p).lower()

    def test_priority_ordered_data_additions(self, plan: tuple[Path, str]) -> None:
        _, text = plan
        for label, pattern in PRIORITY_TIERS:
            assert label in text, f"feasibility plan missing tier label '{label}'"
            assert re.search(pattern, text), (
                f"feasibility plan missing content for tier '{label}' "
                f"(looking for regex '{pattern}')"
            )

    def test_narrowed_plan_keeps_five_slice_contract(self, plan: tuple[Path, str]) -> None:
        _, text = plan
        if "priority 2 only" not in text:
            pytest.skip("narrowed fallback plan not present")
        assert "five-slice contract" in text, (
            "narrowed fallback must explicitly preserve the five-slice contract"
        )
        assert '"heterogeneity" slices' not in text, (
            "narrowed fallback must not invent a non-contract 'heterogeneity' slice"
        )

    def test_five_slice_eval_contract(self, plan: tuple[Path, str]) -> None:
        _, text = plan
        # Accept either an in-line enumeration of all five slices OR an
        # explicit reference to the README's five-slice contract so the
        # plan does not silently drop the contract.
        explicit_reference = bool(
            re.search(r"five[- ]?slice", text)
            and re.search(r"readme", text)
        )
        if explicit_reference:
            return
        missing = [
            name for name, pattern in FIVE_SLICES if not re.search(pattern, text)
        ]
        assert not missing, (
            "feasibility plan must either enumerate all five slices or "
            f"explicitly reference the README five-slice contract; missing: {missing}"
        )

    def test_cost_and_retrain_budget(self, plan: tuple[Path, str]) -> None:
        _, text = plan
        # retrain budget must be stated (either gpu-hours, wall-clock, $).
        assert re.search(
            r"(gpu[- ]?hour|gpu[- ]?hr|wall[- ]?clock|\bcost\b|\bbudget\b|\bretrain\b)",
            text,
        ), "feasibility plan must state a retrain / cost budget"

    def test_explicit_blocker_section(self, plan: tuple[Path, str]) -> None:
        _, text = plan
        assert "blocker" in text, (
            "feasibility plan must contain an explicit 'blocker' section"
        )

    def test_frozen_baseline_reference(self, plan: tuple[Path, str]) -> None:
        _, text = plan
        assert "baseline" in text and (
            "frozen" in text or "pre-revision" in text or "2026-04-15" in text
        ), "feasibility plan must compare against the frozen pre-revision baseline"


# ---------------------------------------------------------------------------
# Blocker-to-action synthesis
# ---------------------------------------------------------------------------


class TestBlockerToActionSynthesis:
    """Each blocker must map to a concrete action with a decision gate."""

    @pytest.fixture
    def synthesis(self) -> tuple[Path, str]:
        # Prefer a dedicated blocker-to-action file; otherwise accept the
        # canonical synthesis / revision log if it contains a lane-c section.
        for cand in SYNTHESIS_CANDIDATES:
            if cand.exists():
                raw = _read(cand)
                lowered = raw.lower()
                if "blocker" in lowered and "lane c" in lowered:
                    return cand, lowered
        pytest.skip(
            "No blocker-to-action synthesis found yet. Expected a Lane C "
            "blocker section in one of: "
            + ", ".join(str(c.relative_to(REPO_ROOT)) for c in SYNTHESIS_CANDIDATES)
        )

    def test_has_action_mapping_verb(self, synthesis: tuple[Path, str]) -> None:
        _, text = synthesis
        # Must map each blocker to an actionable verb.
        assert re.search(
            r"(action|next step|mitigation|decision|if.*then|owner|eta)",
            text,
        ), "synthesis must map blockers to concrete actions"

    def test_names_known_blockers(self, synthesis: tuple[Path, str]) -> None:
        _, text = synthesis
        # At least one of the documented blockers must be explicitly named.
        known = [
            "data curation",
            "retrain",
            "compute",
            "gpu",
            "held-out",
            "ood",
            "mouse",
        ]
        found = [k for k in known if k in text]
        assert found, (
            "synthesis must enumerate at least one of the documented Lane C "
            f"blockers: {known}"
        )

    def test_has_decision_gate(self, synthesis: tuple[Path, str]) -> None:
        _, text = synthesis
        # A decision gate determines go / no-go or scope-cut criteria.
        assert re.search(
            r"(go[- ]?/[- ]?no[- ]?go|criteria|threshold|gate|cut[- ]?scope|de[- ]?prioritise|deprioritize)",
            text,
        ), "synthesis must include an explicit decision gate / criteria"


# ---------------------------------------------------------------------------
# Cross-file invariants
# ---------------------------------------------------------------------------


class TestRevisionLogSync:
    """The REVISION_LOG must reflect Lane C status at the repo level."""

    @pytest.fixture
    def log_text(self) -> str:
        log = REVISION_DIR / "REVISION_LOG.md"
        if not log.exists():
            pytest.skip("REVISION_LOG.md missing")
        return _read(log).lower()

    def test_lane_c_line_present(self, log_text: str) -> None:
        assert "lane c" in log_text, "REVISION_LOG must reference Lane C"

    def test_lane_c_status_truthful(self, log_text: str) -> None:
        # Either still gated/pending, or explicitly marked complete — never
        # silent.
        assert re.search(
            r"lane c[^.\n]{0,80}(pending|not started|gated|complete|in progress|feasibility|data plan)",
            log_text,
        ), "REVISION_LOG must make Lane C status explicit"

    def test_lane_c_summary_matches_integrated_plan(self, log_text: str) -> None:
        has_integrated_plan = "lane-c feasibility and blocker-to-action synthesis" in log_text
        stale_top_summary = (
            "lane c (data expansion) | not started — gated on a3." in log_text
            or "lane c still pending" in log_text
        )
        assert not (has_integrated_plan and stale_top_summary), (
            "REVISION_LOG cannot claim Lane C is still gated/pending in the top "
            "summary once the integrated feasibility section is present"
        )

    def test_head_line_uses_stable_contract(self, log_text: str) -> None:
        assert "current `revision/major` head: run `git rev-parse --short head`." in log_text, (
            "REVISION_LOG should point readers to git for the exact moving HEAD"
        )
