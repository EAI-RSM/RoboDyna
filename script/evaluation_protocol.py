"""Shared fixed-seed protocol setup for local and client policy evaluators."""
from __future__ import annotations

from interactive.base_task_gui import SCENARIO_OVERRIDES
from interactive.experiment_config import evaluation_seeds_for, evaluation_suite


def configure_evaluation(
    task: str,
    args: dict,
    scenario: str | None = None,
) -> tuple[str, str | None, list[int]]:
    """Apply one task scenario and return its shared ten-seed evaluation suite.

    Base tasks default to ``default`` and apply precisely the same feature
    overrides as the human GUI. Household tasks have no variants.
    """
    suite = evaluation_suite(task)
    task = str(task).strip().replace("-", "_")
    if suite == "base":
        selected = str(scenario or "default").strip().lower()
        overrides = SCENARIO_OVERRIDES.get(task, {}).get(selected)
        if overrides is None:
            raise ValueError(
                f"Unknown base scenario {scenario!r} for {task!r}; "
                "choose default, opt1, opt2, or opt1+2."
            )
        args.setdefault("task_args", {}).setdefault(task, {}).update(overrides)
        # Some environments use this marker in addition to task_args.
        args["interactive_scenario"] = selected
        args["interactive_task"] = task
        scenario = selected
    else:
        scenario = None

    seeds = evaluation_seeds_for(task, scenario)
    if not seeds:
        label = scenario or "default"
        raise ValueError(
            f"No fixed evaluation seeds configured for {suite}/{task}/{label} "
            "in task_config/eval_seeds.yml."
        )
    return suite, scenario, seeds
