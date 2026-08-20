from interactive.experiment_config import (
    BASE_TASK_TABLE,
    HOUSEHOLD_TASK_TABLE,
    evaluation_seeds_for,
    load_experiment_config,
    parse_seed_policy,
)


def test_evaluation_seeds_are_not_clamped() -> None:
    policy = parse_seed_policy(
        {"base": {"catch_cuboid": {"default": [10000, 10017]}}}
    )

    assert policy.resolve("base", "catch_cuboid", "default") == [10000, 10017]


def test_gui_uses_the_shared_fixed_evaluation_seeds() -> None:
    cfg = load_experiment_config()
    for _number, task, _label in BASE_TASK_TABLE:
        for scenario in ("default", "opt1", "opt2", "opt1+2"):
            seeds = evaluation_seeds_for(task, scenario)
            assert len(seeds) == 10
            assert cfg.seeds_for("base", task, scenario) == seeds
    for _number, task, _label in HOUSEHOLD_TASK_TABLE:
        seeds = evaluation_seeds_for(task)
        assert len(seeds) == 10
        assert cfg.seeds_for("household", task) == seeds
