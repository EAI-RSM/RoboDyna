from interactive.experiment_config import parse_seed_policy


def test_evaluation_seeds_are_not_clamped() -> None:
    policy = parse_seed_policy(
        {"base": {"catch_cuboid": {"default": [10000, 10017]}}}
    )

    assert policy.resolve("base", "catch_cuboid", "default") == [10000, 10017]
