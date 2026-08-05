# Layout randomization demos

Top-down + side-by-side only (no head-only).

## Success (from layout demo recordings)

| Task | Files | Layout notes |
|------|-------|----------------|
| pour_beer | `success_s{1,3,4}_*` | Right-side tap; varied tap Y / props |
| measure_ingredient | `success_s{1,6,8}_*` | MW corner/top-left; L/R station |
| cook_food | `success_center_lf_*`, `success_current_lf_*`, `success_center_rf_*` | Stove pose + burner variants |

## Failures (from the original 10-seed layout runs)

| Task | Seed | Files | Mode |
|------|------|-------|------|
| pour_beer | 0 | `fail_s0_*` | Foam/beer overflow |
| pour_beer | 5 | `fail_s5_*` | Foam/beer overflow |

`measure_ingredient` seed 4 and early `cook_food` fails were expert/check failures without saved videos in the smoke run; cook seed 2 now passes after plate soft-seat, so no cook fail clip was kept.
