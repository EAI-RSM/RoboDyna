import {
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  H1,
  H2,
  Pill,
  Row,
  Select,
  Stack,
  Stat,
  Table,
  Text,
  useCanvasState,
} from "cursor/canvas";

type MetricRow = {
  key: string;
  type: string;
  meaning: string;
};

type TaskSpec = {
  id: string;
  name: string;
  category: string;
  metrics: MetricRow[];
};

const SHARED: MetricRow[] = [
  { key: "success", type: "bool", meaning: "Same as check_success()" },
  {
    key: "total_time_sim_s",
    type: "float",
    meaning: "Episode sim time (steps × timestep)",
  },
  {
    key: "total_time_wall_s",
    type: "float",
    meaning: "Wall-clock from episode start → metric build",
  },
  { key: "total_steps", type: "int", meaning: "Policy / env step count" },
  {
    key: "option_label",
    type: "str | null",
    meaning: "default / opt1 / opt2 / opt1+2 when applicable",
  },
];

const TASKS: TaskSpec[] = [
  {
    id: "sort_apples_belt",
    name: "sort_apples_belt",
    category: "Sorting / packing",
    metrics: [
      { key: "n_apples", type: "int", meaning: "Total apples" },
      { key: "n_red", type: "int", meaning: "Red apple count" },
      { key: "n_green", type: "int", meaning: "Green apple count" },
      { key: "n_rotten", type: "int", meaning: "Rotten apple count" },
      { key: "red_ok_count", type: "int", meaning: "Red correctly stored" },
      { key: "green_ok_count", type: "int", meaning: "Green correctly stored" },
      {
        key: "rotten_ok_count",
        type: "int",
        meaning: "Rotten correctly discarded to dump",
      },
      {
        key: "red_success_pct",
        type: "float | null",
        meaning: "red_ok_count / n_red (null if n_red=0)",
      },
      {
        key: "green_success_pct",
        type: "float | null",
        meaning: "green_ok_count / n_green (null if n_green=0)",
      },
      {
        key: "rotten_success_pct",
        type: "float | null",
        meaning: "rotten_ok_count / n_rotten (null if n_rotten=0)",
      },
      {
        key: "sorting_accuracy",
        type: "float",
        meaning: "Correct apples / n_apples",
      },
      { key: "macro_f1", type: "float", meaning: "Existing macro-F1" },
      {
        key: "rotten_discarded_ok",
        type: "bool | null",
        meaning: "Rotten in dump (null if no rotten)",
      },
      {
        key: "rotten_in_apple_box",
        type: "bool | null",
        meaning: "Rotten landed in left or right basket",
      },
      {
        key: "wrong_by_color",
        type: "dict",
        meaning:
          "{green_in_red, red_in_green, rotten_in_red, rotten_in_green, fresh_in_dump}",
      },
      {
        key: "missed_count",
        type: "int",
        meaning: "Not settled in any target receptacle",
      },
    ],
  },
  {
    id: "pack_fruits",
    name: "pack_fruits",
    category: "Sorting / packing",
    metrics: [
      { key: "n_apple", type: "int", meaning: "Apple count" },
      { key: "n_orange", type: "int", meaning: "Orange count" },
      { key: "n_distractor", type: "int", meaning: "Black distractor count" },
      {
        key: "apple_ok_count",
        type: "int",
        meaning: "Apples in correct (left) basket",
      },
      {
        key: "orange_ok_count",
        type: "int",
        meaning: "Oranges in correct (right) basket",
      },
      {
        key: "apple_success_pct",
        type: "float | null",
        meaning: "apple_ok_count / n_apple",
      },
      {
        key: "orange_success_pct",
        type: "float | null",
        meaning: "orange_ok_count / n_orange",
      },
      {
        key: "packing_accuracy",
        type: "float",
        meaning: "Correct real fruit / (n_apple + n_orange)",
      },
      {
        key: "wrong_by_color",
        type: "dict",
        meaning: "{apple_in_orange_basket, orange_in_apple_basket}",
      },
      {
        key: "missed_fruit",
        type: "int",
        meaning: "Real fruit not in correct basket",
      },
      {
        key: "distractors_in_basket",
        type: "int",
        meaning: "Black distractors in a basket (info only)",
      },
    ],
  },
  {
    id: "pick_ripe_apple",
    name: "pick_ripe_apple",
    category: "Sorting / packing",
    metrics: [
      { key: "good_in_basket", type: "bool", meaning: "Good apple in basket" },
      {
        key: "spoiled_present",
        type: "bool",
        meaning: "Spoiled apple was spawned",
      },
      {
        key: "spoiled_in_basket",
        type: "bool",
        meaning: "Spoiled apple ended in basket",
      },
      {
        key: "spoiled_discarded_ok",
        type: "bool | null",
        meaning: "Spoiled present and not in basket (null if none)",
      },
      { key: "ripeness_score", type: "float", meaning: "Existing ripeness score" },
      {
        key: "r_grasp",
        type: "float",
        meaning: "Grasp ripeness (−1 if never grasped)",
      },
      { key: "final_score", type: "float", meaning: "Existing final score" },
    ],
  },
  {
    id: "control_quality",
    name: "control_quality",
    category: "Belt / stamp",
    metrics: [
      { key: "n_tiles", type: "int", meaning: "Total tiles" },
      { key: "n_red", type: "int", meaning: "Red tile count" },
      { key: "n_green", type: "int", meaning: "Green tile count" },
      { key: "n_black", type: "int", meaning: "Black outlier count" },
      {
        key: "red_ok_count",
        type: "int",
        meaning: "Red tiles correctly stamped",
      },
      {
        key: "green_ok_count",
        type: "int",
        meaning: "Green tiles correctly stamped",
      },
      {
        key: "red_success_pct",
        type: "float | null",
        meaning: "red_ok_count / n_red",
      },
      {
        key: "green_success_pct",
        type: "float | null",
        meaning: "green_ok_count / n_green",
      },
      {
        key: "black_skipped_ok_count",
        type: "int",
        meaning: "Black tiles correctly skipped",
      },
      {
        key: "black_skip_pct",
        type: "float | null",
        meaning: "black_skipped_ok_count / n_black",
      },
      {
        key: "stamping_accuracy",
        type: "float",
        meaning: "Correct actions / all tiles",
      },
      {
        key: "missed_colored",
        type: "int",
        meaning: "Colored tiles missed",
      },
      { key: "black_press", type: "bool", meaning: "Any invalid black press" },
      {
        key: "black_press_count",
        type: "int",
        meaning: "Count of black presses",
      },
    ],
  },
  {
    id: "dispense_gummy",
    name: "dispense_gummy",
    category: "Belt / stamp",
    metrics: [
      { key: "target_color", type: "str", meaning: "yellow or blue" },
      { key: "total_target", type: "int", meaning: "Expected target count" },
      {
        key: "total_distractor",
        type: "int",
        meaning: "Expected distractor count",
      },
      { key: "target_caught", type: "int", meaning: "Targets in bowl" },
      { key: "target_missed", type: "int", meaning: "Targets missed" },
      {
        key: "target_success_pct",
        type: "float | null",
        meaning: "target_caught / total_target",
      },
      {
        key: "distractor_caught",
        type: "int",
        meaning: "Distractors in bowl",
      },
      {
        key: "distractor_missed",
        type: "int",
        meaning: "Distractors not in bowl",
      },
      {
        key: "distractor_reject_ok",
        type: "bool",
        meaning: "No distractor in bowl",
      },
      {
        key: "catch_accuracy",
        type: "float",
        meaning: "Targets caught + distractors rejected / relevant items",
      },
      {
        key: "invalid_pattern",
        type: "bool",
        meaning: "Invalid layout flag",
      },
    ],
  },
  {
    id: "punch_dual_holes",
    name: "punch_dual_holes",
    category: "Belt / stamp",
    metrics: [
      {
        key: "n_present_left",
        type: "int",
        meaning: "Present (non-missing) left tiles",
      },
      {
        key: "n_present_right",
        type: "int",
        meaning: "Present (non-missing) right tiles",
      },
      {
        key: "n_punched_left",
        type: "int",
        meaning: "Successfully punched left",
      },
      {
        key: "n_punched_right",
        type: "int",
        meaning: "Successfully punched right",
      },
      {
        key: "left_success_pct",
        type: "float | null",
        meaning: "n_punched_left / n_present_left",
      },
      {
        key: "right_success_pct",
        type: "float | null",
        meaning: "n_punched_right / n_present_right",
      },
      {
        key: "punch_accuracy",
        type: "float",
        meaning: "Punched present / all present",
      },
      { key: "n_missed", type: "int", meaning: "Present tiles missed" },
      {
        key: "invalid_empty_press",
        type: "bool",
        meaning: "Empty-slot press occurred",
      },
      {
        key: "invalid_empty_press_count",
        type: "int",
        meaning: "Empty-slot press count",
      },
      {
        key: "punch_score_L",
        type: "float",
        meaning: "Left offset-based score",
      },
      {
        key: "punch_score_R",
        type: "float",
        meaning: "Right offset-based score",
      },
      {
        key: "punch_score_mean",
        type: "float",
        meaning: "Mean punch score",
      },
    ],
  },
  {
    id: "cook_meat",
    name: "cook_meat",
    category: "Manipulation",
    metrics: [
      { key: "n_stations", type: "int", meaning: "1 or 2 stations" },
      {
        key: "stations",
        type: "list[dict]",
        meaning:
          "Per station: grasp_doneness, target_doneness, doneness_error, cooked_ok, under_cooked, over_cooked, on_board, off_pan, station_success",
      },
      {
        key: "n_stations_ok",
        type: "int",
        meaning: "Stations that fully passed",
      },
      {
        key: "station_success_pct",
        type: "float",
        meaning: "n_stations_ok / n_stations",
      },
      {
        key: "cook_accuracy",
        type: "float",
        meaning: "Alias of station_success_pct",
      },
    ],
  },
  {
    id: "whack_moles",
    name: "whack_moles",
    category: "Manipulation",
    metrics: [
      { key: "n_moles", type: "int", meaning: "Moles this episode" },
      { key: "n_touched", type: "int", meaning: "Moles touched" },
      {
        key: "mole_success_pct",
        type: "float",
        meaning: "n_touched / n_moles",
      },
      {
        key: "whack_accuracy",
        type: "float",
        meaning: "mole_success_pct if no rabbit hit, else 0",
      },
      {
        key: "distractor_enabled",
        type: "bool",
        meaning: "Rabbit present",
      },
      { key: "distractor_hit", type: "bool", meaning: "Rabbit touched" },
      {
        key: "distractor_avoided_ok",
        type: "bool | null",
        meaning: "Rabbit present and not hit (null if none)",
      },
    ],
  },
  {
    id: "hit_target",
    name: "hit_target",
    category: "Manipulation",
    metrics: [
      { key: "stuck", type: "bool", meaning: "Tip stuck in board" },
      { key: "hit_center", type: "bool", meaning: "Yellow center hit" },
      { key: "hit_blocker", type: "bool", meaning: "Contacted a blocker" },
      {
        key: "blocker_avoided_ok",
        type: "bool | null",
        meaning: "Blockers present and never hit (null if none)",
      },
      {
        key: "radial_offset",
        type: "float",
        meaning: "Planar radial miss (−1 if no hit)",
      },
      { key: "hit_score", type: "float", meaning: "Existing hit score" },
    ],
  },
  {
    id: "save_goal",
    name: "save_goal",
    category: "Catch / save",
    metrics: [
      {
        key: "keeper_in_zone",
        type: "bool",
        meaning: "Keeper fully in green zone in time",
      },
      {
        key: "ball_blocked",
        type: "bool",
        meaning: "Blocked on keeper front face",
      },
      { key: "goal_conceded", type: "bool", meaning: "Ball entered goal" },
      {
        key: "late_failure",
        type: "bool",
        meaning: "Arrived after deadline",
      },
      { key: "grippers_open", type: "bool", meaning: "Both grippers open" },
      {
        key: "save_ok",
        type: "bool",
        meaning: "Composite of save success components",
      },
    ],
  },
  {
    id: "load_train",
    name: "load_train",
    category: "Catch / place",
    metrics: [
      {
        key: "ball_in_train",
        type: "bool",
        meaning: "Ball seated in some wagon",
      },
      {
        key: "in_allowed_wagon",
        type: "bool",
        meaning: "In an allowed wagon for this mode",
      },
      {
        key: "latched_car_idx",
        type: "int | null",
        meaning: "Wagon index ball ended in",
      },
      {
        key: "target_wagon_idx",
        type: "int | null",
        meaning: "Nominated wagon (opt1 modes)",
      },
      {
        key: "wrong_wagon",
        type: "bool",
        meaning: "In a wagon but not allowed",
      },
      { key: "missed", type: "bool", meaning: "Not in any wagon" },
    ],
  },
  {
    id: "play_billiard",
    name: "play_billiard",
    category: "Manipulation",
    metrics: [
      {
        key: "primary_pocketed",
        type: "bool",
        meaning: "Red ball in a pocket",
      },
      {
        key: "primary_pocket_id",
        type: "int | null",
        meaning: "Pocket id of primary ball",
      },
      {
        key: "in_allowed_pocket",
        type: "bool",
        meaning: "Pocket allowed for current mode",
      },
      {
        key: "wrong_pocket",
        type: "bool",
        meaning: "Pocketed but not allowed",
      },
      {
        key: "distractor_pocketed",
        type: "bool",
        meaning: "Any distractor pocketed",
      },
      {
        key: "robot_ball_contact",
        type: "bool",
        meaning: "Arm/robot touched ball",
      },
      { key: "n_distractors", type: "int", meaning: "Distractor ball count" },
    ],
  },
  {
    id: "drop_ball_hole",
    name: "drop_ball_hole",
    category: "Manipulation",
    metrics: [
      {
        key: "ball_in_box",
        type: "bool",
        meaning: "Through target hole into box",
      },
      {
        key: "ball_stuck_on_platform",
        type: "bool",
        meaning: "Stuck (sticky mode)",
      },
      {
        key: "went_through_dummy",
        type: "bool | null",
        meaning: "Fell via dummy hole if detectable (null if not tracked)",
      },
    ],
  },
  {
    id: "catch_ramp_ball",
    name: "catch_ramp_ball",
    category: "Catch / save",
    metrics: [
      { key: "target_in_cup", type: "bool", meaning: "Red ball in cup" },
      {
        key: "distractor_present",
        type: "bool",
        meaning: "Distractor spawned",
      },
      {
        key: "distractor_in_cup",
        type: "bool",
        meaning: "Distractor in cup",
      },
      {
        key: "distractor_reject_ok",
        type: "bool | null",
        meaning: "Present and not in cup",
      },
      {
        key: "catch_offset",
        type: "float",
        meaning: "Horizontal catch error",
      },
      {
        key: "ball_ball_bounces",
        type: "int",
        meaning: "Ball–ball bounce count",
      },
      {
        key: "drop_wall_bounces",
        type: "int",
        meaning: "Wall bounce count",
      },
    ],
  },
  {
    id: "catch_valley_ball",
    name: "catch_valley_ball",
    category: "Catch / save",
    metrics: [
      { key: "target_in_bowl", type: "bool", meaning: "Red ball in bowl" },
      {
        key: "bowl_behind_line",
        type: "bool",
        meaning: "Bowl past red line",
      },
      {
        key: "arm_ball_contact",
        type: "bool",
        meaning: "Arm touched red ball",
      },
      {
        key: "distractor_present",
        type: "bool",
        meaning: "Black distractor present",
      },
      {
        key: "distractor_in_bowl",
        type: "bool",
        meaning: "Distractor in bowl",
      },
      {
        key: "distractor_reject_ok",
        type: "bool | null",
        meaning: "Present and not in bowl",
      },
      {
        key: "horizontal_offset",
        type: "float",
        meaning: "Catch offset",
      },
    ],
  },
  {
    id: "catch_shelf_marble",
    name: "catch_shelf_marble",
    category: "Catch / save",
    metrics: [
      {
        key: "marble_caught",
        type: "bool",
        meaning: "Result == caught",
      },
      {
        key: "marble_result",
        type: "str",
        meaning: "caught / missed / etc.",
      },
      {
        key: "target_catch_x",
        type: "float",
        meaning: "Planned catch x",
      },
      { key: "bowl_x", type: "float", meaning: "Final bowl x" },
      {
        key: "catch_x_error",
        type: "float",
        meaning: "|bowl_x − target_catch_x|",
      },
    ],
  },
  {
    id: "catch_marbles_trapdoors",
    name: "catch_marbles_trapdoors",
    category: "Catch / save",
    metrics: [
      {
        key: "ball_in_lower_box",
        type: "bool",
        meaning: "Target through door into box",
      },
      {
        key: "used_matching_door",
        type: "bool",
        meaning: "Correct color door",
      },
      {
        key: "used_wrong_door",
        type: "bool",
        meaning: "Wrong color door",
      },
      {
        key: "wrong_door_opened",
        type: "bool",
        meaning: "Any wrong door opened",
      },
      {
        key: "ball_still_on_top",
        type: "bool",
        meaning: "Never dropped",
      },
      {
        key: "distractor_present",
        type: "bool",
        meaning: "Distractor enabled",
      },
      {
        key: "distractor_through_any",
        type: "bool",
        meaning: "Distractor through any door",
      },
      {
        key: "distractor_in_lower_box",
        type: "bool",
        meaning: "Distractor in lower box",
      },
      {
        key: "distractor_reject_ok",
        type: "bool | null",
        meaning: "Present and not through / not in box",
      },
    ],
  },
  {
    id: "marble_shelf_maze",
    name: "marble_shelf_maze",
    category: "Catch / save",
    metrics: [
      { key: "ball_in_bowl", type: "bool", meaning: "Target marble in bowl" },
      {
        key: "ball_on_table",
        type: "bool",
        meaning: "Missed onto table",
      },
      { key: "ball_missed", type: "bool", meaning: "Explicit miss flag" },
      { key: "presses_made", type: "int", meaning: "Button presses" },
      { key: "n_shelves", type: "int", meaning: "Shelf count" },
    ],
  },
  {
    id: "put_cup_belt",
    name: "put_cup_belt",
    category: "Catch / place",
    metrics: [
      {
        key: "cup_between_tools",
        type: "bool",
        meaning: "Seated between yellow tools",
      },
      {
        key: "curtain_present",
        type: "bool",
        meaning: "Curtains enabled",
      },
      { key: "curtain_hit", type: "bool", meaning: "Touched a curtain" },
      {
        key: "curtain_avoided_ok",
        type: "bool | null",
        meaning: "Curtains present and never hit",
      },
      {
        key: "placement_score",
        type: "float",
        meaning: "Existing placement score",
      },
    ],
  },
  {
    id: "place_block_belt",
    name: "place_block_belt",
    category: "Catch / place",
    metrics: [
      {
        key: "placed_before_line",
        type: "bool",
        meaning: "First contact before red line",
      },
      { key: "placed_on_belt", type: "bool", meaning: "On belt" },
      { key: "in_bowl", type: "bool", meaning: "Ends in bowl" },
      {
        key: "blocker_enabled",
        type: "bool",
        meaning: "Blocker present",
      },
      { key: "hit_blocker", type: "bool", meaning: "Hit blocker" },
      {
        key: "avoided_blocker",
        type: "bool",
        meaning: "Cleared blocker lane",
      },
      { key: "tilt_score", type: "float", meaning: "Existing tilt score" },
      { key: "max_tilt_deg", type: "float", meaning: "Peak tilt (deg)" },
    ],
  },
  {
    id: "catch_rat",
    name: "catch_rat",
    category: "Manipulation",
    metrics: [
      { key: "n_rats", type: "int", meaning: "1 or 2" },
      {
        key: "rats_held",
        type: "list[bool]",
        meaning: "Per-rat gripper contact",
      },
      { key: "n_held", type: "int", meaning: "How many held" },
      {
        key: "catch_pct",
        type: "float",
        meaning: "n_held / n_rats",
      },
      {
        key: "catch_accuracy",
        type: "float",
        meaning: "Same as catch_pct (all required rats for success)",
      },
      {
        key: "catch_two_mice",
        type: "bool",
        meaning: "Two-mice mode flag",
      },
      {
        key: "catch_score",
        type: "float",
        meaning: "Existing grasp-offset score",
      },
      {
        key: "opaque_surface",
        type: "bool",
        meaning: "Opaque surface option",
      },
    ],
  },
];

const EVAL_ROWS: MetricRow[] = [
  {
    key: "EpisodeMetrics.metric_detail",
    type: "dict",
    meaning: "Full per-episode metric_detail copied at episode end",
  },
  {
    key: "AggregatedMetrics.success_rate",
    type: "float",
    meaning: "Mean success across episodes",
  },
  {
    key: "AggregatedMetrics.total_time_sim_s",
    type: "mean/std",
    meaning: "Aggregate sim time",
  },
  {
    key: "AggregatedMetrics.total_time_wall_s",
    type: "mean/std",
    meaning: "Aggregate wall time",
  },
  {
    key: "test_report.json[].metric_detail",
    type: "dict",
    meaning: "Each episode row includes metric_detail",
  },
];

function metricTableRows(metrics: MetricRow[]) {
  return metrics.map((m) => [m.key, m.type, m.meaning]);
}

function allTaskRows(tasks: TaskSpec[]) {
  const rows: Array<Array<string>> = [];
  for (const task of tasks) {
    for (const m of task.metrics) {
      rows.push([task.name, task.category, m.key, m.type, m.meaning]);
    }
  }
  return rows;
}

export default function MetricDetailApproval() {
  const [selected, setSelected] = useCanvasState<string>("task", "all");

  const visible =
    selected === "all" ? TASKS : TASKS.filter((t) => t.id === selected);
  const taskFieldCount = TASKS.reduce((n, t) => n + t.metrics.length, 0);
  const selectedTask = selected === "all" ? null : visible[0] ?? null;

  return (
    <Stack gap={20} style={{ padding: 20, maxWidth: 1200 }}>
      <Stack gap={6}>
        <H1>metric_detail approval catalog</H1>
        <Text tone="secondary">
          Proposed keys for all 21 final_task_demos tasks. Shared fields apply
          to every task; task tables are additive. No implementation yet —
          approve or edit this list.
        </Text>
      </Stack>

      <Callout tone="info" title="Locked decisions">
        Time: both sim and wall. Rates: per-class % plus overall accuracy.
        Surfacing: env.info["metric_detail"] and
        EpisodeMetrics / test_report.json.
      </Callout>

      <Row gap={12} wrap>
        <Stat value={String(TASKS.length)} label="Tasks" />
        <Stat value={String(SHARED.length)} label="Shared fields" />
        <Stat value={String(taskFieldCount)} label="Task-specific fields" />
        <Stat
          value={String(SHARED.length + taskFieldCount)}
          label="Total field entries"
        />
      </Row>

      <Card>
        <CardHeader trailing={<Pill tone="neutral">all tasks</Pill>}>
          Shared fields
        </CardHeader>
        <CardBody style={{ paddingTop: 0 }}>
          <Table
            headers={["Key", "Type", "Meaning"]}
            rows={metricTableRows(SHARED)}
            striped
            stickyHeader
          />
        </CardBody>
      </Card>

      <Stack gap={8}>
        <Row gap={12} align="center" justify="space-between">
          <H2>Per-task fields</H2>
          <Select
            value={selected}
            onChange={setSelected}
            options={[
              { value: "all", label: `All tasks (${TASKS.length})` },
              ...TASKS.map((t) => ({
                value: t.id,
                label: `${t.name} (${t.metrics.length})`,
              })),
            ]}
          />
        </Row>
        <Text tone="tertiary" size="small">
          {selected === "all"
            ? `Full catalog · ${taskFieldCount} task-specific rows`
            : `${selectedTask?.name ?? selected} · ${visible[0]?.metrics.length ?? 0} fields`}
        </Text>
      </Stack>

      {selected === "all" ? (
        <Card>
          <CardHeader trailing={<Pill tone="info">{taskFieldCount} rows</Pill>}>
            All tasks · flat table
          </CardHeader>
          <CardBody style={{ paddingTop: 0 }}>
            <Table
              headers={["Task", "Category", "Key", "Type", "Meaning"]}
              rows={allTaskRows(TASKS)}
              striped
              stickyHeader
              style={{ maxHeight: 640 }}
            />
          </CardBody>
        </Card>
      ) : selectedTask ? (
        <Card>
          <CardHeader
            trailing={
              <Row gap={6}>
                <Pill tone="neutral">{selectedTask.category}</Pill>
                <Pill tone="info">{selectedTask.metrics.length} fields</Pill>
              </Row>
            }
          >
            {selectedTask.name}
          </CardHeader>
          <CardBody style={{ paddingTop: 0 }}>
            <Table
              headers={["Key", "Type", "Meaning"]}
              rows={metricTableRows(selectedTask.metrics)}
              striped
              stickyHeader
            />
          </CardBody>
        </Card>
      ) : null}

      <Divider />

      <Stack gap={8}>
        <H2>Eval / report wiring</H2>
        <Text tone="secondary">
          Not task-specific keys — how metric_detail is exported beyond
          env.info.
        </Text>
        <Table
          headers={["Key", "Type", "Meaning"]}
          rows={metricTableRows(EVAL_ROWS)}
          striped
        />
      </Stack>

      <Callout tone="warning" title="Awaiting approval">
        Reply with edits (rename / drop / add fields) or say this catalog is
        approved to start implementation.
      </Callout>
    </Stack>
  );
}
