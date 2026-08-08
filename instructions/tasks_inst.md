<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>RoboDynaExp — Tasks &amp; Metrics</title>
<style>
  :root {
    --bg: #0f1419;
    --panel: #1a222c;
    --panel2: #222c38;
    --border: #334155;
    --text: #e8eef4;
    --muted: #9aabbc;
    --accent: #5b9fd4;
    --basic: #3d8b6e;
    --hh: #b8860b;
    --proposed: #7c6cf0;
    --ok: #2f9e6b;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    background: radial-gradient(1200px 600px at 10% -10%, #1a2a3a 0%, var(--bg) 55%);
    color: var(--text); line-height: 1.45;
  }
  header.hero {
    padding: 2rem 1.5rem 1.25rem; border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, #15202b, transparent);
  }
  h1 { margin: 0 0 0.4rem; font-size: 1.75rem; letter-spacing: -0.02em; }
  h2 { margin: 1.5rem 0 0.75rem; font-size: 1.25rem; }
  h3 { margin: 0; font-size: 1.05rem; font-weight: 600; }
  .muted { color: var(--muted); max-width: 70ch; }
  .wrap { max-width: 1100px; margin: 0 auto; padding: 1rem 1.5rem 3rem; }
  .toolbar {
    display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: center;
    margin: 1rem 0 1.5rem; position: sticky; top: 0; z-index: 5;
    background: rgba(15,20,25,0.92); backdrop-filter: blur(8px);
    padding: 0.75rem 0; border-bottom: 1px solid var(--border);
  }
  input[type=search] {
    flex: 1; min-width: 200px; padding: 0.55rem 0.75rem; border-radius: 8px;
    border: 1px solid var(--border); background: var(--panel); color: var(--text);
  }
  .filters label { margin-right: 0.75rem; color: var(--muted); font-size: 0.9rem; cursor: pointer; }
  .pill {
    display: inline-block; font-size: 0.72rem; padding: 0.15rem 0.5rem; border-radius: 999px;
    border: 1px solid var(--border); color: var(--muted); margin-right: 0.25rem;
  }
  .pill.basic { color: #b7e4d0; border-color: var(--basic); background: rgba(61,139,110,0.15); }
  .pill.hh { color: #f0d48a; border-color: var(--hh); background: rgba(184,134,11,0.15); }
  .pill.proposed { color: #d2ccff; border-color: var(--proposed); background: rgba(124,108,240,0.18); }
  .pill.env { color: #a8c4d8; }
  .pill.ok { color: #b8f0d4; border-color: var(--ok); }
  .pill.dim { color: var(--muted); }
  .pill.cat { color: var(--accent); }
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.75rem; margin: 1rem 0; }
  .stat { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 0.85rem 1rem; }
  .stat b { display: block; font-size: 1.4rem; }
  .stat span { color: var(--muted); font-size: 0.85rem; }
  table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
  th, td { text-align: left; padding: 0.45rem 0.55rem; border-bottom: 1px solid var(--border); vertical-align: top; }
  th { color: var(--muted); font-weight: 600; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em; }
  code { font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 0.88em; color: #c6e0f5; }
  .task {
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 1rem 1.1rem; margin: 0.85rem 0;
  }
  .task header { display: flex; flex-wrap: wrap; gap: 0.5rem 1rem; align-items: baseline; justify-content: space-between; }
  .desc { color: var(--muted); margin: 0.5rem 0 0.75rem; }
  .index { columns: 2; gap: 1.5rem; }
  @media (max-width: 720px) { .index { columns: 1; } }
  .index li { margin: 0.25rem 0; }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
  .callout {
    background: var(--panel2); border-left: 3px solid var(--accent);
    padding: 0.85rem 1rem; border-radius: 0 8px 8px 0; margin: 1rem 0;
    color: var(--muted); font-size: 0.92rem;
  }
  .hidden { display: none !important; }
</style>
</head>
<body>
<header class="hero">
  <div class="wrap">
    <h1>RoboDynaExp — Tasks &amp; Metrics</h1>
    <p class="muted">
      Inventory of <strong>basic</strong> (dynamic suite) and <strong>household</strong> tasks,
      with metrics that can be computed for each: closed-loop eval metrics
      (<code>script/eval_metrics.py</code>), proposed <code>metric_detail</code> keys
      (<code>docs/metrics/metric_detail_approval.*</code>), and in-env scores already exposed by task code.
    </p>
    <div class="stats">
      <div class="stat"><b>23</b><span>basic tasks</span></div>
      <div class="stat"><b>12</b><span>household GUI tasks</span></div>
      <div class="stat"><b>4</b><span>household extras / demos</span></div>
      <div class="stat"><b>21</b><span>tasks in metric_detail catalog</span></div>
    </div>
  </div>
</header>

<div class="wrap">
  <div class="toolbar">
    <input id="q" type="search" placeholder="Filter tasks or metrics…" />
    <div class="filters">
      <label><input type="radio" name="suite" value="all" checked/> All</label>
      <label><input type="radio" name="suite" value="basic"/> Basic</label>
      <label><input type="radio" name="suite" value="household"/> Household</label>
    </div>
  </div>

  <div class="callout">
    <strong>Status legend.</strong>
    <span class="pill ok">implemented</span> = used in closed-loop eval today (SR, MS, RC, penalties).
    <span class="pill proposed">metric_detail catalog</span> = proposed per-task keys (not wired into Python yet).
    <span class="pill env">env / eval only</span> = binary success + obs / in-env scores; no catalog entry.
    Efficiency and Comfort exist in <code>eval_metrics.py</code> but are currently commented out.
  </div>

  <h2>Global metrics (all evaluated tasks)</h2>
  <table>
    <thead><tr><th>Metric</th><th>Status</th><th>Applies to</th><th>Meaning</th></tr></thead>
    <tbody><tr><td>Success Rate (SR)</td><td><span class="pill ok">implemented</span></td><td>All</td><td>Mean of check_success() over episodes</td></tr><tr><td>Route Completion (RC)</td><td><span class="pill ok">implemented</span></td><td>All</td><td>0–100; 100 if success; else EE progress toward dynamic target</td></tr><tr><td>Manipulation Score (MS)</td><td><span class="pill ok">implemented</span></td><td>All</td><td>RC × product of penalty factors</td></tr><tr><td>Penalty: out_of_bounds</td><td><span class="pill ok">implemented</span></td><td>Dynamic tasks reporting OOB</td><td>Factor 0.5 if target leaves workspace</td></tr><tr><td>Penalty: collision</td><td><span class="pill ok">implemented</span></td><td>Tasks with clutter</td><td>Factor 0.8 per unexpected clutter collision</td></tr><tr><td>Efficiency (E_eff)</td><td><span class="pill dim">defined, disabled</span></td><td>All</td><td>(1 − steps/max_steps)×100 if success else 0</td></tr><tr><td>Comfort (C_comf)</td><td><span class="pill dim">defined, disabled</span></td><td>All</td><td>100 × exp(−avg_jerk / 0.02) from EE jerk</td></tr><tr><td>metric_detail.*</td><td><span class="pill dim">proposed</span></td><td>21 basic tasks in catalog</td><td>Per-task keys in docs/metrics/metric_detail_approval.*; not wired yet</td></tr><tr><td>total_time_sim_s / wall_s / steps</td><td><span class="pill dim">proposed (shared)</span></td><td>Catalog tasks</td><td>Shared timing fields in metric_detail</td></tr><tr><td>option_label</td><td><span class="pill dim">proposed (shared)</span></td><td>Catalog tasks with options</td><td>default / opt1 / opt2 / opt1+2</td></tr></tbody>
  </table>

  <h2>Task index</h2>
  <h3>Basic</h3>
  <ol class="index"><li><a href="#catch_marbles_trapdoors"><code>catch_marbles_trapdoors</code></a> — Catch Marbles Trapdoors</li><li><a href="#catch_ramp_ball"><code>catch_ramp_ball</code></a> — Catch Ramp Ball</li><li><a href="#catch_cuboid"><code>catch_cuboid</code></a> — Catch Cuboid</li><li><a href="#catch_shelf_marble"><code>catch_shelf_marble</code></a> — Catch Shelf Marble</li><li><a href="#catch_valley_ball"><code>catch_valley_ball</code></a> — Catch Valley Ball</li><li><a href="#stop_valley_ball"><code>stop_valley_ball</code></a> — Stop Valley Ball</li><li><a href="#cook_meat"><code>cook_meat</code></a> — Cook Meat</li><li><a href="#cook_meat_timer"><code>cook_meat_timer</code></a> — Cook Meat Timer</li><li><a href="#put_cup_belt"><code>put_cup_belt</code></a> — Put Cup Belt</li><li><a href="#dispense_gummy"><code>dispense_gummy</code></a> — Dispense Gummy</li><li><a href="#punch_dual_holes"><code>punch_dual_holes</code></a> — Punch Dual Holes</li><li><a href="#save_goal"><code>save_goal</code></a> — Save Goal</li><li><a href="#hit_target"><code>hit_target</code></a> — Hit Target</li><li><a href="#load_train"><code>load_train</code></a> — Load Train</li><li><a href="#marble_shelf_maze"><code>marble_shelf_maze</code></a> — Marble Shelf Maze</li><li><a href="#pack_fruits"><code>pack_fruits</code></a> — Pack Fruits</li><li><a href="#pick_ripe_apple"><code>pick_ripe_apple</code></a> — Pick Ripe Apple</li><li><a href="#place_block_belt"><code>place_block_belt</code></a> — Place Block Belt</li><li><a href="#play_billiard"><code>play_billiard</code></a> — Play Billiard</li><li><a href="#control_quality"><code>control_quality</code></a> — Control Quality</li><li><a href="#drop_ball_hole"><code>drop_ball_hole</code></a> — Drop Ball Hole</li><li><a href="#sort_apples_belt"><code>sort_apples_belt</code></a> — Sort Apples Belt</li><li><a href="#whack_moles"><code>whack_moles</code></a> — Whack Moles</li></ol>
  <h3>Household (GUI)</h3>
  <ol class="index"><li><a href="#trap_bug"><code>trap_bug</code></a> — Trap Bug</li><li><a href="#boil_milk"><code>boil_milk</code></a> — Boil Milk</li><li><a href="#fill_coffee_jar"><code>fill_coffee_jar</code></a> — Fill Coffee Jar</li><li><a href="#pour_beer"><code>pour_beer</code></a> — Pour Beer</li><li><a href="#cook_food"><code>cook_food</code></a> — Cook Food</li><li><a href="#cook_food_timer"><code>cook_food_timer</code></a> — Cook Food Timer</li><li><a href="#measure_ingredient"><code>measure_ingredient</code></a> — Measure Ingredient</li><li><a href="#make_soup"><code>make_soup</code></a> — Make Soup</li><li><a href="#catch_cup"><code>catch_cup</code></a> — Catch Cup</li><li><a href="#catch_mouse_object_drop"><code>catch_mouse_object_drop</code></a> — Catch Mouse Object Drop</li><li><a href="#stop_ball"><code>stop_ball</code></a> — Stop Ball</li><li><a href="#clean_table"><code>clean_table</code></a> — Clean Table</li></ol>
  <h3>Household extras</h3>
  <ol class="index"><li><a href="#serve_dinner"><code>serve_dinner</code></a> — Serve Dinner</li><li><a href="#make_soup_test"><code>make_soup_test</code></a> — Make Soup Test</li><li><a href="#catch_rolling_cup"><code>catch_rolling_cup</code></a> — Catch Rolling Cup</li><li><a href="#empty_bag"><code>empty_bag</code></a> — Empty Bag</li></ol>

  <h2 id="basic">Basic tasks</h2>
  <p class="muted">Source: <code>script_exp/interactive_task_gui.py</code> TASKS (23).</p>
  
    <section class="task" id="catch_marbles_trapdoors" data-suite="basic" data-name="catch_marbles_trapdoors Catch Marbles Trapdoors">
      <header>
        <h3><code>catch_marbles_trapdoors</code> — Catch Marbles Trapdoors</h3>
        <div class="badges"><span class="pill proposed">metric_detail catalog</span><span class="pill basic">basic</span> <span class="pill cat">Catch / save</span></div>
      </header>
      <p class="desc">Time matching colored key press to drop the target marble through its trapdoor.</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>total_time_sim_s</code></td><td>float</td><td>Episode sim time (steps × timestep) [proposed metric_detail]</td></tr><tr><td><code>total_time_wall_s</code></td><td>float</td><td>Wall-clock from episode start → metric build [proposed metric_detail]</td></tr><tr><td><code>total_steps</code></td><td>int</td><td>Policy / env step count [proposed metric_detail]</td></tr><tr><td><code>option_label</code></td><td>str | null</td><td>default / opt1 / opt2 / opt1+2 when applicable [proposed metric_detail]</td></tr><tr><td><code>ball_in_lower_box</code></td><td>bool</td><td>Target through door into box [proposed metric_detail]</td></tr><tr><td><code>used_matching_door</code></td><td>bool</td><td>Correct color door [proposed metric_detail]</td></tr><tr><td><code>used_wrong_door</code></td><td>bool</td><td>Wrong color door [proposed metric_detail]</td></tr><tr><td><code>wrong_door_opened</code></td><td>bool</td><td>Any wrong door opened [proposed metric_detail]</td></tr><tr><td><code>ball_still_on_top</code></td><td>bool</td><td>Never dropped [proposed metric_detail]</td></tr><tr><td><code>distractor_present</code></td><td>bool</td><td>Distractor enabled [proposed metric_detail]</td></tr><tr><td><code>distractor_through_any</code></td><td>bool</td><td>Distractor through any door [proposed metric_detail]</td></tr><tr><td><code>distractor_in_lower_box</code></td><td>bool</td><td>Distractor in lower box [proposed metric_detail]</td></tr><tr><td><code>distractor_reject_ok</code></td><td>bool | null</td><td>Present and not through / not in box [proposed metric_detail]</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="catch_ramp_ball" data-suite="basic" data-name="catch_ramp_ball Catch Ramp Ball">
      <header>
        <h3><code>catch_ramp_ball</code> — Catch Ramp Ball</h3>
        <div class="badges"><span class="pill proposed">metric_detail catalog</span><span class="pill basic">basic</span> <span class="pill cat">Catch / save</span></div>
      </header>
      <p class="desc">Catch the red ball leaving a ramp in a cup; reject the distractor.</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>total_time_sim_s</code></td><td>float</td><td>Episode sim time (steps × timestep) [proposed metric_detail]</td></tr><tr><td><code>total_time_wall_s</code></td><td>float</td><td>Wall-clock from episode start → metric build [proposed metric_detail]</td></tr><tr><td><code>total_steps</code></td><td>int</td><td>Policy / env step count [proposed metric_detail]</td></tr><tr><td><code>option_label</code></td><td>str | null</td><td>default / opt1 / opt2 / opt1+2 when applicable [proposed metric_detail]</td></tr><tr><td><code>target_in_cup</code></td><td>bool</td><td>Red ball in cup [proposed metric_detail]</td></tr><tr><td><code>distractor_present</code></td><td>bool</td><td>Distractor spawned [proposed metric_detail]</td></tr><tr><td><code>distractor_in_cup</code></td><td>bool</td><td>Distractor in cup [proposed metric_detail]</td></tr><tr><td><code>distractor_reject_ok</code></td><td>bool | null</td><td>Present and not in cup [proposed metric_detail]</td></tr><tr><td><code>catch_offset</code></td><td>float</td><td>Horizontal catch error [proposed metric_detail]</td></tr><tr><td><code>ball_ball_bounces</code></td><td>int</td><td>Ball–ball bounce count [proposed metric_detail]</td></tr><tr><td><code>drop_wall_bounces</code></td><td>int</td><td>Wall bounce count [proposed metric_detail]</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="catch_cuboid" data-suite="basic" data-name="catch_cuboid Catch Cuboid">
      <header>
        <h3><code>catch_cuboid</code> — Catch Cuboid</h3>
        <div class="badges"><span class="pill proposed">metric_detail catalog</span><span class="pill basic">basic</span> <span class="pill cat">Manipulation</span></div>
      </header>
      <p class="desc">Grasp cuboid(s) during timed pop-up windows.</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>total_time_sim_s</code></td><td>float</td><td>Episode sim time (steps × timestep) [proposed metric_detail]</td></tr><tr><td><code>total_time_wall_s</code></td><td>float</td><td>Wall-clock from episode start → metric build [proposed metric_detail]</td></tr><tr><td><code>total_steps</code></td><td>int</td><td>Policy / env step count [proposed metric_detail]</td></tr><tr><td><code>option_label</code></td><td>str | null</td><td>default / opt1 / opt2 / opt1+2 when applicable [proposed metric_detail]</td></tr><tr><td><code>n_cuboids</code></td><td>int</td><td>1 or 2 [proposed metric_detail]</td></tr><tr><td><code>cuboids_held</code></td><td>list[bool]</td><td>Per-cuboid gripper contact [proposed metric_detail]</td></tr><tr><td><code>n_held</code></td><td>int</td><td>How many held [proposed metric_detail]</td></tr><tr><td><code>catch_pct</code></td><td>float</td><td>n_held / n_cuboids [proposed metric_detail]</td></tr><tr><td><code>catch_accuracy</code></td><td>float</td><td>Same as catch_pct (all required cuboids for success) [proposed metric_detail]</td></tr><tr><td><code>catch_two_cuboids</code></td><td>bool</td><td>Two-cuboids mode flag [proposed metric_detail]</td></tr><tr><td><code>catch_score</code></td><td>float</td><td>Existing grasp-offset score [proposed metric_detail]</td></tr><tr><td><code>opaque_surface</code></td><td>bool</td><td>Opaque surface option [proposed metric_detail]</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="catch_shelf_marble" data-suite="basic" data-name="catch_shelf_marble Catch Shelf Marble">
      <header>
        <h3><code>catch_shelf_marble</code> — Catch Shelf Marble</h3>
        <div class="badges"><span class="pill proposed">metric_detail catalog</span><span class="pill basic">basic</span> <span class="pill cat">Catch / save</span></div>
      </header>
      <p class="desc">Slide a bowl on a belt to catch a marble rolling off tilted shelves.</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>total_time_sim_s</code></td><td>float</td><td>Episode sim time (steps × timestep) [proposed metric_detail]</td></tr><tr><td><code>total_time_wall_s</code></td><td>float</td><td>Wall-clock from episode start → metric build [proposed metric_detail]</td></tr><tr><td><code>total_steps</code></td><td>int</td><td>Policy / env step count [proposed metric_detail]</td></tr><tr><td><code>option_label</code></td><td>str | null</td><td>default / opt1 / opt2 / opt1+2 when applicable [proposed metric_detail]</td></tr><tr><td><code>marble_caught</code></td><td>bool</td><td>Result == caught [proposed metric_detail]</td></tr><tr><td><code>marble_result</code></td><td>str</td><td>caught / missed / etc. [proposed metric_detail]</td></tr><tr><td><code>target_catch_x</code></td><td>float</td><td>Planned catch x [proposed metric_detail]</td></tr><tr><td><code>bowl_x</code></td><td>float</td><td>Final bowl x [proposed metric_detail]</td></tr><tr><td><code>catch_x_error</code></td><td>float</td><td>|bowl_x − target_catch_x| [proposed metric_detail]</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="catch_valley_ball" data-suite="basic" data-name="catch_valley_ball Catch Valley Ball">
      <header>
        <h3><code>catch_valley_ball</code> — Catch Valley Ball</h3>
        <div class="badges"><span class="pill proposed">metric_detail catalog</span><span class="pill basic">basic</span> <span class="pill cat">Catch / save</span></div>
      </header>
      <p class="desc">Push an open box past a line to catch the red ball from a curved ramp.</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>total_time_sim_s</code></td><td>float</td><td>Episode sim time (steps × timestep) [proposed metric_detail]</td></tr><tr><td><code>total_time_wall_s</code></td><td>float</td><td>Wall-clock from episode start → metric build [proposed metric_detail]</td></tr><tr><td><code>total_steps</code></td><td>int</td><td>Policy / env step count [proposed metric_detail]</td></tr><tr><td><code>option_label</code></td><td>str | null</td><td>default / opt1 / opt2 / opt1+2 when applicable [proposed metric_detail]</td></tr><tr><td><code>target_in_bowl</code></td><td>bool</td><td>Red ball in bowl [proposed metric_detail]</td></tr><tr><td><code>bowl_behind_line</code></td><td>bool</td><td>Bowl past red line [proposed metric_detail]</td></tr><tr><td><code>arm_ball_contact</code></td><td>bool</td><td>Arm touched red ball [proposed metric_detail]</td></tr><tr><td><code>distractor_present</code></td><td>bool</td><td>Black distractor present [proposed metric_detail]</td></tr><tr><td><code>distractor_in_bowl</code></td><td>bool</td><td>Distractor in bowl [proposed metric_detail]</td></tr><tr><td><code>distractor_reject_ok</code></td><td>bool | null</td><td>Present and not in bowl [proposed metric_detail]</td></tr><tr><td><code>horizontal_offset</code></td><td>float</td><td>Catch offset [proposed metric_detail]</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="stop_valley_ball" data-suite="basic" data-name="stop_valley_ball Stop Valley Ball">
      <header>
        <h3><code>stop_valley_ball</code> — Stop Valley Ball</h3>
        <div class="badges"><span class="pill env">env / eval only</span><span class="pill basic">basic</span> <span class="pill cat">Basic</span></div>
      </header>
      <p class="desc">Hold a ping-pong bat mid-air so the red ball hits the circular head before the table.</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>total_time_sim_s / total_steps</code></td><td>float/int</td><td>Shared timing metrics (proposed)</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="cook_meat" data-suite="basic" data-name="cook_meat Cook Meat">
      <header>
        <h3><code>cook_meat</code> — Cook Meat</h3>
        <div class="badges"><span class="pill proposed">metric_detail catalog</span><span class="pill basic">basic</span> <span class="pill cat">Manipulation</span></div>
      </header>
      <p class="desc">Cook steak on a pan to target doneness; return cooked steak to the board.</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>total_time_sim_s</code></td><td>float</td><td>Episode sim time (steps × timestep) [proposed metric_detail]</td></tr><tr><td><code>total_time_wall_s</code></td><td>float</td><td>Wall-clock from episode start → metric build [proposed metric_detail]</td></tr><tr><td><code>total_steps</code></td><td>int</td><td>Policy / env step count [proposed metric_detail]</td></tr><tr><td><code>option_label</code></td><td>str | null</td><td>default / opt1 / opt2 / opt1+2 when applicable [proposed metric_detail]</td></tr><tr><td><code>n_stations</code></td><td>int</td><td>1 or 2 stations [proposed metric_detail]</td></tr><tr><td><code>stations</code></td><td>list[dict]</td><td>Per station: grasp_doneness, target_doneness, doneness_error, cooked_ok, under_cooked, over_cooked, on_board, off_pan, station_success [proposed metric_detail]</td></tr><tr><td><code>n_stations_ok</code></td><td>int</td><td>Stations that fully passed [proposed metric_detail]</td></tr><tr><td><code>station_success_pct</code></td><td>float</td><td>n_stations_ok / n_stations [proposed metric_detail]</td></tr><tr><td><code>cook_accuracy</code></td><td>float</td><td>Alias of station_success_pct [proposed metric_detail]</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="cook_meat_timer" data-suite="basic" data-name="cook_meat_timer Cook Meat Timer">
      <header>
        <h3><code>cook_meat_timer</code> — Cook Meat Timer</h3>
        <div class="badges"><span class="pill env">env / eval only</span><span class="pill basic">basic</span> <span class="pill cat">Basic</span></div>
      </header>
      <p class="desc">Same as cook_meat, plus a pie timer that tracks doneness (green→yellow→red).</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>grasp_doneness / doneness</code></td><td>float</td><td>Latched / live cooking progress in [0,1]</td></tr><tr><td><code>target_doneness_range</code></td><td>tuple</td><td>Success band for doneness</td></tr><tr><td><code>cook_accuracy</code></td><td>float</td><td>Inherit cook_meat doneness accuracy (proposed)</td></tr><tr><td><code>timer_phase / fill</code></td><td>str/float</td><td>Pie timer green→yellow→red from doneness</td></tr><tr><td><code>total_time_sim_s / total_steps</code></td><td>float/int</td><td>Shared timing metrics (proposed)</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="put_cup_belt" data-suite="basic" data-name="put_cup_belt Put Cup Belt">
      <header>
        <h3><code>put_cup_belt</code> — Put Cup Belt</h3>
        <div class="badges"><span class="pill proposed">metric_detail catalog</span><span class="pill basic">basic</span> <span class="pill cat">Catch / place</span></div>
      </header>
      <p class="desc">Place a cup in the slot between yellow tools on a belt; optional curtains.</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>total_time_sim_s</code></td><td>float</td><td>Episode sim time (steps × timestep) [proposed metric_detail]</td></tr><tr><td><code>total_time_wall_s</code></td><td>float</td><td>Wall-clock from episode start → metric build [proposed metric_detail]</td></tr><tr><td><code>total_steps</code></td><td>int</td><td>Policy / env step count [proposed metric_detail]</td></tr><tr><td><code>option_label</code></td><td>str | null</td><td>default / opt1 / opt2 / opt1+2 when applicable [proposed metric_detail]</td></tr><tr><td><code>cup_between_tools</code></td><td>bool</td><td>Seated between yellow tools [proposed metric_detail]</td></tr><tr><td><code>curtain_present</code></td><td>bool</td><td>Curtains enabled [proposed metric_detail]</td></tr><tr><td><code>curtain_hit</code></td><td>bool</td><td>Touched a curtain [proposed metric_detail]</td></tr><tr><td><code>curtain_avoided_ok</code></td><td>bool | null</td><td>Curtains present and never hit [proposed metric_detail]</td></tr><tr><td><code>placement_score</code></td><td>float</td><td>Existing placement score [proposed metric_detail]</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="dispense_gummy" data-suite="basic" data-name="dispense_gummy Dispense Gummy">
      <header>
        <h3><code>dispense_gummy</code> — Dispense Gummy</h3>
        <div class="badges"><span class="pill proposed">metric_detail catalog</span><span class="pill basic">basic</span> <span class="pill cat">Belt / stamp</span></div>
      </header>
      <p class="desc">Operate dispenser / moving bowl; collect only target-colored gummies.</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>total_time_sim_s</code></td><td>float</td><td>Episode sim time (steps × timestep) [proposed metric_detail]</td></tr><tr><td><code>total_time_wall_s</code></td><td>float</td><td>Wall-clock from episode start → metric build [proposed metric_detail]</td></tr><tr><td><code>total_steps</code></td><td>int</td><td>Policy / env step count [proposed metric_detail]</td></tr><tr><td><code>option_label</code></td><td>str | null</td><td>default / opt1 / opt2 / opt1+2 when applicable [proposed metric_detail]</td></tr><tr><td><code>target_color</code></td><td>str</td><td>yellow or blue [proposed metric_detail]</td></tr><tr><td><code>total_target</code></td><td>int</td><td>Expected target count [proposed metric_detail]</td></tr><tr><td><code>total_distractor</code></td><td>int</td><td>Expected distractor count [proposed metric_detail]</td></tr><tr><td><code>target_caught</code></td><td>int</td><td>Targets in bowl [proposed metric_detail]</td></tr><tr><td><code>target_missed</code></td><td>int</td><td>Targets missed [proposed metric_detail]</td></tr><tr><td><code>target_success_pct</code></td><td>float | null</td><td>target_caught / total_target [proposed metric_detail]</td></tr><tr><td><code>distractor_caught</code></td><td>int</td><td>Distractors in bowl [proposed metric_detail]</td></tr><tr><td><code>distractor_missed</code></td><td>int</td><td>Distractors not in bowl [proposed metric_detail]</td></tr><tr><td><code>distractor_reject_ok</code></td><td>bool</td><td>No distractor in bowl [proposed metric_detail]</td></tr><tr><td><code>catch_accuracy</code></td><td>float</td><td>Targets caught + distractors rejected / relevant items [proposed metric_detail]</td></tr><tr><td><code>invalid_pattern</code></td><td>bool</td><td>Invalid layout flag [proposed metric_detail]</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="punch_dual_holes" data-suite="basic" data-name="punch_dual_holes Punch Dual Holes">
      <header>
        <h3><code>punch_dual_holes</code> — Punch Dual Holes</h3>
        <div class="badges"><span class="pill proposed">metric_detail catalog</span><span class="pill basic">basic</span> <span class="pill cat">Belt / stamp</span></div>
      </header>
      <p class="desc">Both arms punch present tiles on two belts.</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>total_time_sim_s</code></td><td>float</td><td>Episode sim time (steps × timestep) [proposed metric_detail]</td></tr><tr><td><code>total_time_wall_s</code></td><td>float</td><td>Wall-clock from episode start → metric build [proposed metric_detail]</td></tr><tr><td><code>total_steps</code></td><td>int</td><td>Policy / env step count [proposed metric_detail]</td></tr><tr><td><code>option_label</code></td><td>str | null</td><td>default / opt1 / opt2 / opt1+2 when applicable [proposed metric_detail]</td></tr><tr><td><code>n_present_left</code></td><td>int</td><td>Present (non-missing) left tiles [proposed metric_detail]</td></tr><tr><td><code>n_present_right</code></td><td>int</td><td>Present (non-missing) right tiles [proposed metric_detail]</td></tr><tr><td><code>n_punched_left</code></td><td>int</td><td>Successfully punched left [proposed metric_detail]</td></tr><tr><td><code>n_punched_right</code></td><td>int</td><td>Successfully punched right [proposed metric_detail]</td></tr><tr><td><code>left_success_pct</code></td><td>float | null</td><td>n_punched_left / n_present_left [proposed metric_detail]</td></tr><tr><td><code>right_success_pct</code></td><td>float | null</td><td>n_punched_right / n_present_right [proposed metric_detail]</td></tr><tr><td><code>punch_accuracy</code></td><td>float</td><td>Punched present / all present [proposed metric_detail]</td></tr><tr><td><code>n_missed</code></td><td>int</td><td>Present tiles missed [proposed metric_detail]</td></tr><tr><td><code>invalid_empty_press</code></td><td>bool</td><td>Empty-slot press occurred [proposed metric_detail]</td></tr><tr><td><code>invalid_empty_press_count</code></td><td>int</td><td>Empty-slot press count [proposed metric_detail]</td></tr><tr><td><code>punch_score_L</code></td><td>float</td><td>Left offset-based score [proposed metric_detail]</td></tr><tr><td><code>punch_score_R</code></td><td>float</td><td>Right offset-based score [proposed metric_detail]</td></tr><tr><td><code>punch_score_mean</code></td><td>float</td><td>Mean punch score [proposed metric_detail]</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="save_goal" data-suite="basic" data-name="save_goal Save Goal">
      <header>
        <h3><code>save_goal</code> — Save Goal</h3>
        <div class="badges"><span class="pill proposed">metric_detail catalog</span><span class="pill basic">basic</span> <span class="pill cat">Catch / save</span></div>
      </header>
      <p class="desc">Place a keeper before the deadline to block the ball from the goal.</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>total_time_sim_s</code></td><td>float</td><td>Episode sim time (steps × timestep) [proposed metric_detail]</td></tr><tr><td><code>total_time_wall_s</code></td><td>float</td><td>Wall-clock from episode start → metric build [proposed metric_detail]</td></tr><tr><td><code>total_steps</code></td><td>int</td><td>Policy / env step count [proposed metric_detail]</td></tr><tr><td><code>option_label</code></td><td>str | null</td><td>default / opt1 / opt2 / opt1+2 when applicable [proposed metric_detail]</td></tr><tr><td><code>keeper_in_zone</code></td><td>bool</td><td>Keeper fully in green zone in time [proposed metric_detail]</td></tr><tr><td><code>ball_blocked</code></td><td>bool</td><td>Blocked on keeper front face [proposed metric_detail]</td></tr><tr><td><code>goal_conceded</code></td><td>bool</td><td>Ball entered goal [proposed metric_detail]</td></tr><tr><td><code>late_failure</code></td><td>bool</td><td>Arrived after deadline [proposed metric_detail]</td></tr><tr><td><code>grippers_open</code></td><td>bool</td><td>Both grippers open [proposed metric_detail]</td></tr><tr><td><code>save_ok</code></td><td>bool</td><td>Composite of save success components [proposed metric_detail]</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="hit_target" data-suite="basic" data-name="hit_target Hit Target">
      <header>
        <h3><code>hit_target</code> — Hit Target</h3>
        <div class="badges"><span class="pill proposed">metric_detail catalog</span><span class="pill basic">basic</span> <span class="pill cat">Manipulation</span></div>
      </header>
      <p class="desc">Stick hits a moving target’s yellow center; avoid blockers.</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>total_time_sim_s</code></td><td>float</td><td>Episode sim time (steps × timestep) [proposed metric_detail]</td></tr><tr><td><code>total_time_wall_s</code></td><td>float</td><td>Wall-clock from episode start → metric build [proposed metric_detail]</td></tr><tr><td><code>total_steps</code></td><td>int</td><td>Policy / env step count [proposed metric_detail]</td></tr><tr><td><code>option_label</code></td><td>str | null</td><td>default / opt1 / opt2 / opt1+2 when applicable [proposed metric_detail]</td></tr><tr><td><code>stuck</code></td><td>bool</td><td>Tip stuck in board [proposed metric_detail]</td></tr><tr><td><code>hit_center</code></td><td>bool</td><td>Yellow center hit [proposed metric_detail]</td></tr><tr><td><code>hit_blocker</code></td><td>bool</td><td>Contacted a blocker [proposed metric_detail]</td></tr><tr><td><code>blocker_avoided_ok</code></td><td>bool | null</td><td>Blockers present and never hit (null if none) [proposed metric_detail]</td></tr><tr><td><code>radial_offset</code></td><td>float</td><td>Planar radial miss (−1 if no hit) [proposed metric_detail]</td></tr><tr><td><code>hit_score</code></td><td>float</td><td>Existing hit score [proposed metric_detail]</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="load_train" data-suite="basic" data-name="load_train Load Train">
      <header>
        <h3><code>load_train</code> — Load Train</h3>
        <div class="badges"><span class="pill proposed">metric_detail catalog</span><span class="pill basic">basic</span> <span class="pill cat">Catch / place</span></div>
      </header>
      <p class="desc">Drop a ball into an allowed wagon of a circling toy train.</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>total_time_sim_s</code></td><td>float</td><td>Episode sim time (steps × timestep) [proposed metric_detail]</td></tr><tr><td><code>total_time_wall_s</code></td><td>float</td><td>Wall-clock from episode start → metric build [proposed metric_detail]</td></tr><tr><td><code>total_steps</code></td><td>int</td><td>Policy / env step count [proposed metric_detail]</td></tr><tr><td><code>option_label</code></td><td>str | null</td><td>default / opt1 / opt2 / opt1+2 when applicable [proposed metric_detail]</td></tr><tr><td><code>ball_in_train</code></td><td>bool</td><td>Ball seated in some wagon [proposed metric_detail]</td></tr><tr><td><code>in_allowed_wagon</code></td><td>bool</td><td>In an allowed wagon for this mode [proposed metric_detail]</td></tr><tr><td><code>latched_car_idx</code></td><td>int | null</td><td>Wagon index ball ended in [proposed metric_detail]</td></tr><tr><td><code>target_wagon_idx</code></td><td>int | null</td><td>Nominated wagon (opt1 modes) [proposed metric_detail]</td></tr><tr><td><code>wrong_wagon</code></td><td>bool</td><td>In a wagon but not allowed [proposed metric_detail]</td></tr><tr><td><code>missed</code></td><td>bool</td><td>Not in any wagon [proposed metric_detail]</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="marble_shelf_maze" data-suite="basic" data-name="marble_shelf_maze Marble Shelf Maze">
      <header>
        <h3><code>marble_shelf_maze</code> — Marble Shelf Maze</h3>
        <div class="badges"><span class="pill proposed">metric_detail catalog</span><span class="pill basic">basic</span> <span class="pill cat">Catch / save</span></div>
      </header>
      <p class="desc">Tilt shelves to route a marble through a zig-zag into a bowl.</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>total_time_sim_s</code></td><td>float</td><td>Episode sim time (steps × timestep) [proposed metric_detail]</td></tr><tr><td><code>total_time_wall_s</code></td><td>float</td><td>Wall-clock from episode start → metric build [proposed metric_detail]</td></tr><tr><td><code>total_steps</code></td><td>int</td><td>Policy / env step count [proposed metric_detail]</td></tr><tr><td><code>option_label</code></td><td>str | null</td><td>default / opt1 / opt2 / opt1+2 when applicable [proposed metric_detail]</td></tr><tr><td><code>ball_in_bowl</code></td><td>bool</td><td>Target marble in bowl [proposed metric_detail]</td></tr><tr><td><code>ball_on_table</code></td><td>bool</td><td>Missed onto table [proposed metric_detail]</td></tr><tr><td><code>ball_missed</code></td><td>bool</td><td>Explicit miss flag [proposed metric_detail]</td></tr><tr><td><code>presses_made</code></td><td>int</td><td>Button presses [proposed metric_detail]</td></tr><tr><td><code>n_shelves</code></td><td>int</td><td>Shelf count [proposed metric_detail]</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="pack_fruits" data-suite="basic" data-name="pack_fruits Pack Fruits">
      <header>
        <h3><code>pack_fruits</code> — Pack Fruits</h3>
        <div class="badges"><span class="pill proposed">metric_detail catalog</span><span class="pill basic">basic</span> <span class="pill cat">Sorting / packing</span></div>
      </header>
      <p class="desc">Sort apples/oranges from belts into matching baskets.</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>total_time_sim_s</code></td><td>float</td><td>Episode sim time (steps × timestep) [proposed metric_detail]</td></tr><tr><td><code>total_time_wall_s</code></td><td>float</td><td>Wall-clock from episode start → metric build [proposed metric_detail]</td></tr><tr><td><code>total_steps</code></td><td>int</td><td>Policy / env step count [proposed metric_detail]</td></tr><tr><td><code>option_label</code></td><td>str | null</td><td>default / opt1 / opt2 / opt1+2 when applicable [proposed metric_detail]</td></tr><tr><td><code>n_apple</code></td><td>int</td><td>Apple count [proposed metric_detail]</td></tr><tr><td><code>n_orange</code></td><td>int</td><td>Orange count [proposed metric_detail]</td></tr><tr><td><code>n_distractor</code></td><td>int</td><td>Black distractor count [proposed metric_detail]</td></tr><tr><td><code>apple_ok_count</code></td><td>int</td><td>Apples in correct (left) basket [proposed metric_detail]</td></tr><tr><td><code>orange_ok_count</code></td><td>int</td><td>Oranges in correct (right) basket [proposed metric_detail]</td></tr><tr><td><code>apple_success_pct</code></td><td>float | null</td><td>apple_ok_count / n_apple [proposed metric_detail]</td></tr><tr><td><code>orange_success_pct</code></td><td>float | null</td><td>orange_ok_count / n_orange [proposed metric_detail]</td></tr><tr><td><code>packing_accuracy</code></td><td>float</td><td>Correct real fruit / (n_apple + n_orange) [proposed metric_detail]</td></tr><tr><td><code>wrong_by_color</code></td><td>dict</td><td>{apple_in_orange_basket, orange_in_apple_basket} [proposed metric_detail]</td></tr><tr><td><code>missed_fruit</code></td><td>int</td><td>Real fruit not in correct basket [proposed metric_detail]</td></tr><tr><td><code>distractors_in_basket</code></td><td>int</td><td>Black distractors in a basket (info only) [proposed metric_detail]</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="pick_ripe_apple" data-suite="basic" data-name="pick_ripe_apple Pick Ripe Apple">
      <header>
        <h3><code>pick_ripe_apple</code> — Pick Ripe Apple</h3>
        <div class="badges"><span class="pill proposed">metric_detail catalog</span><span class="pill basic">basic</span> <span class="pill cat">Sorting / packing</span></div>
      </header>
      <p class="desc">Pick a good apple into the basket; ignore spoiled fruit.</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>total_time_sim_s</code></td><td>float</td><td>Episode sim time (steps × timestep) [proposed metric_detail]</td></tr><tr><td><code>total_time_wall_s</code></td><td>float</td><td>Wall-clock from episode start → metric build [proposed metric_detail]</td></tr><tr><td><code>total_steps</code></td><td>int</td><td>Policy / env step count [proposed metric_detail]</td></tr><tr><td><code>option_label</code></td><td>str | null</td><td>default / opt1 / opt2 / opt1+2 when applicable [proposed metric_detail]</td></tr><tr><td><code>good_in_basket</code></td><td>bool</td><td>Good apple in basket [proposed metric_detail]</td></tr><tr><td><code>spoiled_present</code></td><td>bool</td><td>Spoiled apple was spawned [proposed metric_detail]</td></tr><tr><td><code>spoiled_in_basket</code></td><td>bool</td><td>Spoiled apple ended in basket [proposed metric_detail]</td></tr><tr><td><code>spoiled_discarded_ok</code></td><td>bool | null</td><td>Spoiled present and not in basket (null if none) [proposed metric_detail]</td></tr><tr><td><code>ripeness_score</code></td><td>float</td><td>Existing ripeness score [proposed metric_detail]</td></tr><tr><td><code>r_grasp</code></td><td>float</td><td>Grasp ripeness (−1 if never grasped) [proposed metric_detail]</td></tr><tr><td><code>final_score</code></td><td>float</td><td>Existing final score [proposed metric_detail]</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="place_block_belt" data-suite="basic" data-name="place_block_belt Place Block Belt">
      <header>
        <h3><code>place_block_belt</code> — Place Block Belt</h3>
        <div class="badges"><span class="pill proposed">metric_detail catalog</span><span class="pill basic">basic</span> <span class="pill cat">Catch / place</span></div>
      </header>
      <p class="desc">Place a top-heavy block on a belt so it rides upright into the exit bowl.</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>total_time_sim_s</code></td><td>float</td><td>Episode sim time (steps × timestep) [proposed metric_detail]</td></tr><tr><td><code>total_time_wall_s</code></td><td>float</td><td>Wall-clock from episode start → metric build [proposed metric_detail]</td></tr><tr><td><code>total_steps</code></td><td>int</td><td>Policy / env step count [proposed metric_detail]</td></tr><tr><td><code>option_label</code></td><td>str | null</td><td>default / opt1 / opt2 / opt1+2 when applicable [proposed metric_detail]</td></tr><tr><td><code>placed_before_line</code></td><td>bool</td><td>First contact before red line [proposed metric_detail]</td></tr><tr><td><code>placed_on_belt</code></td><td>bool</td><td>On belt [proposed metric_detail]</td></tr><tr><td><code>in_bowl</code></td><td>bool</td><td>Ends in bowl [proposed metric_detail]</td></tr><tr><td><code>blocker_enabled</code></td><td>bool</td><td>Blocker present [proposed metric_detail]</td></tr><tr><td><code>hit_blocker</code></td><td>bool</td><td>Hit blocker [proposed metric_detail]</td></tr><tr><td><code>avoided_blocker</code></td><td>bool</td><td>Cleared blocker lane [proposed metric_detail]</td></tr><tr><td><code>tilt_score</code></td><td>float</td><td>Existing tilt score [proposed metric_detail]</td></tr><tr><td><code>max_tilt_deg</code></td><td>float</td><td>Peak tilt (deg) [proposed metric_detail]</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="play_billiard" data-suite="basic" data-name="play_billiard Play Billiard">
      <header>
        <h3><code>play_billiard</code> — Play Billiard</h3>
        <div class="badges"><span class="pill proposed">metric_detail catalog</span><span class="pill basic">basic</span> <span class="pill cat">Manipulation</span></div>
      </header>
      <p class="desc">Strike the red ball into an allowed pocket without robot–ball contact.</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>total_time_sim_s</code></td><td>float</td><td>Episode sim time (steps × timestep) [proposed metric_detail]</td></tr><tr><td><code>total_time_wall_s</code></td><td>float</td><td>Wall-clock from episode start → metric build [proposed metric_detail]</td></tr><tr><td><code>total_steps</code></td><td>int</td><td>Policy / env step count [proposed metric_detail]</td></tr><tr><td><code>option_label</code></td><td>str | null</td><td>default / opt1 / opt2 / opt1+2 when applicable [proposed metric_detail]</td></tr><tr><td><code>primary_pocketed</code></td><td>bool</td><td>Red ball in a pocket [proposed metric_detail]</td></tr><tr><td><code>primary_pocket_id</code></td><td>int | null</td><td>Pocket id of primary ball [proposed metric_detail]</td></tr><tr><td><code>in_allowed_pocket</code></td><td>bool</td><td>Pocket allowed for current mode [proposed metric_detail]</td></tr><tr><td><code>wrong_pocket</code></td><td>bool</td><td>Pocketed but not allowed [proposed metric_detail]</td></tr><tr><td><code>distractor_pocketed</code></td><td>bool</td><td>Any distractor pocketed [proposed metric_detail]</td></tr><tr><td><code>robot_ball_contact</code></td><td>bool</td><td>Arm/robot touched ball [proposed metric_detail]</td></tr><tr><td><code>n_distractors</code></td><td>int</td><td>Distractor ball count [proposed metric_detail]</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="control_quality" data-suite="basic" data-name="control_quality Control Quality">
      <header>
        <h3><code>control_quality</code> — Control Quality</h3>
        <div class="badges"><span class="pill proposed">metric_detail catalog</span><span class="pill basic">basic</span> <span class="pill cat">Belt / stamp</span></div>
      </header>
      <p class="desc">Stamp red/green tiles; skip black outliers.</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>total_time_sim_s</code></td><td>float</td><td>Episode sim time (steps × timestep) [proposed metric_detail]</td></tr><tr><td><code>total_time_wall_s</code></td><td>float</td><td>Wall-clock from episode start → metric build [proposed metric_detail]</td></tr><tr><td><code>total_steps</code></td><td>int</td><td>Policy / env step count [proposed metric_detail]</td></tr><tr><td><code>option_label</code></td><td>str | null</td><td>default / opt1 / opt2 / opt1+2 when applicable [proposed metric_detail]</td></tr><tr><td><code>n_tiles</code></td><td>int</td><td>Total tiles [proposed metric_detail]</td></tr><tr><td><code>n_red</code></td><td>int</td><td>Red tile count [proposed metric_detail]</td></tr><tr><td><code>n_green</code></td><td>int</td><td>Green tile count [proposed metric_detail]</td></tr><tr><td><code>n_black</code></td><td>int</td><td>Black outlier count [proposed metric_detail]</td></tr><tr><td><code>red_ok_count</code></td><td>int</td><td>Red tiles correctly stamped [proposed metric_detail]</td></tr><tr><td><code>green_ok_count</code></td><td>int</td><td>Green tiles correctly stamped [proposed metric_detail]</td></tr><tr><td><code>red_success_pct</code></td><td>float | null</td><td>red_ok_count / n_red [proposed metric_detail]</td></tr><tr><td><code>green_success_pct</code></td><td>float | null</td><td>green_ok_count / n_green [proposed metric_detail]</td></tr><tr><td><code>black_skipped_ok_count</code></td><td>int</td><td>Black tiles correctly skipped [proposed metric_detail]</td></tr><tr><td><code>black_skip_pct</code></td><td>float | null</td><td>black_skipped_ok_count / n_black [proposed metric_detail]</td></tr><tr><td><code>stamping_accuracy</code></td><td>float</td><td>Correct actions / all tiles [proposed metric_detail]</td></tr><tr><td><code>missed_colored</code></td><td>int</td><td>Colored tiles missed [proposed metric_detail]</td></tr><tr><td><code>black_press</code></td><td>bool</td><td>Any invalid black press [proposed metric_detail]</td></tr><tr><td><code>black_press_count</code></td><td>int</td><td>Count of black presses [proposed metric_detail]</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="drop_ball_hole" data-suite="basic" data-name="drop_ball_hole Drop Ball Hole">
      <header>
        <h3><code>drop_ball_hole</code> — Drop Ball Hole</h3>
        <div class="badges"><span class="pill proposed">metric_detail catalog</span><span class="pill basic">basic</span> <span class="pill cat">Manipulation</span></div>
      </header>
      <p class="desc">Guide a ball through the target hole of a rotating sorter into a container.</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>total_time_sim_s</code></td><td>float</td><td>Episode sim time (steps × timestep) [proposed metric_detail]</td></tr><tr><td><code>total_time_wall_s</code></td><td>float</td><td>Wall-clock from episode start → metric build [proposed metric_detail]</td></tr><tr><td><code>total_steps</code></td><td>int</td><td>Policy / env step count [proposed metric_detail]</td></tr><tr><td><code>option_label</code></td><td>str | null</td><td>default / opt1 / opt2 / opt1+2 when applicable [proposed metric_detail]</td></tr><tr><td><code>ball_in_box</code></td><td>bool</td><td>Through target hole into box [proposed metric_detail]</td></tr><tr><td><code>ball_stuck_on_platform</code></td><td>bool</td><td>Stuck (sticky mode) [proposed metric_detail]</td></tr><tr><td><code>went_through_dummy</code></td><td>bool | null</td><td>Fell via dummy hole if detectable (null if not tracked) [proposed metric_detail]</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="sort_apples_belt" data-suite="basic" data-name="sort_apples_belt Sort Apples Belt">
      <header>
        <h3><code>sort_apples_belt</code> — Sort Apples Belt</h3>
        <div class="badges"><span class="pill proposed">metric_detail catalog</span><span class="pill basic">basic</span> <span class="pill cat">Sorting / packing</span></div>
      </header>
      <p class="desc">Sort red/green apples into bins; rotten → dump.</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>total_time_sim_s</code></td><td>float</td><td>Episode sim time (steps × timestep) [proposed metric_detail]</td></tr><tr><td><code>total_time_wall_s</code></td><td>float</td><td>Wall-clock from episode start → metric build [proposed metric_detail]</td></tr><tr><td><code>total_steps</code></td><td>int</td><td>Policy / env step count [proposed metric_detail]</td></tr><tr><td><code>option_label</code></td><td>str | null</td><td>default / opt1 / opt2 / opt1+2 when applicable [proposed metric_detail]</td></tr><tr><td><code>n_apples</code></td><td>int</td><td>Total apples [proposed metric_detail]</td></tr><tr><td><code>n_red</code></td><td>int</td><td>Red apple count [proposed metric_detail]</td></tr><tr><td><code>n_green</code></td><td>int</td><td>Green apple count [proposed metric_detail]</td></tr><tr><td><code>n_rotten</code></td><td>int</td><td>Rotten apple count [proposed metric_detail]</td></tr><tr><td><code>red_ok_count</code></td><td>int</td><td>Red correctly stored [proposed metric_detail]</td></tr><tr><td><code>green_ok_count</code></td><td>int</td><td>Green correctly stored [proposed metric_detail]</td></tr><tr><td><code>rotten_ok_count</code></td><td>int</td><td>Rotten correctly discarded to dump [proposed metric_detail]</td></tr><tr><td><code>red_success_pct</code></td><td>float | null</td><td>red_ok_count / n_red (null if n_red=0) [proposed metric_detail]</td></tr><tr><td><code>green_success_pct</code></td><td>float | null</td><td>green_ok_count / n_green (null if n_green=0) [proposed metric_detail]</td></tr><tr><td><code>rotten_success_pct</code></td><td>float | null</td><td>rotten_ok_count / n_rotten (null if n_rotten=0) [proposed metric_detail]</td></tr><tr><td><code>sorting_accuracy</code></td><td>float</td><td>Correct apples / n_apples [proposed metric_detail]</td></tr><tr><td><code>macro_f1</code></td><td>float</td><td>Existing macro-F1 [proposed metric_detail]</td></tr><tr><td><code>rotten_discarded_ok</code></td><td>bool | null</td><td>Rotten in dump (null if no rotten) [proposed metric_detail]</td></tr><tr><td><code>rotten_in_apple_box</code></td><td>bool | null</td><td>Rotten landed in left or right basket [proposed metric_detail]</td></tr><tr><td><code>wrong_by_color</code></td><td>dict</td><td>{green_in_red, red_in_green, rotten_in_red, rotten_in_green, fresh_in_dump} [proposed metric_detail]</td></tr><tr><td><code>missed_count</code></td><td>int</td><td>Not settled in any target receptacle [proposed metric_detail]</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="whack_moles" data-suite="basic" data-name="whack_moles Whack Moles">
      <header>
        <h3><code>whack_moles</code> — Whack Moles</h3>
        <div class="badges"><span class="pill proposed">metric_detail catalog</span><span class="pill basic">basic</span> <span class="pill cat">Manipulation</span></div>
      </header>
      <p class="desc">Strike moles with mallets; do not touch the rabbit.</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>total_time_sim_s</code></td><td>float</td><td>Episode sim time (steps × timestep) [proposed metric_detail]</td></tr><tr><td><code>total_time_wall_s</code></td><td>float</td><td>Wall-clock from episode start → metric build [proposed metric_detail]</td></tr><tr><td><code>total_steps</code></td><td>int</td><td>Policy / env step count [proposed metric_detail]</td></tr><tr><td><code>option_label</code></td><td>str | null</td><td>default / opt1 / opt2 / opt1+2 when applicable [proposed metric_detail]</td></tr><tr><td><code>n_moles</code></td><td>int</td><td>Moles this episode [proposed metric_detail]</td></tr><tr><td><code>n_touched</code></td><td>int</td><td>Moles touched [proposed metric_detail]</td></tr><tr><td><code>mole_success_pct</code></td><td>float</td><td>n_touched / n_moles [proposed metric_detail]</td></tr><tr><td><code>whack_accuracy</code></td><td>float</td><td>mole_success_pct if no rabbit hit, else 0 [proposed metric_detail]</td></tr><tr><td><code>distractor_enabled</code></td><td>bool</td><td>Rabbit present [proposed metric_detail]</td></tr><tr><td><code>distractor_hit</code></td><td>bool</td><td>Rabbit touched [proposed metric_detail]</td></tr><tr><td><code>distractor_avoided_ok</code></td><td>bool | null</td><td>Rabbit present and not hit (null if none) [proposed metric_detail]</td></tr></tbody>
      </table>
    </section>

  <h2 id="household">Household tasks</h2>
  <p class="muted">Source: <code>script_hh_exp/household_task_gui.py</code> TASKS (12). Shared head camera + step cutoff via <code>HOUSEHOLD_TASKS</code> in <code>envs/utils/household_view.py</code>.</p>
  
    <section class="task" id="trap_bug" data-suite="household" data-name="trap_bug Trap Bug">
      <header>
        <h3><code>trap_bug</code> — Trap Bug</h3>
        <div class="badges"><span class="pill env">env / eval only</span><span class="pill hh">household</span> <span class="pill cat">Household</span></div>
      </header>
      <p class="desc">Trap a scurrying bug under a glass box (office shelf).</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>total_time_sim_s / total_steps</code></td><td>float/int</td><td>Shared timing + eval MS/RC</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="boil_milk" data-suite="household" data-name="boil_milk Boil Milk">
      <header>
        <h3><code>boil_milk</code> — Boil Milk</h3>
        <div class="badges"><span class="pill env">env / eval only</span><span class="pill hh">household</span> <span class="pill cat">Household</span></div>
      </header>
      <p class="desc">Boil milk and shut the stove off at the rim without overflow.</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>liquid_level / overflowed</code></td><td>float/bool</td><td>Fill state from obs</td></tr><tr><td><code>total_time_sim_s / total_steps</code></td><td>float/int</td><td>Shared timing + eval MS/RC</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="fill_coffee_jar" data-suite="household" data-name="fill_coffee_jar Fill Coffee Jar">
      <header>
        <h3><code>fill_coffee_jar</code> — Fill Coffee Jar</h3>
        <div class="badges"><span class="pill env">env / eval only</span><span class="pill hh">household</span> <span class="pill cat">Household</span></div>
      </header>
      <p class="desc">Press the dispenser lid; fill the jar to the marked line.</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>fill_level</code></td><td>float</td><td>Dispensed fill progress</td></tr><tr><td><code>total_time_sim_s / total_steps</code></td><td>float/int</td><td>Shared timing + eval MS/RC</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="pour_beer" data-suite="household" data-name="pour_beer Pour Beer">
      <header>
        <h3><code>pour_beer</code> — Pour Beer</h3>
        <div class="badges"><span class="pill env">env / eval only</span><span class="pill hh">household</span> <span class="pill cat">Household</span></div>
      </header>
      <p class="desc">Draft-tap pour with foam control; overflow fails.</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>beer_level / foam / overflowed</code></td><td>float/bool</td><td>Pour state from obs</td></tr><tr><td><code>total_time_sim_s / total_steps</code></td><td>float/int</td><td>Shared timing + eval MS/RC</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="cook_food" data-suite="household" data-name="cook_food Cook Food">
      <header>
        <h3><code>cook_food</code> — Cook Food</h3>
        <div class="badges"><span class="pill env">env / eval only</span><span class="pill hh">household</span> <span class="pill cat">Household</span></div>
      </header>
      <p class="desc">Board → pan → cook via knob → shut off; success = target doneness after stove off.</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>grasp_doneness / doneness</code></td><td>float</td><td>Latched / live cooking progress</td></tr><tr><td><code>target_doneness_range</code></td><td>tuple</td><td>Food-type success band</td></tr><tr><td><code>fire_intensity / stove_on</code></td><td>float/bool</td><td>Stove state from obs</td></tr><tr><td><code>cook_steps</code></td><td>int</td><td>Per-episode sampled cook duration (speed)</td></tr><tr><td><code>total_time_sim_s / total_steps</code></td><td>float/int</td><td>Shared timing + eval MS/RC</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="cook_food_timer" data-suite="household" data-name="cook_food_timer Cook Food Timer">
      <header>
        <h3><code>cook_food_timer</code> — Cook Food Timer</h3>
        <div class="badges"><span class="pill env">env / eval only</span><span class="pill hh">household</span> <span class="pill cat">Household</span></div>
      </header>
      <p class="desc">Same as cook_food, plus a stove-gated pie timer.</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>grasp_doneness / doneness</code></td><td>float</td><td>Latched / live cooking progress</td></tr><tr><td><code>timer_phase / fill</code></td><td>str/float</td><td>Pie timer gated by stove-on cooking</td></tr><tr><td><code>target_doneness_range</code></td><td>tuple</td><td>Food-type success band</td></tr><tr><td><code>total_time_sim_s / total_steps</code></td><td>float/int</td><td>Shared timing + eval MS/RC</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="measure_ingredient" data-suite="household" data-name="measure_ingredient Measure Ingredient">
      <header>
        <h3><code>measure_ingredient</code> — Measure Ingredient</h3>
        <div class="badges"><span class="pill env">env / eval only</span><span class="pill hh">household</span> <span class="pill cat">Household</span></div>
      </header>
      <p class="desc">Fill a marked jar with oil, then weigh it on a scale.</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>fill_level / scale_reading</code></td><td>float</td><td>Oil + weight signals</td></tr><tr><td><code>total_time_sim_s / total_steps</code></td><td>float/int</td><td>Shared timing + eval MS/RC</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="make_soup" data-suite="household" data-name="make_soup Make Soup">
      <header>
        <h3><code>make_soup</code> — Make Soup</h3>
        <div class="badges"><span class="pill env">env / eval only</span><span class="pill hh">household</span> <span class="pill cat">Household</span></div>
      </header>
      <p class="desc">Tip board veggies into a pot, then turn the stove on.</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>total_time_sim_s / total_steps</code></td><td>float/int</td><td>Shared timing + eval MS/RC</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="catch_cup" data-suite="household" data-name="catch_cup Catch Cup">
      <header>
        <h3><code>catch_cup</code> — Catch Cup</h3>
        <div class="badges"><span class="pill env">env / eval only</span><span class="pill hh">household</span> <span class="pill cat">Household</span></div>
      </header>
      <p class="desc">Push a pillow under a tipping mug so it lands softly.</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>total_time_sim_s / total_steps</code></td><td>float/int</td><td>Shared timing + eval MS/RC</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="catch_mouse_object_drop" data-suite="household" data-name="catch_mouse_object_drop Catch Mouse Object Drop">
      <header>
        <h3><code>catch_mouse_object_drop</code> — Catch Mouse Object Drop</h3>
        <div class="badges"><span class="pill env">env / eval only</span><span class="pill hh">household</span> <span class="pill cat">Household</span></div>
      </header>
      <p class="desc">Pillow-lined basket catches a shelf object knocked by a mouse.</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>total_time_sim_s / total_steps</code></td><td>float/int</td><td>Shared timing + eval MS/RC</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="stop_ball" data-suite="household" data-name="stop_ball Stop Ball">
      <header>
        <h3><code>stop_ball</code> — Stop Ball</h3>
        <div class="badges"><span class="pill env">env / eval only</span><span class="pill hh">household</span> <span class="pill cat">Household</span></div>
      </header>
      <p class="desc">Block a TT ball falling from a shelf before the table edge.</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>total_time_sim_s / total_steps</code></td><td>float/int</td><td>Shared timing + eval MS/RC</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="clean_table" data-suite="household" data-name="clean_table Clean Table">
      <header>
        <h3><code>clean_table</code> — Clean Table</h3>
        <div class="badges"><span class="pill env">env / eval only</span><span class="pill hh">household</span> <span class="pill cat">Household</span></div>
      </header>
      <p class="desc">Wipe a coffee spill before it reaches the laptop.</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>total_time_sim_s / total_steps</code></td><td>float/int</td><td>Shared timing + eval MS/RC</td></tr></tbody>
      </table>
    </section>

  <h2 id="extras">Household extras / aliases</h2>
  <p class="muted">Present in demos, <code>HOUSEHOLD_TASKS</code>, or configs but not in the household GUI list.</p>
  
    <section class="task" id="serve_dinner" data-suite="household" data-name="serve_dinner Serve Dinner">
      <header>
        <h3><code>serve_dinner</code> — Serve Dinner</h3>
        <div class="badges"><span class="pill env">env / eval only</span><span class="pill hh">household</span> <span class="pill cat">Household</span></div>
      </header>
      <p class="desc">Turn off a lit stove; tip meatballs from pan onto a plate. (in HOUSEHOLD_TASKS / demos)</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>total_time_sim_s / total_steps</code></td><td>float/int</td><td>Shared timing + eval MS/RC</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="make_soup_test" data-suite="household" data-name="make_soup_test Make Soup Test">
      <header>
        <h3><code>make_soup_test</code> — Make Soup Test</h3>
        <div class="badges"><span class="pill env">env / eval only</span><span class="pill hh">household</span> <span class="pill cat">Household</span></div>
      </header>
      <p class="desc">Alias of make_soup (legacy name; in HOUSEHOLD_TASKS).</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>total_time_sim_s / total_steps</code></td><td>float/int</td><td>Shared timing + eval MS/RC</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="catch_rolling_cup" data-suite="household" data-name="catch_rolling_cup Catch Rolling Cup">
      <header>
        <h3><code>catch_rolling_cup</code> — Catch Rolling Cup</h3>
        <div class="badges"><span class="pill env">env / eval only</span><span class="pill hh">household</span> <span class="pill cat">Household</span></div>
      </header>
      <p class="desc">Stop a rolling cup and stand it upright (office demos; not in HOUSEHOLD_TASKS frozenset).</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>total_time_sim_s / total_steps</code></td><td>float/int</td><td>Shared timing + eval MS/RC</td></tr></tbody>
      </table>
    </section>

    <section class="task" id="empty_bag" data-suite="household" data-name="empty_bag Empty Bag">
      <header>
        <h3><code>empty_bag</code> — Empty Bag</h3>
        <div class="badges"><span class="pill env">env / eval only</span><span class="pill hh">household</span> <span class="pill cat">Household</span></div>
      </header>
      <p class="desc">Tip a grocery bag and catch a rolling apple (KitchenL demos).</p>
      <table>
        <thead><tr><th>Metric</th><th>Type</th><th>Meaning</th></tr></thead>
        <tbody><tr><td><code>success / SR</code></td><td>bool / float</td><td>Binary success; aggregated as Success Rate</td></tr><tr><td><code>manipulation_score (MS)</code></td><td>float</td><td>Route completion × penalties (eval_metrics.py)</td></tr><tr><td><code>route_completion (RC)</code></td><td>float</td><td>0–100 route progress</td></tr><tr><td><code>total_time_sim_s / total_steps</code></td><td>float/int</td><td>Shared timing + eval MS/RC</td></tr></tbody>
      </table>
    </section>

  <p class="muted" style="margin-top:2rem">
    Generated from GUI task lists, <code>HOUSEHOLD_TASKS</code>, <code>script/eval_metrics.py</code>,
    and <code>docs/metrics/metric_detail_approval.canvas.tsx</code>.
  </p>
</div>

<script>
const q = document.getElementById('q');
const radios = document.querySelectorAll('input[name=suite]');
function apply() {
  const query = (q.value || '').trim().toLowerCase();
  const suite = document.querySelector('input[name=suite]:checked').value;
  document.querySelectorAll('section.task').forEach(sec => {
    const suiteOk = suite === 'all' || sec.dataset.suite === suite;
    const text = (sec.innerText || '').toLowerCase();
    const name = (sec.dataset.name || '').toLowerCase();
    const queryOk = !query || text.includes(query) || name.includes(query);
    sec.classList.toggle('hidden', !(suiteOk && queryOk));
  });
}
q.addEventListener('input', apply);
radios.forEach(r => r.addEventListener('change', apply));
</script>
</body>
</html>
