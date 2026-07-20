# marble_shelf_maze demo versions

Each file is a side-by-side video: left = `head_camera` (current/overhead view), right = `front_camera` (robot's front view).

- **v1** (`marble_shelf_maze_v1_demo.mp4`): baseline settings -- shelves raised ~10cm, shelf thickness (z) halved to 0.00625m, both glass sheets transparent (open-frame), correct tilt direction with turn-arrow decals on the buttons. `natural_ball_stop` and `smooth_tilt` both off (default): marble snaps to shelf center on landing, shelf tilts snap to 45deg over a fixed 0.5s.
- **v2** (`marble_shelf_maze_v2_demo.mp4`): same geometry as v1, but with `natural_ball_stop: true` and `smooth_tilt: true` enabled -- the marble settles wherever real friction/gravity leave it after landing (no re-centering snap), and each button press produces a slow, visibly gradual 0->45deg tilt (30 deg/s) instead of an instant sweep.
- **v3** (`marble_shelf_maze_v3_demo.mp4`): 3-episode montage with `n_shelves_min: 3` / `n_shelves_max: 6` -- the number of shelves (and thus the length + left/right pattern of the required button-press combination) is drawn randomly per episode instead of being fixed at 4. Episode 0 = 3 shelves, episode 1 = 4 shelves, episode 2 = 6 shelves in this recording; the bowl always sits under whichever shelf ends up on the bottom, so a deeper stack needs a longer combo and a shallower one needs a shorter one. `n_shelves` (2-6) also works as a plain fixed value if randomization isn't wanted.

Add new versions here (`v4`, `v5`, ...) with a one-line description of what changed, each time a new demo is generated.
