# Seat and wet a filter-paper disk in a Buchner funnel before filtration

The whole task (objects, scoring and scripted demo) is one task file for the `yam_arena` embodiment:
`buchner_filter_paper.py`.

## Offline check (no GPU, a few seconds)

```bash
python tasks/buchner-filter-paper/check_demo.py
```

Checks the demo's IK, prints the gripper opening the task needs, and tests the scorer on made-up states.

## Play the scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=tasks/buchner-filter-paper/buchner_filter_paper.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Prepare the Buchner funnel for filtration: lay the filter paper flat on the funnel's perforated plate so it covers every hole, then wet it with the wash bottle so it seals against the plate, and put the wash bottle back down upright. Do not add the sample." \
  --scorer task_score \
  --max-steps 900 --store-frames --log-dir ~/yam-eval/buchner-demo
```

The left arm grasps the paper sheet at its centre, as the example task grasps its cube, and sets it down on the
plate with the finger tips passing beside the plate's open sides; the right arm then holds the wash bottle over the seated paper for 2.5 s and puts it back.
Set `BUCHNER_DEBUG=1` to print the paper's pose and the wetting state once a second.
About 23 s.

## Run a trial (Dream-1)

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=tasks/buchner-filter-paper/buchner_filter_paper.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Prepare the Buchner funnel for filtration: lay the filter paper flat on the funnel's perforated plate so it covers every hole, then wet it with the wash bottle so it seals against the plate, and put the wash bottle back down upright. Do not add the sample." \
  --scorer task_score \
  --max-steps 2400 --store-frames --log-dir ~/yam-eval/buchner-dream
```

## Scoring

`task_score` reports the task file's `score()` at the end of the run: 0.4 for the paper seated (centred
within 4 mm, flat within 3 degrees, resting on the plate; 0.15 if it is in the funnel but not seated),
+0.4 for wetting after seating, +0.2 for the wash bottle put back upright; 0 if the sample beaker is lifted
over the funnel or tipped. Wetting is counted over the episode (`score()` is called every control step), so
run at `-E control_hz=20`: the 1 s hold is 20 steps.

## Runtime and seeds

- Run limit: 2,400 steps (2 minutes at 20 Hz)
- Wall-clock per trial on one L40S: about 3 minutes, plus about 1 minute of start-up
- Fixed layout (`-E object_jitter=0`); no seed needed