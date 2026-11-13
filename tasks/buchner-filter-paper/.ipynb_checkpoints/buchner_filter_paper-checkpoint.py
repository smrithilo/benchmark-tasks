"""On the station table, a vacuum-filtration bench.

- Filter paper (in front of the left arm): a white rectangular sheet, 4.5 cm front to back x 3.5 cm side to side and
  8 mm thick, lying flat and centred on a grey paper post (2 x 2 cm, 3 cm tall), overhanging it all round. Loose.
- Filter plate (on the midline, reachable by both arms): a porcelain Buchner-style filter plate, 5 cm front to back x
  3 cm side to side, its top 4 cm above the table, on a dark stand of the same width. The dark grey centre (3.5 x 2 cm)
  is the perforated area. A 5 mm lip runs along its front and back edges; the sheet is 5 mm wider than the plate, so
  it overhangs the plate's open sides by 2.5 mm. Fixed.
- Wash bottle (in front of the right arm): a pale blue squeeze bottle, 4 cm across and 8 cm tall, upright. Held
  upright over the funnel, it dispenses water onto whatever is on the plate. Loose.
- Sample beaker (in front of the right arm, further right): an amber beaker of slurry, 6 cm across and 7 cm tall.
  Loose; it must not be touched, since the sample goes in only after the funnel is prepared.

Order: the paper goes on the plate first, flat, covering every hole; only then is it wetted.

Run it:  inspect-robots run --embodiment yam_arena -E task=tasks/buchner-filter-paper/buchner_filter_paper.py \\
           --instruction "Prepare the Buchner funnel for filtration: ..." --scorer task_score ...
"""

import math
import os

import numpy as np

from yam_arena.demo import FINGER_CLOSED, FINGER_OPEN, Waypoint, pick_place_waypoints
from yam_arena.task_file import TABLE_TOP, Box, Cylinder

SCENE = "station"
TOP = TABLE_TOP[SCENE]

# ---------------------------------------------------------------- funnel (fixed: no mass)
FUNNEL_X, FUNNEL_Y = 0.25, 0.0
STAND_H = 0.035
PLATE_X, PLATE_Y, PLATE_T = 0.05, 0.03, 0.005  # plate between the lips (front to back), side to side, thickness
PLATE_TOP = TOP + STAND_H + PLATE_T  # 4 cm above the table
HOLES_X, HOLES_Y = 0.035, 0.02  # perforated area
RIM_T, RIM_H = 0.004, 0.005  # RIM_H: above the plate
RIM_TOP = PLATE_TOP + RIM_H

PORCELAIN = (0.93, 0.93, 0.90)
HOLES_GREY = (0.40, 0.40, 0.42)
STAND_GREY = (0.25, 0.25, 0.28)


def _funnel() -> list[Box]:
    """Lips only on the front and back edges: the fingers close along world y and pass down beside the open sides."""
    out_x = PLATE_X + 2 * RIM_T
    off_x = PLATE_X / 2 + RIM_T / 2
    z_lo = PLATE_TOP - PLATE_T
    rim_z, rim_h = (z_lo + RIM_TOP) / 2, RIM_TOP - z_lo
    return [
        Box("funnel_stand", pos=(FUNNEL_X, FUNNEL_Y, TOP + STAND_H / 2), size=(out_x, PLATE_Y, STAND_H), color=STAND_GREY),
        Box("funnel_plate", pos=(FUNNEL_X, FUNNEL_Y, PLATE_TOP - PLATE_T / 2), size=(PLATE_X, PLATE_Y, PLATE_T), color=PORCELAIN),
        # the perforated area, 0.15 mm proud of the plate so it shows; the paper rests on it
        Box("perforated_area", pos=(FUNNEL_X, FUNNEL_Y, PLATE_TOP - PLATE_T / 2 + 0.00015), size=(HOLES_X, HOLES_Y, PLATE_T), color=HOLES_GREY),
        Box("funnel_rim_back", pos=(FUNNEL_X - off_x, FUNNEL_Y, rim_z), size=(RIM_T, PLATE_Y, rim_h), color=PORCELAIN),
        Box("funnel_rim_front", pos=(FUNNEL_X + off_x, FUNNEL_Y, rim_z), size=(RIM_T, PLATE_Y, rim_h), color=PORCELAIN),
    ]


# ---------------------------------------------------------------- loose objects
POST_TOP = TOP + 0.03
POST = Box("paper_post", pos=(0.22, 0.30, TOP + 0.015), size=(0.02, 0.02, 0.03), color=(0.55, 0.55, 0.55))
PAPER = Box("filter_paper", pos=(0.22, 0.30, POST_TOP + 0.004 + 0.0005), size=(0.045, 0.035, 0.008), mass=0.03, color=(0.98, 0.98, 0.98))
BOTTLE = Cylinder("wash_bottle", pos=(0.25, -0.25, TOP + 0.04 + 0.0005), radius=0.02, height=0.08, mass=0.08, color=(0.80, 0.88, 0.95))
BEAKER = Cylinder("sample_beaker", pos=(0.17, -0.37, TOP + 0.035 + 0.0005), radius=0.03, height=0.07, mass=0.2, color=(0.85, 0.55, 0.15))

OBJECTS = [*_funnel(), POST, PAPER, BOTTLE, BEAKER]
FREE = {o.name: o for o in (PAPER, BOTTLE, BEAKER)}
PAPER_T = PAPER.size[2]

# ---------------------------------------------------------------- scoring
CENTRE_TOL = 0.004  # paper centre within 4 mm of the funnel axis: covers every hole (5 mm of sheet past the holes each side)
FLAT_DEG = 3.0
SEAT_Z_TOL = 0.002
WET_STEPS = 20  # 1 s at 20 Hz (-E control_hz=20)


def _tilt_deg(s) -> float:
    x, y = float(s.quat_xyzw[0]), float(s.quat_xyzw[1])
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))


def _off_axis(s) -> float:
    return math.hypot(float(s.pos[0]) - FUNNEL_X, float(s.pos[1]) - FUNNEL_Y)


def seated(paper) -> bool:
    """Flat on the plate, centred, at rest."""
    return (
        _off_axis(paper) <= CENTRE_TOL
        and _tilt_deg(paper) <= FLAT_DEG
        and abs(float(paper.pos[2]) - (PLATE_TOP + PAPER_T / 2)) <= SEAT_Z_TOL
        and paper.resting()
    )


def in_funnel(paper) -> bool:
    return _off_axis(paper) < PLATE_Y / 2 + 0.005 and float(paper.low[2]) < RIM_TOP + 0.01


def dispensing(bottle) -> bool:
    """Upright over the funnel, its bottom 0.5 to 6 cm above the rim: water would land on the plate."""
    z = float(bottle.low[2])
    return _off_axis(bottle) <= 0.02 and _tilt_deg(bottle) <= 15.0 and RIM_TOP + 0.005 <= z <= RIM_TOP + 0.06


def bottle_put_back(bottle) -> bool:
    """Upright, at rest, standing on the table clear of the funnel."""
    return (
        _tilt_deg(bottle) <= 10.0
        and bottle.resting()
        and float(bottle.low[2]) <= TOP + 0.005
        and _off_axis(bottle) > PLATE_X / 2 + RIM_T + BOTTLE.radius + 0.01
    )


def sample_touched(beaker) -> bool:
    """Lifted over the funnel (the sample went in) or tipped over (spilled)."""
    over = _off_axis(beaker) < PLATE_X / 2 + BEAKER.radius and float(beaker.low[2]) > TOP + 0.01
    return over or _tilt_deg(beaker) > 30.0


# score() is called after every control step; it keeps what happened so far in this episode.
_state: dict = {}


def _reset_state() -> None:
    _state.clear()
    _state.update(wet_run=0, wet_done=False, wet_early=False, sample=False, last={})


def _new_episode(objects) -> bool:
    """A reset puts the loose objects back where they started in one step; nothing moves that far that fast."""
    last = _state.get("last")
    if not last:
        return True
    for name, spec in FREE.items():
        p = np.asarray(objects[name].pos, dtype=float)
        jump = np.linalg.norm(p - last[name])
        home = math.hypot(p[0] - spec.pos[0], p[1] - spec.pos[1])
        if jump > 0.05 and home < 0.035:
            return True
    return False


def _update(objects) -> None:
    if _new_episode(objects):
        _reset_state()
    paper, bottle, beaker = objects["filter_paper"], objects["wash_bottle"], objects["sample_beaker"]
    if dispensing(bottle):
        if seated(paper):
            _state["wet_run"] += 1
            if _state["wet_run"] >= WET_STEPS:
                _state["wet_done"] = True
        elif not _state["wet_done"]:
            _state["wet_early"] = True  # water on the bare plate or on a paper that isn't seated
            _state["wet_run"] = 0
    else:
        _state["wet_run"] = 0
    if sample_touched(beaker):
        _state["sample"] = True
    _state["last"] = {n: np.asarray(objects[n].pos, dtype=float).copy() for n in FREE}


DEBUG = os.environ.get("BUCHNER_DEBUG") == "1"  # BUCHNER_DEBUG=1: print the paper's pose once a second


def _debug(objects) -> None:
    _state["n"] = _state.get("n", 0) + 1
    if _state["n"] % 20:
        return
    p, b = objects["filter_paper"], objects["wash_bottle"]
    print(
        f"[buchner t={_state['n'] / 20:5.1f}s] paper off-axis {_off_axis(p) * 1000:5.1f} mm (<= 4), tilt {_tilt_deg(p):5.1f} deg (<= 3), "
        f"above seat {(float(p.pos[2]) - PLATE_TOP - PAPER_T / 2) * 1000:6.1f} mm (|.| <= 2), resting {p.resting()}, "
        f"seated {seated(p)} | bottle dispensing {dispensing(b)} tilt {_tilt_deg(b):4.1f} | wet_done {_state['wet_done']} wet_early {_state['wet_early']}",
        flush=True,
    )


def score(objects) -> float:
    """0.4 paper seated (0.15 if inside the funnel but not seated); +0.4 wetted after it was seated, while it is
    still seated; +0.2 wash bottle put back upright after wetting. Wetting before the paper is seated forfeits
    both wetting credits; touching the sample scores 0."""
    _update(objects)
    if DEBUG:
        _debug(objects)
    if _state["sample"]:
        return 0.0
    paper, bottle = objects["filter_paper"], objects["wash_bottle"]
    is_seated = seated(paper)
    total = 0.4 if is_seated else (0.15 if in_funnel(paper) else 0.0)
    wet_ok = _state["wet_done"] and not _state["wet_early"]
    if wet_ok and is_seated:
        total += 0.4
    if wet_ok and bottle_put_back(bottle):
        total += 0.2
    return round(min(1.0, total), 3)


def success(objects) -> bool:
    """Full score: seated, wetted after seating, bottle put back, sample untouched; must hold 1 s. Reads the
    episode record that score() keeps and doesn't update it, so each step is counted once."""
    if not _state or _state["sample"] or _state["wet_early"] or not _state["wet_done"]:
        return False
    return seated(objects["filter_paper"]) and bottle_put_back(objects["wash_bottle"])


def distance(objects) -> float:
    return _off_axis(objects["filter_paper"])


# ---------------------------------------------------------------- demo
TRAVEL_Z = TOP + 0.14  # as in the example task: straight-down grips over the midline stay within joint4's limit
# Paper, as the example grasps its cube: TCP at the sheet's centre, the finger tips passing down beside the post.
# The fingers close along world y across the sheet's 3.5 cm side (a 4.5 cm sheet was knocked off by the fingers).
PAPER_PICK_Z = PAPER.pos[2]
# Set down like the example: the sheet's centre 3 mm above where it rests, the finger tips beside the plate's open sides.
PAPER_RELEASE_Z = PLATE_TOP + PAPER_T / 2 + 0.003
# Bottle: as in the example, grasped at its centre, so its weight sits between the pads.
BOTTLE_POUR_Z = RIM_TOP + 0.02 + BOTTLE.height / 2  # bottle bottom 2 cm above the rim
RIGHT_WAIT = 9.0  # the right arm reaches the funnel after the left arm has set the paper down and left
LET_GO_S = 1.0  # fingers stay open and still this long before rising, so the sheet settles off the pads


def demo():
    """Left arm: paper from its post onto the plate, then out of the way. Right arm: wait, take the wash bottle
    over the seated paper, hold it there 2.5 s (wetting), put it back where it was."""
    left = pick_place_waypoints((PAPER.pos[0], PAPER.pos[1], PAPER_PICK_Z), (FUNNEL_X, FUNNEL_Y, PAPER_RELEASE_Z), travel_z=TRAVEL_Z)
    k = next(i for i, w in enumerate(left) if w.label == "release")
    left.insert(k + 1, Waypoint(LET_GO_S, left[k].pos, FINGER_OPEN, "let go", left[k].yaw))
    left.append(Waypoint(1.5, (PAPER.pos[0], PAPER.pos[1], TRAVEL_Z), FINGER_OPEN, "clear of the funnel"))
    bx, by, bz = BOTTLE.pos
    right = [
        Waypoint(RIGHT_WAIT, (bx, by, TRAVEL_Z), FINGER_OPEN, "wait above the bottle"),
        Waypoint(1.5, (bx, by, bz), FINGER_OPEN, "descend"),
        Waypoint(0.8, (bx, by, bz), FINGER_CLOSED, "grip"),
        Waypoint(1.2, (bx, by, TRAVEL_Z), FINGER_CLOSED, "lift"),
        Waypoint(1.8, (FUNNEL_X, FUNNEL_Y, TRAVEL_Z), FINGER_CLOSED, "carry to the funnel"),
        Waypoint(1.2, (FUNNEL_X, FUNNEL_Y, BOTTLE_POUR_Z), FINGER_CLOSED, "lower over the paper"),
        Waypoint(2.5, (FUNNEL_X, FUNNEL_Y, BOTTLE_POUR_Z), FINGER_CLOSED, "squeeze: wet the paper"),
        Waypoint(1.2, (FUNNEL_X, FUNNEL_Y, TRAVEL_Z), FINGER_CLOSED, "lift"),
        Waypoint(1.8, (bx, by, TRAVEL_Z), FINGER_CLOSED, "carry back"),
        Waypoint(1.2, (bx, by, bz + 0.005), FINGER_CLOSED, "set down"),
        Waypoint(0.6, (bx, by, bz + 0.005), FINGER_OPEN, "release"),
        Waypoint(1.0, (bx, by, TRAVEL_Z), FINGER_OPEN, "retreat"),
    ]
    return {"left": left, "right": right}