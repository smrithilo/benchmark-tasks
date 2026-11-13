"""Offline checks, no Isaac Sim or GPU: the demo's IK, the gripper opening, and the scorer on made-up states.

    python tasks/buchner-filter-paper/check_demo.py
"""

import math
from pathlib import Path

import numpy as np

from yam_arena import station, task_file
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP
from yam_arena.demo import ArmPlan

TASK = Path(__file__).with_name("buchner_filter_paper.py")
DT = 1.0 / 20
task = task_file.load(TASK)
m = task.module
ok = True

# 1) the demo is within reach
for arm, waypoints in task.demo().items():
    plan = ArmPlan(list(waypoints), station.BASES[arm])
    for t in np.arange(0.0, plan.duration + DT, DT):
        plan.action7(float(t))
    good = plan.worst_ik_mm < 5.0 and plan.worst_ik_deg < 3.0
    ok &= good
    print(f"IK {arm:5s}: {plan.duration:5.1f} s, worst {plan.worst_ik_mm:.1f} mm / {plan.worst_ik_deg:.1f} deg  {'PASS' if good else 'FAIL'}")

# 2) the gripper opens wide enough (FINGER_OPEN is one finger's travel; the opening is printed both ways)
print(f"gripper: FINGER_OPEN={FINGER_OPEN} FINGER_CLOSED={FINGER_CLOSED} PAD_TIP_PAST_TCP={PAD_TIP_PAST_TCP}")
print(f"  needs to open past {m.PAPER.size[1] * 100:.1f} cm (paper) and {2 * m.BOTTLE.radius * 100:.1f} cm (bottle)")


# 3) the scorer
def st(spec, pos=None, tilt_deg=0.0):
    pos = np.array(spec.pos if pos is None else pos, dtype=float)
    if hasattr(spec, "size"):
        h = np.array(spec.size) / 2
    else:
        h = np.array([spec.radius, spec.radius, spec.height / 2])
    half = math.radians(tilt_deg) / 2
    return task_file.ObjectState(pos, np.array([math.sin(half), 0, 0, math.cos(half)]), np.zeros(3), np.zeros(3), pos - h, pos + h)


def objs(paper=None, bottle=None, beaker=None):
    return {"filter_paper": paper or st(m.PAPER), "wash_bottle": bottle or st(m.BOTTLE), "sample_beaker": beaker or st(m.BEAKER)}


SEATED = st(m.PAPER, (m.FUNNEL_X + 0.002, m.FUNNEL_Y, m.PLATE_TOP + m.PAPER_T / 2))
CROOKED = st(m.PAPER, (m.FUNNEL_X + 0.010, m.FUNNEL_Y, m.PLATE_TOP + m.PAPER_T / 2 + 0.006), tilt_deg=15)
POURING = st(m.BOTTLE, (m.FUNNEL_X, m.FUNNEL_Y, m.RIM_TOP + 0.02 + m.BOTTLE.height / 2))
LIFTED = st(m.BOTTLE, (m.BOTTLE.pos[0], m.BOTTLE.pos[1] + 0.03, m.TOP + 0.10))


def carry(paper, a, b, n=10):
    """The bottle moved from state a to state b over n steps, as an arm would (no reset-sized jumps)."""
    return [objs(paper, st(m.BOTTLE, a.pos + (b.pos - a.pos) * k / n)) for k in range(1, n + 1)]


HOME = st(m.BOTTLE)


def run(states):
    m._reset_state()
    for s in states:
        out = m.score(s)
    return out


cases = {
    "nothing done": ([objs()], 0.0),
    "paper seated, not wetted": ([objs(SEATED)], 0.4),
    "paper in the funnel, crooked": ([objs(CROOKED)], 0.15),
    "seated, wetted 1 s, bottle back": (
        [objs(SEATED)] + carry(SEATED, HOME, POURING) + [objs(SEATED, POURING)] * 20 + carry(SEATED, POURING, HOME), 1.0
    ),
    "seated, wetted 1 s, bottle left held": (
        [objs(SEATED)] + carry(SEATED, HOME, POURING) + [objs(SEATED, POURING)] * 20 + carry(SEATED, POURING, LIFTED), 0.8
    ),
    "wetted only 0.5 s": ([objs(SEATED)] + [objs(SEATED, POURING)] * 10 + [objs(SEATED)], 0.4),
    "wetted the bare plate first": ([objs(None, POURING)] * 5 + [objs(SEATED)] + [objs(SEATED, POURING)] * 20 + [objs(SEATED)], 0.4),
    "touched the sample": ([objs(SEATED)] + [objs(SEATED, POURING)] * 20 + [objs(SEATED, None, st(m.BEAKER, (m.FUNNEL_X, 0.0, m.TOP + 0.12)))], 0.0),
    "reset clears the last episode": (
        carry(None, HOME, POURING) + [objs(None, POURING)] * 5 + [objs(SEATED, POURING)]  # wetted too early, then reset:
        + [objs()] + [objs(SEATED)] + carry(SEATED, HOME, POURING) + [objs(SEATED, POURING)] * 20 + carry(SEATED, POURING, HOME),
        1.0,
    ),
}
for name, (states, want) in cases.items():
    got = run(states)
    good = abs(got - want) < 1e-6
    ok &= good
    print(f"score {name:38s} {got:.2f} (want {want:.2f})  {'PASS' if good else 'FAIL'}")
m._reset_state()
print("all checks passed" if ok else "SOME CHECKS FAILED")
