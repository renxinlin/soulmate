# -*- coding: utf-8 -*-
"""深度诊断：任务到底去了哪个槽"""
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from com.robot.scheduler.base import BaseTask
from com.robot.scheduler.time_wheel import HierarchicalTimeWheel


class T(BaseTask):
    is_periodic = False

    def next_run_at(self, now=None):
        return None


fired = []


def on_due(task):
    print("FIRE:", task.name, flush=True)
    fired.append(task.name)


w = HierarchicalTimeWheel(on_due=on_due, tick_interval=0.2)

t = T(name="a", callback=None)
t.created_at = time.time()
t.run_at = t.created_at + 0.5
print("before add, cursor=", w.seconds.cursor, flush=True)
w.add(t)
print("after add, entries=", dict(w._entries), flush=True)
for i, slot in enumerate(w.seconds.slots[:5]):
    if slot:
        print(f"  slot{i}: {[x.name for x in slot]}", flush=True)
w.start()
time.sleep(1.0)
print("--- after 1s ---", flush=True)
for i, slot in enumerate(w.seconds.slots[:5]):
    if slot:
        print(f"  slot{i}: {[x.name for x in slot]}", flush=True)
print("entries=", dict(w._entries), flush=True)
print("fired=", fired, flush=True)
for layer in (w.days, w.hours, w.minutes, w.seconds):
    for i, slot in enumerate(layer.slots):
        for x in slot:
            if x.task_id == t.task_id:
                print(f"FOUND at layer={layer.unit} slot={i}", flush=True)
w.stop()
