# -*- coding: utf-8 -*-
"""隔离测试时间轮"""
import os
import sys
import threading
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
    fired.append(task.name)


w = HierarchicalTimeWheel(on_due=on_due, tick_interval=0.2)
t = T(name="a", callback=None)
t.created_at = time.time()
t.run_at = t.created_at + 0.5
w.add(t)
print("entries:", len(w._entries), "slot1 len:", len(w.seconds.slots[1]))
w.start()
time.sleep(2)
print("fired:", fired)
print("entries after:", len(w._entries))
for th in threading.enumerate():
    print("thread:", th.name)
w.stop()
