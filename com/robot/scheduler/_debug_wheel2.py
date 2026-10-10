# -*- coding: utf-8 -*-
"""隔离测试时间轮 advance"""
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
    fired.append(task.name)


w = HierarchicalTimeWheel(on_due=on_due, tick_interval=0.2)

orig_advance = w.advance


def traced_advance():
    print("advance called, seconds.cursor=", w.seconds.cursor,
          "slot0_len=", len(w.seconds.slots[0]),
          "slot1_len=", len(w.seconds.slots[1]),
          "slot2_len=", len(w.seconds.slots[2]))
    orig_advance()
    print("  after: cursor=", w.seconds.cursor)


w.advance = traced_advance

t = T(name="a", callback=None)
t.created_at = time.time()
t.run_at = t.created_at + 0.5
w.add(t)
print("inserted. cursor=", w.seconds.cursor,
      "slot0=", len(w.seconds.slots[0]),
      "slot1=", len(w.seconds.slots[1]),
      "slot2=", len(w.seconds.slots[2]),
      "entries=", len(w._entries))
w.start()
time.sleep(2)
print("fired:", fired)
w.stop()
