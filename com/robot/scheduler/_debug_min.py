# -*- coding: utf-8 -*-
"""最小复现：时间轮是否触发"""
import logging
import os
import sys
import tempfile
import time

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)s %(message)s")
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from com.robot.scheduler import Scheduler, DelayTask

rt = tempfile.mkdtemp(prefix="dbg_")
s = Scheduler(workers=4, runners=4, max_concurrency=2, tick_interval=0.2,
              disk_dir=os.path.join(rt, "disk"), state_file=os.path.join(rt, "s.json"))
print("before start, wheel._running =", s.wheel._running)
s.start()
print("after start, wheel._running =", s.wheel._running)


def cb(ctx):
    print(">>> CALLBACK RAN <<<", ctx.task.name)
    return 1


t = s.submit(DelayTask("x", cb, delay=0.5, timeout=5))
print("submitted", t, "run_at in", s.get_task(t).run_at - time.time(), "s")
time.sleep(2.0)
print("state:", s.get_task(t).state)
print("wheel count:", s.wheel.count())
s.shutdown()
import shutil
shutil.rmtree(rt, ignore_errors=True)
