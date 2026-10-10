# -*- coding: utf-8 -*-
"""
调度框架全特性验证脚本
=====================
覆盖：超时(stop_event) / 钩子(start/end/timeout/cancel) / 五类任务 /
信号量并发控制 / 4 层时间轮 / >7 天磁盘缓存 + 扫描回捞 / JSON 元信息持久化重启还原。

运行（在项目根目录 C:\\Users\\tyt\\PyCharmMiscProject 下）::

    python -m com.robot.scheduler.demo

所有回调/钩子均为模块级函数（可被 import 还原，用于持久化验证）。
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
import time

# 使 demo 可从项目根目录独立运行
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from com.robot.scheduler import (                       # noqa: E402
    Scheduler, DelayTask, PeriodicTask, ScheduledTask,
    ScheduledPeriodicTask, DelayPeriodicTask, TaskState,
)

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    mark = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail else ""))


# ----------------------------------------------------------------------
# 模块级回调（可 import，供持久化还原）
# ----------------------------------------------------------------------
_count = 0
_count_lock = threading.Lock()
_trace: list = []
_trace_lock = threading.Lock()


def cb_ping(ctx):
    print(f"    [cb_ping] {ctx.task.name} 执行 | "
          f"stop_event={ctx.stop_event.is_set()} semaphore={ctx.semaphore is not None}")
    return "pong"


def cb_coop(ctx):
    """协作式超时回调：周期检查 ctx.stop_event。"""
    for _ in range(50):
        if ctx.stop_event.is_set():
            print(f"    [cb_coop] {ctx.task.name} 检测到 stop_event，协作退出")
            return "coop-stopped"
        time.sleep(0.05)
    return "coop-done"


def cb_slow(ctx):
    """不协作回调：闷头睡 3 秒，超时只能标记状态、无法强杀。"""
    time.sleep(3)
    return "slow-done"


def cb_count(ctx):
    global _count
    with _count_lock:
        _count += 1
        n = _count
    print(f"    [cb_count] {ctx.task.name} 第 {n} 次执行")
    return n


def cb_trace(ctx):
    """记录起止时间用于并发验证。"""
    with _trace_lock:
        _trace.append((ctx.task.name, "start", time.time()))
    time.sleep(0.5)
    with _trace_lock:
        _trace.append((ctx.task.name, "end", time.time()))
    return "traced"


# ----------------------------------------------------------------------
# 模块级钩子
# ----------------------------------------------------------------------
def hook_start(ctx):
    print(f"    [hook:start]   {ctx.task.name}")


def hook_end(ctx):
    print(f"    [hook:end]     {ctx.task.name} -> {ctx.task.state.value}")


def hook_timeout(ctx):
    print(f"    [hook:timeout] {ctx.task.name} 超时, stop_event={ctx.stop_event.is_set()}")


def hook_cancel(ctx):
    print(f"    [hook:cancel]  {ctx.task.name} 已取消")


def hook_error(ctx):
    print(f"    [hook:error]   {ctx.task.name} err={ctx.error}")


HOOKS = {"start": hook_start, "end": hook_end,
         "timeout": hook_timeout, "cancel": hook_cancel, "error": hook_error}


# ----------------------------------------------------------------------
def main() -> None:
    global _count
    runtime = tempfile.mkdtemp(prefix="sched_demo_")
    disk_dir = os.path.join(runtime, "disk")
    state_file = os.path.join(runtime, "state.json")
    sched = Scheduler(workers=8, runners=8, max_concurrency=1,
                      tick_interval=0.2, scan_interval=86400.0,
                      disk_dir=disk_dir, state_file=state_file)
    sched.start()
    print("=" * 70)
    print("1) 超时机制（stop_event）+ 超时钩子")
    print("=" * 70)
    # 协作式超时
    t1 = sched.submit(DelayTask("协作超时", cb_coop, delay=0.4, timeout=1.0, hooks=HOOKS))
    time.sleep(1.8)
    s1 = sched.get_task(t1).state
    check("协作式回调在 stop_event 置位后退出 → TIMEOUT",
          s1 is TaskState.TIMEOUT, f"state={s1.value}")
    # 非协作超时：runner 及时返回 TIMEOUT，残留线程后台收尾
    t2 = sched.submit(DelayTask("硬超时", cb_slow, delay=0.4, timeout=0.7, hooks=HOOKS))
    time.sleep(1.6)
    s2 = sched.get_task(t2).state
    check("不协作回调：超时后 runner 立即标记 TIMEOUT",
          s2 is TaskState.TIMEOUT, f"state={s2.value}")

    print("=" * 70)
    print("2) 定时任务类 ScheduledTask（绝对时间点执行一次）")
    print("=" * 70)
    t3 = sched.submit(ScheduledTask(
        "定时一次", cb_ping, run_at=time.time() + 1.0, timeout=5, hooks=HOOKS))
    time.sleep(1.8)
    s3 = sched.get_task(t3).state
    check("定时任务到点执行 → SUCCESS", s3 is TaskState.SUCCESS, f"state={s3.value}")
    check("一次性任务不再调度(next_run_at=None)",
          sched.get_task(t3).next_run_at() is None)

    print("=" * 70)
    print("3) 周期任务类 PeriodicTask（固定间隔重复 + cancel 钩子）")
    print("=" * 70)
    _count = 0
    t4 = sched.submit(PeriodicTask("周期计数", cb_count, interval=0.8,
                                   start_after=0.3, timeout=3, hooks=HOOKS))
    time.sleep(2.2)
    sched.cancel(t4)
    time.sleep(0.3)
    check("周期任务运行约 2 秒执行 ≥2 次", _count >= 2, f"count={_count}")
    check("取消后状态为 CANCELLED",
          sched.get_task(t4).state is TaskState.CANCELLED,
          sched.get_task(t4).state.value)

    print("=" * 70)
    print("4) 定时周期任务类 ScheduledPeriodicTask（cron 表达式）")
    print("=" * 70)
    t5 = sched.submit(ScheduledPeriodicTask("每日08:30", cb_ping,
                                            cron="30 8 * * *", timeout=5,
                                            hooks=HOOKS))
    import datetime as _dt
    nxt = _dt.datetime.fromtimestamp(sched.get_task(t5).run_at)
    check("cron 下一次触发为当天/次日 08:30", nxt.hour == 8 and nxt.minute == 30,
          str(nxt))
    # 手动把触发点拉到 1 秒后验证真实执行
    task5 = sched.get_task(t5)
    sched.wheel.cancel(t5)
    task5.run_at = time.time() + 1.0
    sched.submit(task5)
    time.sleep(1.8)
    # 周期任务执行后会自动重新排程到下一个 cron 时刻（明天 08:30）
    rearmed = task5.run_at is not None and task5.run_at > time.time() + 3600
    check("cron 任务到点执行并重新排程", rearmed,
          f"next_run_at={_dt.datetime.fromtimestamp(task5.run_at)}")
    sched.cancel(t5)

    print("=" * 70)
    print("5) 延时周期任务类 DelayPeriodicTask（delay 后首次，再按间隔重复）")
    print("=" * 70)
    _count = 0
    t6 = sched.submit(DelayPeriodicTask("延时周期", cb_count, delay=0.4,
                                        interval=0.8, timeout=3, hooks=HOOKS))
    time.sleep(2.4)
    sched.cancel(t6)
    check("延时周期任务 ≥2 次执行", _count >= 2, f"count={_count}")

    print("=" * 70)
    print("6) 信号量并发控制（max_concurrency=1 串行执行）")
    print("=" * 70)
    with _trace_lock:
        _trace.clear()
    ta = sched.submit(DelayTask("并发A", cb_trace, delay=0.3, timeout=5))
    tb = sched.submit(DelayTask("并发B", cb_trace, delay=0.3, timeout=5))
    time.sleep(2.0)
    with _trace_lock:
        starts = sorted([t for _, op, t in _trace if op == "start"])
        ends = sorted([t for _, op, t in _trace if op == "end"])
    serial = len(starts) == 2 and len(ends) == 2 and ends[0] <= starts[1] + 0.05
    check("两个任务未重叠执行（信号量串行）", serial,
          f"start={starts} end={ends}")
    check("两个任务均已 SUCCESS",
          sched.get_task(ta).state is TaskState.SUCCESS and
          sched.get_task(tb).state is TaskState.SUCCESS)

    print("=" * 70)
    print("7) >7 天任务：磁盘缓存 → 扫描回捞 → 时间轮")
    print("=" * 70)
    t7 = sched.submit(DelayTask("远期任务8天", cb_ping, delay=8 * 86400,
                                timeout=5, hooks=HOOKS))
    sum_before = sched.summary()
    check("8 天后任务进入磁盘缓存", sum_before["disk_cached"] == 1,
          f"summary={sum_before}")
    # 模拟时间推移：任务进入 7 天窗口（剩余 2 秒），更新磁盘元信息
    task7 = sched.get_task(t7)
    task7.run_at = time.time() + 2.0
    sched.disk.save(task7)
    # 模拟扫描任务：提前一天定时检测，回捞进入窗口的任务
    due = sched.disk.scan_due(7 * 86400)
    check("扫描回捞进入 7 天窗口的任务", len(due) == 1, f"due={len(due)}")
    if due:
        sched.submit(due[0])
    time.sleep(0.3)
    check("回捞任务已进入时间轮",
          sched.wheel.count() >= 1, f"wheel_pending={sched.wheel.count()}")
    time.sleep(2.2)
    check("回捞任务到期执行 → SUCCESS",
          sched.get_task(t7).state is TaskState.SUCCESS,
          sched.get_task(t7).state.value)
    check("磁盘缓存已清理", sched.summary()["disk_cached"] == 0,
          str(sched.summary()))

    print("=" * 70)
    print("8) 内存任务 JSON 元信息持久化 → 重启还原")
    print("=" * 70)
    t8 = sched.submit(DelayTask("重启还原任务", cb_ping, delay=3.0, timeout=5,
                                hooks={"start": hook_start, "end": hook_end}))
    saved = sched.save_state()
    check("状态落盘(JSON 元信息)", saved >= 1, f"saved={saved}")
    # 模拟进程重启：新调度器加载同一份元信息
    sched.shutdown()
    sched2 = Scheduler(workers=8, runners=8, max_concurrency=2,
                       tick_interval=0.2, scan_interval=86400.0,
                       disk_dir=disk_dir, state_file=state_file)
    sched2.start()
    loaded = sched2.load_state()
    rt = sched2.get_task(t8)
    check("重启后还原任务(同一 task_id)", loaded >= 1 and rt is not None,
          f"loaded={loaded}")
    if rt is not None:
        check("还原任务保留 run_at / 回调 / 超时",
              rt.run_at is not None and rt.callback is cb_ping and rt.timeout == 5,
              f"run_at={rt.run_at}")
        time.sleep(3.5)
        check("还原任务重新入队并执行 → SUCCESS",
              rt.state is TaskState.SUCCESS, rt.state.value)

    print("=" * 70)
    print(f"验证结果：{PASS} 通过 / {FAIL} 失败")
    print("=" * 70)
    sched2.shutdown()
    shutil.rmtree(runtime, ignore_errors=True)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
