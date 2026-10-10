# -*- coding: utf-8 -*-
"""
com.robot.scheduler —— 机器人任务调度框架
========================================
五类任务（每类一个文件） + 统一抽象层 + 调度引擎：

* ``base.py``                  抽象层：TaskState / TaskContext / BaseTask（超时·钩子·持久化骨架）
* ``scheduled_task.py``        定时任务类（绝对时间点执行一次）
* ``periodic_task.py``         周期任务类（固定间隔重复）
* ``delay_task.py``            延时任务类（延迟一次）
* ``scheduled_periodic_task.py`` 定时周期任务类（cron 表达式）
* ``delay_periodic_task.py``   延时周期任务类（延迟后按间隔重复）
* ``time_wheel.py``            四层时间轮（天/时/分/秒），处理 7 天以内任务
* ``disk_store.py``            >7 天任务磁盘缓存 + 索引 + 提前一天扫描回捞
* ``task_store.py``            内存任务 JSON 元信息持久化 / 重启还原工厂
* ``scheduler.py``             调度器：线程池 + 信号量 + 时间轮 + 扫描器 + 持久化

快速开始::

    from com.robot.scheduler import Scheduler, DelayTask

    sched = Scheduler(workers=8, max_concurrency=4)
    sched.start()
    sched.submit(DelayTask("demo", my_callback, delay=2, timeout=5))
"""
from .base import BaseTask, TaskContext, TaskState, to_epoch
from .delay_periodic_task import DelayPeriodicTask
from .delay_task import DelayTask
from .disk_store import DiskTaskStore
from .periodic_task import PeriodicTask
from .scheduled_periodic_task import ScheduledPeriodicTask
from .scheduled_task import ScheduledTask
from .scheduler import Scheduler
from .task_store import TaskStore, restore_task
from .time_wheel import HierarchicalTimeWheel

__all__ = [
    "Scheduler",
    "BaseTask",
    "TaskState",
    "TaskContext",
    "ScheduledTask",
    "PeriodicTask",
    "DelayTask",
    "ScheduledPeriodicTask",
    "DelayPeriodicTask",
    "HierarchicalTimeWheel",
    "DiskTaskStore",
    "TaskStore",
    "restore_task",
    "to_epoch",
]
