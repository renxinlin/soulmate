# -*- coding: utf-8 -*-
"""
内存任务持久化 TaskStore（JSON 结构元信息）
==========================================
将内存中的任务（时间轮 + 待执行 + 执行中）以 **JSON 结构元信息** 落盘，
重启后再加载还原。只保存结构元信息，因此：

* 回调 / 钩子必须以 "模块.限定名" 形式可被 import（模块级函数/类方法），
  lambda、闭包等无法持久化（load 时会被标记为不可恢复并跳过）。
* 只保存 PENDING / RUNNING 的任务；已结束（终态）任务不落盘。

``restore_task(data)`` 为全局还原工厂，供 TaskStore 与 DiskTaskStore 共用。
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from .base import BaseTask, TaskState
from .delay_task import DelayTask
from .delay_periodic_task import DelayPeriodicTask
from .periodic_task import PeriodicTask
from .scheduled_periodic_task import ScheduledPeriodicTask
from .scheduled_task import ScheduledTask

logger = logging.getLogger(__name__)

_TASK_CLASSES = {
    cls.__name__: cls
    for cls in (ScheduledTask, PeriodicTask, DelayTask,
                ScheduledPeriodicTask, DelayPeriodicTask)
}


def restore_task(data: Dict[str, Any]) -> BaseTask:
    """按 JSON 元信息还原任务实例。"""
    cls = _TASK_CLASSES.get(data.get("type", ""))
    if cls is None:
        raise ValueError(f"未知任务类型: {data.get('type')!r}")
    return cls.from_dict(data)


class TaskStore:
    """内存任务元信息的 JSON 持久化存储。"""

    def __init__(self, path: str) -> None:
        self.path = path

    # ------------------------------------------------------------------
    def save(self, tasks: Dict[str, BaseTask]) -> int:
        """把任务元信息写入 JSON（原子写：临时文件 + 改名）。"""
        items: List[Dict[str, Any]] = []
        for task in tasks.values():
            if task.state in (TaskState.PENDING, TaskState.RUNNING):
                try:
                    items.append(task.to_dict())
                except Exception:                       # noqa: BLE001
                    logger.warning("任务 %s 序列化失败，跳过", task.task_id)
        directory = os.path.dirname(self.path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"version": 1, "tasks": items}, fh,
                          ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except Exception:
            os.unlink(tmp)
            raise
        return len(items)

    # ------------------------------------------------------------------
    def load(self) -> List[Dict[str, Any]]:
        """读取 JSON，返回原始元信息列表（不还原对象）。"""
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload.get("tasks", [])

    # ------------------------------------------------------------------
    def restore_all(self) -> List[BaseTask]:
        """读取并还原全部任务；单个任务还原失败不阻塞整体。"""
        tasks: List[BaseTask] = []
        for data in self.load():
            try:
                tasks.append(restore_task(data))
            except Exception as exc:                    # noqa: BLE001
                logger.warning("任务元信息还原失败(%s): %s，已跳过",
                               data.get("task_id"), exc)
        return tasks

    # ------------------------------------------------------------------
    def clear(self) -> None:
        if os.path.exists(self.path):
            os.remove(self.path)
