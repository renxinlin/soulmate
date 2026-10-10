# -*- coding: utf-8 -*-
"""
远程期任务磁盘缓存 DiskTaskStore
================================
调度规则：**触发时刻距今 > 7 天** 的任务先落到磁盘缓存（每条任务一个 JSON），
由扫描任务（默认每天一次，可配置）**提前检测**：当任务进入 7 天窗口
（``run_at <= now + horizon``）时，从磁盘加载并交还调度器进入内存时间轮。

结构::

    disk_dir/
    ├── index.json          # {task_id: run_at} 轻量索引，扫描只读索引
    └── <task_id>.json      # 任务 JSON 元信息（与 TaskStore 同一格式）

原子性：写文件采用 临时文件 + os.replace；加载成功后立即删除磁盘文件，
保证同一任务不会被重复加载（扫描线程 + 取消操作共用一把锁）。
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from typing import Dict, List, Optional

from .base import BaseTask
from .task_store import restore_task

logger = logging.getLogger(__name__)


class DiskTaskStore:
    """>7 天任务的磁盘缓存 + 索引 + 扫描加载。"""

    def __init__(self, directory: str) -> None:
        self.directory = directory
        self.index_path = os.path.join(directory, "index.json")
        self._lock = threading.RLock()
        os.makedirs(directory, exist_ok=True)

    # ------------------------------------------------------------------
    # 基础读写
    # ------------------------------------------------------------------
    def _read_index(self) -> Dict[str, float]:
        if not os.path.exists(self.index_path):
            return {}
        try:
            with open(self.index_path, "r", encoding="utf-8") as fh:
                return json.load(fh).get("tasks", {})
        except (json.JSONDecodeError, OSError):
            logger.exception("磁盘缓存索引损坏，按空索引处理")
            return {}

    def _write_index(self, index: Dict[str, float]) -> None:
        fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"version": 1, "tasks": index},
                          fh, ensure_ascii=False, indent=2)
            os.replace(tmp, self.index_path)
        except Exception:
            os.unlink(tmp)
            raise

    # ------------------------------------------------------------------
    def save(self, task: BaseTask) -> None:
        """把远期任务缓存到磁盘。"""
        path = os.path.join(self.directory, f"{task.task_id}.json")
        fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(task.to_dict(), fh, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
            with self._lock:
                index = self._read_index()
                index[task.task_id] = task.run_at
                self._write_index(index)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    # ------------------------------------------------------------------
    def delete(self, task_id: str) -> None:
        """删除磁盘缓存（任务进入内存轮或已取消时调用）。"""
        with self._lock:
            path = os.path.join(self.directory, f"{task_id}.json")
            if os.path.exists(path):
                os.remove(path)
            index = self._read_index()
            if task_id in index:
                index.pop(task_id)
                self._write_index(index)

    # ------------------------------------------------------------------
    def load_task(self, task_id: str) -> Optional[BaseTask]:
        """按 ID 加载并删除磁盘文件；返回还原的任务或 None。"""
        with self._lock:
            path = os.path.join(self.directory, f"{task_id}.json")
            if not os.path.exists(path):
                return None
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            task = restore_task(data)
            self.delete(task_id)
            return task

    # ------------------------------------------------------------------
    # 扫描：进入 7 天窗口的任务交还内存
    # ------------------------------------------------------------------
    def scan_due(self, horizon: float, now: Optional[float] = None) -> List[BaseTask]:
        """
        扫描磁盘缓存中 ``run_at <= now + horizon`` 的任务并加载。
        horizon 即 7 天窗口（秒）；返回的任务已从磁盘删除，可直接入时间轮。
        """
        now = now if now is not None else time.time()
        due: List[BaseTask] = []
        with self._lock:
            index = self._read_index()
            for task_id, run_at in index.items():
                if run_at <= now + horizon:
                    task = self.load_task(task_id)
                    if task is not None:
                        due.append(task)
        return due

    # ------------------------------------------------------------------
    def list_index(self) -> Dict[str, float]:
        """只读索引（task_id -> run_at），供运维/展示。"""
        with self._lock:
            return dict(self._read_index())
