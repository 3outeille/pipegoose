import random
from abc import ABC, abstractclassmethod
from dataclasses import dataclass
from queue import Queue
from time import sleep
from typing import Dict, List

import torch.distributed.rpc as rpc

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode

SessionId = str


class Handshake(ABC):
    def __init__(self, parallel_context: ParallelContext, parallel_mode: ParallelMode):
        self.parallel_context = parallel_context
        self.parallel_mode = parallel_mode

        self._session_id: str = None
        self._queue: Dict[SessionId, Queue] = {}
        self._ranks_confirmed: Dict[SessionId, List[int]] = set()

        self._data = None
        self._clock_idx = 0

    @abstractclassmethod
    def initiate(self):
        raise NotImplementedError

    def _generate_session_id(self) -> int:
        return random.randint(0, 9999)

    @abstractclassmethod
    def confirm(self):
        raise NotImplementedError

    @abstractclassmethod
    def is_initiated(self):
        raise NotImplementedError

    @abstractclassmethod
    def is_confirmed(self):
        raise NotImplementedError

    @abstractclassmethod
    def is_all_confirmed(self):
        raise NotImplementedError

    @abstractclassmethod
    def wait_until_all_confirmed(self):
        raise NotImplementedError


@dataclass
class SessionMetadata:
    src_rank: int
    parallel_mode: ParallelMode


class SchedulerHandshake(Handshake):
    NUM_SECONDS_IDLE = 0.5
    # TODO: make this configurable
    MASTER_RANK = 0

    progress = None
    clock_idx = 0

    # def pipeline_progress(self):
    #     return SchedulerHandshake._PIPELINE_TASKS

    # @pipeline_progress.setter
    # def pipeline_progress(self, data):
    #     SchedulerHandshake._recv_execution_plan(data)

    @staticmethod
    def _recv_execution_plan(data):
        SchedulerHandshake.progress = data

    def initiate(self, data):
        rank = self.parallel_context.get_local_rank(self.parallel_mode)
        world_size = self.parallel_context.get_world_size(self.parallel_mode)
        for dst in range(world_size):
            if dst == rank:
                continue

            worker_name = self.parallel_context.get_worker_name(dst)
            rpc.rpc_sync(to=worker_name, func=SchedulerHandshake._recv_execution_plan, args=(data,))

        # self.pipeline_progress = data
        SchedulerHandshake._recv_execution_plan(data)

    def confirm(self, task):
        # TODO: only non-scheduler ranks should confirm
        master_worker_name = self.parallel_context.get_worker_name(self.MASTER_RANK)
        rank = self.parallel_context.get_local_rank(self.parallel_mode)
        rpc.rpc_sync(master_worker_name, func=SchedulerHandshake._recv_confirm, args=(task, rank))

    @staticmethod
    def _recv_confirm(task, src):
        clock_idx = SchedulerHandshake.clock_idx
        progress = SchedulerHandshake.progress
        progress[clock_idx][task] = True

    def is_initiated(self) -> bool:
        return self.progress is not None

    def is_confirmed(self) -> bool:
        raise NotImplementedError

    def is_all_confirmed(self) -> bool:
        num_confirmed = len(self._ranks_confirmed[self.session_id])
        local_world_size = self.parallel_context.get_world_size(self.parallel_mode)
        return num_confirmed == local_world_size

    def wait_until_all_confirmed(self):
        if self.parallel_context.is_first_rank() is True:
            while True:
                while self._queue.empty() is True:
                    sleep(self.NUM_SECONDS_IDLE)

                new_rank_confirmed = self._queue.get()
                self._ranks_confirmed[self.session_id].add(new_rank_confirmed)

                if self.is_all_confirmed() is True:
                    break
        else:
            pass


def recv_handshake():
    pass
