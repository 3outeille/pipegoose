import random
from abc import ABC, abstractclassmethod
from dataclasses import dataclass
from queue import Queue
from time import sleep
from typing import Dict, List

import torch
import torch.distributed.rpc as rpc

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.pipeline_parallel2.sync.func import recv_execution_plan

SessionId = str


# def trigger():
#     # worker_name = self.parallel_context.get_worker_name(rank=1)
#     worker_name = "RPC_GLOBAL_WORKER_1"
#     # task = torch.tensor([1, 2])
#     # from pipegoose.nn.pipeline_parallel2.sync.func import recv_execution_plan
#     rpc.rpc_sync(
#         to=worker_name,
#         func=recv_execution_plan,
#         # args=(task,)
#         # func=torch.add,
#         args=(torch.ones(2), 3)
#     )


class Handshake(ABC):
    def __init__(self, parallel_context: ParallelContext, parallel_mode: ParallelMode):
        self.parallel_context = parallel_context
        self.parallel_mode = parallel_mode

        self._session_id: str = None
        self._queue: Dict[SessionId, Queue] = {}
        self._ranks_confirmed: Dict[SessionId, List[int]] = set()

        self._data = None

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

    @property
    def session_id(self) -> str:
        return self._session_id

    def initiate(self, data):
        # NOTE: broadcast expected tasks to all corresponding ranks
        for task in data:
            # microbatch_idx, partition_idx = task

            # NOTE: get the first rank of the pipeline stage
            # the rank that we do confirmation
            # is the last rank of a tensor parallel group
            # self.parallel_context._ranks_in_group[ParallelMode.TENSOR][-1]
            worker_name = self.parallel_context.get_worker_name(rank=1)
            task = torch.tensor(task)

            rpc.rpc_sync(to=worker_name, func=recv_execution_plan, args=(task,))
            break

        self._data = data

    def _set_session(self, session_id: torch.Tensor):
        self._session_id = session_id

    def confirm(self):
        # TODO: only non-scheduler ranks should confirm
        master_worker_name = self.parallel_context.get_worker_name(self.MASTER_RANK)
        rank = self.parallel_context.get_rank(self.parallel_mode)
        rpc.rpc_sync(master_worker_name, func=self._recv_confirm, args=(rank,))

    def _recv_confirm(self, rank: int):
        self._queue[self.session_id].put(rank)

    def is_initiated(self) -> bool:
        # return self.session_id is not None
        # data = get_execution_plan()
        return self._data is not None

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
