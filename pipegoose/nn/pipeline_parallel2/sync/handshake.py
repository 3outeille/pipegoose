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

SessionId = str


class Handshake(ABC):
    def __init__(self, parallel_context: ParallelContext, parallel_mode: ParallelMode):
        self.parallel_context = parallel_context
        self.parallel_mode = parallel_mode

        self._session_id: str = None
        self._queue: Dict[SessionId, Queue] = {}
        self._ranks_confirmed: Dict[SessionId, List[int]] = set()

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

    def initiate(self):
        session_id = torch.tensor(self._generate_session_id())
        ranks_in_group = self.parallel_context.get_ranks_in_group(self.parallel_mode)
        local_rank = self.parallel_context.get_local_rank(self.parallel_mode)

        self._session_id = session_id
        self._queue[session_id] = Queue()

        # TODO: broadcast session id, and sender rank
        for other_rank in ranks_in_group:
            if local_rank == other_rank:
                # NOTE: only send to other ranks
                continue

            worker_name = self.parallel_context.get_worker_name(other_rank)
            # TODO: define a common mapping format
            rpc.rpc_sync(worker_name, func=self._set_session, args=(session_id,))

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
        return self.session_id is not None

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
