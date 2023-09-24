import time
from copy import deepcopy

import pytest

from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.pipeline_parallel2._utils import get_partition_idx
from pipegoose.nn.pipeline_parallel2.sync.handshake import ProgressTracker
from pipegoose.testing.utils import init_parallel_context, spawn


def get_gpipe_schedules(n_partitions, n_microbatches):
    n_clock_cycles = n_partitions + n_microbatches - 1
    schedules = []
    for clock_idx in range(n_clock_cycles):
        start_partrition = max(clock_idx + 1 - n_microbatches, 0)
        end_partition = min(clock_idx + 1, n_partitions)

        tasks = []
        for partition_idx in range(start_partrition, end_partition):
            microbatch_idx = clock_idx - partition_idx
            tasks.append((microbatch_idx, partition_idx))

        schedules.append(tasks)

    return schedules


def schedules_to_progress(schedules):
    return {i: {item: False for item in sublist} for i, sublist in enumerate(schedules)}


def run_progress_tracker(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    N_MICROBATCHES = 4
    MICROBATCH_IDX = 0

    schedules = get_gpipe_schedules(pipeline_parallel_size, N_MICROBATCHES)
    PROGRESS = schedules_to_progress(schedules)
    INITIAL_PROGRESS = deepcopy(PROGRESS)

    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    tracker = ProgressTracker(parallel_context, ParallelMode.GLOBAL)

    if rank == tracker.MASTER_RANK:
        tracker.initiate(PROGRESS)
        assert tracker.is_initiated() is True
        assert tracker.progress == PROGRESS
        assert tracker.clock_idx == 0

        # NOTE: wait until all workers are confirmed
        time.sleep(5)
        assert tracker.is_all_confirmed(clock_idx=0) is True

        # NOTE: after all workers are confirmed,
        # the clock index should be incremented
        assert tracker.clock_idx == 1
        assert tracker.progress != INITIAL_PROGRESS
    else:
        # NOTE: wait until the tracker is initiated
        time.sleep(2)
        assert tracker.is_initiated() is True
        # NOTE: other workers may updated the progress
        # so the progress should be updated
        # TODO: if haven't confirmed any task, clock_idx should be 0
        # assert tracker.progress == PROGRESS
        # assert handshake.clock_idx == 0

        task = (MICROBATCH_IDX, get_partition_idx(parallel_context))
        tracker.confirm(task)
        assert tracker.is_confirmed(task, 0) is True

        # NOTE: wait until all workers are confirmed
        time.sleep(5)
        assert tracker.clock_idx == 1
        assert tracker.progress != INITIAL_PROGRESS

    parallel_context.destroy()


@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
@pytest.mark.parametrize("pipeline_parallel_size", [2, 4])
@pytest.mark.parametrize("data_parallel_size", [1, 2])
def test_progress_tracker(tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    world_size = tensor_parallel_size * pipeline_parallel_size * data_parallel_size

    spawn(
        run_progress_tracker,
        world_size=world_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    )
