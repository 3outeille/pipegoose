import pytest
import torch
from torch.distributed import ProcessGroup
from torch.multiprocessing import Process

from pipegoose.distributed.context import ParallelContext
from pipegoose.distributed.mode import ParallelMode

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

backend = ["gloo", pytest.param("nccl", marks=skip_if_no_cuda)]


def run_worker(rank, world_size, seed, backend, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    parallel_context = ParallelContext(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        local_world_size=world_size,
        host="localhost",
        port=12355,
        backend="gloo",
        seed=seed,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    )

    # try:
    #     # assert parallel_context.rank == rank
    #     parallel_modes = [
    #         ParallelMode.TENSOR,
    #         ParallelMode.PIPELINE,
    #         ParallelMode.DATA,
    #     ]

    #     assert parallel_context.tensor_parallel_size == tensor_parallel_size
    #     assert parallel_context.pipeline_parallel_size == pipeline_parallel_size
    #     assert parallel_context.data_parallel_size == data_parallel_size

    #     assert parallel_context.get_global_rank() == rank

    #     for parallel_mode in parallel_modes:
    #         assert parallel_context.is_initialized(parallel_mode) is True
    #         assert type(parallel_context.get_local_rank(parallel_mode)) == int
    #         assert type(parallel_context.get_world_size(parallel_mode)) == int
    # except Exception as e:
    #     pytest.fail(f"assertion failed: {e}")

    return parallel_context


@pytest.mark.parametrize("backend", backend)
def test_parallel_context_single_process_cpu(backend):
    TENSOR_PARALLEL_SIZE = 1
    PIPELINE_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1
    SEED = 69
    RANK = 0
    WORLD_SIZE = 1

    parallel_context = run_worker(
        RANK, WORLD_SIZE, SEED, backend, TENSOR_PARALLEL_SIZE, PIPELINE_PARALLEL_SIZE, DATA_PARALLEL_SIZE
    )

    parallel_modes = [
        ParallelMode.GLOBAL,
        ParallelMode.TENSOR,
        ParallelMode.PIPELINE,
        ParallelMode.DATA,
    ]

    assert parallel_context.tensor_parallel_size == TENSOR_PARALLEL_SIZE
    assert parallel_context.pipeline_parallel_size == PIPELINE_PARALLEL_SIZE
    assert parallel_context.data_parallel_size == DATA_PARALLEL_SIZE

    assert parallel_context.get_global_rank() == RANK

    for parallel_mode in parallel_modes:
        assert parallel_context.is_initialized(parallel_mode) is True
        assert type(parallel_context.get_local_rank(parallel_mode)) == int
        assert type(parallel_context.get_world_size(parallel_mode)) == int
        assert isinstance(parallel_context.get_group(parallel_mode), ProcessGroup)
        assert isinstance(parallel_context.get_ranks_in_group(parallel_mode), list)

    if torch.cuda.is_available():
        assert isinstance(torch.cuda.current_device(), int)

    # TODO: test seed

    torch.distributed.destroy_process_group()


def test_parallel_context_multiprocess_cpu():
    TENSOR_PARALLEL_SIZE = 2
    PIPELINE_PARALLEL_SIZE = 2
    DATA_PARALLEL_SIZE = 2
    SEED = 69
    WORLD_SIZE = 8

    processes = []

    for rank in range(WORLD_SIZE):
        p = Process(
            target=run_worker,
            args=(
                rank,
                WORLD_SIZE,
                SEED,
                TENSOR_PARALLEL_SIZE,
                PIPELINE_PARALLEL_SIZE,
                DATA_PARALLEL_SIZE,
            ),
            daemon=True,
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
        assert p.exitcode == 0
