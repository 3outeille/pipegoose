import pytest
import torch
import torch.distributed as dist

from pipegoose.distributed._initializers.initialize_pipeline import (
    PipelineParallelGroupInitializer,
)
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.testing.utils import spawn


def init_tensor_parallel_group(rank, world_size, host, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    init_method = f"tcp://{host}:{port}"

    torch.distributed.init_process_group(
        rank=rank,
        world_size=world_size,
        backend="gloo",
        init_method=init_method,
    )

    result = PipelineParallelGroupInitializer(
        rank,
        world_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    ).init_dist_group()

    assert isinstance(result["local_rank"], int)
    assert isinstance(result["local_world_size"], int)
    # assert isinstance(result["process_group"], ProcessGroup)
    assert isinstance(result["ranks_in_group"], list)
    assert result["parallel_mode"] == ParallelMode.PIPELINE

    dist.barrier()
    dist.destroy_process_group(result["process_group"])
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize(
    "world_size, tensor_parallel_size, pipeline_parallel_size, data_parallel_size",
    [(1, 1, 1, 1), (8, 2, 2, 2)],
)
def test_init_tensor_parallel_group(world_size, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    spawn(
        init_tensor_parallel_group,
        nprocs=world_size,
        host="localhost",
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    )
