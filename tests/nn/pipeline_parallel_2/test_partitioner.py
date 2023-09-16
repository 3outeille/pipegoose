import pytest
from torch import nn
from transformers import AutoModelForCausalLM

from pipegoose.nn.pipeline_parallel2.partitioner import (
    PartitionPolicy,
    get_model_partition,
)
from pipegoose.testing.utils import init_parallel_context, spawn

MODEL_NAME = "sshleifer/tiny-gpt2"


def run_model_partitioner(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    module = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    policy = PartitionPolicy.UNIFORM
    partition = get_model_partition(module, policy, parallel_context)

    assert isinstance(partition, nn.Module)
    assert partition != module


@pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
def test_naive_partitioning(pipeline_parallel_size):
    TENSOR_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1

    spawn(
        run_model_partitioner,
        world_size=pipeline_parallel_size,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=DATA_PARALLEL_SIZE,
    )
