import pytest
import torch

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.tensor_parallel.parallelize import (
    ParallelizeEmbedding,
    ParallelizeLinear,
)
from pipegoose.testing.utils import spawn


def init_parallel_context(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    parallel_context = ParallelContext(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        local_world_size=world_size,
        host="localhost",
        port=port,
        seed=69,
        backend="gloo",
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    )

    return parallel_context


def run_parallelize_embedding(
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, embedding, input, output
):
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    world_size = parallel_context.get_world_size(parallel_mode=ParallelMode.TENSOR)

    parallelized_embedding = ParallelizeEmbedding(embedding, parallel_context).parallelize()
    parallel_output = parallelized_embedding(input)

    assert torch.allclose(parallel_output, output)

    # NOTE: since we already test the backward pass
    # of ParallelEmbedding in another test, we don't
    # need to test it here


@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
def test_parallelize_embedding(model, tensor_parallel_size):
    PIPELINE_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1

    input = torch.arange(0, 10)
    embedding = model.get_input_embeddings()
    output = embedding(input)

    spawn(
        run_parallelize_embedding,
        world_size=tensor_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        embedding=embedding,
        input=input.detach(),
        output=output.detach(),
    )


def run_parallelize_linear(
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, module_name, module, input, output
):
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    # TODO: make this based on parallel mapping
    parallelized_module = ParallelizeLinear(module, parallel_context).parallelize(module_name)
    parallel_output = parallelized_module(input)

    torch.allclose(parallel_output, output, rtol=1e-4)

    # NOTE: since we already test the backward pass of
    # ColumnParallelLinear, and RowParallelLinear in another test,
    # we don't need to test it here.


@pytest.mark.parametrize("tensor_parallel_size, MODULE_NAME, get_module", [
    (1, "transformer.h.0.mlp.dense_h_to_4h", lambda model: model.h[0].mlp.dense_h_to_4h),
    (2, "transformer.h.0.mlp.dense_h_to_4h", lambda model: model.h[0].mlp.dense_h_to_4h),
    (1, "transformer.h.0.mlp.dense_4h_to_h", lambda model: model.h[0].mlp.dense_4h_to_h),
    (2, "transformer.h.0.mlp.dense_4h_to_h", lambda model: model.h[0].mlp.dense_4h_to_h),
])
def test_parallelize_linear(model, tensor_parallel_size, MODULE_NAME, get_module):
    PIPELINE_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1

    module = get_module(model)
    input_size = module.weight.shape[1]

    input_tensor = torch.randn(10, input_size)
    output = module(input_tensor)

    spawn(
        run_parallelize_linear,
        world_size=tensor_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        module_name=MODULE_NAME,
        module=module,
        input=input_tensor.detach(),
        output=output.detach(),
    )


@pytest.mark.skip
def test_parallelize_attention():
    pass


@pytest.mark.skip
def test_parallelize_layer_norm():
    pass
