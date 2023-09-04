# NOTE: since we already test the backward pass
# of these modules in another tensor_parallel tests, we don't
# need to test it here

import pytest
import torch

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.tensor_parallel.parallelize import (
    ParallelizeEmbedding,
    ParallelizeLinear,
    ParallelizeLayerNorm,
    ParallelizeLMHead
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
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, module_name, module, input, output
):
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    world_size = parallel_context.get_world_size(parallel_mode=ParallelMode.TENSOR)

    parallelized_module = ParallelizeEmbedding(module_name, module, parallel_context).parallelize()
    parallel_output = parallelized_module(input)

    assert torch.allclose(parallel_output, output)


@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
def test_parallelize_embedding(model, tensor_parallel_size):
    PIPELINE_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1
    MODULE_NAME = "transformer.word_embeddings"

    input = torch.arange(0, 10)
    module = model.get_input_embeddings()
    output = module(input)

    spawn(
        run_parallelize_embedding,
        world_size=tensor_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        module_name=MODULE_NAME,
        module=module,
        input=input.detach(),
        output=output.detach(),
    )


def run_parallelize_linear(
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, module_name, module, input, output
):
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    parallelized_module = ParallelizeLinear(module_name, module, parallel_context).parallelize()
    parallel_output = parallelized_module(input)

    torch.allclose(parallel_output, output, rtol=1e-4)


MODULE_NAMES = [
    "transformer.h.0.mlp.dense_h_to_4h",
    "transformer.h.0.mlp.dense_4h_to_h",
    "transformer.h.0.self_attention.query_key_value",
    "transformer.h.0.self_attention.dense"
]

GET_MODULE_FUNCTIONS = [
    lambda model: model.transformer.h[0].mlp.dense_h_to_4h,
    lambda model: model.transformer.h[0].mlp.dense_4h_to_h,
    lambda model: model.transformer.h[0].self_attention.query_key_value,
    lambda model: model.transformer.h[0].self_attention.dense,
]


@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
@pytest.mark.parametrize("MODULE_NAME, get_module", list(zip(MODULE_NAMES, GET_MODULE_FUNCTIONS)))
def test_parallelize_linear(model, tensor_parallel_size, MODULE_NAME, get_module):
    # NOTE: This is parallelizing two dense layers in an MLP
    # and all query, key, value, and head projections in self-attention
    PIPELINE_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1

    module = get_module(model)
    input_size = module.weight.shape[1]

    input = torch.randn(10, input_size)
    output = module(input)

    spawn(
        run_parallelize_linear,
        world_size=tensor_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        module_name=MODULE_NAME,
        module=module,
        input=input.detach(),
        output=output.detach(),
    )


@pytest.mark.skip
def test_parallelize_attention():
    pass


def run_parallelize_layernorm(
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, module_name, module, input, output
):
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    # TODO: make this based on parallel mapping
    parallelized_module = ParallelizeLayerNorm(module_name, module, parallel_context).parallelize()
    parallel_output = parallelized_module(input)

    torch.allclose(parallel_output, output)


@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
def test_parallelize_layer_norm(model, tensor_parallel_size):
    DATA_PARALLEL_SIZE = 1
    PIPELINE_PARALLEL_SIZE = 1

    MODULE_NAME = "transformer.word_embeddings_layernorm"
    module = model.transformer.word_embeddings_layernorm

    BATCH_SIZE = 10
    SEQ_LEN = 5
    HIDDEN_SIZE = module.normalized_shape[0]
    input = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
    output = module(input)

    spawn(
        run_parallelize_layernorm,
        world_size=tensor_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        module_name=MODULE_NAME,
        module=module,
        input=input.detach(),
        output=output.detach(),
    )


def test_parallelize_positional_embedding():
    pass


def run_parallelize_lm_head(
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, module_name, module, input, output
):
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )

    parallelized_module = ParallelizeLMHead(module_name, module, parallel_context).parallelize()
    parallel_output = parallelized_module(input)

    torch.allclose(parallel_output, output)


@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
def test_parallelize_lm_head(model, tensor_parallel_size):
    DATA_PARALLEL_SIZE = 1
    PIPELINE_PARALLEL_SIZE = 1

    MODULE_NAME = "lm_head"
    module = model.lm_head

    BATCH_SIZE = 10
    HIDDEN_SIZE = module.weight.shape[1]

    input = torch.randn(BATCH_SIZE, HIDDEN_SIZE)
    output = module(input)

    spawn(
        run_parallelize_lm_head,
        world_size=tensor_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        module_name=MODULE_NAME,
        module=module,
        input=input.detach(),
        output=output.detach(),
    )
