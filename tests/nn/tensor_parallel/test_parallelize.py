import pytest
import torch
from transformers import AutoModel

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.tensor_parallel._utils import VocabUtility, is_splitable
from pipegoose.nn.tensor_parallel.parallelize import (
    ParallelizeEmbedding,
    ParallelizeLinear,
)
from pipegoose.testing.utils import spawn

MODEL_NAME = "bigscience/bloom-560m"


@pytest.fixture
def model():
    return AutoModel.from_pretrained(MODEL_NAME)


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

    def get_new_embedding_size(vocab_size):
        padding_size = 0
        while not is_splitable(vocab_size + padding_size, parallel_context):
            padding_size += 1

        new_vocab_size = vocab_size + padding_size
        new_partition_size = new_vocab_size // world_size
        return new_vocab_size, new_partition_size

    vocab_size, embedding_dim = embedding.weight.size()
    new_vocab_size, new_partition_size = get_new_embedding_size(vocab_size)
    vocab_start_idx, vocab_end_idx = VocabUtility.get_vocab_range_from_global_vocab_size(world_size, rank, new_vocab_size)

    parallelized_embedding = ParallelizeEmbedding(embedding, parallel_context).parallelize()
    parallel_output = parallelized_embedding(input)

    assert parallelized_embedding.vocab_start_idx == vocab_start_idx
    assert parallelized_embedding.vocab_end_idx == vocab_end_idx
    assert parallelized_embedding.weight.shape == (new_partition_size, embedding_dim)
    assert torch.allclose(parallel_output, output)

    # NOTE: since we already test the backward pass
    # of ParallelEmbedding in another test, we don't
    # need to test it here


@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
def test_parallelize_embedding(model, tensor_parallel_size):
    input = torch.arange(0, 10)
    embedding = model.get_input_embeddings()
    output = embedding(input)

    spawn(
        run_parallelize_embedding,
        world_size=tensor_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        embedding=embedding,
        input=input.detach(),
        output=output.detach(),
    )


def run_parallelize_column_linear(
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, linear, input, output
):
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    parallelized_linear = ParallelizeLinear(linear, parallel_context).parallelize()
    parallel_output = parallelized_linear(input)

    torch.allclose(parallel_output, output, rtol=1e-4)

    # NOTE: since we already test the backward pass
    # of ColumnParallelLinear in another test, we don't
    # need to test it here


@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
def test_parallelize_column_linear(model, tensor_parallel_size):
    # NOTE: this is column parallel linear
    linear = model.h[0].mlp.dense_h_to_4h
    input_size = linear.weight.shape[1]

    input = torch.randn(10, input_size)
    output = linear(input)

    spawn(
        run_parallelize_column_linear,
        world_size=tensor_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        linear=linear,
        input=input.detach(),
        output=output.detach(),
    )


def run_parallelize_row_linear(
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, linear, input, output
):
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    # TODO: make this based on parallel mapping
    parallelized_linear = ParallelizeLinear(linear, parallel_context)._parallelize_row_linear(linear)
    parallel_output = parallelized_linear(input)

    torch.allclose(parallel_output, output, rtol=1e-4)


@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
def test_parallelize_row_linear(model, tensor_parallel_size):
    linear = model.h[0].mlp.dense_4h_to_h
    input_size = linear.weight.shape[1]

    input = torch.randn(10, input_size)
    output = linear(input)

    spawn(
        run_parallelize_row_linear,
        world_size=tensor_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        linear=linear,
        input=input.detach(),
        output=output.detach(),
    )
