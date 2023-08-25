import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.tensor_parallel.loss import VocabParallelCrossEntropy
from pipegoose.testing.utils import spawn


def check_equal(A, B):
    assert torch.allclose(A, B, rtol=1e-3, atol=1e-1) or torch.allclose(A, B)


def run_parallel_cross_entropy(
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, logits, targets, loss
):
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

    local_rank = parallel_context.get_local_rank(parallel_mode=ParallelMode.TENSOR)
    ranks_in_group = parallel_context.get_ranks_in_group(parallel_mode=ParallelMode.TENSOR)

    if local_rank in ranks_in_group:
        N_LABELS = logits.shape[-1]

        def get_partition(logits):
            local_world_size = parallel_context.get_world_size(parallel_mode=ParallelMode.TENSOR)
            per_partition = N_LABELS // local_world_size
            chunks = torch.split(logits, per_partition, dim=-1)
            return chunks[local_rank]

        parallel_logits = get_partition(logits)
        parallel_predicted_logits = VocabParallelCrossEntropy.apply(parallel_logits, targets, parallel_context)

        predicted_logits = logits[:, torch.arange(targets.size(-1)), targets.view(-1)].squeeze()
        assert torch.allclose(parallel_predicted_logits, predicted_logits)

        # parallel_output.sum().backward()

        # assert torch.allclose(parallel_embedding.weight.grad.data, get_partition(grads))


@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
def test_parallel_cross_entropy(tensor_parallel_size):
    torch.manual_seed(69)

    BATCH_SIZE = 1
    SEQ_LEN = 2
    VOCAB_SIZE = 4
    logits = torch.randn(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

    loss = F.cross_entropy(
        rearrange(logits, "batch_size seq_len vocab_size -> (batch_size seq_len) vocab_size"),
        rearrange(targets, "batch_size seq_len -> (batch_size seq_len)"),
    )

    spawn(
        run_parallel_cross_entropy,
        world_size=tensor_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        logits=logits,
        targets=targets,
        loss=loss,
    )
