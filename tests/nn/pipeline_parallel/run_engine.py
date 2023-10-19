from copy import deepcopy
from functools import reduce

import torch
from torch import nn

from pipegoose.nn.pipeline_parallel._utils import get_partition_idx, is_last_stage
from pipegoose.nn.pipeline_parallel._worker import WorkerManager
from pipegoose.nn.pipeline_parallel.pipeline_engine import PipelineEngine
from pipegoose.nn.pipeline_parallel.scheduler import GPipeScheduler
from pipegoose.testing.utils import init_parallel_context, spawn


def run_pipeline_engine(
    rank,
    world_size,
    port,
    tensor_parallel_size,
    pipeline_parallel_size,
    data_parallel_size,
    n_microbatches,
    model,
    inputs,
    outputs,
    grads,
):
    forward_timeline = []
    backward_timeline = []

    def backward_hook(module, grad_input, grad_output):
        backward_timeline.append((module.microbatch_idx - 1, module.partition_idx))
        module.microbatch_idx -= 1

    class Function(nn.Module):
        def __init__(self, partition_idx):
            super().__init__()
            self.partition_idx = partition_idx
            self.microbatch_idx = 0
            self.net = model[self.partition_idx]
            self.register_backward_hook(backward_hook)

        def forward(self, input):
            forward_timeline.append((self.microbatch_idx, self.partition_idx))
            self.microbatch_idx += 1
            return self.net(input)

    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    scheduler = GPipeScheduler(n_microbatches, pipeline_parallel_size)
    worker_manager = WorkerManager()
    partition_idx = get_partition_idx(parallel_context)
    partition = Function(partition_idx)
    pipeline_engine = PipelineEngine(
        module=partition,
        scheduler=scheduler,
        worker_manager=worker_manager,
        parallel_context=parallel_context,
    )
    [(microbatch_idx, partition_idx) for microbatch_idx in range(n_microbatches)]
    EXPECTED_FORWARD_TIMELINE = [(microbatch_idx, partition_idx) for microbatch_idx in range(n_microbatches)]
    # EXPECTED_BACKWARD_TIMELINE = [(microbatch_idx, partition_idx) for microbatch_idx in range(n_microbatches, -1, -1)]
    p_outputs = pipeline_engine.run(inputs)

    assert forward_timeline == EXPECTED_FORWARD_TIMELINE

    if is_last_stage(parallel_context):
        assert torch.allclose(torch.cat(p_outputs, dim=0), outputs)

    for output in p_outputs:
        output.sum().backward(retain_graph=True)

    for param in partition.parameters():
        assert param.grad is not None

    for p, ground_grad in zip(partition.parameters(), grads[partition_idx]):
        assert torch.allclose(p.grad, ground_grad)


if __name__ == "__main__":
    TENSOR_PARALLEL_SIZE = 1
    PIPELINE_PARALLEL_SIZE = 4
    DATA_PARALLEL_SIZE = 1

    BATCH_SIZE = 32
    N_MICROBATCHES = 6
    SEQ_LEN = 10
    HIDDEN_DIM = 5
    WORLD_SIZE = TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE * DATA_PARALLEL_SIZE

    inputs = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, requires_grad=False)
    model = nn.ModuleList([nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU()) for _ in range(PIPELINE_PARALLEL_SIZE)])
    ORIG_MODEL = deepcopy(model)
    outputs = reduce(lambda inputs, layer: layer(inputs), model, inputs)

    outputs.sum().backward()

    grads = [[p.grad for p in layer.parameters()] for layer in model]

    spawn(
        run_pipeline_engine,
        world_size=WORLD_SIZE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        n_microbatches=N_MICROBATCHES,
        model=ORIG_MODEL,
        inputs=inputs.detach(),
        outputs=outputs.detach(),
        grads=grads,
    )
