import pytest
import torch
from torch import nn

from pipegoose.nn.pipeline_parallel2._job.forward import (
    ConfirmCompleteATaskToProgressTracker,
    CreateForwardOutputPackageCallback,
    ForwardJob,
    SaveActivationIfTrainingCallback,
    SaveInputActivationsCallback,
    SendForwardPackageCallback,
)
from pipegoose.nn.pipeline_parallel2._job.job_type import JobType
from pipegoose.nn.pipeline_parallel2._package import Package
from pipegoose.nn.pipeline_parallel2._utils import sleep
from pipegoose.nn.pipeline_parallel2.queue import SavedActivation, get_input_activations
from pipegoose.testing.utils import init_pipeline_context, spawn

# NOTE: use for creating a forward job
function = nn.Linear(2, 4)


def test_the_output_package_of_a_forward_job(forward_package, parallel_context, pipeline_context):
    cbs = [CreateForwardOutputPackageCallback(parallel_context, pipeline_context)]
    forward_job = ForwardJob(function, forward_package, cbs)

    output = forward_job.compute()

    assert forward_job.output == output
    assert isinstance(output, Package)
    assert isinstance(output.data, torch.Tensor)
    assert output.metadata.job_type == JobType.FORWARD

    for key in vars(output.metadata.training).keys():
        assert getattr(output.metadata.training, key) == getattr(forward_job.input.metadata.training, key)


@pytest.mark.skip
def test_destination_of_output_package(forward_package, pipeline_context):
    # # NOTE: (microbatch_idx, partition_idx) -> (microbatch_idx, next_partition_idx)
    # OUTPUT_DESTINATION = {
    #     (0, 0): (0, 1),
    #     (0, 1): (0, 2),
    #     (1, 0): (1, 1),
    #     (1, 1): (1, 2),
    #     (2, 0): (2, 1),
    #     (2, 1): (2, 2),
    #     (3, 0): (3, 1),
    #     (3, 1): (3, 2),
    #     (4, 0): (4, 1),
    #     (4, 1): (4, 2),
    # }

    # OUTPUT_SRC_DST_RANK_MAPPING = {
    #     (0): (0, 1),
    # }

    # forward_job = ForwardJob(
    #     function, forward_package, cbs=[CreateForwardOutputPackageCallback()]
    # )
    # ORIG_MICROBATCH_IDX = forward_job.input.metadata.microbatch_idx
    # ORIG_PARTITION_IDX = forward_job.input.metadata.partition_idx

    # output = forward_job.compute()

    # assert forward_job.output == output
    # assert isinstance(output, Package)
    # assert isinstance(output.data, torch.Tensor)
    # assert output.metadata.job_type == JobType.FORWARD

    # assert OUTPUT_DESTINATION[(ORIG_MICROBATCH_IDX, ORIG_PARTITION_IDX)] == (
    #     output.metadata.microbatch_idx,
    #     output.metadata.partition_idx,
    # )
    # for key in vars(output.metadata.training).keys():
    #     assert getattr(output.metadata.training, key) == getattr(forward_job.input.metadata.training, key)

    # # NOTE: we expect the metadata of the output package to
    # # indicate which node executed it, and the destination node
    # src, dst = output.metadata.src, output.metadata.dst
    # assert isinstance(src, int)
    # assert isinstance(dst, int)
    # assert (src, dst) == OUTPUT_SRC_DST_RANK_MAPPING[ORIG_MICROBATCH_IDX]
    pass


def test_forward_job_save_input_activations_for_backward_pass(forward_package, parallel_context, pipeline_context):
    MICROBATCH_IDX = forward_package.metadata.microbatch_idx
    PARTITION_IDX = forward_package.metadata.partition_idx
    CALLBACKS = [CreateForwardOutputPackageCallback(parallel_context, pipeline_context), SaveInputActivationsCallback()]

    forward_job = ForwardJob(function, forward_package, CALLBACKS)

    with pytest.raises(Exception):
        # NOTE: only save the input activations after the forward pass
        saved_activations = get_input_activations(MICROBATCH_IDX, PARTITION_IDX)

    forward_job.compute()
    saved_activations = get_input_activations(MICROBATCH_IDX, PARTITION_IDX)

    assert isinstance(saved_activations, torch.Tensor)
    assert torch.equal(saved_activations, forward_package.data)
    assert saved_activations.requires_grad is True


def test_forward_job_save_output_activations_for_backward_pass(forward_package, parallel_context, pipeline_context):
    MICROBATCH_IDX = forward_package.metadata.microbatch_idx
    PARTITION_IDX = forward_package.metadata.partition_idx
    CALLBACKS = [CreateForwardOutputPackageCallback(parallel_context, pipeline_context), SaveActivationIfTrainingCallback()]

    key = SavedActivation.get_key(MICROBATCH_IDX, PARTITION_IDX)
    forward_job = ForwardJob(function, forward_package, CALLBACKS)

    output = forward_job.compute()
    saved_activations = SavedActivation.get_saved_activations(key)

    assert isinstance(saved_activations, torch.Tensor)
    assert torch.equal(saved_activations, output.data)
    assert saved_activations.requires_grad is True

    with pytest.raises(KeyError):
        # NOTE: we expect the saved activations to be removed
        # after retrieving them
        SavedActivation.get_saved_activations(key)


def run_forward_job_send_output_to_the_next_pipeline_stage(
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, forward_package
):
    pipeline_context, parallel_context = init_pipeline_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    callbacks = [
        CreateForwardOutputPackageCallback(parallel_context, pipeline_context),
        SendForwardPackageCallback(parallel_context),
    ]
    forward_job = ForwardJob(function, forward_package, cbs=callbacks)

    forward_job.compute()

    RECV_RANK = forward_job.output.metadata.dst
    if world_size > 1 and rank == RECV_RANK:
        from pipegoose.nn.pipeline_parallel2._comm import RECV_QUEUE

        sleep(0.1)
        assert RECV_QUEUE.qsize() == 1

        received_package = RECV_QUEUE.get()

        assert isinstance(received_package, Package)
        assert received_package.metadata.dst == rank
        # NEXT_PARTITION_IDX = PARTITION_IDX + 1
        # received_package = RECV_QUEUE[(MICROBATCH_IDX, NEXT_PARTITION_IDX)]


@pytest.mark.parametrize(
    "pipeline_parallel_size",
    [
        1,
        2,
        # TODO: fix this, it can't work with 3, 5
        3,
        5,
    ],
)
def test_forward_job_send_output_to_the_next_pipeline_stage(forward_package, pipeline_parallel_size):
    TENSOR_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1

    spawn(
        run_forward_job_send_output_to_the_next_pipeline_stage,
        world_size=pipeline_parallel_size,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=DATA_PARALLEL_SIZE,
        forward_package=forward_package,
    )


def run_confirm_a_forward_job_after_completing_it(
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, forward_package
):
    import torch.distributed as dist

    from pipegoose.distributed.parallel_mode import ParallelMode
    from pipegoose.nn.pipeline_parallel2.sync.handshake import ProgressTracker
    from pipegoose.nn.pipeline_parallel2.sync.progress_tracker import (
        get_progresses_from_pipeline_context,
    )

    MASTER_RANK = 0

    pipeline_context, parallel_context = init_pipeline_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )

    tracker = ProgressTracker(MASTER_RANK, parallel_context=parallel_context, parallel_mode=ParallelMode.GLOBAL)
    progresses = get_progresses_from_pipeline_context(pipeline_context)
    tracker.initiate(progresses)
    dist.barrier()

    callbacks = [ConfirmCompleteATaskToProgressTracker(parallel_context)]
    forward_job = ForwardJob(function, forward_package, callbacks)
    forward_job.compute()

    assert tracker.is_all_confirmed(clock_idx=0) is True


def test_confirm_a_forward_job_after_completing_it(forward_package):
    TENSOR_PARALLEL_SIZE = 1
    PIPELINE_PARALLEL_SIZE = 2
    DATA_PARALLEL_SIZE = 1

    WORLD_SIZE = TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE * DATA_PARALLEL_SIZE

    spawn(
        run_confirm_a_forward_job_after_completing_it,
        world_size=WORLD_SIZE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        forward_package=forward_package,
    )
