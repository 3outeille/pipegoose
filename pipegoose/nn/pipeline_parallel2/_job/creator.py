from abc import ABC, abstractmethod
from typing import Any, Callable, Union

import torch
import torch.distributed.rpc as rpc

from pipegoose.nn.pipeline_parallel2._comm import (
    get_pipeline_context,
    set_pipeline_context,
)
from pipegoose.nn.pipeline_parallel2._job.backward import (
    BackwardJob,
    CreateBackwardOutputPackageCallback,
    SendBackwardPackageCallback,
)
from pipegoose.nn.pipeline_parallel2._job.forward import (
    CreateForwardOutputPackageCallback,
    ForwardJob,
    SendForwardPackageCallback,
)
from pipegoose.nn.pipeline_parallel2._job.job import Job
from pipegoose.nn.pipeline_parallel2._job.job_type import JobType
from pipegoose.nn.pipeline_parallel2._package import Metadata, Package
from pipegoose.nn.pipeline_parallel2.pipeline_context import PipelineContext
from pipegoose.nn.pipeline_parallel2.queue import JobQueue


class JobCreator(ABC):
    """A base class for creating a job from a package."""

    @abstractmethod
    def create(self) -> Job:
        raise NotImplementedError("not implemented")


class _ForwardJobCreator(JobCreator):
    """Put a forward job into job queue for a worker to execute."""

    CBS = [CreateForwardOutputPackageCallback, SendForwardPackageCallback]

    @classmethod
    def create(cls, function: Callable, package: Package, pipeline_context: PipelineContext) -> ForwardJob:
        job = ForwardJob(function, package, cbs=cls.CBS, pipeline_context=pipeline_context)
        return job


class _BackwardJobCreator(JobCreator):
    CBS = [CreateBackwardOutputPackageCallback, SendBackwardPackageCallback]

    @classmethod
    def create(cls, function: Callable, package: Package, pipeline_context: PipelineContext) -> BackwardJob:
        job = BackwardJob(function, package, cbs=cls.CBS, pipeline_context=pipeline_context)
        return job


def create_job(function: Callable, package: Package, pipeline_context: PipelineContext) -> Union[ForwardJob, BackwardJob]:
    """Create a job based on the package."""
    assert isinstance(package, Package), f"package must be an instance of Package, got {type(package)}"
    assert isinstance(
        pipeline_context, PipelineContext
    ), f"pipeline_context must be an instance of PipelineContext, got {type(pipeline_context)}"

    JOB_TYPE_TO_CREATOR = {
        JobType.FORWARD: _ForwardJobCreator,
        JobType.BACKWARD: _BackwardJobCreator,
    }

    job_type = package.metadata.job_type
    job = JOB_TYPE_TO_CREATOR[job_type].create(function, package, pipeline_context)

    return job


def _create_backward_job_and_put_to_pending_queue(grad_input: torch.Tensor, metadata: Metadata):
    """Create a backward job and put it to pending queue."""
    # NOTE: construct backward package
    data = torch.randn(2, 4)
    package = Package(data, metadata)
    package.metadata.job_type = JobType.BACKWARD

    # NOTE: construct backward job
    def backward_function(self):
        pass

    parallel_context = get_pipeline_context()

    backward_job = create_job(backward_function, package, parallel_context)

    # NOTE : put the backward job to pending queue
    JobQueue.PENDING_JOBS.put(backward_job)


def schedule_backward_job(package: Package, pipeline_context: PipelineContext) -> Package:
    assert isinstance(package, Package), f"package must be an instance of Package, got {type(package)}"

    class _ScheduleBackwardJob(torch.autograd.Function):
        @staticmethod
        def forward(ctx, metadata: Metadata, pipeline_context: PipelineContext, input: torch.Tensor):
            # NOTE: can't assign metadata attribute to ctx
            # AttributeError: attribute 'metadata' of 'torch._C._FunctionBase'
            # objects is not writable
            ctx.package_meta = metadata
            ctx.pipeline_context = pipeline_context
            return input

        @staticmethod
        def backward(ctx: Any, grad_input: torch.Tensor):
            metadata = ctx.package_meta
            pipeline_context = ctx.pipeline_context
            parallel_context = pipeline_context.parallel_context

            # TODO: because forward job and backward job are in the same node
            # rpc isn't necessary
            rpc.rpc_sync(
                # NOTE: the backward job create in the same node
                # as the forward job
                to=parallel_context.get_worker_name(metadata.src),
                func=_create_backward_job_and_put_to_pending_queue,
                args=(grad_input, metadata),
            )

            return (None, None, None)

    set_pipeline_context(pipeline_context)

    data = package.data
    metadata = package.metadata
    data = _ScheduleBackwardJob.apply(metadata, pipeline_context, data)
    package.data = data

    return package
