import torch

from pipegoose.nn.pipeline_parallel2._job.job import Job
from pipegoose.nn.pipeline_parallel2.exception import PipelineGradientFlowError
from pipegoose.nn.pipeline_parallel2.queue import SavedActivation

# class CreateBackwardOutputPackageCallback(Callback):
#     """Create a new package for the output of a backward job."""

#     def after_compute(self):
#         data = self.job.output
#         orig_metadata = self.job.input.metadata

#         package = Package(data, orig_metadata)
#         package.metadata.partition_idx -= 1

#         self.job.output = package


# class SendBackwardPackageCallback(Callback):
#     pass


class BackwardJob(Job):
    """Do backward pass."""

    def run_compute(self) -> torch.Tensor:
        microbatch_idx = self.input.metadata.microbatch_idx
        partition_idx = self.input.metadata.partition_idx
        key = SavedActivation.get_key(microbatch_idx, partition_idx)
        output = SavedActivation.get_saved_activations(key)
        prev_grad = self.input.data

        rank = self.pipeline_context.parallel_context.get_global_rank()

        # NOTE: just for testing the pipeline engine, exepct to refactor this out
        output = output.detach().requires_grad_(True)
        from pipegoose.nn.pipeline_parallel2.queue import _INPUT_ACTIVATIONS

        input = _INPUT_ACTIVATIONS[key]

        print(f"executing backward job, rank={rank}, microbatch_idx={microbatch_idx}, partition_idx={partition_idx}")

        # with torch.enable_grad():
        torch.autograd.backward(output, grad_tensors=prev_grad)

        if input.grad is None:
            raise PipelineGradientFlowError("Gradients can't flow back to the input of the pipeline stage")

        # TODO: remove this, since the grads is stored in module's weights
        # and we do gradient accumulation, we don't need return grads or send to other stages
        return input.grad
