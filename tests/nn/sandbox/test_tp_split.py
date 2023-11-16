from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.nn import TensorParallel
from transformers import AutoModelForCausalLM
from copy import deepcopy
import torch.distributed as dist
from pipegoose.utils.logger import Logger
import torch

def get_model_params_size(model, fp_bytes=4):
    params_size = 0
    for p in model.parameters():
        params_size += p.numel()
    params_gb = params_size * fp_bytes / 2**30
    return params_gb

if __name__ == "__main__":

    DATA_PARALLEL_SIZE = 1
    TENSOR_PARALLEL_SIZE = 2
    PIPELINE_PARALLEL_SIZE = 1
    MODEL = "bigscience/bloom-560m"

    parallel_context = ParallelContext.from_torch(
        data_parallel_size=DATA_PARALLEL_SIZE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
    )

    rank = parallel_context.get_global_rank()

    Logger()(f"rank={rank}, initialized parallel_context")

    model = AutoModelForCausalLM.from_pretrained(MODEL)
    ref_model = deepcopy(model)
    if rank == 0:
        Logger()(f"(word embeddings) = {model.transformer.word_embeddings.weight.data.shape}")

    # Logger()(f"rank={rank}, model size before parallelizing: {round(get_model_params_size(model), 3)} GB")

    dist.barrier()

    model_tp = TensorParallel(model, parallel_context).parallelize()
    model.to("cuda")
    device = next(model_tp.parameters()).device
    
    # if rank == 0:
        # Logger()(f"After splitting = {model}")

    # Logger()(f"rank={rank}, model size after parallelizing: {round(get_model_params_size(model), 3)} GB")
    # Logger()(f"(word embeddings) (rank={rank}) = {model_tp.transformer.word_embeddings.weight.data.shape}")
    dist.barrier()

    model_undo_tp = TensorParallel(model_tp, parallel_context).deparallelize()
    
    dist.barrier()

    # if rank == 0:
    #     del model_tp
    #     Logger()(f"(word embeddings) = {model_undo_tp.transformer.word_embeddings.weight.data.shape}")

    # # Sanity check
    # if rank == 0:
    #     ref_model_state_dict = ref_model.state_dict()
    #     model_undo_tp_state_dict = model_undo_tp.state_dict()

    #     assert sorted(list(ref_model_state_dict.keys())) == sorted(list(model_undo_tp_state_dict.keys()))

    #     for name in ref_model_state_dict.keys():
    #         data = ref_model_state_dict[name]
    #         data_tp = model_undo_tp_state_dict[name]

    #         assert data.shape == data_tp.shape, name

    #         torch.testing.assert_close(data, data_tp, msg=lambda msg: f"{name}:\n{msg}")
