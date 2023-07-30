from abc import ABC, abstractclassmethod

import torch
from torch import nn

from pipegoose.distributed.context import ParallelContext

# from pipegoose.distributed.mode import ParallelMode


class ParallelizeModule(ABC):
    def __init__(self, module: nn.Module, parallel_context: ParallelContext):
        self.module = module
        self.parallel_context = parallel_context

    @abstractclassmethod
    def parallelize(self):
        raise NotImplementedError

    @abstractclassmethod
    def deparallelize(self):
        raise NotImplementedError


class ParallelizeLinear(ParallelizeModule):
    def parallelize(self):
        # tensor_parallel_size = parallel_context.tensor_paralell_size
        pass

    def deparallelize(self):
        pass


class ParallelizeEmbedding(ParallelizeModule):
    def parallelize(self):
        vocab_size, embedding_size = self.module.weight.size()


class ParallelizeLayerNorm(ParallelizeModule):
    pass


class ParallelizeAttention(ParallelizeModule):
    pass


class TensorParallel:
    """Turn a sequential model into a tensor-parallel model.

    Inspired by OSLO's TensorParallel: https://github.com/EleutherAI/oslo/blob/00e3be56446df37a0372a93a094863ffc89a2f8b/oslo/torch/nn/parallel/tensor_parallel/tensor_parallel.py#L51
    """

    def __init__(self, module: nn.Module, parallel_context: ParallelContext):
        super().__init__()
        self.module = module
        self.parallel_context = parallel_context

    @torch.no_grad()
    def parallelize(self):
        pass

    def _parallelize_embedding(self):
        pass

    def _parallize_layernorm(self):
        for _, module in self.module.named_modules():
            if isinstance(module, nn.LayerNorm):
                pass
