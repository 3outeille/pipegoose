from abc import ABC, abstractclassmethod

import torch
from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.tensor_parallel._utils import VocabUtility, is_splitable
from pipegoose.nn.tensor_parallel.embedding import ParallelEmbedding
from pipegoose.nn.tensor_parallel.linear import ColumnParallelLinear


def _update_model_arguments(module: nn.Module, **kwargs):
    for key, value in kwargs.items():
        setattr(module, key, value)


def get_partition(data: torch.Tensor, parallel_context: ParallelContext, dim: int):
    rank = parallel_context.get_local_rank(ParallelMode.TENSOR)
    chunks = torch.chunk(data, parallel_context.get_world_size(ParallelMode.TENSOR), dim=dim)
    return chunks[rank].contiguous()


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
    def parallelize(self) -> nn.Module:
        self._parallelize_column_linear()
        return self.module

    def deparallelize(self):
        pass

    def _parallelize_column_linear(self):
        self.module.weight.data = get_partition(self.module.weight, self.parallel_context, dim=0)

        if self.module.bias is not None:
            self.module.bias.data = get_partition(self.module.bias, self.parallel_context, dim=0)

        self.module.__class__ = ColumnParallelLinear
        _update_model_arguments(
            module=self.module,
            # NOTE: make this based on parallel mapping
            # column parallel don't gather the output
            gather_output=True,
            parallel_context=self.parallel_context,
        )

    def _parallelize_row_linear(self):
        pass


class ParallelizeEmbedding(ParallelizeModule):
    # TODO: refactor to staticmethod
    def parallelize(self) -> nn.Module:
        """Parallelize nn.Embedding module."""
        assert isinstance(self.module, nn.Embedding), "only parallelize nn.Embedding"
        self._resize_vocab_size()
        self._split_weight()
        return self.module

    def deparallelize(self):
        pass

    def _split_weight(self):
        """Split weight into chunks and assign to each process."""
        world_size = self.parallel_context.get_world_size(ParallelMode.TENSOR)
        rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR)

        vocab_size = self.module.weight.shape[0]
        vocab_start_idx, vocab_end_idx = VocabUtility.get_vocab_range_from_global_vocab_size(world_size, rank, vocab_size)
        weight_chunks = torch.chunk(self.module.weight, world_size, dim=0)

        self.module.weight.data = weight_chunks[rank]
        self.module.__class__ = ParallelEmbedding

        _update_model_arguments(
            module=self.module,
            parallel_context=self.parallel_context,
            vocab_start_idx=vocab_start_idx,
            vocab_end_idx=vocab_end_idx,
            world_size=world_size,
        )

    def _resize_vocab_size(self):
        """Pad embedding size to make it splittable across GPUs"""
        padding_size = 0

        vocab_size, embedding_dim = self.module.weight.size()
        while not is_splitable(vocab_size + padding_size, self.parallel_context):
            padding_size += 1

        if padding_size > 0:
            padding = torch.zeros((padding_size, embedding_dim))
            new_embeddings = torch.cat([self.module.weight, padding], dim=0)

            self.module.weight.data = new_embeddings


class ParallelizeLayerNorm(ParallelizeModule):
    pass


class ParallelizeAttention(ParallelizeModule):
    pass
