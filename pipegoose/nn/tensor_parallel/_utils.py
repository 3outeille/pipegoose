from typing import Tuple

from torch import nn

from pipegoose.distributed.parallel_mode import ParallelMode


def is_splitable(size, parallel_context):
    world_size = parallel_context.get_world_size(ParallelMode.TENSOR)
    return True if size % world_size == 0 else False


def is_linear_parallelizable(module: nn.Module) -> bool:
    pass


def _is_column_parallel(module: nn.Module) -> bool:
    pass


def _is_row_parallel(module: nn.Module) -> bool:
    pass


def get_vocab_range_idx(partition_size: int, rank: int) -> Tuple[int, int]:
    start_idx = rank * partition_size
    end_idx = start_idx + partition_size
    return start_idx, end_idx


class VocabUtility:
    @staticmethod
    def get_vocab_range_idx_from_partition_size(partition_size: int, rank: int) -> Tuple[int, int]:
        start_idx = rank * partition_size
        end_idx = start_idx + partition_size
        return start_idx, end_idx

    @staticmethod
    def get_vocab_range_from_global_vocab_size(world_size, rank, vocab_size):
        partition_size = vocab_size // world_size
        return VocabUtility.get_vocab_range_idx_from_partition_size(partition_size, rank)


def _update_model_arguments(module: nn.Module, **kwargs):
    for key, value in kwargs.items():
        setattr(module, key, value)
