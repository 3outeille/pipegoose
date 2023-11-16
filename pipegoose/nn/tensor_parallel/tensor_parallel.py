from typing import List, Optional, Tuple

import torch
from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.nn.parallel import Parallel
from pipegoose.nn.tensor_parallel.parallelizer import (
    EmbeddingParallelizer,
    LayerNormParallelizer,
    LinearParallelizer,
    LMHeadParallelizer,
    ModuleParallelizer,
)

from pipegoose.utils.logger import Logger

class TensorParallel(Parallel):
    """Turn a 🤗 transformers model into a tensor parallel model."""

    # PARALLELIZERS = [EmbeddingParallelizer, LinearParallelizer, LayerNormParallelizer, LMHeadParallelizer]
    PARALLELIZERS = [EmbeddingParallelizer]

    def __init__(self, module: nn.Module, parallel_context: ParallelContext):
        self.module = module
        self.parallel_context = parallel_context

    @torch.no_grad()
    def parallelize(self) -> nn.Module:
        module = self.module

        if self.parallel_context.tensor_parallel_size > 1:
            # NOTE: because module.named_modules returns a leaf more than once,
            # this could potentially lead to the weight of a module being split
            # multiple times. so we filter out and retain the non-repetitive modules (leaf modules)
            leaf_modules = self._get_leaf_modules(module)
            for module_name, leaf_module in leaf_modules:
                parallelizer = self._find_parallelizer(leaf_module)
                if parallelizer is not None:
                    parallelizer(module_name, leaf_module, module, self.parallel_context).parallelize()

            self._save_metadata(module, self.parallel_context)

        return module

    def _get_leaf_modules(self, model: nn.Module) -> List[Tuple[str, nn.Module]]:
        leaf_modules = []
        for module_name, module in model.named_modules():
            if list(module.children()):
                continue
            leaf_modules.append((module_name, module))

        return leaf_modules

    def _find_parallelizer(self, module: nn.Module) -> Optional[ModuleParallelizer]:
        for parallelizer in self.PARALLELIZERS:
            if parallelizer.is_parallelizable(module):
                return parallelizer
        return None

    def _find_deparallelizer(self, module: nn.Module) -> Optional[ModuleParallelizer]:
        for parallelizer in self.PARALLELIZERS:
            if parallelizer.is_deparallelizable(module):
                return parallelizer
        return None

    @torch.no_grad()
    def deparallelize(self) -> nn.Module:
        module = self.module

        if self.parallel_context.tensor_parallel_size > 1:
            # NOTE: because module.named_modules returns a leaf more than once,
            # this could potentially lead to the weight of a module being split
            # multiple times. so we filter out and retain the non-repetitive modules (leaf modules)
            leaf_modules = self._get_leaf_modules(module)
           
            for module_name, leaf_module in leaf_modules:
                parallelizer = self._find_deparallelizer(leaf_module)
                if parallelizer is not None:
                    Logger()(f"deparallelizing {module_name}")
                    parallelizer(module_name, leaf_module, module, self.parallel_context).deparallelize()

            # TODO: update metadata ?