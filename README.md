# 🚧 PipeGoose - Pipeline Parallelism for transformers model - WIP

This project is actively under development.

```python:
from transformer import AutoModel, AutoTokenizer
from pipegoose import Pipeline

model = AutoModel.from_pretrained("bloom-560")
tokenizer = AutoTokenizer.from_pretrained("bloom-560")

pipeline = Pipeline(model, tokenizer, partrition=partrition_func)

pipeline.fit(dataloader, n_microbatches=16)
```

**Implementation Details**

- Supports training `transformers` model.
- Implements parallel compute and data transfer using separate CUDA streams.
- Gradient checkpointing will be implemented by enforcing virtual dependency in the backpropagation graph, ensuring that the activation for gradient checkpoint will be recomputed just in time for each (micro-batch, partition).
- Custom algorithms for model partitioning with two default partitioning models based on elapsed time and GPU memory consumption per layer.
- Potential support includes:
    - Callbacks within the pipeline: `Callback(function, microbatch_idx, partition_idx)` for before and after the forward, backward, and recompute steps (for gradient checkpointing).
    - Mixed precision training.