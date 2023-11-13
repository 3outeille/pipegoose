from copy import deepcopy

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn import DataParallel, TensorParallel
from pipegoose.optim import DistributedOptimizer
from logger import Logger


def get_model_params_size(model, fp_bytes=4):
    params_size = 0
    for p in model.parameters():
        params_size += p.numel()
    params_gb = params_size * fp_bytes / 2**30
    return params_gb


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    import wandb

    DATA_PARALLEL_SIZE = 1
    TENSOR_PARALLEL_SIZE = 1
    PIPELINE_PARALLEL_SIZE = 1
    MODEL = "bigscience/bloom-560m"
    DATASET = "imdb"
    NUM_EPOCHS = 3
    LR = 1e-4
    SEED = 69
    BATCH_SIZE = 4
    CONTEXT_LENGTH = 1024

    rank = 0
    torch.cuda.empty_cache()
    set_seed(SEED)

    Logger()(f"device_count: {torch.cuda.device_count()}")
    Logger()(f"is available: {torch.cuda.is_available()}")

    train_dataset = load_dataset("imdb", split="train[:130]")
    train_dataset = train_dataset.map(lambda x: {"text": x["text"][:10]})  # for demonstration purposes

    # train_sampler = DistributedSampler(train_dataset, num_replicas=DATA_PARALLEL_SIZE, rank=0, seed=SEED)
    # train_sampler = SequentialSampler(train_dataset)
    # train_dataloader = DataLoader(
    #     train_dataset, batch_size=BATCH_SIZE // DATA_PARALLEL_SIZE, shuffle=False, sampler=train_sampler
    # )
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE // DATA_PARALLEL_SIZE, shuffle=False
    )

    model_cpu = AutoModelForCausalLM.from_pretrained(MODEL)
    model_gpu = deepcopy(model_cpu).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})


    optim_cpu = SGD(model_cpu.parameters(), lr=LR)
    optim_gpu = SGD(model_gpu.parameters(), lr=LR)

    model_cpu.train()
    model_gpu.train()
    step = 0

    if rank == 0:

        def get_time_name():
            import datetime

            today = datetime.datetime.now()
            return today.strftime("%d/%m/%Y_%H:%M:%S")

        wandb.init(
            project="pipegoose",
            name=f"{get_time_name()}.test_dp_tp_zero1_converegence",
            config={
                "data_parallel_size": DATA_PARALLEL_SIZE,
                "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
                "pipeline_parallel_size": PIPELINE_PARALLEL_SIZE,
                "model": MODEL,
                "dataset": DATASET,
                "epochs": NUM_EPOCHS,
                "learning_rate": LR,
                "seed": SEED,
                "batch_size": BATCH_SIZE,
            },
        )

    for epoch in range(NUM_EPOCHS):
        # train_sampler.set_epoch(epoch)
        Logger()(f"rank={rank}, epoch={epoch}")

        for batch in train_dataloader:
            inputs = tokenizer(batch["text"], padding=True, truncation=True, max_length=CONTEXT_LENGTH, return_tensors="pt")
            
            inputs_cpu = {name: tensor for name, tensor in inputs.items()}
            inputs_gpu = {name: tensor.to("cuda") for name, tensor in inputs.items()}
            
            labels_cpu = inputs_cpu["input_ids"]
            labels_gpu = inputs_gpu["input_ids"]

            outputs_cpu = model_cpu(**inputs_cpu, labels=labels_cpu)
            outputs_gpu = model_gpu(**inputs_gpu, labels=labels_gpu)

            optim_cpu.zero_grad()
            outputs_cpu.loss.backward()
            optim_cpu.step()

            optim_gpu.zero_grad()
            outputs_gpu.loss.backward()
            optim_gpu.step()

            if rank == 0:
                Logger()(f"epoch={epoch}, step={step}, rank={rank}, train_loss_cpu={outputs_cpu.loss}, train_loss_gpu={outputs_gpu.loss}")
                wandb.log({"train_loss_cpu": outputs_cpu.loss, "train_loss_gpu": outputs_gpu.loss, "step": step, "epoch": epoch})

            step += 1

    wandb.finish()
    # model.cpu()
