#!/usr/bin/env python3

import os
import os, torch, time
import transformers
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.debug.profiler as xp


os.environ.setdefault("XLA_HLO_DEBUG", "1")
os.environ.setdefault("XLA_FLAGS",
                      "--xla_hlo_profile=true "
                      "--xla_dump_to=/tmp/xla_dumps "
                      "--xla_dump_hlo_as_text "
                      "--xla_auto_spmd_partitioning=true "
                      "--xla_async_eager_clone=true "
                      )
os.environ.setdefault("XLA_CAPTURE_PERF_COUNTER", "1")


def train_loop(rank):
    device = xm.xla_device()
    xm.master_print(f"[{rank}] device={device}")

    MODEL = "meta-llama/Llama-3.1-8B"
    SEQ = 1024
    BATCH = 1
    WARMUP = 5
    PROFILE = 10

    #create model
    tok = transformers.AutoTokenizer.from_pretrained(MODEL, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})

    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.resize_token_embeddings(len(tok))
    model.to(device)


    #load training data
    ds = load_dataset("c4", "en", split="train", streaming=False)
    subset = ds.shuffle(seed=0).select(range((WARMUP + PROFILE) * 2))

    def tok_fn(ex):
        t = tok(ex["text"], truncation=True, max_length=SEQ, padding="max_length")
        return {"input_ids": t["input_ids"], "attention_mask": t["attention_mask"]}

    subset = subset.map(tok_fn, remove_columns=subset.column_names)
    subset.set_format("torch")
    loader = DataLoader(subset, batch_size=BATCH, shuffle=True)
    it = iter(loader)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

    #warm up for profiling
    for _ in range(WARMUP):
        b = next(it)
        out = model(b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device), labels=b["input_ids"].to(device))
        out.loss.backward()
        xm.optimizer_step(opt)
        opt.zero_grad()
        xm.mark_step()

    # profiling
    xp.start()
    for i in range(PROFILE):
        b = next(it)
        out = model(b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device), labels=b["input_ids"].to(device))
        out.loss.backward()
        xm.optimizer_step(opt)
        opt.zero_grad()
        xm.mark_step()
        if i % 10 == 0: xm.master_print(f"[{rank}] step {i}/{PROFILE}")

    xp.stop()
    xm.master_print(f"[{rank}] done.")

def _mp(rank):
    train_loop(rank)

if __name__ == "__main__":
    xmp.spawn(_mp, args=(), nprocs=8)