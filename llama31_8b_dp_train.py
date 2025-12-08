#!/usr/bin/env python3

import os
import time

# ---------------------------------------------------------------------
# Environment: MUST be set BEFORE importing torch_xla
# ---------------------------------------------------------------------
os.environ.setdefault("PJRT_DEVICE", "TPU")  # use PJRT runtime on TPU
os.environ.setdefault("XLA_HLO_DEBUG", "1")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_hlo_profile=true "
    "--xla_dump_to=/tmp/xla_dumps "
    "--xla_dump_hlo_as_text "
)
os.environ.setdefault("XLA_CAPTURE_PERF_COUNTER", "1")
# Optional: more verbose XLA logs
# os.environ.setdefault("PT_XLA_DEBUG_LEVEL", "1")

import torch
import transformers
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.debug.profiler as xp

# ---------------------------------------------------------------------
# Global experiment config
# ---------------------------------------------------------------------
MODEL = "meta-llama/Llama-3.1-8B"

SEQ = 1024
BATCH = 1
WARMUP_STEPS = 5
PROFILE_STEPS = 10
LR = 1e-5

LOGDIR_BASE = "/tmp/xla_traces"         # XLA per-rank traces (for XProf / TB profile)
TB_LOGDIR_BASE = "/tmp/tb_logs_dp_c4"   # TensorBoard scalars


# ---------------------------------------------------------------------
# Model & tokenizer
# ---------------------------------------------------------------------
def build_model_and_tokenizer(device):
    """Load Llama-3.1-8B, freeze all but lm_head, move to TPU."""
    tok = transformers.AutoTokenizer.from_pretrained(
        MODEL, use_fast=True, trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})

    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.resize_token_embeddings(len(tok))

    # Freeze everything except lm_head to reduce trainable state
    for p in model.parameters():
        p.requires_grad = False
    for p in model.lm_head.parameters():
        p.requires_grad = True

    model.to(device=device, dtype=torch.bfloat16)
    return model, tok


# ---------------------------------------------------------------------
# C4 streaming dataset (NO full download)
# ---------------------------------------------------------------------
class C4StreamingDataset(IterableDataset):
    """Stream C4 from HF without downloading the whole dataset.

    We only iterate over `max_samples` examples and tokenize on the fly.
    """

    def __init__(self, tokenizer, seq_len, max_samples):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_samples = max_samples

    def __iter__(self):
        # streaming=True: iterate directly from remote storage
        ds = load_dataset(
            "c4",
            "en",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        for i, ex in enumerate(ds):
            if i >= self.max_samples:
                break

            t = self.tokenizer(
                ex["text"],
                truncation=True,
                max_length=self.seq_len,
                padding="max_length",
            )
            # Convert to tensors here (no multiprocessing DataLoader workers)
            yield {
                "input_ids": torch.tensor(t["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(t["attention_mask"], dtype=torch.long),
            }


def make_streaming_dataloader(tok):
    """Create a small streaming C4 DataLoader for training."""
    max_samples = (WARMUP_STEPS + PROFILE_STEPS) * BATCH * 2

    stream_ds = C4StreamingDataset(tok, SEQ, max_samples)
    loader = DataLoader(
        stream_ds,
        batch_size=BATCH,
        num_workers=0,   # important for IterableDataset + XLA
    )
    return iter(loader)


# ---------------------------------------------------------------------
# Training & profiling loop
# ---------------------------------------------------------------------
def train_loop(rank: int):
    # Discover devices and choose this rank's device
    devices = xm.get_xla_supported_devices()
    world_size = len(devices)
    device = xm.xla_device()

    xm.master_print(f"[DP-C4][rank {rank}/{world_size}] device={device}")

    # TensorBoard writer (only rank 0 writes scalars)
    writer = None
    if rank == 0:
        tb_logdir = os.path.join(TB_LOGDIR_BASE, "rank0")
        os.makedirs(tb_logdir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_logdir)
        xm.master_print(f"[DP-C4] TensorBoard logdir: {tb_logdir}")

    global_step = 0
    tokens_per_step_per_core = BATCH * SEQ

    # Make sure all ranks reach this point
    xm.rendezvous("dp_c4_init")

    # -----------------------------------------------------------------
    # Model, tokenizer, data, optimizer
    # -----------------------------------------------------------------
    if rank == 0:
        xm.master_print("[DP-C4] Loading model & tokenizer on all ranks...")

    model, tok = build_model_and_tokenizer(device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
    )
    data_iter = make_streaming_dataloader(tok)

    # -----------------------------------------------------------------
    # Warmup (not the main profiled region)
    # -----------------------------------------------------------------
    for step in range(1, WARMUP_STEPS + 1):
        model.train()
        opt.zero_grad()

        batch = next(data_iter)
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)

        t0 = time.time()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            labels=input_ids,
        )
        loss = outputs.loss
        loss.backward()
        xm.optimizer_step(opt, barrier=True)
        xm.mark_step()

        step_time_ms = (time.time() - t0) * 1000.0
        loss_val = float(loss)
        tokens_per_sec_core = tokens_per_step_per_core / (step_time_ms / 1000.0)
        tokens_per_sec_global = tokens_per_sec_core * world_size

        if rank == 0:
            xm.master_print(
                f"[DP-C4][warmup] step {step}/{WARMUP_STEPS}, "
                f"loss={loss_val:.4f}, step_time_ms={step_time_ms:.1f}, "
                f"tokens/s/core={tokens_per_sec_core:.1f}"
            )

            if writer is not None:
                writer.add_scalar("train/loss", loss_val, global_step)
                writer.add_scalar("train/step_time_ms", step_time_ms, global_step)
                writer.add_scalar(
                    "train/tokens_per_sec_per_core", tokens_per_sec_core, global_step
                )
                writer.add_scalar(
                    "train/global_tokens_per_sec", tokens_per_sec_global, global_step
                )

        global_step += 1

    # -----------------------------------------------------------------
    # Profiling region: per-rank trace (great for XProf / TB Profile)
    # -----------------------------------------------------------------
    rank_logdir = os.path.join(LOGDIR_BASE, f"dp_c4_rank{rank}")
    os.makedirs(rank_logdir, exist_ok=True)

    xm.master_print(
        f"[DP-C4][rank {rank}] starting trace into {rank_logdir} "
        f"for {PROFILE_STEPS} steps"
    )

    # Start XLA trace capture for this rank
    xp.start_trace(rank_logdir)

    for step in range(1, PROFILE_STEPS + 1):
        model.train()
        opt.zero_grad()

        batch = next(data_iter)
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)

        t0 = time.time()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            labels=input_ids,
        )
        loss = outputs.loss
        loss.backward()
        xm.optimizer_step(opt, barrier=True)
        xm.mark_step()

        step_time_ms = (time.time() - t0) * 1000.0
        loss_val = float(loss)
        tokens_per_sec_core = tokens_per_step_per_core / (step_time_ms / 1000.0)
        tokens_per_sec_global = tokens_per_sec_core * world_size

        if rank == 0:
            xm.master_print(
                f"[DP-C4][profile] step {step}/{PROFILE_STEPS}, "
                f"loss={loss_val:.4f}, step_time_ms={step_time_ms:.1f}, "
                f"tokens/s/core={tokens_per_sec_core:.1f}"
            )

            if writer is not None:
                writer.add_scalar("train/loss", loss_val, global_step)
                writer.add_scalar("train/step_time_ms", step_time_ms, global_step)
                writer.add_scalar(
                    "train/tokens_per_sec_per_core", tokens_per_sec_core, global_step
                )
                writer.add_scalar(
                    "train/global_tokens_per_sec", tokens_per_sec_global, global_step
                )

        global_step += 1

    xp.stop_trace()
    xm.master_print(f"[DP-C4][rank {rank}] profiling done.")

    if writer is not None:
        writer.flush()
        writer.close()

    xm.rendezvous("dp_c4_done")


def _mp_entry(rank: int):
    train_loop(rank)


if __name__ == "__main__":
    # PJRT mode: nprocs must be 1 or omitted.
    # Omitting nprocs => spawn as many workers as devices (e.g. 4 on v6e-4).
    xmp.spawn(_mp_entry, args=())