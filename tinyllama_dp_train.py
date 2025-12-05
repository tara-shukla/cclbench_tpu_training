import os
import time

import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoTokenizer, AutoModelForCausalLM

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.debug.profiler as xp
import torch_xla.runtime as xr


# ------------------------
# Config
# ------------------------

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

BATCH_PER_CORE = 1     # per TPU core
SEQ_LEN = 128
NUM_STEPS = 20
LEARNING_RATE = 1e-4

PROFILE_START_STEP = 5
PROFILE_END_STEP = 10

LOGDIR = "/home/apk67/tinyllama_profile_dp"
TB_LOGDIR = os.path.join(LOGDIR, "events")


def _mp_worker(index):
    """
    One worker per TPU device. Data parallel:
      - each worker has its own model replica
      - gradients are allreduced in xm.optimizer_step
    PJRT passes `index` = 0..(num_devices-1).
    """
    rank = index  # logical rank for this process

    # Ask PJRT how many devices we actually have
    world_size = xr.global_runtime_device_count()

    # Logical XLA device for this worker
    device = xm.xla_device()  # deprecation warning is fine
    print(f"[rank {rank}] Using device {device}, world_size={world_size}")

    # Only rank 0 writes logs & controls tracing
    if rank == 0:
        os.makedirs(LOGDIR, exist_ok=True)
        os.makedirs(TB_LOGDIR, exist_ok=True)
        writer = SummaryWriter(log_dir=TB_LOGDIR)
        print("[rank 0] Loading tokenizer and model...")
        xp.start_server(9012)
    else:
        writer = None

    # ------------------------
    # Load tokenizer + model
    # ------------------------
    # All ranks load the same model (data parallel)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,  # `torch_dtype` deprecation warning is OK here
        device_map=None,
    )

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    model.to(device)

    vocab_size = tokenizer.vocab_size if tokenizer.vocab_size is not None else 32000
    if rank == 0:
        print(f"[rank 0] Vocab size: {vocab_size}")
        print("[rank 0] Starting TinyLlama DP profiling run...")

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    trace_started = False
    global_step = 0  # only meaningful on rank 0

    # Different seeds per rank just to decorrelate a bit
    torch.manual_seed(1234 + rank)

    for step in range(NUM_STEPS):
        step_start = time.time()

        # Synthetic batch (per core)
        input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(BATCH_PER_CORE, SEQ_LEN),
            device=device,
            dtype=torch.long,
        )
        labels = input_ids.clone()

        # Forward
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Optimizer step across all replicas (includes gradient sync)
        xm.optimizer_step(optimizer, barrier=True)
        xm.mark_step()

        step_time = time.time() - step_start
        tokens_per_core = BATCH_PER_CORE * SEQ_LEN
        global_tokens = tokens_per_core * world_size
        global_tokens_per_sec = global_tokens / step_time

        # Just log rank 0's loss; it's representative in DP
        if rank == 0:
            print(
                f"[rank 0] Step {step} | "
                f"loss={loss.item():.4f} | "
                f"step_time={step_time:.3f}s | "
                f"global_tokens/s={global_tokens_per_sec:.1f}"
            )

            if writer is not None:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/step_time", step_time, global_step)
                writer.add_scalar(
                    "train/global_tokens_per_sec",
                    global_tokens_per_sec,
                    global_step,
                )
                global_step += 1

            # ---- XLA trace: only rank 0 controls it ----
            if step == PROFILE_START_STEP and not trace_started:
                print(f"[rank 0] >>> Starting XLA trace at step {step}, logdir={LOGDIR}")
                xp.start_trace(LOGDIR)
                trace_started = True

            if step == PROFILE_END_STEP and trace_started:
                print(f"[rank 0] >>> Stopping XLA trace at step {step}")
                xp.stop_trace()
                trace_started = False

        # xm.optimizer_step(..., barrier=True) already syncs ranks; no extra barrier needed

    if trace_started and rank == 0:
        print("[rank 0] >>> Stopping XLA trace at end of run")
        xp.stop_trace()

    if writer is not None and rank == 0:
        writer.close()
        print("[rank 0] TinyLlama DP profiling complete.")


def main():
    # PJRT requires nprocs=None or 1; None = "use all devices"
    xmp.spawn(_mp_worker)


if __name__ == "__main__":
    main()
