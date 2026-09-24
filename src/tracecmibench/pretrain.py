"""Phase 1: train one decoder on a corpus's training split by next-token prediction.

    python -m tracecmibench.pretrain --corpus <dir> --ordering end --grain session --max-len 64 \
        --n-layers 6 --d-model 256 --n-heads 8 --ff-mult 2 --dropout 0.0 --rope-theta 10000 \
        --lr 3e-4 --adam-beta1 0.9 --adam-beta2 0.95 --warmup-steps 500 --lr-schedule cosine \
        --weight-decay 0.1 --batch-size 256 --steps 12000 --grad-clip 1.0 --amp bf16 \
        --val-every 1000 --val-batches 50 --checkpoint-every 1000 --entropy-order 2 \
        --seed 0 --device cuda --output-folder <run-dir>

Nothing causal-specific: plain maximum likelihood (Eq. 1). One model per
corpus, trained on the `end-session` view and shared by every arm scored on
that corpus (D-CB-17). A checkpoint is written before every validation and
`model.pt` at the end; `results.json.model_sha256` is the hash every arm and
comparison record carries. The oracle score (Eq. 22) is reported against an
independent order-k entropy floor (D-CB-7) and never gated; the budget
fields (`configs_tried_on_val`, `total_steps`, `wall_clock_s`) are what the
external baseline's record is compared with (PRD scenario 21).
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from .constants import CHECKPOINT_FMT, DEVICES, GRAINS, LR_SCHEDULES, MODEL_PT, ORACLE_IN_REGIME, ORDERINGS
from .corpus import Corpus
from .data import SequenceStore, endless_batches
from .entropy import oracle_score, order_k_floor
from .log import log
from .model import DecoderLM, lr_at
from .record import RunRecord
from .vocab import Vocab

AMP_MODES = ("none", "bf16")


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validate(model, store, batch_size, val_batches, device, amp):
    """Mean NLL per non-PAD target over the first `val_batches` length-sorted batches."""
    model.eval()
    total, count = 0.0, 0
    for k, idx in enumerate(store.batches(batch_size)):
        if k >= val_batches:
            break
        ids, pad = store.collate(idx, device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(amp == "bf16")):
            loss, n = model.loss(ids, pad)
        total += float(loss) * n
        count += n
    model.train()
    return total / max(count, 1), count


def pretrain(args, out_dir):
    seed_all(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda but CUDA is not available")
    corpus = Corpus(args.corpus, args.ordering, args.grain)
    vocab = Vocab.from_model_vocab(corpus.vocab_json())
    train = SequenceStore.from_corpus(corpus, "train", vocab, args.max_len)
    val = SequenceStore.from_corpus(corpus, "val", vocab, args.max_len)
    floor = order_k_floor(train, args.entropy_order, vocab.size)
    log_alphabet = math.log(len(vocab.predictable_ids))
    log({"event": "data", "train_sequences": len(train), "train_tokens": train.n_tokens, "val_sequences": len(val),
         "truncated_train": train.n_truncated, "folded_train": train.n_folded, "vocab_size": vocab.size,
         "entropy_floor": floor["entropy_floor"], "entropy_order": args.entropy_order, "log_alphabet": log_alphabet})

    model = DecoderLM(vocab.size, args.n_layers, args.d_model, args.n_heads, args.ff_mult, args.dropout, args.rope_theta).to(device)
    log({"event": "model", "params": model.n_params, **model.config})
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(args.adam_beta1, args.adam_beta2))
    stream = endless_batches(train, args.batch_size, args.seed)
    val_curve = []
    checkpoints = []
    t0 = time.monotonic()
    tokens_seen = 0
    running, running_n = 0.0, 0
    model.train()
    for step in range(args.steps):
        lr = lr_at(step, args.lr, args.warmup_steps, args.steps, args.lr_schedule)
        for g in opt.param_groups:
            g["lr"] = lr
        if step % args.checkpoint_every == 0 and step > 0:
            path = out_dir / CHECKPOINT_FMT.format(step=step)
            sha = model.save(path, {"step": step, "vocab_size": vocab.size})
            checkpoints.append({"step": step, "file": path.name, "sha256": sha})
        if step % args.val_every == 0 and step > 0:
            vloss, vn = validate(model, val, args.batch_size, args.val_batches, device, args.amp)
            val_curve.append({"step": step, "loss": vloss, "n_targets": vn})
            log({"event": "val", "step": step, "val_loss": vloss, "train_loss": running / max(running_n, 1), "lr": lr,
                 "tok_pos_per_s": tokens_seen / max(time.monotonic() - t0, 1e-9)})
            running, running_n = 0.0, 0
        ids, pad = train.collate(next(stream), device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(args.amp == "bf16")):
            loss, n = model.loss(ids, pad)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        tokens_seen += n
        running += float(loss)
        running_n += 1
    train_wall = time.monotonic() - t0
    vloss, vn = validate(model, val, args.batch_size, args.val_batches, device, args.amp)
    val_curve.append({"step": args.steps, "loss": vloss, "n_targets": vn})
    model_path = out_dir / MODEL_PT
    model_sha = model.save(model_path, {"step": args.steps, "vocab_size": vocab.size})
    oracle = oracle_score(vloss, floor["entropy_floor"], log_alphabet, ORACLE_IN_REGIME)
    oracle["entropy_order"] = args.entropy_order
    results = {
        "model_file": MODEL_PT, "model_sha256": model_sha, "params": model.n_params, "model_config": model.config,
        "vocab_size": vocab.size, "train_sequences": len(train), "train_tokens": train.n_tokens,
        "truncated_train": train.n_truncated, "folded_train": train.n_folded,
        "train_loss_last": running / max(running_n, 1), "val_curve": val_curve, "checkpoints": checkpoints,
        "oracle": oracle, "entropy": floor,
        "budget": {"configs_tried_on_val": 1, "total_steps": args.steps, "wall_clock_s": round(train_wall, 3)},
        "tok_pos_per_s": tokens_seen / max(train_wall, 1e-9), "tokens_seen": tokens_seen,
    }
    log({"event": "pretrain_done", "model_sha256": model_sha, "val_loss": vloss, "eps_hat": oracle["eps_hat"],
         "in_regime": oracle["in_regime"], "tok_pos_per_s": results["tok_pos_per_s"]})
    return results


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", required=True)
    p.add_argument("--ordering", required=True, choices=ORDERINGS)
    p.add_argument("--grain", required=True, choices=GRAINS)
    p.add_argument("--max-len", required=True, type=int)
    p.add_argument("--n-layers", required=True, type=int)
    p.add_argument("--d-model", required=True, type=int)
    p.add_argument("--n-heads", required=True, type=int)
    p.add_argument("--ff-mult", required=True, type=float, help="SwiGLU hidden = ff_mult * d_model (2 gives the 4.2M anchor at |X| = 1000)")
    p.add_argument("--dropout", required=True, type=float)
    p.add_argument("--rope-theta", required=True, type=float)
    p.add_argument("--lr", required=True, type=float)
    p.add_argument("--adam-beta1", required=True, type=float)
    p.add_argument("--adam-beta2", required=True, type=float)
    p.add_argument("--warmup-steps", required=True, type=int)
    p.add_argument("--lr-schedule", required=True, choices=LR_SCHEDULES)
    p.add_argument("--weight-decay", required=True, type=float)
    p.add_argument("--batch-size", required=True, type=int)
    p.add_argument("--steps", required=True, type=int)
    p.add_argument("--grad-clip", required=True, type=float, help="0 disables clipping")
    p.add_argument("--amp", required=True, choices=AMP_MODES)
    p.add_argument("--val-every", required=True, type=int)
    p.add_argument("--val-batches", required=True, type=int)
    p.add_argument("--checkpoint-every", required=True, type=int)
    p.add_argument("--entropy-order", required=True, type=int, help="order k of the n-gram entropy floor (D-CB-7)")
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--device", required=True, choices=DEVICES)
    p.add_argument("--output-folder", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    with RunRecord(args.output_folder, "pretrain", vars(args)) as rec:
        results = pretrain(args, rec.out_dir)
        rec.note(tok_pos_per_s=results["tok_pos_per_s"])
        rec.finish(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
