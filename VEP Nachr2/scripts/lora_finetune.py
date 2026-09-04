"""
LoRA fine-tune of ESM-2 150M for 3-class GOF/NNE/LOF direction (idea #1).

Parameter-efficient fine-tuning (Sledzieski et al. 2024): freezes the ESM-2
backbone and trains low-rank adapters (LoRA) on the attention query/key/value
projections plus a freshly-initialised classification head. With r=8 this is
~1% of the model's parameters, which is the right inductive bias for a 760-
variant dataset.

Input representation: the *mutated* reference sequence (wildtype human sequence
with the variant AA substituted at the human-numbered position). The model must
learn to attend to the single changed residue and map it to direction.

CV discipline: leave-one-subunit-out (StratifiedGroupKFold by gene), the same
honest split as the classical pipeline. LoRA adapters + head are trained from
scratch on the training genes only for each fold; test-gene variants are never
seen during fine-tuning.

Usage:
    python scripts/lora_finetune.py --smoke          # 1 fold, 1 epoch (sanity)
    python scripts/lora_finetune.py                  # 1 seed, full CV
    python scripts/lora_finetune.py --seeds 42 123 456 789 1011
"""

import argparse
import os
import time
from pathlib import Path

# Must be set before torch/CUDA init: avoids allocator fragmentation on the
# laptop RTX 4050 (6 GB, WDDM unified-memory), which otherwise reports spurious
# "16 GiB allocated" OOMs on a ~600 MB model.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from vep_nachr2.config import RESULTS_DIR, DEFAULT_SEEDS
from vep_nachr2.data.loader import load_mutation_data, make_subunit_group_key
from vep_nachr2.data.reference import load_all_reference_sequences
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score


MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
LORA_TARGET_MODULES = ["query", "key", "value"]  # canonical ESM-2 LoRA targets
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
N_EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
MAX_LEN = 1024


def build_mutant_sequence(ref: str, position: int, variant_aa: str) -> str:
    """Substitute variant_aa into the reference at the 1-based position."""
    pos = position - 1
    if pos < 0 or pos >= len(ref):
        return ref
    return ref[:pos] + variant_aa + ref[pos + 1:]


def make_sequences(df: pd.DataFrame, ref_seqs: dict[str, str]) -> list[str]:
    """Build one mutated-sequence string per variant row."""
    seqs = []
    for _, row in df.iterrows():
        gene = str(row["subunit"]).upper()
        ref = ref_seqs.get(gene, "")
        if not ref:
            seqs.append("")
            continue
        seqs.append(build_mutant_sequence(ref, int(row["position"]),
                                          str(row["variant_aa"]).upper()))
    return seqs


def tokenize(tok, seqs: list[str]):
    enc = tok(seqs, padding=True, truncation=True, max_length=MAX_LEN,
              return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def train_one_fold(model, device, X_ids, X_mask, y, class_weights, n_epochs=N_EPOCHS):
    """Train LoRA model on a fold's data. Returns the trained (eval-mode) model."""
    from transformers import get_linear_schedule_with_warmup
    from torch.optim import AdamW

    model.train()
    # Trade ~20% speed for a much smaller activation graph on the 6 GB GPU.
    if hasattr(model, "base_model") and hasattr(model.base_model,
                                                "gradient_checkpointing_enable"):
        model.base_model.gradient_checkpointing_enable()

    ds = TensorDataset(X_ids, X_mask, torch.as_tensor(y, dtype=torch.long))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    # Freeze non-LoRA / non-head parameters already handled by peft; optimiser
    # sees only trainable params.
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)
    total_steps = len(loader) * n_epochs
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=0,
                                            num_training_steps=total_steps)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    for _ in range(n_epochs):
        for ids, mask, labels in loader:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            out = model(input_ids=ids, attention_mask=mask)
            loss = loss_fn(out.logits, labels)
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()

    model.eval()
    return model


@torch.no_grad()
def predict_fold(model, device, X_ids, X_mask):
    model.eval()
    preds = []
    ds = TensorDataset(X_ids, X_mask)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    for ids, mask in loader:
        ids, mask = ids.to(device), mask.to(device)
        out = model(input_ids=ids, attention_mask=mask)
        preds.append(out.logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS[:1])
    parser.add_argument("--smoke", action="store_true",
                        help="1 fold, 1 epoch sanity run")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    print("Loading data...")
    df = load_mutation_data()
    y = df["effect"].map({"LOF": 0, "No net effect": 1, "GOF": 2}).values.astype(np.int64)
    ref_seqs = load_all_reference_sequences("human")
    seqs = make_sequences(df, ref_seqs)
    groups = make_subunit_group_key(df)

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import get_peft_model, LoraConfig, TaskType

    print(f"Loading {MODEL_NAME} (device={device})...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    all_f1 = []

    for seed in args.seeds:
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        split_pairs = list(outer.split(np.zeros(len(y)), y, groups))
        if args.smoke:
            split_pairs = split_pairs[:1]

        fold_f1 = []
        for fold_idx, (train_idx, test_idx) in enumerate(split_pairs):
            print(f"  Seed {seed} fold {fold_idx + 1}/{len(split_pairs)}: "
                  f"train={len(train_idx)} test={len(test_idx)}", flush=True)

            torch.cuda.empty_cache()
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, num_labels=3
            )
            cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=LORA_R,
                             lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                             target_modules=LORA_TARGET_MODULES)
            model = get_peft_model(model, cfg).to(device)

            tr_ids, tr_mask = tokenize(tok, [seqs[i] for i in train_idx])
            te_ids, te_mask = tokenize(tok, [seqs[i] for i in test_idx])

            # Class weights = inverse frequency (matches pipeline imbalance).
            classes, counts = np.unique(y[train_idx], return_counts=True)
            cw = torch.zeros(3, dtype=torch.float32)
            for c, n in zip(classes, counts):
                cw[c] = len(train_idx) / (3.0 * n)

            n_epochs = 1 if args.smoke else N_EPOCHS
            model = train_one_fold(model, device, tr_ids, tr_mask, y[train_idx], cw, n_epochs)
            preds = predict_fold(model, device, te_ids, te_mask)

            f1 = f1_score(y[test_idx], preds, average="macro",
                          labels=[0, 1, 2], zero_division=0)
            fold_f1.append(f1)
            print(f"    macro F1: {f1:.4f}", flush=True)

            del model
            torch.cuda.empty_cache()

        seed_f1 = float(np.mean(fold_f1))
        all_f1.append(seed_f1)
        print(f"  Seed {seed} mean macro F1: {seed_f1:.4f}")

    mean = float(np.mean(all_f1))
    std = float(np.std(all_f1)) if len(all_f1) > 1 else 0.0
    print(f"\nLoRA ESM-2 3-class macro F1: {mean:.4f} +/- {std:.4f}")

    out_dir = RESULTS_DIR / "new_ideas"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"seed": args.seeds, "macro_f1_mean": all_f1}).to_csv(
        out_dir / "lora_results.csv", index=False
    )
    print(f"Total time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
