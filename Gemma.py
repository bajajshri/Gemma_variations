import json 
import argparse
import math 
import os
import time
from dataclasses import asdict, dataclass
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F

LN2 = math.log(2)

@dataclass
class ModelConfig:            
    vocab_size: int = 4096    
    d_model: int = 512        # embedding size
    n_layers: int = 6          
    d_ff: int = 2048          # GeGLU hidden size
    n_heads: int = 8
    n_kv_heads: int = 8       
    head_dim: int = 64
    context: int = 512              
    rope_base: float = 10000.0  # based on notes (b)
    norm_eps: float = 1e-6


def init_params(cfg: ModelConfig, device, seed=0) -> dict:
    g = torch.Generator().manual_seed(seed)
    # initalize weights (Transformers are initalized to a small std)
    def w(*shape, std = 0.02):
        return (torch.randn(*shape, generator=g)*std).to(device).requires_grad_()
    def zeros(n):  # Gemma RMSNorm scales by (1 + w), so w = 0 starts as the identity
            return torch.zeros(n, device=device, requires_grad=True)
    d, hd, out_std = cfg.d_model, cfg.head_dim,  0.02 / math.sqrt(2 * cfg.n_layers)
    p = {"embed": w(cfg.vocab_size, d)}
    for l in range(cfg.n_layers):
        p.update({
            f"{l}.attn_norm": zeros(d),
            f"{l}.wq": w(cfg.n_heads * hd, d),
            f"{l}.wk": w(cfg.n_kv_heads * hd, d),
            f"{l}.wv": w(cfg.n_kv_heads * hd, d),
            f"{l}.wo": w(d, cfg.n_heads * hd, std = out_std),
            f"{l}.ffn_norm": zeros(d),
            f"{l}.w_gate": w(cfg.d_ff, d),  # Geglu
            f"{l}.w_up": w(cfg.d_ff, d),    #Geglu
            f"{l}.w_down": w(d, cfg.d_ff, std = out_std),
        }
        )
    p["final_norm"] = zeros(d)
    return p

def rms_norm(x,w, eps):
    xf = x.float()  # upcast to avoid numerical issues
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * (1.0 + w.float())).type_as(x)  # back to old precision

def rope_tables(T, hd, base, device):
    inv_freq = 1.0 / base ** (torch.arange(0, hd, 2, device=device).float() / hd)
    angles = torch.outer(torch.arange(T, device=device).float(), inv_freq)
    return angles.cos(), angles.sin()

def apply_rope(x, cos, sin):
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).type_as(x)

def forward(p , idx: torch.Tensor, cfg: ModelConfig, dropout = 0.0):
    drop = lambda t: F.dropout(t, dropout, training=dropout > 0)
    B,T = idx.shape 
    H, Hkv, hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim 
    cos, sin = rope_tables(T, hd, cfg.rope_base, idx.device)
    x = F.embedding(idx, p["embed"]) * math.sqrt(cfg.d_model)   # scale just the input to get 1 std
    for l in range (cfg.n_layers):
        h = rms_norm(x, p[f"{l}.attn_norm"], cfg.norm_eps)
        q = F.linear(h, p[f"{l}.wq"]).view(B, T, H, hd).transpose(1, 2)
        k = F.linear(h, p[f"{l}.wk"]).view(B, T, Hkv, hd).transpose(1, 2)
        v = F.linear(h, p[f"{l}.wv"]).view(B, T, Hkv, hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k ,cos, sin)
        if Hkv != H:                                                  # multi-query / grouped-query
            k, v = k.repeat_interleave(H // Hkv, 1), v.repeat_interleave(H // Hkv, 1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout)
        x = x + drop(F.linear(a.transpose(1, 2).reshape(B, T, H * hd), p[f"{l}.wo"]))
        # --- GeGLU feed-forward block (pre-norm) ---
        h = rms_norm(x, p[f"{l}.ffn_norm"], cfg.norm_eps)
        gate = F.gelu(F.linear(h, p[f"{l}.w_gate"]), approximate="tanh")
        x = x + drop(F.linear(gate * F.linear(h, p[f"{l}.w_up"]), p[f"{l}.w_down"]))
    x = rms_norm(x, p["final_norm"], cfg.norm_eps)
    return F.linear(x, p["embed"])      



# =============================================================================
# Data
# =============================================================================
def read_char_counts(jsonl_path):
    with open(jsonl_path, encoding="utf-8") as f:
        return [len(json.loads(line)["text"]) for line in f]


def load_data(data_dir, n_val_books=5):
    with open(os.path.join(data_dir, "tokenized_meta.json")) as f:
        meta = json.load(f)
    eos = meta["eos_id"]

    train = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    train_chars = sum(read_char_counts(os.path.join(data_dir, "train.jsonl")))
    train_text_tokens = len(train) - meta["train_books"]          # minus one </s> per book
    chars_per_token = train_chars / train_text_tokens

    # Validation: split val.bin back into books at </s>, pair each with its character count
    val = np.fromfile(os.path.join(data_dir, "val.bin"), dtype=np.uint16).astype(np.int64)
    ends = np.flatnonzero(val == eos)
    starts = np.concatenate([[0], ends[:-1] + 1])
    val_books = [val[s:e] for s, e in zip(starts, ends)]
    val_chars = read_char_counts(os.path.join(data_dir, "val.jsonl"))
    assert len(val_books) == len(val_chars), "val.bin and val.jsonl disagree on the number of books"
    if len(val_books) > n_val_books:
        print(f"Using the first {n_val_books} of {len(val_books)} held-out books for validation")
        val_books, val_chars = val_books[:n_val_books], val_chars[:n_val_books]
    elif len(val_books) < n_val_books:
        print(f"WARNING: only {len(val_books)} validation books; the assignment asks for {n_val_books}")
    return meta, train, chars_per_token, val_books, val_chars

def get_batch(data, batch_size, T, rng, device):
    ix = rng.integers(0, len(data) - T - 1, size=batch_size)
    chunk = torch.from_numpy(np.stack([data[i:i + T + 1] for i in ix]).astype(np.int64))
    if device == "cuda":
        chunk = chunk.pin_memory().to(device, non_blocking=True)
    return chunk[:, :-1], chunk[:, 1:]                             # input, next-token target


def build_val_windows(books, T, device):
    """Cut each book into consecutive windows; every token except a book's first is predicted once."""
    X, Y, owner = [], [], []
    for b, ids in enumerate(books):
        for i in range(0, len(ids) - 1, T):
            tgt = ids[i + 1:i + T + 1]
            n = len(tgt)
            x = np.zeros(T, np.int64)
            y = np.full(T, -100, np.int64)       # -100 = ignored padding in the last window
            x[:n], y[:n] = ids[i:i + n], tgt
            X.append(x); Y.append(y); owner.append(b)
    to = lambda a: torch.from_numpy(np.stack(a)).to(device)
    return to(X), to(Y), torch.tensor(owner)

@torch.no_grad()
def evaluate(params, cfg, val, amp_ctx, batch_size):
    X, Y, owner, books, chars = val
    nats = torch.zeros(len(books), dtype=torch.float64)
    for i in range(0, len(X), batch_size):
        with amp_ctx:
            logits = forward(params, X[i:i + batch_size], cfg)
        ce = F.cross_entropy(logits.float().transpose(1, 2), Y[i:i + batch_size],
                             ignore_index=-100, reduction="none")    # (b, T), 0 where ignored
        nats.index_add_(0, owner[i:i + batch_size], ce.sum(1).double().cpu())
    n_tok = np.array([len(b) for b in books], dtype=np.float64)
    n_pred = n_tok - 1
    n_char = np.array(chars, dtype=np.float64)
    per_book_loss = nats.numpy() / n_pred
    loss = nats.sum().item() / n_pred.sum()
    # bpc = (loss / ln 2) * n_tok / n_char, with n_char = len(cleaned text) before normalization
    bpc = loss / LN2 * n_tok.sum() / n_char.sum()
    per_book_bpc = (per_book_loss / LN2 * n_tok / n_char).tolist()
    return loss, bpc, per_book_bpc

# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default = "clean")   
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr", type=float, default=6e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="residual + attention dropout during training (0.1 helps against overfitting)")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--n-val-books", type=int, default=5, help="held-out books used for validation BPC")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--tiny", action="store_true", help="tiny model for a quick CPU smoke test") 
    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    latest_path = os.path.join(args.ckpt_dir, "latest.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    meta, train, chars_per_token, val_books, val_chars = load_data(args.data_dir, args.n_val_books)
    cfg = ModelConfig(vocab_size=meta["vocab_size"])
    if args.tiny:
            cfg = ModelConfig(vocab_size=meta["vocab_size"], d_model=64, n_layers=2, d_ff=256,
                              n_heads=4, n_kv_heads=4, head_dim=16, context=64)
    val_X, val_Y, val_owner = build_val_windows(val_books, cfg.context, device)
    val = (val_X, val_Y, val_owner, val_books, val_chars)
    print(f"train: {len(train):,} tokens ({chars_per_token:.3f} chars/token) | "
              f"val: {len(val_books)} books, {sum(map(len, val_books)):,} tokens, {sum(val_chars):,} chars")

    # --- mixed precision: bf16 on Ampere+, fp16 + loss scaling on T4 ---
    use_bf16 = device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    amp_ctx = torch.autocast("cuda", dtype=amp_dtype) if device == "cuda" else nullcontext()
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and not use_bf16))

    # --- model and optimizer ---
    params = init_params(cfg, device, seed=args.seed)
    n_params = sum(t.numel() for t in params.values())
    print(f"model: {n_params / 1e6:.2f}M parameters | device: {device}"
            + (f" ({torch.cuda.get_device_name()}, {amp_dtype})" if device == "cuda" else ""))
    decay = [t for t in params.values() if t.dim() >= 2]       # matrices
    no_decay = [t for t in params.values() if t.dim() < 2]     # norm scales
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},  # matrices
            {"params": no_decay, "weight_decay": 0.0}],            # norm scales RMS 
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=(device == "cuda"))
    # --- resume ---
    step = 0
    history = {"train_step": [], "train_loss": [], "val": []}
    best_bpc = float("inf")
    if os.path.exists(latest_path):
        ck = torch.load(latest_path, map_location=device, weights_only=False)
        with torch.no_grad():
            for k, t in params.items():
                t.copy_(ck["params"][k])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, history, best_bpc = ck["step"], ck["history"], ck["best_bpc"]
        print(f"resumed from {latest_path} at step {step} (best val BPC so far {best_bpc:.4f})")

    # --- training loop ---
    pending = []                          # losses kept on GPU, synced every log_every steps
    t_log = time.time()
    tokens_per_step = args.batch_size * cfg.context
    while step < args.max_steps:
        lr = lr_at(step, args)
        for g in opt.param_groups:
            g["lr"] = lr

        rng = np.random.default_rng([args.seed, step])          # same batches after a resume
        x, y = get_batch(train, args.batch_size, cfg.context, rng, device)
        with amp_ctx:
            logits = forward(params, x, cfg, dropout=args.dropout)   # evaluation/sampling use 0
        loss = F.cross_entropy(logits.float().view(-1, cfg.vocab_size), y.reshape(-1))

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params.values(), args.grad_clip)
        scaler.step(opt)
        scaler.update()
        step += 1
        pending.append(loss.detach())

        if step % args.log_every == 0 or step == args.max_steps:
            losses = torch.stack(pending).float().cpu().tolist()
            history["train_step"] += list(range(step - len(losses) + 1, step + 1))
            history["train_loss"] += losses
            pending = []
            dt = time.time() - t_log
            t_log = time.time()
            avg = sum(losses) / len(losses)
            tok_s = len(losses) * tokens_per_step / dt
            eta_h = (args.max_steps - step) * dt / len(losses) / 3600
            print(f"step {step:6d} | loss {avg:.4f} | bpc {avg / LN2 / chars_per_token:.4f} | "
                    f"lr {lr:.2e} | {tok_s / 1e3:.1f}k tok/s | ETA {eta_h:.2f} h", flush=True)

        if step % args.eval_every == 0 or step == args.max_steps:
            t_eval = time.time()
            v_loss, v_bpc, per_book = evaluate(params, cfg, val, amp_ctx, args.eval_batch_size)
            history["val"].append({"step": step, "loss": v_loss, "bpc": v_bpc, "per_book_bpc": per_book})
            is_best = v_bpc < best_bpc
            best_bpc = min(best_bpc, v_bpc)
            print(f"  >> val @ {step}: loss {v_loss:.4f} | BPC {v_bpc:.4f}"
                    f"{'  (new best)' if is_best else ''} | per book "
                    + ", ".join(f"{b:.3f}" for b in per_book) + f" | {time.time() - t_eval:.0f}s", flush=True)

            weights = {k: t.detach() for k, t in params.items()}
            if is_best:
                save_atomic({"params": weights, "cfg": asdict(cfg), "step": step, "val_bpc": v_bpc},
                            os.path.join(args.ckpt_dir, "best.pt"))
            save_atomic({"params": weights, "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                            "step": step, "history": history, "best_bpc": best_bpc,
                            "cfg": asdict(cfg), "args": vars(args)}, latest_path)
            with open(os.path.join(args.ckpt_dir, "history.json"), "w") as f:
                json.dump(history, f)
            plot_curves(history, chars_per_token, os.path.join(args.ckpt_dir, "curves.png"))
            t_log = time.time()            # don't count eval time in throughput

    print(f"Done. Best validation BPC: {best_bpc:.4f} -> {os.path.join(args.ckpt_dir, 'best.pt')}")


if __name__ == "__main__":
    main()