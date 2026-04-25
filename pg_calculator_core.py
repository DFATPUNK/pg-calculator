from __future__ import annotations

import math
from dataclasses import asdict, dataclass

LN2 = math.log(2.0)

# Fixed planning assumptions for the minimal-input calculator.
ARTIFACT_CAP_BYTES = 16_000_000
TRAIN_CAP_SECONDS = 600.0
EVAL_CAP_SECONDS = 600.0
SAFETY_MARGIN = 0.05
CODE_BYTES_ASSUMPTION = 20_000
SCALE_OVERHEAD_FRAC = 0.03
CTRL_MUL = 4.0
INCLUDE_SKIP = True
EXTRA_MUL = 1.0

@dataclass
class CalculatorInputs:
    V: int
    d: int
    L: int
    m: float
    H: int
    K: int
    tokens_per_byte: float
    val_loss_estimate: float
    weight_bits: float
    compression_ratio: float
    train_seconds: float
    eval_seconds: float

@dataclass
class CalculatorResult:
    V: int
    d: int
    L: int
    m: float
    H: int
    K: int
    d_kv: int
    N_params: int
    N_embedding: int
    N_per_layer_attn: int
    N_per_layer_mlp: int
    N_per_layer_ctrl: int
    N_skip: int
    N_extra: int
    quantized_model_bytes_est: int
    total_artifact_bytes_est: int
    tokens_per_byte_est: float
    val_loss_est: float
    val_bpb_pred: float
    artifact_ok: bool
    train_runtime_ok: bool
    eval_runtime_ok: bool
    eligible: bool

class CalculatorError(ValueError):
    pass

def _compute_params(V: int, d: int, L: int, m: float, H: int, K: int) -> tuple[int, dict[str, int]]:
    if H <= 0 or K <= 0:
        raise CalculatorError("H and K must be > 0")
    if H % K != 0:
        raise CalculatorError("H must be divisible by K for GQA")

    d_kv = d * K // H
    n_emb = V * d
    n_attn = d * d + d * d_kv + d * d_kv + d * d + H
    n_mlp = int(2 * d * (m * d))
    n_ctrl = int(CTRL_MUL * d)
    n_skip = (min(L // 2, L - (L // 2)) * d) if INCLUDE_SKIP else 0
    n_extra = int(EXTRA_MUL * L * d)

    total = n_emb + L * (n_attn + n_mlp + n_ctrl) + n_skip + n_extra
    return total, {
        "d_kv": d_kv,
        "N_embedding": n_emb,
        "N_per_layer_attn": n_attn,
        "N_per_layer_mlp": n_mlp,
        "N_per_layer_ctrl": n_ctrl,
        "N_skip": n_skip,
        "N_extra": n_extra,
    }

def _estimate_artifact_bytes(N_params: int, weight_bits: float, compression_ratio: float) -> tuple[int, int]:
    if compression_ratio <= 0:
        raise CalculatorError("compression_ratio must be > 0")
    if weight_bits <= 0:
        raise CalculatorError("weight_bits must be > 0")
    raw_bytes = N_params * (weight_bits / 8.0)
    with_overhead = raw_bytes * (1.0 + SCALE_OVERHEAD_FRAC)
    compressed_model = int(round(with_overhead / compression_ratio))
    total_artifact = compressed_model + CODE_BYTES_ASSUMPTION
    return compressed_model, total_artifact

def calculate(inputs: CalculatorInputs) -> CalculatorResult:
    N_params, parts = _compute_params(inputs.V, inputs.d, inputs.L, inputs.m, inputs.H, inputs.K)
    model_bytes, total_artifact = _estimate_artifact_bytes(N_params, inputs.weight_bits, inputs.compression_ratio)
    val_bpb = (inputs.val_loss_estimate / LN2) * inputs.tokens_per_byte

    artifact_limit = int(ARTIFACT_CAP_BYTES * (1.0 - SAFETY_MARGIN))
    train_limit = TRAIN_CAP_SECONDS * (1.0 - SAFETY_MARGIN)
    eval_limit = EVAL_CAP_SECONDS * (1.0 - SAFETY_MARGIN)

    artifact_ok = total_artifact <= artifact_limit
    train_ok = inputs.train_seconds <= train_limit
    eval_ok = inputs.eval_seconds <= eval_limit

    return CalculatorResult(
        V=inputs.V,
        d=inputs.d,
        L=inputs.L,
        m=inputs.m,
        H=inputs.H,
        K=inputs.K,
        d_kv=parts["d_kv"],
        N_params=N_params,
        N_embedding=parts["N_embedding"],
        N_per_layer_attn=parts["N_per_layer_attn"],
        N_per_layer_mlp=parts["N_per_layer_mlp"],
        N_per_layer_ctrl=parts["N_per_layer_ctrl"],
        N_skip=parts["N_skip"],
        N_extra=parts["N_extra"],
        quantized_model_bytes_est=model_bytes,
        total_artifact_bytes_est=total_artifact,
        tokens_per_byte_est=inputs.tokens_per_byte,
        val_loss_est=inputs.val_loss_estimate,
        val_bpb_pred=val_bpb,
        artifact_ok=artifact_ok,
        train_runtime_ok=train_ok,
        eval_runtime_ok=eval_ok,
        eligible=artifact_ok and train_ok and eval_ok,
    )

def result_to_dict(result: CalculatorResult) -> dict:
    return asdict(result)