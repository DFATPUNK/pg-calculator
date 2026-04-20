#!/usr/bin/env python3
"""Tiny Parameter Golf calculator (minimal-input version)."""

from __future__ import annotations

import argparse
import json

from pg_calculator_core import CalculatorInputs, CalculatorError, calculate, result_to_dict


def main() -> None:
    p = argparse.ArgumentParser(description="Tiny Parameter Golf calculator (minimal inputs)")

    # Only requested inputs.
    p.add_argument("--V", type=int, required=True, help="Vocabulary size")
    p.add_argument("--d", type=int, required=True, help="Model width / hidden size")
    p.add_argument("--L", type=int, required=True, help="Number of layers")
    p.add_argument("--m", type=float, required=True, help="MLP multiplier")
    p.add_argument("--H", type=int, required=True, help="Number of query heads")
    p.add_argument("--K", type=int, required=True, help="Number of KV heads")

    p.add_argument("--tokens-per-byte", type=float, required=True, help="Estimated T/B for chosen tokenizer family")
    p.add_argument("--val-loss-estimate", type=float, required=True, help="Estimated val_loss (nats/token)")
    p.add_argument("--weight-bits", type=float, required=True, help="Average stored bits per parameter")
    p.add_argument("--compression-ratio", type=float, required=True, help="Estimated compression ratio after packing")
    p.add_argument("--train-seconds", type=float, required=True, help="Estimated training runtime in seconds")
    p.add_argument("--eval-seconds", type=float, required=True, help="Estimated eval runtime in seconds")

    p.add_argument("--json", action="store_true", help="Print JSON only")
    args = p.parse_args()

    try:
        result = calculate(
            CalculatorInputs(
                V=args.V,
                d=args.d,
                L=args.L,
                m=args.m,
                H=args.H,
                K=args.K,
                tokens_per_byte=args.tokens_per_byte,
                val_loss_estimate=args.val_loss_estimate,
                weight_bits=args.weight_bits,
                compression_ratio=args.compression_ratio,
                train_seconds=args.train_seconds,
                eval_seconds=args.eval_seconds,
            )
        )
    except CalculatorError as e:
        raise SystemExit(f"error: {e}") from e

    output = result_to_dict(result)

    if args.json:
        print(json.dumps(output, indent=2))
        return

    print("=== Tiny Parameter Golf Calculator (minimal inputs) ===")
    print(f"N_params: {result.N_params:,}")
    print(
        f"parts: emb={result.N_embedding:,}, attn/layer={result.N_per_layer_attn:,}, "
        f"mlp/layer={result.N_per_layer_mlp:,}, ctrl/layer={result.N_per_layer_ctrl:,}, "
        f"skip={result.N_skip:,}, extra={result.N_extra:,}"
    )
    print(f"quantized_model_bytes_est: {result.quantized_model_bytes_est:,}")
    print(f"total_artifact_bytes_est: {result.total_artifact_bytes_est:,}")
    print(f"tokens_per_byte_est: {result.tokens_per_byte_est:.6f}")
    print(f"val_loss_est: {result.val_loss_est:.6f}")
    print(f"val_bpb_pred: {result.val_bpb_pred:.6f}")
    print(
        f"eligible: {result.eligible} "
        f"(artifact_ok={result.artifact_ok}, train_runtime_ok={result.train_runtime_ok}, eval_runtime_ok={result.eval_runtime_ok})"
    )


if __name__ == "__main__":
    main()