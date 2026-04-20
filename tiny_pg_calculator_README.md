# tiny_pg_calculator.py (minimal-input version)

This calculator now accepts **only** the requested inputs:

- `V, d, L, m, H, K`
- `tokens_per_byte`
- `val_loss_estimate`
- `weight_bits`
- `compression_ratio`
- `train_seconds`
- `eval_seconds`

All other planning knobs are fixed as internal assumptions in the script.

## Quick start

```bash
python tiny_pg_calculator.py \
  --V 8192 --d 512 --L 11 --m 4 --H 8 --K 4 \
  --tokens-per-byte 0.26834 \
  --val-loss-estimate 2.7918 \
  --weight-bits 6.0 --compression-ratio 1.60 \
  --train-seconds 560 --eval-seconds 520
```

## JSON output

```bash
python tiny_pg_calculator.py ... --json
```

Useful for wiring into a web frontend.

## Fixed internal assumptions

- artifact cap: `16,000,000` bytes
- train/eval caps: `600s` each
- safety margin: `5%`
- code bytes assumption: `20,000`
- quant metadata overhead: `3%`
- param estimator structure constants:
  - `ctrl_mul = 4.0`
  - `include_skip = True`
  - `extra_mul = 1.0`

If you want to tune these in your deployed app, expose them as advanced fields in your UI.
