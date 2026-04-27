#!/usr/bin/env python3
from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from pg_calculator_core import (
    CalculatorError,
    CalculatorInputs,
    calculate,
    result_to_dict,
)

app = Flask(__name__)


DEFAULTS = {
    "V": 8192,
    "d": 512,
    "L": 11,
    "m": 4,
    "H": 8,
    "K": 4,
    "tokens_per_byte": 0.26834,
    "val_loss_estimate": 2.7918,
    "weight_bits": 6.0,
    "compression_ratio": 1.6,
    "train_seconds": 560,
    "eval_seconds": 520,
}


def _coerce_float(name: str, data: dict) -> float:
    try:
        return float(data[name])
    except Exception as e:
        raise CalculatorError(f"Invalid value for {name}") from e


def _coerce_int(name: str, data: dict) -> int:
    try:
        return int(data[name])
    except Exception as e:
        raise CalculatorError(f"Invalid value for {name}") from e


def _parse_inputs(data: dict) -> CalculatorInputs:
    return CalculatorInputs(
        V=_coerce_int("V", data),
        d=_coerce_int("d", data),
        L=_coerce_int("L", data),
        m=_coerce_float("m", data),
        H=_coerce_int("H", data),
        K=_coerce_int("K", data),
        tokens_per_byte=_coerce_float("tokens_per_byte", data),
        val_loss_estimate=_coerce_float("val_loss_estimate", data),
        weight_bits=_coerce_float("weight_bits", data),
        compression_ratio=_coerce_float("compression_ratio", data),
        train_seconds=_coerce_float("train_seconds", data),
        eval_seconds=_coerce_float("eval_seconds", data),
    )


@app.get("/")
def root():
    return render_template("pg_calculator.html", defaults=DEFAULTS)


@app.get("/pg-calculator")
def pg_calculator_page():
    return render_template("pg_calculator.html", defaults=DEFAULTS)


@app.post("/api/pg-calculator")
def pg_calculator_api():
    payload = request.get_json(force=True, silent=True) or {}

    try:
        result = calculate(_parse_inputs(payload))
        return jsonify({"ok": True, "result": result_to_dict(result)})
    except CalculatorError as e:
        return jsonify({"ok": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)