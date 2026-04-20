# Deploying `/pg-calculator`

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-pg-calculator.txt
python pg_calculator_web.py
```

Open: `http://localhost:8000/pg-calculator`

## Production hint

Use any WSGI host, e.g. gunicorn:

```bash
gunicorn -w 2 -b 0.0.0.0:8000 pg_calculator_web:app
```

Then reverse-proxy `/pg-calculator` and `/api/pg-calculator` from your domain.

## Notes

This is the minimal-input version (no advanced knobs in UI yet).