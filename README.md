## Quickstart (CPU)
```bash
python -m venv .venv
# activate venv...
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy matplotlib

python -m scripts.benchmark_all