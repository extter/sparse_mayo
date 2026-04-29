# Computational Imaging — Group A

## Project: Sparse-Views CT Reconstruction

### Methods
- Total Variation (TV) regularization
- End-to-end neural network (UNet)
- Diffusion model with DPS
- Plug-and-Play with Half Quadratic Splitting (HQS)

### How to run
1. Install dependencies: `pip install -r requirements.txt`
2. Preprocess data: `python scripts/preprocess.py`
3. Run a method: `python scripts/run_tv.py`
4. Compare all results: `python scripts/compare_all.py`
