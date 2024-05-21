# selective-ssms-and-linear-cdes
Theoretical Foundations of Deep Selective State-Space Models: The experiments

## Dependencies

### The Jax Environment

```angular2html
conda create -n jax_cde python=3.11
conda activate jax_cde
conda install pre-commit numpy scikit-learn matplotlib
pip install -U "jax[cuda12]"
pip install diffrax signax==0.1.1
pre-commit install
```


### The PyTorch Environment

- Python 3.11
- pre-commit