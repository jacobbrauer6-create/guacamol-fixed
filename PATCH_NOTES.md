# GuacaMol — Community Patch Notes

## Version 0.5.5.post1

### Why this patch exists

`guacamol==0.5.4` fails on modern Python/NumPy environments:

1. **Hard-pinned `rdkit-pypi` version requirement**
   - The package `rdkit-pypi` was renamed to `rdkit` on PyPI around 2022.
   - `guacamol` pinned an old version that conflicts with the current `rdkit` package.
   - Solution: Removed version pin. Accepts `rdkit >=2022.3` under either name.

2. **NumPy 2.0: removed `np.bool`, `np.int`, `np.float` type aliases**
   - Error: `AttributeError: module 'numpy' has no attribute 'bool'`
   - These aliases were deprecated in NumPy 1.20 and removed in 2.0.
   - Solution: All instances replaced with Python built-in `bool`, `int`, `float`.

3. **scikit-learn 1.x API changes**
   - `check_is_fitted` and `NotFittedError` import paths changed.
   - Solution: Updated import paths; removed deprecated validator usage.

4. **RDKit 2022+ API changes**
   - Some internal `Chem.MolToSmiles` default arguments changed.
   - Solution: Added explicit flags where behaviour was relied upon.

### Changes summary

| File | Change |
|------|--------|
| `setup.py` | Replaced by `pyproject.toml` |
| `guacamol/common_scoring_functions.py` | `np.bool/np.float` → builtins; sklearn fix |
| `guacamol/scoring_function.py` | `np.float` → `float` throughout |
| `guacamol/utils/chemistry.py` | RDKit API fixes; numpy fixes |
| `pyproject.toml` | Removed rdkit-pypi pin; Python 3.9-3.12 classifiers |

### Install

```bash
pip install rdkit fcd_torch
pip install git+https://github.com/YOUR_USERNAME/guacamol.git
```

### Fork and PR

- **PR to original repo**: https://github.com/BenevolentAI/guacamol/pulls
- **This fork**: https://github.com/YOUR_USERNAME/guacamol
