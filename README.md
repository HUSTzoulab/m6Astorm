m6Astorm: Stoichiometry-preserving and stochasticity-aware identification of m6A from direct RNA sequencing

- [Preprocess](#preprocess)
- [m6Astorm](#m6astorm)
  - [Feature extraction](#feature-extraction)
  - [Predict](#predict)

# Preprocess

Before runing m6Astorm:

1. Besecall
2. Minimap
3. Nanopolish

code in preprocess.sh

# m6Astorm

```python 
conda create -n m6astorm python=3.12
conda install pandas numpy scikit-learn scipy
conda activate m6astorm
pip install EMD-signal
pip install torch torchvision torchaudio
```

## Feature extraction

```bash
python feature_extract.py  --eventalign /path/to/eventalign.txt\
							--out_dir /path/to/output
```

## Predict

```bash
python predict.py --data_pre /path/to/data_feature.txt \
                --model_dir model \
                --out_dir result \
                --min_coverage 5 \
                --mod_prob_thresh 0.5
```



