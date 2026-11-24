# Cross Validation Techniques in Python (scikit-learn)

Hands-on notebooks covering **every major cross-validation / model-selection technique**: Hold-Out, K-Fold, Leave-One-Out, Leave-P-Out, Repeated K-Fold, **Stratified K-Fold**, Repeated Stratified K-Fold and **Shuffle Split**. Everything runs on a small fruits dataset (Apple vs Orange) with a Decision Tree classifier.

> By **[Dataloves](mailto:ask.dataloves@gmail.com)**: GenAI & Data Science Training Institute

---

## Repository structure

```
.
├── kfold_cross_validation_231126.ipynb     # Part 1: Hold-Out vs K-Fold vs LOOCV
├── STRATIFIEDKFOLD_24-11-2026.ipynb        # Part 2: LOO, LPO, Repeated, Stratified, ShuffleSplit
├── NOTES.md                                # Class notes + industry practice + modern sklearn
├── images/                                 # Handwritten class notes (jpeg/png)
├── data/
│   └── fruits.csv                          # (add your dataset here)
├── requirements.txt
└── README.md
```

> 📒 **Full notes:** see [NOTES.md](NOTES.md) for the cleaned class notes (from the handwritten pages in `images/`), **industry practice**, and the **modern scikit-learn way** (`cross_validate`, `Pipeline`, `GroupKFold`, `TimeSeriesSplit`, nested CV).

## Why cross validation?

A single `train_test_split` gives **one** score, and that score depends on which rows happened to land in the test set. Cross validation tests the model on **several different splits** and averages the results, so the estimate is more stable and less biased.

```
Hold-Out:    [ train ............ | test ]            -> 1 score (luck of the split)
K-Fold (4):  [test| train | train | train ]  -> score 1
             [train |test| train | train ]  -> score 2
             [train | train |test| train ]  -> score 3
             [train | train | train |test]  -> score 4   => average
```

## Techniques covered

| # | Technique | scikit-learn class | One-line idea |
|---|---|---|---|
| 1 | Hold-Out | `train_test_split` | One random split; fast but unstable |
| 2 | K-Fold | `KFold` | `k` equal folds; each fold is the test set once |
| 3 | Leave-One-Out (LOOCV) | `LeaveOneOut` | K-Fold with `k = n`; one sample tested at a time |
| 4 | Leave-P-Out (LPO) | `LeavePOut` | Every possible combination of `p` samples as test set |
| 5 | Repeated K-Fold | `RepeatedKFold` | K-Fold repeated `n_repeats` times with different shuffles |
| 6 | **Stratified K-Fold** | `StratifiedKFold` | K-Fold that keeps the **class ratio** same in every fold |
| 7 | Repeated Stratified K-Fold | `RepeatedStratifiedKFold` | Stratified K-Fold repeated several times |
| 8 | Shuffle Split | `ShuffleSplit` | Random train/test splits, repeated `n_splits` times |

### 1. K-Fold and the role of `shuffle`

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=4)                    # folds are consecutive blocks of rows
kf = KFold(n_splits=4, shuffle=True)      # rows are shuffled first, then split
```

- Without shuffle, folds are **contiguous chunks**. If your file is sorted by class (all Apples, then all Oranges), some folds can contain only one class.
- With `shuffle=True`, rows are randomly assigned to folds. Every sample is still tested **exactly once**, and test folds never overlap.
- Use `random_state=42` to make the shuffle reproducible.

### 2. Leave-One-Out and Leave-P-Out

```python
from sklearn.model_selection import LeaveOneOut, LeavePOut

loo = LeaveOneOut()          # n splits
lpo = LeavePOut(p=2)         # C(n, 2) splits
```

- `LeavePOut(p=1)` is the same as `LeaveOneOut()`.
- `LeaveOneOut()` is the same as `KFold(n_splits=n)`.
- Number of splits for LPO = **C(n, p)**, so it grows very fast. For n = 60: p=1 gives 60, p=2 gives 1,770 and p=3 gives 34,220 model fits. Use only on small datasets.

### 3. Repeated K-Fold

```python
from sklearn.model_selection import RepeatedKFold
rkf = RepeatedKFold(n_splits=4, n_repeats=5)   # 4 x 5 = 20 train/test runs
```

Repeating with fresh shuffles reduces the dependence on one particular fold arrangement and gives a more reliable mean score.

### 4. Stratified K-Fold (most important for classification)

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=3, shuffle=True)
for ix_train, ix_test in skf.split(X, y):      # note: y is required
    print(y.iloc[ix_test].value_counts())
```

Normal `KFold` can produce folds with very different class counts (e.g. 12 Apples and 3 Oranges in one fold, 2 and 13 in another). **StratifiedKFold keeps the Apple:Orange ratio of the full dataset in every fold**, which gives fair and stable scores, especially on **imbalanced** data.

| | KFold | StratifiedKFold |
|---|---|---|
| Needs `y` in `.split()` | No | **Yes** |
| Class ratio per fold | Random | **Preserved** |
| Best for | Regression, balanced data | **Classification**, imbalanced data |

`RepeatedStratifiedKFold(n_splits=3, n_repeats=5)` does the same thing repeatedly.

### 5. Shuffle Split

```python
from sklearn.model_selection import ShuffleSplit
ss = ShuffleSplit(n_splits=10, test_size=0.25, random_state=42)
```

Each split is an independent random train/test split. Unlike K-Fold, test sets **can overlap** between splits and some samples may never be tested. You control the train/test sizes and the number of splits freely, which is useful for large datasets.

## Evaluation metrics used

Scores are computed on **every fold** and then **averaged**:

| Metric | Function | Note |
|---|---|---|
| Accuracy | `accuracy_score` | Misleading on imbalanced data |
| Balanced accuracy | `balanced_accuracy_score` | Average recall across classes |
| Cohen's Kappa | `cohen_kappa_score` | Agreement corrected for chance |
| F1-score | `f1_score(..., pos_label='Orange')` | Harmonic mean of precision and recall |
| Recall | `recall_score(..., pos_label='Apple')` | Sensitivity for the chosen class |

Standard evaluation loop used in the notebooks:

```python
accuracy, bal_accuracy, f_measure = [], [], []
model = DecisionTreeClassifier(criterion='entropy')

for ix_train, ix_test in skf.split(X, y):
    x_train, x_test = X.iloc[ix_train], X.iloc[ix_test]
    y_train, y_test = y.iloc[ix_train], y.iloc[ix_test]
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy.append(accuracy_score(y_test, y_pred))
    bal_accuracy.append(balanced_accuracy_score(y_test, y_pred))
    f_measure.append(f1_score(y_test, y_pred, pos_label='Orange'))

np.mean(accuracy), np.mean(bal_accuracy), np.mean(f_measure)
```

## Which technique should I use?

| Situation | Recommended |
|---|---|
| Classification, imbalanced classes | **StratifiedKFold** (or RepeatedStratifiedKFold) |
| Regression or well-balanced data | KFold with `shuffle=True` |
| Very small dataset | LeaveOneOut / LeavePOut |
| Want a more stable estimate | RepeatedKFold / RepeatedStratifiedKFold |
| Large dataset, limited compute | ShuffleSplit or a small `k` (5 or 10) |
| Quick sanity check only | Hold-Out |

Rule of thumb: **5 or 10 folds**, with stratification for classification.

## Common mistakes to avoid

1. **Per-fold metrics with Leave-One-Out.** Each test set has one sample, so per-fold balanced accuracy, F1 and kappa are not meaningful (and often warn or return 0). Collect all predictions first, then compute the metric once on the pooled results.
2. **Forgetting `y` in `StratifiedKFold.split(X, y)`.** It cannot stratify without labels.
3. **Data leakage.** Fit scalers, encoders and feature selection **inside** each training fold (use a `Pipeline`), never on the whole dataset before splitting.
4. **No `random_state`.** Results change every run; set it for reproducible experiments.
5. **Using LeavePOut on big data.** C(n, p) explodes quickly.
6. **Reporting only accuracy** on imbalanced classes; also check balanced accuracy, F1 and kappa.

## Shortcut: `cross_val_score` (one-liner)

Once you understand the manual loop, scikit-learn can do it for you:

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    DecisionTreeClassifier(criterion='entropy'),
    X, y, cv=skf, scoring='balanced_accuracy'
)
print(scores, scores.mean(), scores.std())
```

## Industry notes (short version)

- Default: **5 or 10-fold, stratified** for classification, inside a `Pipeline`.
- Keep a **final test set** untouched; use CV only for tuning and model selection.
- Several rows per user/patient → `GroupKFold`. Time-ordered data → `TimeSeriesSplit`.
- Tuning and reporting on the same folds is optimistic → use **nested CV**.
- Report **mean ± std** of several metrics, not just accuracy.

Details and code in [NOTES.md](NOTES.md).

## Dataset

`fruits.csv` columns:

| Column | Description |
|---|---|
| `sphericity` | How round the fruit is (feature) |
| `weight` | Weight of the fruit (feature) |
| `label` | `Apple` or `Orange` (target) |

The dataset is not included. Place it in `data/fruits.csv` and load it with a relative path so the notebooks run on any machine:

```python
df = pd.read_csv("data/fruits.csv")
```

## Getting started

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

## Tech stack

Python • NumPy • Pandas • Matplotlib • Seaborn • scikit-learn • Jupyter

## Contact

**Dataloves**
- Phone: +91 91404 97404
- Email: ask.dataloves@gmail.com

If this repo helped you, give it a ⭐
