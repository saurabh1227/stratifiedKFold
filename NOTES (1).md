# Cross Validation: Class Notes + Industry Practice

Cleaned-up version of the handwritten class notes (see `images/`), extended with **how this is done in real projects** and **updated scikit-learn practice**.

---

## Part 1: Class notes (from the handwritten pages)

### 1.1 Hold-Out method
![Hold-out and K-Fold notes](images/class-notes-holdout-kfold-loo.jpeg)

- Split data once into **Train | Test**, train on train, report **accuracy** on test.
- In scikit-learn: `X_train, X_test, y_train, y_test = train_test_split(X, y)`
- Problem: the score depends on which rows fell into the test set (one lucky or unlucky split).

### 1.2 K-Fold cross validation
![K-Fold notes](images/class-notes-kfold.jpeg)

- Divide the data into `K` folds, e.g. **5-fold CV = 5 folds of 20% each**.
- **1 fold = test, K-1 folds = train** (5-fold: 80% train / 20% test; 4-fold: 75% / 25%).
- Repeat K times so each fold is the test set once, then **average** the K scores.

| Run | Test fold | Train folds | Accuracy (example) |
|---|---|---|---|
| 1 | Fold 1 | F2 + F3 + F4 | 92% |
| 2 | Fold 2 | F1 + F3 + F4 | 86% |
| 3 | Fold 3 | F1 + F2 + F4 | 90% |
| 4 | Fold 4 | F1 + F2 + F3 | 88% |
| | | **Average** | **89%** |

(The numbers are illustrative, as in the class board; the last value was hard to read.)

**Advantage of K-Fold:** every sample is used for testing exactly once, and for training K-1 times.

**Getting the data for each fold:** `kf.split(X, y)` returns **indexes**, not data:

```python
for ix_train, ix_test in kf.split(X, y):
    X_train, X_test = X.iloc[ix_train], X.iloc[ix_test]
    y_train, y_test = y.iloc[ix_train], y.iloc[ix_test]
```

### 1.3 Leave-One-Out (n-fold CV)

- **1 sample = test, n-1 samples = train**, repeated for all n samples.
- This is K-Fold with **K = n**, so it is also called **n-fold CV**.
- Each fold's score is either 1 or 0 (right or wrong), as noted on the board.
- `LeaveOneOut()` ≡ `KFold(n_splits=n)` ≡ `LeavePOut(p=1)`.

### 1.4 Model selection map (whiteboard)
![Model selection notes](images/class-notes-model-selection.png)

```
Model selection (train/test split strategies)
├── KFold              ├── StratifiedKFold
├── LeaveOneOut        ├── RepeatedKFold
└── LeavePOut          └── ShuffleSplit
```

**Metrics, each averaged over folds:** `accuracy_score`, `balanced_accuracy_score`, `cohen_kappa_score`, `f1_score`.

---

## Part 2: Industry notes

### 2.1 What teams actually do

| Practice | Why |
|---|---|
| **5-fold or 10-fold (stratified)** is the default | Good bias/variance balance at reasonable cost |
| Keep a **final untouched test set** | CV is for model selection and tuning; the test set gives one honest final number |
| Report **mean ± standard deviation** | A 90% average with ±8% spread is a very different story than ±1% |
| Use a **`Pipeline`** for scaling, encoding, imputation, feature selection | Fitting them on all data before splitting leaks test information (data leakage) |
| Always set **`random_state`** | Reproducible experiments and fair model comparison |
| LOOCV / LeavePOut only for **tiny datasets** | Cost is n (or C(n,p)) model fits; high variance in the estimate |
| Log CV results (MLflow, Weights & Biases, or a simple CSV) | Comparable experiments across the team |

### 2.2 Pick the splitter by the data, not by habit

| Data situation | Use | Why |
|---|---|---|
| Classification, imbalanced classes | `StratifiedKFold` | Keeps class ratio in each fold |
| Several rows per patient / user / device | `GroupKFold`, `StratifiedGroupKFold` | Same entity must not appear in train **and** test |
| Time series, forecasting, logs | `TimeSeriesSplit` | Train only on the past, test on the future; **never shuffle** |
| Regression, balanced data | `KFold(shuffle=True)` | Simple and enough |
| Large data, cheap sanity check | `ShuffleSplit` / hold-out | Fewer fits |
| Hyperparameter tuning | `GridSearchCV`, `RandomizedSearchCV` | CV is built in |
| Unbiased score of a tuned model | **Nested CV** | Tuning on the same folds you report on is optimistic |

Typical real-world cases: **fraud detection** (very imbalanced → stratified + F1/PR-AUC), **medical data** (many records per patient → group split), **demand or stock forecasting** (time-ordered → `TimeSeriesSplit`).

### 2.3 Metrics: the `average` argument (the "average" on the whiteboard)

For multi-class problems, `f1_score`, `precision_score` and `recall_score` need an `average`:

| `average=` | Meaning | Use when |
|---|---|---|
| `'binary'` | One positive class (`pos_label=...`) | Two classes, one of interest |
| `'macro'` | Plain mean over classes | All classes equally important |
| `'weighted'` | Mean weighted by class size | Imbalanced, want overall view |
| `'micro'` | Pool all predictions | Overall performance across all samples |

Rules of thumb:
- Imbalanced data: **don't rely on accuracy**; look at balanced accuracy, F1 (macro), Kappa, PR-AUC.
- With Leave-One-Out, **pool all predictions and score once** (`cross_val_predict`); a per-fold F1 on a single sample is meaningless.

---

## Part 3: Updated, modern scikit-learn way

The class loop (`for ix_train, ix_test in kf.split(...)`) is the best way to **learn**. In projects you will usually write this instead:

### 3.1 `cross_validate`: many metrics in one call

```python
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

model = make_pipeline(StandardScaler(), DecisionTreeClassifier(criterion="entropy", random_state=42))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

res = cross_validate(
    model, X, y, cv=cv,
    scoring=["accuracy", "balanced_accuracy", "f1_macro"],
    return_train_score=True,
)
for k, v in res.items():
    if k.startswith(("test_", "train_")):
        print(f"{k:28s} {v.mean():.3f} ± {v.std():.3f}")
```

Comparing `train_*` and `test_*` also shows **over-fitting** (e.g. decision trees often get 100% on train).

### 3.2 Pooled predictions for Leave-One-Out

```python
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import classification_report

y_pred = cross_val_predict(model, X, y, cv=LeaveOneOut())
print(classification_report(y, y_pred))
```

### 3.3 Grouped data and time series

```python
from sklearn.model_selection import GroupKFold, TimeSeriesSplit, cross_val_score

# same patient/user never in both train and test
cross_val_score(model, X, y, groups=patient_id, cv=GroupKFold(n_splits=5))

# train on past, test on future
cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=5))
```

### 3.4 Nested CV (tune + honest estimate)

```python
from sklearn.model_selection import GridSearchCV, cross_val_score

inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)   # tuning
outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)   # evaluation

search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid={"max_depth": [2, 3, 4, 5, None], "criterion": ["gini", "entropy"]},
    scoring="balanced_accuracy", cv=inner,
)
scores = cross_val_score(search, X, y, cv=outer, scoring="balanced_accuracy")
print(scores.mean(), scores.std())
```

### 3.5 Repeated CV for small datasets

```python
from sklearn.model_selection import RepeatedStratifiedKFold
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)  # 50 fits, steadier mean
```

---

## Quick cheat sheet

```
Learn it        -> manual loop with kf.split(X, y) and iloc
Classification  -> StratifiedKFold(5 or 10, shuffle=True, random_state=...)
Grouped rows    -> GroupKFold / StratifiedGroupKFold
Time order      -> TimeSeriesSplit
Tuning          -> GridSearchCV inside nested CV
Preprocessing   -> always inside a Pipeline
Report          -> mean ± std of several metrics, plus a final hold-out test
```

## Interview questions this covers

1. Why is K-Fold better than a single hold-out split?
2. What is the difference between K-Fold and Stratified K-Fold? When does it matter?
3. How are Leave-One-Out and K-Fold related? Why is LOOCV expensive?
4. What is data leakage in cross validation, and how does a `Pipeline` prevent it?
5. How would you cross-validate time-series data, or data with several rows per customer?
6. What is nested cross validation and why use it?
7. Which averaging (`macro`, `micro`, `weighted`) would you use for imbalanced multi-class F1?
