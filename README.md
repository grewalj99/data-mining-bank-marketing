# **Bank Marketing Data Analysis Project**

This project performs a full data-mining workflow on the *Bank Marketing* dataset, including:

- **Preprocessing**
- **Dimensionality Reduction (PCA)**
- **Clustering**
- **Outlier Detection**
- **Classification (Random Forest)**
- **Feature Selection (Lasso)**
- **Data Augmentation (SMOTE)**
- **Hyperparameter Tuning**
- **Performance Analysis**

---

## Project Structure

All non-exploratory logic (preprocessing, clustering, outlier detection, feature
selection, model training, evaluation) lives in the `src/` package, not in the
notebooks. Each notebook imports and calls these functions rather than
reimplementing them, so there is a single, tested implementation shared by the
notebooks and the CLI below.

```
config.yaml           # all pipeline parameters (paths, hyperparameters, search space)
src/
  config.py            # loads config.yaml
  data.py               # raw/processed CSV loading
  preprocessing.py       # cleaning, imputation, encoding, scaling
  clustering.py          # KMeans + silhouette k-sweep
  outliers.py            # Isolation Forest
  features.py             # L1-logistic feature selection
  train.py                 # Random Forest training, SMOTE, hyperparameter search
  evaluate.py                # metrics + plotting
  pipeline.py                 # CLI entrypoint tying the stages together
notebooks/                      # EDA and analysis, calling into src/
```

## Reproducing this Project

```bash
pip install -r requirements.txt

# Run the full pipeline (preprocess -> cluster -> outliers -> train -> tune)
python -m src.pipeline

# Or run a subset of stages
python -m src.pipeline --stages preprocess train
```

This regenerates `data/processed/bank_marketing_preprocessed.csv` and prints a
JSON summary of clustering, outlier, and classification metrics for every
model variant described below — the same numbers reported in this README.
All parameters (Random Forest hyperparameters, SMOTE/feature-selection
settings, the k range for clustering, Isolation Forest contamination, the
hyperparameter search space) live in `config.yaml`.

---

## 1. Preprocessing

Cleaned and prepared the dataset using:

- Handling missing values  
- One-hot encoding for categorical variables  
- Standardization for numerical attributes  
- Saving the result as `bank_marketing_preprocessed.csv`

This processed dataset is used for all subsequent analysis.

---

## 2. Clustering

### **K-Means clustering** 
File: `clustering_analysis_kmeans.ipynb`

Applied KMeans clustering with different values of k (2 to 10) and evaluated cluster quality using the Silhouette Score.
### Key Results
- Optimal number of clusters: k = 2
- Best Silhouette Score ≈ 0.30
- Higher k values decreased cluster quality
- PCA visualization shows two broad groups but not sharply separated
- Clusters do not correspond to the target label, so the dataset is not naturally clusterable in a way that predicts outcomes
- Clustering is still useful for understanding structural patterns

---

## 3. Outlier Detection

### **Isolation Forest** 
File: `outlier_detection_isolation_forest.ipynb`

### Key Results
- With contamination = 0.05, PCA visualization shows outliers concentrated in sparse regions
- Cross-tabulation revealed:

| is_outlier | y=0 | y=1 |
|-----------|-----|-----|
| 0         | ~91% | ~9% |
| 1         | ~49% | ~51% |

- Many outliers belong to class y=1, meaning anomalies may represent meaningful minority-class patterns rather than noise.
- Therefore, outliers were kept for downstream tasks.

---

## 4. Feature Selection

### **L1 Logisitic Regression** 
File: `classification_random_forest.ipynb`

### Key Results
- Reduced many weak one-hot encoded features.
- Computational efficiency improved.
- However, Random Forest performs best with more features, so F1 and recall slightly decreased after feature reduction.

### **Performance Comparison**

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|------|
| RF Baseline | ~0.90 | ~0.67 | ~0.25 | ~0.37 | ~0.81 |
| RF + Feature Selection | ~0.903 | ~0.69 | ~0.25 | ~0.37 | ~0.81 |
| RF + SMOTE | ~0.886 | ~0.49 | ~0.57 | ~0.53 | ~0.80 |
| RF + Feature Selection + SMOTE | ~0.883 | ~0.49 | ~0.57 | ~0.52 | ~0.80 |

- Feature Selection helps interpretability but did not improve Random Forest performance.
- SMOTE was far more impactful for improving F1.

---

## 6. Classification

### **Random Forest** 
File: `classification_random_forest.ipynb`

### Key Results
Baseline Random Forest:

| Metric | Value |
|--------|--------|
| Test Accuracy | ~0.90 |
| Precision | ~0.67 |
| Recall | ~0.25 |
| F1 | ~0.37 |
| AUC | ~0.81 |

- Low recall due to dataset imbalance
- Classification improved with SMOTE and tuning

After Data Augmentation (SMOTE):

The dataset is imbalanced so SMOTE was applied on the training set.

- Recall improved dramatically (≈ 0.25 to ≈ 0.57)
- F1-score improved substantially
- Slight accuracy drop, which is expected

---

## 7. Hyperparameter Tuning

### **RandomizedSearchCV for Random Forest**
File: `classification_random_forest.ipynb`

### Key Results
- Best parameters: n_estimators=400, min_samples_split=5, min_samples_leaf=2, max_features=sqrt, max_depth=None, bootstrap=False
- Optimizing for F1 improved CV results but test-set F1 did not exceed SMOTE baseline
- Random Forest already performed near-optimal with default parameters

---

