# Student-Performance-Clustering---Unsupervised-ML

Unsupervised machine learning project comparing four clustering algorithms on the Student Performance Factors dataset (6,607 students, 20 features). Achieved **A+**.

## Overview
This project applies K-Means, Agglomerative Hierarchical Clustering, DBSCAN, and Gaussian Mixture Model to discover natural student groupings.

## My Findings
- K-Means, Hierarchical, and GMM consistently identified **5 student clusters**
- Most significant finding: students with learning disabilities underperform despite highest family income
- Peer influence emerged as the strongest driver of cluster separation
- DBSCAN was unsuitable due to uniform density distribution
- GMM assigned all 6,607 students with 100% certainty. That was confirmed through log-likelihood analysis

## Algorithms Implemented
- **K-Means** - centroid-based clustering with Elbow Method and Silhouette Score for k selection
- **Agglomerative Hierarchical Clustering** 0 Ward linkage, dendrogram analysis, linkage methods compared
- **DBSCAN** - density-based clustering, K-Distance Graph analysis
- **Gaussian Mixture Model** - probabilistic clustering with EM algorithm, BIC/AIC for model selection

## Evaluation Metrics
| Metric | K-Means | Hierarchical | DBSCAN | GMM |
|--------|---------|--------------|--------|-----|
| Silhouette Score | 0.1102 | 0.1089 | N/A | 0.1102 |
| Davies-Bouldin | 2.4570 | 2.4731 | N/A | 2.4570 |
| Calinski-Harabasz | 439.7 | 434.0 | N/A | 439.7 |
| ANOVA p-value | <0.001 | <0.001 | N/A | <0.001 |

## Streamlit App Features
- Dataset & Preprocessing analysis
- Individual algorithm results pages
- Interactive Model Comparison
- **Student Profile Predictor** - enter student details to predict cluster using trained GMM
- Cluster Explorer
- Dataset Explorer with CSV download

## How to Run
```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn scipy
streamlit run Streamlit.py
```
Keep all files and folders together (data/, plots/, Streamlit.py) before running.

## Dataset
Nguyen, L. (2024). Student Performance Factors. Kaggle.
https://www.kaggle.com/datasets/lainguyn123/student-performance-factors

## Tech Stack
Python, Scikit-learn, Streamlit, Pandas, NumPy, Matplotlib, Seaborn, SciPy
