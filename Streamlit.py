import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle


# PATHS
st.set_page_config(page_title="Student Performance Clustering", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOT_DIR = os.path.join(BASE_DIR, "plots")

# LOAD DATA & MODEL
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "student_preprocessed.csv"))
    X_scaled = pd.read_csv(os.path.join(DATA_DIR, "X_scaled.csv"))
    gmm_labelled = pd.read_csv(os.path.join(DATA_DIR, "gmm_labelled.csv"))
    return df, X_scaled, gmm_labelled

# load saved model
@st.cache_resource
def load_gmm_and_scaler():
    with open(os.path.join(DATA_DIR, 'gmm_model.pkl'), 'rb') as f:
        gmm_model = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'scaler.pkl'), 'rb') as f:
        scaler_model = pickle.load(f)
    return gmm_model, scaler_model

try:
    df, X_scaled, gmm_labelled = load_data()
    gmm, scaler = load_gmm_and_scaler()
except Exception as e:
    st.error(f"Error loading data or model: {e}")
    st.stop()

cluster_names = {
    0: 'At-Risk Borderline Group',
    1: 'General Students B',
    2: 'Students with Learning Disabilities',
    3: 'General Students C',
    4: 'General Students A'
}

# SIDEBAR NAVIGATION
page = st.sidebar.selectbox("Navigation", [
    "Home",
    "Dataset & Preprocessing",
    "K-Means Clustering",
    "Agglomerative Hierarchical Clustering",
    "DBSCAN",
    "Gaussian Mixture Model",
    "Model Comparison",
    "Student Profile Predictor",
    "Cluster Explorer",
    "Dataset Explorer"
])

# HOME
if page == "Home":
    st.title("Student Performance Clustering Analysis")
    st.subheader("Unsupervised Machine Learning | AI Assignment")
    st.markdown("---")
    st.markdown("""
    This app presents the results of applying four unsupervised clustering
    algorithms to the **Student Performance Factors** dataset obtained from Kaggle, which
    contains **6,607 student records** across **20 features** covering academic,
    lifestyle, and socioeconomic factors.
    """)

    st.markdown("### Algorithms Implemented")
    col1, col2, col3, col4 = st.columns(4)
    col1.info("**K-Means**\nCentroid-based clustering")
    col2.info("**Hierarchical**\nBottom-up agglomerative clustering")
    col3.warning("**DBSCAN**\nDensity-based clustering")
    col4.info("**GMM**\nProbabilistic clustering")

    st.markdown("### Important Finding")
    st.success("""
    K-Means, Hierarchical Clustering, and GMM independently identified **5 consistent
    student clusters**. The most significant finding was a distinct cluster of students
    with learning disabilities showing consistently lower academic outcomes despite
    higher family income. DBSCAN was found unsuitable for this dataset due to its
    uniform density distribution.
    """)

    st.markdown("### Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", "6,607")
    col2.metric("Total Features", "20")
    col3.metric("Numerical Features", "7")
    col4.metric("Categorical Features", "13")

    st.markdown("### Interactive Features")
    col1, col2, col3 = st.columns(3)
    col1.info("**Student Profile Predictor**\nEnter student details to find which cluster they belong to")
    col2.info("**Cluster Explorer**\nSelect a cluster to explore its characteristics")
    col3.info("**Dataset Explorer**\nFilter and download clustered student data")

# DATASET & PREPROCESSING
elif page == "Dataset & Preprocessing":
    st.title("Dataset & Preprocessing")
    st.markdown("---")

    st.markdown("### Dataset Source")
    st.markdown("""
    The **Student Performance Factors** dataset was obtained from Kaggle.
    It contains 6,607 student records across 20 features showing academic,
    behavioural, lifestyle, and socioeconomic factors.
    """)

    st.markdown("### Missing Values")
    missing_data = pd.DataFrame({
        'Feature': ['Teacher_Quality', 'Parental_Education_Level', 'Distance_from_Home'],
        'Missing Count': [78, 90, 67],
        'Missing %': ['1.18%', '1.36%', '1.01%'],
        'Strategy': ['Mode imputation', 'Mode imputation', 'Mode imputation']
    })
    st.dataframe(missing_data, use_container_width=True)

    st.markdown("### Imputation Strategy Comparison")
    st.image(os.path.join(PLOT_DIR, "imputation_comparison.png"), use_container_width=True)
    st.caption("Mode imputation and row deletion produced near-identical distributions. Mode imputation selected to retain all 6,607 records.")

    st.markdown("### Numerical Feature Distributions")
    st.image(os.path.join(PLOT_DIR, "01_numerical_distributions.png"), use_container_width=True)

    st.markdown("### Categorical Feature Distributions")
    st.image(os.path.join(PLOT_DIR, "02_categorical_distributions.png"), use_container_width=True)

    st.markdown("### Correlation Heatmap")
    st.image(os.path.join(PLOT_DIR, "03_correlation_heatmap.png"), use_container_width=True)
    st.caption("Attendance (r=0.581) and Hours_Studied (r=0.445) show the strongest correlations with Exam_Score.")

    st.markdown("### Why Attendance Does Not Drive Clusters")
    st.image(os.path.join(PLOT_DIR, "attendance_vs_ld_distribution.png"), use_container_width=True)
    st.caption("Despite high correlation with Exam_Score, Attendance has a uniform continuous distribution with no natural boundary for cluster formation.")

    st.markdown("### Exam Score vs Categorical Features")
    st.image(os.path.join(PLOT_DIR, "04_boxplots_exam_score.png"), use_container_width=True)

    st.markdown("### Preprocessing Pipeline")
    steps = pd.DataFrame({
        'Step': ['1', '2', '3', '4', '5', '6'],
        'Method': ['Missing Value Imputation', 'Ordinal Encoding', 'Binary Encoding',
                   'One-Hot Encoding', 'StandardScaler', 'PCA (visualisation only)'],
        'Details': [
            'Mode imputation for Teacher_Quality, Parental_Education_Level, Distance_from_Home',
            '7 features with natural order encoded as Low=0, Medium=1, High=2',
            '5 yes/no features encoded as 0/1',
            'Peer_Influence encoded into 3 binary columns (no natural order)',
            'All 21 features normalised to mean=0, std=1',
            'Applied to 6 continuous features only. 34.62% variance in 2D'
        ]
    })
    st.dataframe(steps, use_container_width=True)

    st.markdown("### PCA Scree Plot")
    st.image(os.path.join(PLOT_DIR, "05_pca_scree.png"), use_container_width=True)

    st.markdown("### PCA 2D Pre-Clustering Scatter")
    st.image(os.path.join(PLOT_DIR, "06_pca_scatter_preclustering.png"), use_container_width=True)
    st.caption("Diffuse cloud expected. PCA captures 34.62% of variance. Clustering performed on full 21-feature matrix.")

# ============================================================
# K-MEANS (YASH)
# ============================================================
elif page == "K-Means Clustering":
    st.title("K-Means Clustering")
    st.markdown("---")
    st.markdown("""
    K-Means groups students into k clusters by assigning each point to the nearest 
    cluster centroid. Cluster centres are updated iteratively until stable groups are formed.
    """)

    st.markdown("### Optimal k Selection")
    st.image(os.path.join(PLOT_DIR, "kmeans_k_selection.png"), use_container_width=True)
    st.caption("Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score, and Inertia all consistently indicate k=5 as optimal.")

    st.markdown("### Elbow Method")
    st.image(os.path.join(PLOT_DIR, "kmeans_elbow.png"), use_container_width=True)
    st.caption("Inertia decreases sharply up to k=5 before flattening, confirming k=5 as the elbow point.")

    st.markdown("### Final Model Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Optimal k", "5")
    col2.metric("Silhouette Score", "0.1102")
    col3.metric("Davies-Bouldin", "2.4570")
    col4.metric("Calinski-Harabasz", "439.7")
    col1, col2 = st.columns(2)
    col1.metric("Inertia", "109561.66")
    col2.metric("ANOVA p-value", "< 0.001")

    st.markdown("### Cluster Visualisation")
    col1, col2 = st.columns(2)
    with col1:
        st.image(os.path.join(PLOT_DIR, "kmeans_pca_scatter.png"), use_container_width=True)
        st.caption("PCA 2D scatter")
    with col2:
        st.image(os.path.join(PLOT_DIR, "kmeans_tsne.png"), use_container_width=True)
        st.caption("t-SNE - reveals true 5-cluster separation")

    st.markdown("### Cluster Profiles")
    st.image(os.path.join(PLOT_DIR, "kmeans_cluster_profiles.png"), use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.image(os.path.join(PLOT_DIR, "kmeans_exam_by_cluster.png"), use_container_width=True)
    with col2:
        st.image(os.path.join(PLOT_DIR, "kmeans_cluster_sizes.png"), use_container_width=True)

    st.markdown("### Radar Profiles")
    st.image(os.path.join(PLOT_DIR, "kmeans_radar.png"), use_container_width=True)

    st.markdown("### Feature Importance")
    st.image(os.path.join(PLOT_DIR, "kmeans_feature_importance.png"), use_container_width=True)
    st.caption("Peer_Positive (0.34), Peer_Neutral (0.25), Learning_Disabilities (0.22), Internet_Access (0.19).")

# ============================================================
# HIERARCHICAL
# ============================================================
elif page == "Agglomerative Hierarchical Clustering":
    st.title("Agglomerative Hierarchical Clustering")
    st.markdown("---")
    st.markdown("""
    Agglomerative Hierarchical Clustering starts by treating each data point as
    its own cluster and iteratively merges the closest pairs using Ward linkage.
    """)

    st.markdown("### Dendrogram")
    st.image(os.path.join(PLOT_DIR, "hierarchical_dendrogram_full.png"), use_container_width=True)
    st.caption("Dendrogram constructed on 500-sample subset. Longest vertical lines suggest optimal cut point.")

    st.markdown("### Optimal k Selection")
    st.image(os.path.join(PLOT_DIR, "hierarchical_k_selection.png"), use_container_width=True)
    st.caption("All three metrics consistently indicate k=5 as optimal.")

    st.markdown("### Final Model Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Optimal k", "5")
    col2.metric("Silhouette Score", "0.1089")
    col3.metric("Davies-Bouldin", "2.4731")
    col4.metric("Calinski-Harabasz", "434.0")
    col1, col2 = st.columns(2)
    col1.metric("Linkage Method", "Ward")
    col2.metric("ANOVA p-value", "< 0.001")

    st.markdown("### Linkage Method Comparison")
    linkage_df = pd.DataFrame({
        'Linkage Method': ['Ward', 'Complete', 'Average', 'Single'],
        'Silhouette Score': [0.1089, 0.0594, 0.1127, 'N/A'],
        'Davies-Bouldin': [2.4731, 3.1838, 2.0484, 'N/A'],
        'Note': ['Valid — selected', 'Valid', 'Degenerate (6589/15/3)', 'Degenerate — chaining']
    })
    st.dataframe(linkage_df, use_container_width=True)
    st.caption("Average linkage showed best metric scores but produced imbalanced clusters.")

    st.markdown("### Cluster Visualisation")
    col1, col2 = st.columns(2)
    with col1:
        st.image(os.path.join(PLOT_DIR, "hierarchical_pca_scatter.png"), use_container_width=True)
        st.caption("PCA 2D scatter")
    with col2:
        st.image(os.path.join(PLOT_DIR, "hierarchical_tsne.png"), use_container_width=True)
        st.caption("t-SNE - reveals true 5-cluster separation")

    st.markdown("### Cluster Profiles")
    st.image(os.path.join(PLOT_DIR, "hierarchical_cluster_profiles.png"), use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.image(os.path.join(PLOT_DIR, "hierarchical_exam_by_cluster.png"), use_container_width=True)
    with col2:
        st.image(os.path.join(PLOT_DIR, "hierarchical_cluster_sizes.png"), use_container_width=True)

    st.markdown("### Radar Profiles")
    st.image(os.path.join(PLOT_DIR, "hierarchical_radar.png"), use_container_width=True)

    st.markdown("### Feature Importance")
    st.image(os.path.join(PLOT_DIR, "hierarchical_feature_importance.png"), use_container_width=True)
    st.caption("Peer_Positive (0.34), Peer_Neutral (0.25), Learning_Disabilities (0.22), Internet_Access (0.19).")

# ============================================================
# DBSCAN (NILEYIS)
# ============================================================
elif page == "DBSCAN":
    st.title("DBSCAN Clustering")
    st.markdown("---")
    st.markdown("""
    DBSCAN groups data points based on density, where clusters are formed from 
    regions with high concentration of points and noise represents outliers.
    Unlike K-Means, DBSCAN does not require specifying the number of clusters.
    """)

    st.markdown("### K-Distance Graph")
    st.image(os.path.join(PLOT_DIR, "dbscan_kdistance.png"), use_container_width=True)
    st.caption("No clear elbow observed. This indicates uniform density distribution, making it difficult to determine an optimal eps value. eps ≈ 4.0 was selected.")

    st.markdown("### Final Model Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("eps", "4.0")
    col2.metric("min_samples", "4")
    col3.metric("Clusters Found", "2-3")
    col4.metric("Noise Points", "139")

    col1, col2 = st.columns(2)
    col1.metric("Silhouette Score", "N/A")
    col2.metric("ANOVA p-value", "N/A")

    st.markdown("### Cluster Visualisation")
    col1, col2 = st.columns(2)
    with col1:
        st.image(os.path.join(PLOT_DIR, "dbscan_pca_scatter.png"), use_container_width=True)
        st.caption("PCA 2D scatter")
    with col2:
        st.image(os.path.join(PLOT_DIR, "dbscan_tsne.png"), use_container_width=True)
        st.caption("t-SNE visualisation")

    st.markdown("### Cluster Profiles")
    st.image(os.path.join(PLOT_DIR, "dbscan_cluster_profiles.png"), use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.image(os.path.join(PLOT_DIR, "dbscan_exam_by_cluster.png"), use_container_width=True)
    with col2:
        st.image(os.path.join(PLOT_DIR, "dbscan_cluster_sizes.png"), use_container_width=True)

    st.markdown("### Feature Importance")
    st.image(os.path.join(PLOT_DIR, "dbscan_feature_importance.png"), use_container_width=True)
    st.caption("Learning_Disabilities (0.65), Peer_Negative (0.23) and Previous_Scores (0.10) are the most influential features.")

    st.warning("DBSCAN produced highly imbalanced clusters with most data points grouped into a single cluster.")

    st.info("Due to uniform density distribution, DBSCAN is not suitable for this dataset.")

# ============================================================
# GMM
# ============================================================
elif page == "Gaussian Mixture Model":
    st.title("Gaussian Mixture Model (GMM)")
    st.markdown("---")
    st.markdown("""
    GMM models data as a mixture of Gaussian distributions, assigning each
    data point a **probability** of belonging to each cluster.
    """)

    st.markdown("### Optimal k Selection")
    st.image(os.path.join(PLOT_DIR, "gmm_k_selection.png"), use_container_width=True)
    st.caption("BIC, AIC, and Silhouette Score all indicate k=5 as optimal.")

    st.markdown("### Covariance Type Comparison")
    cov_df = pd.DataFrame({
        'Covariance Type': ['Full', 'Tied', 'Diagonal', 'Spherical'],
        'Silhouette Score': [0.1102, 0.0771, 0.1102, 0.1102],
        'Davies-Bouldin': [2.4570, 3.4809, 2.4570, 2.4570],
        'BIC': [-23892.2, 112915.3, -16397.1, 380711.0],
        'Selected': ['Yes', 'No', 'No', 'No']
    })
    st.dataframe(cov_df, use_container_width=True)

    st.markdown("### Final Model Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Optimal k", "5")
    col2.metric("Silhouette Score", "0.1102")
    col3.metric("Davies-Bouldin", "2.4570")
    col4.metric("Calinski-Harabasz", "439.7")
    col1, col2, col3 = st.columns(3)
    col1.metric("BIC", "-23892.2")
    col2.metric("AIC", "-32482.2")
    col3.metric("ANOVA p-value", "< 0.001")

    st.markdown("### Soft Assignment Analysis")
    st.image(os.path.join(PLOT_DIR, "gmm_soft_assignments.png"), use_container_width=True)
    st.caption("100% of students assigned with >90% certainty.")

    st.markdown("### Cluster Visualisation")
    col1, col2 = st.columns(2)
    with col1:
        st.image(os.path.join(PLOT_DIR, "gmm_pca_scatter.png"), use_container_width=True)
        st.caption("PCA 2D scatter")
    with col2:
        st.image(os.path.join(PLOT_DIR, "gmm_tsne.png"), use_container_width=True)
        st.caption("t-SNE - reveals true 5-cluster separation")

    st.markdown("### Cluster Profiles")
    st.image(os.path.join(PLOT_DIR, "gmm_cluster_profiles.png"), use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.image(os.path.join(PLOT_DIR, "gmm_exam_by_cluster.png"), use_container_width=True)
    with col2:
        st.image(os.path.join(PLOT_DIR, "gmm_cluster_sizes.png"), use_container_width=True)

    st.markdown("### Radar Profiles")
    st.image(os.path.join(PLOT_DIR, "gmm_radar.png"), use_container_width=True)

    st.markdown("### Feature Importance")
    st.image(os.path.join(PLOT_DIR, "gmm_feature_importance.png"), use_container_width=True)
    st.caption("Identical top features to K-Means and Hierarchical, which validates consistency across models.")

# ============================================================
# MODEL COMPARISON
# ============================================================
elif page == "Model Comparison":
    st.title("Model Comparison")
    st.markdown("---")

    st.markdown("### Evaluation Metrics")
    comparison_df = pd.DataFrame({
        'Metric': ['Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz',
                   'Inertia', 'BIC', 'AIC', 'n_clusters', 'Noise Points',
                   'ANOVA F-statistic', 'ANOVA p-value'],
        'K-Means': [0.1102, 2.4570, 439.7, 109561.66, 'N/A', 'N/A', 5, 'N/A', 31.26, '<0.001'],
        'Hierarchical': [0.1089, 2.4731, 434.0, 'N/A', 'N/A', 'N/A', 5, 'N/A', 32.56, '<0.001'],
        'DBSCAN': ['N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', '3 (degenerate)', 139, 'N/A', 'N/A'],
        'GMM': [0.1102, 2.4570, 439.7, 'N/A', -23892.2, -32482.2, 5, 'N/A', 31.26, '<0.001']
    })
    st.dataframe(comparison_df, use_container_width=True)

    st.success("K-Means, Hierarchical, and GMM identified the same 5 student clusters and strongly validated the discovered student profiles.")
    st.warning("DBSCAN was unsuitable for this dataset due to uniform density distribution.")
    st.info("Top features across all models: Peer_Positive (0.34), Peer_Neutral (0.25), Learning_Disabilities (0.22), Internet_Access (0.19).")

    st.markdown("### Cluster Summary")
    cluster_summary = pd.DataFrame({
        'Cluster': ['Students with Learning Disabilities', 'At-Risk Borderline Group',
                    'General Students A', 'General Students B', 'General Students C'],
        'Size': [640, 499, 2149, 2184, 1135],
        'Mean Exam Score': [66.32, 66.54, 67.36, 67.83, 66.68],
        'Important Characteristic': [
            'LD=1.00, lowest attendance, lowest exam score despite high family income',
            'Lowest motivation, most tutoring sessions, still underperforming',
            'Average across all features',
            'Highest exam score, highest motivation',
            'Lower family income, average performance'
        ]
    })
    st.dataframe(cluster_summary, use_container_width=True)

# STUDENT PROFILE PREDICTOR
elif page == "Student Profile Predictor":
    st.title("Student Profile Predictor")
    st.markdown("---")
    st.markdown("""
    Enter a student's details to predict which cluster they belong to
    using the trained **Gaussian Mixture Model (k=5, full covariance)**.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Academic Factors**")
        hours_studied = st.slider("Hours Studied per Week", 1, 44, 20)
        attendance = st.slider("Attendance (%)", 60, 100, 80)
        previous_scores = st.slider("Previous Scores", 50, 100, 75)
        tutoring_sessions = st.slider("Tutoring Sessions", 0, 8, 1)
    with col2:
        st.markdown("**Lifestyle Factors**")
        sleep_hours = st.slider("Sleep Hours per Night", 4, 10, 7)
        physical_activity = st.slider("Physical Activity (hours/week)", 0, 6, 3)
        extracurricular = st.selectbox("Extracurricular Activities", ["No", "Yes"])
        internet_access = st.selectbox("Internet Access", ["Yes", "No"])
    with col3:
        st.markdown("**Socioeconomic Factors**")
        parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
        access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
        motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
        family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
        learning_disabilities = st.selectbox("Learning Disabilities", ["No", "Yes"])
        peer_influence = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"])

    with st.expander("Additional Details"):
        col1, col2 = st.columns(2)
        with col1:
            teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"])
            school_type = st.selectbox("School Type", ["Public", "Private"])
            gender = st.selectbox("Gender", ["Male", "Female"])
        with col2:
            parental_education = st.selectbox("Parental Education Level",
                                              ["High School", "College", "Postgraduate"])
            distance_from_home = st.selectbox("Distance from Home",
                                              ["Near", "Moderate", "Far"])

    if st.button("Predict Cluster", type="primary"):
        ordinal_map = {'Low': 0, 'Medium': 1, 'High': 2}
        education_map = {'High School': 0, 'College': 1, 'Postgraduate': 2}
        distance_map = {'Near': 0, 'Moderate': 1, 'Far': 2}

        input_data = {
            'Hours_Studied': hours_studied,
            'Attendance': attendance,
            'Parental_Involvement': ordinal_map[parental_involvement],
            'Access_to_Resources': ordinal_map[access_to_resources],
            'Extracurricular_Activities': 1 if extracurricular == 'Yes' else 0,
            'Sleep_Hours': sleep_hours,
            'Previous_Scores': previous_scores,
            'Motivation_Level': ordinal_map[motivation_level],
            'Internet_Access': 1 if internet_access == 'Yes' else 0,
            'Tutoring_Sessions': tutoring_sessions,
            'Family_Income': ordinal_map[family_income],
            'Teacher_Quality': ordinal_map[teacher_quality],
            'School_Type': 1 if school_type == 'Private' else 0,
            'Physical_Activity': physical_activity,
            'Learning_Disabilities': 1 if learning_disabilities == 'Yes' else 0,
            'Parental_Education_Level': education_map[parental_education],
            'Distance_from_Home': distance_map[distance_from_home],
            'Gender': 1 if gender == 'Female' else 0,
            'Peer_Negative': 1 if peer_influence == 'Negative' else 0,
            'Peer_Neutral': 1 if peer_influence == 'Neutral' else 0,
            'Peer_Positive': 1 if peer_influence == 'Positive' else 0
        }

        input_df = pd.DataFrame([input_data])

        # exact structure match
        input_df = input_df.reindex(columns=X_scaled.columns, fill_value=0)
        input_scaled = scaler.transform(input_df)

        cluster_id = gmm.predict(input_scaled)[0]
        probs = gmm.predict_proba(input_scaled)[0]
        label = cluster_names[cluster_id]

        st.markdown("---")
        if label == 'Students with Learning Disabilities':
            st.error(f"**Predicted Cluster: {label}**")
        elif label == 'At-Risk Borderline Group':
            st.warning(f"**Predicted Cluster: {label}**")
        else:
            st.success(f"**Predicted Cluster: {label}**")

        prob_df = pd.DataFrame({
            'Cluster': [cluster_names[i] for i in range(5)],
            'Probability': [f"{p * 100:.1f}%" for p in probs]
        }).sort_values('Probability', ascending=False).reset_index(drop=True)
        st.dataframe(prob_df, use_container_width=True)

        descriptions = {
            'Students with Learning Disabilities': "Students in this group have recorded learning disabilities and consistently show lower attendance and exam scores despite higher family income. Specialised support beyond general resources is recommended.",
            'At-Risk Borderline Group': "This student shares characteristics with the at-risk group - low motivation despite high tutoring attendance. Motivational counselling may be more effective than additional tutoring.",
            'General Students A': "Average performing student across academic and lifestyle features. Standard academic support is appropriate.",
            'General Students B': "High performing student with strong motivation. Enrichment programmes and advanced challenges are recommended.",
            'General Students C': "Average performing student with lower family income. Resource support and scholarship opportunities may help."
        }
        st.info(descriptions[label])

# CLUSTER EXPLORER
elif page == "Cluster Explorer":
    st.title("Cluster Explorer")
    st.markdown("---")
    st.markdown("Select a cluster to explore its characteristics.")

    selected_cluster = st.selectbox("Select Cluster", [
        'Students with Learning Disabilities',
        'At-Risk Borderline Group',
        'General Students A',
        'General Students B',
        'General Students C'
    ])

    cluster_info = {
        'Students with Learning Disabilities': {
            'size': 640, 'score': 66.32, 'attendance': 79.05,
            'motivation': 0.90, 'tutoring': 1.52, 'family_income': 0.82,
            'ld': '100%',
            'desc': 'This cluster exclusively comprises students with learning disabilities (99.8%). Despite having the highest family income among all clusters, they consistently show the lowest attendance and exam scores.',
            'rec': 'Specialised educational interventions, individualised learning plans, and dedicated support beyond general academic resources.'
        },
        'At-Risk Borderline Group': {
            'size': 499, 'score': 66.54, 'attendance': 80.72,
            'motivation': 0.86, 'tutoring': 1.54, 'family_income': 0.78,
            'ld': '10%',
            'desc': 'Lowest motivation level (0.86) yet attends the most tutoring sessions (1.54). Despite additional support, exam scores remain below average.',
            'rec': 'Motivational counselling, mentoring programmes, and psychological support rather than additional tutoring.'
        },
        'General Students A': {
            'size': 2149, 'score': 67.36, 'attendance': 80.34,
            'motivation': 0.92, 'tutoring': 1.50, 'family_income': 0.78,
            'ld': '0%',
            'desc': 'Large general student group with average performance across all features.',
            'rec': 'Standard academic support and encouragement.'
        },
        'General Students B': {
            'size': 2184, 'score': 67.83, 'attendance': 79.84,
            'motivation': 0.91, 'tutoring': 1.47, 'family_income': 0.80,
            'ld': '0%',
            'desc': 'Highest performing general cluster with best mean exam scores (67.83) and motivation.',
            'rec': 'Advanced enrichment programmes and peer mentoring opportunities.'
        },
        'General Students C': {
            'size': 1135, 'score': 66.68, 'attendance': 79.77,
            'motivation': 0.90, 'tutoring': 1.50, 'family_income': 0.77,
            'ld': '0%',
            'desc': 'Lowest family income (0.77) among non-LD clusters but average academic performance.',
            'rec': 'Resource support and scholarship opportunities to address socioeconomic barriers.'
        }
    }

    data = cluster_info[selected_cluster]

    if selected_cluster == 'Students with Learning Disabilities':
        st.error(f"### {selected_cluster}")
    elif selected_cluster == 'At-Risk Borderline Group':
        st.warning(f"### {selected_cluster}")
    else:
        st.success(f"### {selected_cluster}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cluster Size", f"{data['size']:,} students")
    col2.metric("Mean Exam Score", f"{data['score']:.2f}")
    col3.metric("Mean Attendance", f"{data['attendance']:.1f}%")
    col4.metric("Learning Disabilities", data['ld'])

    col1, col2, col3 = st.columns(3)
    col1.metric("Motivation Level", f"{data['motivation']:.2f}")
    col2.metric("Tutoring Sessions", f"{data['tutoring']:.2f}")
    col3.metric("Family Income (encoded)", f"{data['family_income']:.2f}")

    st.markdown("### Description")
    st.info(data['desc'])

    st.markdown("### Recommendation for Educators")
    st.success(data['rec'])


# DATASET EXPLORER
elif page == "Dataset Explorer":
    st.title("Dataset Explorer")
    st.markdown("---")
    st.markdown("Filter the clustered dataset and download results.")

    col1, col2 = st.columns(2)
    with col1:
        selected_clusters = st.multiselect(
            "Filter by Cluster",
            options=gmm_labelled['Cluster_Label'].unique().tolist(),
            default=gmm_labelled['Cluster_Label'].unique().tolist()
        )
    with col2:
        score_range = st.slider(
            "Filter by Exam Score Range",
            int(gmm_labelled['Exam_Score'].min()),
            int(gmm_labelled['Exam_Score'].max()),
            (int(gmm_labelled['Exam_Score'].min()),
             int(gmm_labelled['Exam_Score'].max()))
        )

    filtered = gmm_labelled[
        (gmm_labelled['Cluster_Label'].isin(selected_clusters)) &
        (gmm_labelled['Exam_Score'].between(score_range[0], score_range[1]))
    ]

    st.markdown(f"Showing **{len(filtered):,}** of **{len(gmm_labelled):,}** records")
    st.dataframe(filtered, use_container_width=True)

    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name='filtered_student_clusters.csv',
        mime='text/csv'
    )

    st.markdown("### Cluster Distribution in Filtered Data")
    cluster_counts = filtered['Cluster_Label'].value_counts()
    st.bar_chart(cluster_counts)