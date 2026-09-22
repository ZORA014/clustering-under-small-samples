import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    davies_bouldin_score, silhouette_score,
    adjusted_rand_score, adjusted_mutual_info_score
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold, GridSearchCV
)
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import cdist

#全局配置
CONFIG = {
    "data_path": "data.csv",
    "random_state": 42,
    "n_repeats": 10,
    "n_splits": 5,
    "n_jobs": 1,  # 关闭多进程，防止loky子进程崩溃
}

# 创建输出目录，图+表
for folder in ["figures", "tables"]:
    os.makedirs(folder, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 1.加载&预处理数据
def load_preprocess(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    cat_col = ["Class"]
    num_col = [c for c in df.columns if c not in cat_col]
    le = LabelEncoder()
    df["Class_label"] = le.fit_transform(df["Class"])
    y = df["Class_label"].values
    X_raw = df[num_col].copy()
    return df, X_raw, y, num_col, le

df, X_raw, y, num_col, le = load_preprocess(CONFIG["data_path"])
print(f"数据集大小：{X_raw.shape}，类别数:{len(np.unique(y))}")

#表
# 1.类别计数图
plt.figure(figsize=(10,5))
sns.countplot(data=df, x="Class", hue="Class", palette="Set2")
plt.title("各类乳腺组织样本计数")
plt.tight_layout()
plt.savefig("figures/eda_class_count.png", dpi=150, bbox_inches="tight")
plt.show()

# 2.相关性热力图
plt.figure(figsize=(10,8))
sns.heatmap(df[num_col].corr(), annot=True, fmt=".2f", cmap="Set2", linewidths=0.8)
plt.title("特征相关性热力图")
plt.tight_layout()
plt.savefig("figures/eda_corr_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# 3.9个数值特征KDE分布图
fig, axes = plt.subplots(3,3,figsize=(16,10))
axes = axes.flatten()
for idx, col in enumerate(num_col):
    sns.kdeplot(data=df, x=col, hue="Class", multiple="stack", ax=axes[idx], palette="Set2")
    axes[idx].set_title(f"KDE‑{col}")
plt.tight_layout()
plt.savefig("figures/eda_kde_all_features.png", dpi=150, bbox_inches="tight")
plt.show()

#2.Gap‑Statistic 辅助选K
def gap_statistic(X, k_max=10, n_refs=5):
    gaps = []
    sks = []
    n_samples, n_feat = X.shape
    # 获取每个特征的实际最小、最大，构造包围盒
    col_min = X.min(axis=0)
    col_max = X.max(axis=0)

    for k in range(1, k_max+1):
        km = KMeans(n_clusters=k, random_state=CONFIG["random_state"], n_init=10)
        km.fit(X)
        disp = sum(np.min(cdist(X, km.cluster_centers_, "euclidean"), axis=1)**2)/X.shape[0]
        ref_disps = []
        for _ in range(n_refs):
            # 在真实数据包围盒内均匀采样，修复bug
            rand_data = np.random.uniform(low=col_min, high=col_max, size=(n_samples, n_feat))
            km_rand = KMeans(n_clusters=k, random_state=CONFIG["random_state"], n_init=10)
            km_rand.fit(rand_data)
            d = sum(np.min(cdist(rand_data, km_rand.cluster_centers_, "euclidean"),axis=1)**2)/rand_data.shape[0]
            ref_disps.append(d)
        ref_mean = np.mean(np.log(ref_disps))
        gap = ref_mean - np.log(disp)
        gaps.append(gap)
        sks.append(np.std(np.log(ref_disps)) * np.sqrt(1+1/n_refs))
    return np.arange(1, k_max+1), np.array(gaps), np.array(sks)

scaler_global = StandardScaler()
X_scaled = scaler_global.fit_transform(X_raw)
k_arr, gap_vals, sk_vals = gap_statistic(X_scaled, k_max=9, n_refs=5)

plt.figure(figsize=(8,4))
plt.errorbar(k_arr, gap_vals, yerr=sk_vals, fmt="bx-")
plt.xlabel("k")
plt.ylabel("Gap statistic")
plt.title("Gap‑Statistic for K‑Means")
plt.tight_layout()
plt.savefig("figures/gap_stat.png", dpi=150, bbox_inches="tight")
plt.show()

#3.计算不同k下 Silhouette & ARI 曲线
ari_list = []
sil_list = []
k_candidates = list(range(2,9))
for k in k_candidates:
    km = KMeans(n_clusters=k, random_state=CONFIG["random_state"], n_init=10)
    lab = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, lab)
    ari = adjusted_rand_score(y, lab)
    sil_list.append(sil)
    ari_list.append(ari)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(k_candidates, sil_list, "go-")
plt.xlabel("k")
plt.ylabel("Silhouette score")
plt.title("Silhouette vs k")
plt.subplot(1,2,2)
plt.plot(k_candidates, ari_list, "ro-")
plt.xlabel("k")
plt.ylabel("ARI (vs true label)")
plt.title("ARI vs k")
plt.tight_layout()
plt.savefig("figures/k_metric_curve.png", dpi=150, bbox_inches="tight")
plt.show()

df_k_metric = pd.DataFrame({
    "k":k_candidates,
    "silhouette":sil_list,
    "ARI":ari_list
})
df_k_metric.to_csv("tables/k_metrics.csv", index=False, encoding="utf‑8‑sig")
print("\n==== K选择指标 ====")
print(df_k_metric)

#  4.网格搜索SVC，n_jobs=1避免windows崩溃
pipe_svc = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel="rbf"))
])
param_grid = {
    "svc__C": [0.1, 1, 10],
    "svc__gamma": [0.01, 0.1, 1]
}
cv = RepeatedStratifiedKFold(
    n_splits=CONFIG["n_splits"],
    n_repeats=CONFIG["n_repeats"],
    random_state=CONFIG["random_state"]
)
grid = GridSearchCV(
    estimator=pipe_svc,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=CONFIG["n_jobs"],
    verbose=1
)
grid.fit(X_raw, y)
print("\n==== SVC GridSearch 最优 ====")
print(f"best params: {grid.best_params_}")
print(f"best cv mean acc: {grid.best_score_:.4f} ± {grid.cv_results_['std_test_score'][grid.best_index_]:.4f}")

best_svc_kwargs = {}
for key, val in grid.best_params_.items():
    real_key = key.replace("svc__", "")
    best_svc_kwargs[real_key] = val
print(f"剥离前缀后SVC参数: {best_svc_kwargs}")

# ===================== 5.不同PCA维度对比（Pipeline防止泄露） =====================
def eval_pca_dim(n_comp, X, y, cv_obj, svc_kwargs):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_comp, random_state=CONFIG["random_state"])),
        ("svc", SVC(**svc_kwargs))
    ])
    scores = []
    for train_idx, test_idx in cv_obj.split(X,y):
        pipe.fit(X.iloc[train_idx], y[train_idx])
        s = pipe.score(X.iloc[test_idx], y[test_idx])
        scores.append(s)
    return np.mean(scores), np.std(scores)

cv_single = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=CONFIG["random_state"])
dim_list = [2,5,None]
dim_names = ["PCA‑2","PCA‑5","Full‑Feature"]
res_rows = []
for dim,name in zip(dim_list, dim_names):
    mu, std = eval_pca_dim(dim, X_raw, y, cv_single, best_svc_kwargs)
    res_rows.append({"setting":name, "mean_acc":round(mu,4), "std_acc":round(std,4)})

df_dim_compare = pd.DataFrame(res_rows)
df_dim_compare.to_csv("tables/dimension_compare.csv", index=False, encoding="utf‑8‑sig")
print("\n==== PCA维度对比（mean±std） ====")
print(df_dim_compare)

#6.DBSCAN k‑distance绘图
def plot_kdistance(X_scaled_data, k=7):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_scaled_data)
    d,_ = nn.kneighbors(X_scaled_data)
    dists = np.sort(d[:,-1])
    plt.figure(figsize=(8,4))
    plt.plot(dists, "b-")
    plt.xlabel("sample index sorted")
    plt.ylabel(f"{k}‑th nearest neighbor distance")
    plt.title("K‑Distance plot for DBSCAN eps selection")
    plt.tight_layout()
    plt.savefig("figures/k_distance.png", dpi=150, bbox_inches="tight")
    plt.show()

plot_kdistance(X_scaled, k=7)

#7.多聚类算法对比 + 恢复KMeans / DBSCAN对比散点图
def evaluate_clustering(labels_, X_, y_true_):
    """
    返回两套指标：
    full：全部样本(仅适用于无噪声算法，如KMeans/Agglomerative)
    filtered：剔除噪声标签-1之后的样本，用于DBSCAN
    """
    mask_no_noise = labels_ != -1
    labels_filt = labels_[mask_no_noise]
    X_filt = X_[mask_no_noise]
    y_filt = y_true_[mask_no_noise]

    # 全部样本口径（DBSCAN时‑1被当作普通簇，参考即可）
    dbi_full = davies_bouldin_score(X_, labels_)
    ari_full = adjusted_rand_score(y_true_, labels_)
    ami_full = adjusted_mutual_info_score(y_true_, labels_)

    # 过滤噪声口径：用于和KMeans横向公平对比
    sil_filt = np.nan
    dbi_filt = np.nan
    ari_filt = np.nan
    ami_filt = np.nan
    n_cluster_filt = len(np.unique(labels_filt))
    if n_cluster_filt >=2:
        sil_filt = silhouette_score(X_filt, labels_filt)
        dbi_filt = davies_bouldin_score(X_filt, labels_filt)
        ari_filt = adjusted_rand_score(y_filt, labels_filt)
        ami_filt = adjusted_mutual_info_score(y_filt, labels_filt)

    return {
        # 原始全部样本
        "DBI_full(含噪声)": dbi_full,
        "ARI_full(含噪声)": ari_full,
        "AMI_full(含噪声)": ami_full,
        # 剔除噪声，用于公平横向对比
        "silhouette_filtered(无噪声)": sil_filt,
        "DBI_filtered(无噪声)": dbi_filt,
        "ARI_filtered(无噪声)": ari_filt,
        "AMI_filtered(无噪声)": ami_filt,
        "n_cluster_after_filter": n_cluster_filt,
        "n_noise_points": np.sum(labels_ == -1)
    }


# 2维PCA投影
pca_2d = PCA(n_components=2, random_state=CONFIG["random_state"])
x_pca_vis = pca_2d.fit_transform(X_scaled)
k_km=4
km = KMeans(n_clusters=k_km, random_state=CONFIG["random_state"], n_init=10)
lab_km = km.fit_predict(X_scaled)
db = DBSCAN(eps=1.1, min_samples=7)
lab_db = db.fit_predict(X_scaled)
fig, axes = plt.subplots(1,2,figsize=(14,6))
axes[0].scatter(x_pca_vis[:,0], x_pca_vis[:,1], c=lab_km, cmap="Set2", s=45)
axes[0].set_title(f"PCA + K‑Means (k={k_km})")
axes[0].set_xlabel("PC1");axes[0].set_ylabel("PC2")

axes[1].scatter(x_pca_vis[:,0], x_pca_vis[:,1], c=lab_db, cmap="Set2", s=45)
axes[1].set_title("PCA + DBSCAN")
axes[1].set_xlabel("PC1");axes[1].set_ylabel("PC2")
plt.tight_layout()
plt.savefig("figures/cluster_kmeans_dbscan_compare.png", dpi=150, bbox_inches="tight")
plt.show()

# 聚类指标评估
cluster_results = []
res_km = evaluate_clustering(lab_km, X_scaled, y)
res_km["algorithm"] = f"KMeans(k={k_km})"
cluster_results.append(res_km)

agg = AgglomerativeClustering(n_clusters=4)
lab_agg = agg.fit_predict(X_scaled)
res_agg = evaluate_clustering(lab_agg, X_scaled, y)
res_agg["algorithm"] = "Agglomerative(k=4)"
cluster_results.append(res_agg)

res_db = evaluate_clustering(lab_db, X_scaled, y)
res_db["algorithm"] = "DBSCAN(eps=1.1,min_samples=7)"
cluster_results.append(res_db)

df_cluster_eval = pd.DataFrame(cluster_results)
df_cluster_eval.to_csv("tables/cluster_compare.csv", index=False, encoding="utf‑8‑sig")
print("\n==== 聚类算法对比 ====")
print(df_cluster_eval.round(4))

print("\n>>> 全部运行完成，图表输出到figures/，表格输出tables/")
print("重要提示：本数据集样本很小，所有ARI全部<0.35，说明无监督聚类很难还原真实6类乳腺组织标签。")
