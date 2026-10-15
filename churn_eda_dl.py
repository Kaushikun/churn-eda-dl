# Auto-exported from notebooks/churn_eda_dl.ipynb. Run from inside the src/ folder: python churn_eda_dl.py

# Telco Customer Churn: Exploratory Data Analysis, Visualization and Deep Learning (MLP)
# ---------- Imports and global configuration ----------
import os, warnings, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
SEED = 42
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
random.seed(SEED); np.random.seed(SEED)

sns.set_theme(style="whitegrid", context="notebook", palette="Set2")
plt.rcParams.update({"figure.dpi": 110, "axes.titleweight": "bold"})

FIG_DIR = "../figures"; os.makedirs(FIG_DIR, exist_ok=True)
def savefig(name):
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/{name}.png", dpi=150, bbox_inches="tight"); plt.show()


# Stage 1: ACQUIRE: Load the dataset
DATA_PATH = "../data/Telco-Customer-Churn.csv"
df = pd.read_csv(DATA_PATH)
print("Shape (rows, columns):", df.shape)
df.head()


# Stage 2: PARSE: Understand structure and data types
df.info()

# Field groups used throughout the notebook
target = "Churn"
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]   # TotalCharges is stored as text, fixed below
binary_like  = ["SeniorCitizen"]
cat_cols = [c for c in df.columns if c not in numeric_cols + [target, "customerID", "SeniorCitizen"]]
print(len(df.columns), "fields in total")
print("Categorical:", cat_cols)


# Data cleaning (part of the Filter stage)
# --- 1. Data type problem: TotalCharges is text because 11 rows hold a blank string ---
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce")

# --- 2. Missing values ---
miss = df.isna().sum(); print("Missing values per column:\n", miss[miss > 0])
print("\nTenure of rows with missing TotalCharges:", df.loc[df.TotalCharges.isna(), "tenure"].unique())

# All 11 blanks belong to brand-new customers (tenure = 0) who have not been billed yet.
# So the correct value is 0 (not the mean/median), which avoids inventing charges.
df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

# --- 3. Duplicates ---
print("Duplicate rows       :", df.duplicated().sum())
print("Duplicate customerIDs:", df["customerID"].duplicated().sum())

# --- 4. Drop identifier (carries no predictive information) ---
df = df.drop(columns="customerID")

# --- 5. Consistency: 'No internet service' / 'No phone service' are equivalent to 'No' ---
for c in ["MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]:
    df[c] = df[c].replace({"No internet service": "No", "No phone service": "No"})

# --- 6. Target encoding for analysis ---
df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)
print("Missing values after cleaning:", int(df.isna().sum().sum()))

# --- 7. Outlier detection with the IQR rule (1.5 x IQR) ---
rows = []
for c in numeric_cols:
    q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    rows.append([c, lo, hi, ((df[c] < lo) | (df[c] > hi)).sum()])
pd.DataFrame(rows, columns=["feature", "lower_fence", "upper_fence", "n_outliers"])


# **Outlier decision:** no feature has points outside the IQR fences. Charges and tenure are bounded business quantities, so we keep every row and apply standardization later.

# Stage 3: FILTER: Descriptive statistics
df[numeric_cols + ["SeniorCitizen"]].describe().T.round(2)

# Skewness and kurtosis (shape of each numeric distribution)
shape = pd.DataFrame({"skew": df[numeric_cols].skew(), "kurtosis": df[numeric_cols].kurt()}).round(3)
shape


# Stage 4: MINE: Statistical analysis (CO4)
# Frequency tables for categorical features
for c in ["Contract", "InternetService", "PaymentMethod"]:
    print(df[c].value_counts(normalize=True).mul(100).round(1).to_string(), "\n")
print("Churn rate overall: %.2f%%" % (df.ChurnFlag.mean()*100))


# Stage 5: REPRESENT: Data visualizations
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
counts = df["Churn"].value_counts()
sns.countplot(data=df, x="Churn", order=["No","Yes"], ax=ax[0], palette=["#66c2a5","#fc8d62"])
for p in ax[0].patches:
    ax[0].annotate(f"{int(p.get_height())}", (p.get_x()+p.get_width()/2, p.get_height()), ha="center", va="bottom")
ax[0].set_title("Customers by churn status"); ax[0].set_ylabel("Number of customers")
ax[1].pie(counts[["No","Yes"]], labels=["Stayed","Churned"], autopct="%1.1f%%", colors=["#66c2a5","#fc8d62"], startangle=90)
ax[1].set_title("Share of churners")
savefig("v1_churn_distribution")


# **What it shows:** how many customers stayed vs left. **Why:** class balance decides which evaluation metrics and training strategy we need.
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, c in zip(axes, numeric_cols):
    sns.histplot(df[c], kde=True, bins=30, ax=ax, color="#8da0cb")
    ax.axvline(df[c].mean(), color="red", ls="--", label=f"mean={df[c].mean():.1f}")
    ax.axvline(df[c].median(), color="green", ls=":", label=f"median={df[c].median():.1f}")
    ax.set_title(f"Distribution of {c}"); ax.legend()
savefig("v2_numeric_histograms")


# **What it shows:** the shape of tenure, monthly charges and total charges. **Why:** shape tells us about skew, modality and whether scaling or transforms are needed.
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, c in zip(axes, numeric_cols):
    sns.boxplot(data=df, x="Churn", y=c, order=["No","Yes"], ax=ax, palette=["#66c2a5","#fc8d62"])
    ax.set_title(f"{c} by Churn")
savefig("v3_boxplots_by_churn")
print(df.groupby("Churn")[numeric_cols].median().round(1))


# **What it shows:** the median, spread and range of each numeric feature, split by churn. **Why:** to see which numeric variables separate churners from non-churners.
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, c in zip(axes.ravel(), ["Contract","InternetService","PaymentMethod","TechSupport","OnlineSecurity","PaperlessBilling"]):
    rate = df.groupby(c)["ChurnFlag"].mean().mul(100).sort_values(ascending=False)
    sns.barplot(x=rate.values, y=rate.index, ax=ax, palette="rocket_r")
    ax.axvline(df.ChurnFlag.mean()*100, color="k", ls="--", lw=1)
    for i, v in enumerate(rate.values): ax.text(v+0.5, i, f"{v:.1f}%", va="center")
    ax.set_title(f"Churn rate by {c}"); ax.set_xlabel("Churn rate (%)"); ax.set_ylabel("")
savefig("v4_churn_rate_by_category")


# **What it shows:** the percentage of customers who churn within each category (dashed line = overall 26.5%). **Why:** categorical features dominate this dataset, and rates are easier to compare than raw counts.
# Numeric view of the data (binary Yes/No mapped to 1/0) so that all fields can be correlated
corr_df = df.copy()
for c in ["Partner","Dependents","PhoneService","PaperlessBilling","MultipleLines","OnlineSecurity","OnlineBackup",
          "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]:
    corr_df[c] = (corr_df[c] == "Yes").astype(int)
corr_df["gender"] = (corr_df["gender"] == "Male").astype(int)
corr_df["Fiber"] = (corr_df["InternetService"] == "Fiber optic").astype(int)
corr_df["MonthToMonth"] = (corr_df["Contract"] == "Month-to-month").astype(int)
corr_df["ElectronicCheck"] = (corr_df["PaymentMethod"] == "Electronic check").astype(int)
corr_cols = ["ChurnFlag","tenure","MonthlyCharges","TotalCharges","SeniorCitizen","Partner","Dependents","PaperlessBilling",
             "OnlineSecurity","TechSupport","StreamingTV","Fiber","MonthToMonth","ElectronicCheck"]
corr = corr_df[corr_cols].corr()
plt.figure(figsize=(11, 8.5))
sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)), annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=.5, cbar_kws={"label":"Pearson r"})
plt.title("Correlation heatmap (lower triangle)")
savefig("v5_correlation_heatmap")
print(corr["ChurnFlag"].drop("ChurnFlag").sort_values().round(3).to_string())


# **What it shows:** pairwise Pearson correlation. **Why:** to find the strongest linear relationships with churn and detect multicollinearity.
plt.figure(figsize=(9, 5.5))
sns.scatterplot(data=df.sample(2500, random_state=SEED), x="tenure", y="MonthlyCharges", hue="Churn", hue_order=["No","Yes"],
                palette=["#66c2a5","#fc8d62"], alpha=.55, s=28)
plt.title("Tenure vs MonthlyCharges coloured by churn (2,500-customer sample)")
savefig("v6_scatter_tenure_monthly")


# **What it shows:** every point is a customer. **Why:** to see two numeric drivers together and spot clusters.
g = sns.pairplot(df.sample(1500, random_state=SEED), vars=numeric_cols, hue="Churn", hue_order=["No","Yes"],
                 palette=["#66c2a5","#fc8d62"], plot_kws={"alpha":.5, "s":14}, diag_kind="kde")
g.fig.suptitle("Pair plot of numeric features", y=1.02, fontweight="bold")
savefig("v7_pairplot")


# **What it shows:** all pairwise scatter plots and KDEs per class. **Why:** a one-shot multivariate check for separability.
df["TenureGroup"] = pd.cut(df["tenure"], bins=[-1,12,24,48,72], labels=["0-12","13-24","25-48","49-72"])
pivot = df.pivot_table(index="Contract", columns="TenureGroup", values="ChurnFlag", aggfunc="mean").mul(100)
plt.figure(figsize=(7, 3.8))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={"label":"Churn rate (%)"})
plt.title("Churn rate (%) by contract type and tenure group"); plt.ylabel(""); plt.xlabel("Tenure (months)")
savefig("v8_heatmap_contract_tenure")


# **What it shows:** churn rate in each contract-by-tenure cell. **Why:** a two-factor view of the two strongest drivers.

# 4.2 Probability distributions (CO4)
# ---- Continuous: fit candidate distributions to MonthlyCharges and tenure; compare with Kolmogorov-Smirnov ----
cands = {"norm": stats.norm, "lognorm": stats.lognorm, "gamma": stats.gamma, "expon": stats.expon, "weibull_min": stats.weibull_min}
res = []
for col in ["MonthlyCharges", "tenure"]:
    x = df[col][df[col] > 0].values
    for name, dist in cands.items():
        params = dist.fit(x)
        ks = stats.kstest(x, name, args=params)
        res.append([col, name, round(ks.statistic, 4), ks.pvalue])
fit_tbl = pd.DataFrame(res, columns=["feature", "distribution", "KS_stat", "p_value"])
fit_tbl.sort_values(["feature","KS_stat"])

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
# Panel A: MonthlyCharges with normal vs gamma fits
x = df["MonthlyCharges"].values; grid = np.linspace(x.min(), x.max(), 300)
axes[0].hist(x, bins=40, density=True, alpha=.5, color="#8da0cb", label="Observed")
axes[0].plot(grid, stats.norm.pdf(grid, *stats.norm.fit(x)), "r-", label="Normal fit")
axes[0].plot(grid, stats.gamma.pdf(grid, *stats.gamma.fit(x)), "g--", label="Gamma fit")
axes[0].set_title("MonthlyCharges: observed vs fitted"); axes[0].legend()
# Panel B: Q-Q plot against normal
stats.probplot(x, dist="norm", plot=axes[1]); axes[1].set_title("Q-Q plot: MonthlyCharges vs Normal")
# Panel C: Q-Q plot for TotalCharges after log transform
stats.probplot(np.log1p(df["TotalCharges"]), dist="norm", plot=axes[2]); axes[2].set_title("Q-Q plot: log(1+TotalCharges)")
savefig("v9_distribution_fits_qq")

# Normality tests (D'Agostino K^2 is reliable for large n; Shapiro on a 5000-sample)
for c in numeric_cols:
    k2 = stats.normaltest(df[c]); print(f"{c:15s} D'Agostino K2 p = {k2.pvalue:.2e} -> {'NOT normal' if k2.pvalue<0.05 else 'normal'}")


# **Interpretation:** none of the three numeric features is normal (p << 0.05), and `MonthlyCharges` is multimodal so no single parametric curve fits well (all KS p-values are tiny). The Q-Q plots bend away from the diagonal. `TotalCharges` is right-skewed (skew 0.96), but a log transform over-corrects it (skew becomes about -0.82), so the log is not an automatic fix. For a neural network we rely on standardization instead. For a neural network we still standardize, which is enough since the network does not assume normality.
# ---- Discrete 1: Bernoulli / Binomial model for churn ----
p = df.ChurnFlag.mean(); n = 20
k = np.arange(0, 16)
pmf = stats.binom.pmf(k, n, p)
# Empirical: draw many random groups of 20 customers and count churners in each group
rng = np.random.default_rng(SEED)
sim = np.array([df.ChurnFlag.sample(n, random_state=int(s)).sum() for s in rng.integers(0, 10**6, 5000)])
emp = np.bincount(sim, minlength=16)[:16] / len(sim)

# ---- Discrete 2: number of add-on services per customer (0..6) ----
addons = ["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
df["n_addons"] = (df[addons] == "Yes").sum(axis=1)
inet = df[df.InternetService != "No"]["n_addons"]       # only internet customers can buy add-ons
pb = inet.mean() / 6
obs = inet.value_counts(normalize=True).sort_index().reindex(range(7), fill_value=0)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].bar(k-0.2, emp, width=.4, label="Simulated (5000 groups)", color="#8da0cb")
axes[0].bar(k+0.2, pmf, width=.4, label=f"Binomial(n=20, p={p:.3f})", color="#fc8d62")
axes[0].set_title("Churners among 20 random customers"); axes[0].set_xlabel("Number of churners"); axes[0].set_ylabel("Probability"); axes[0].legend()
axes[1].bar(np.arange(7)-0.2, obs.values, width=.4, label="Observed", color="#66c2a5")
axes[1].bar(np.arange(7)+0.2, stats.binom.pmf(np.arange(7), 6, pb), width=.4, label=f"Binomial(6, {pb:.2f})", color="#e78ac3")
axes[1].set_title("Add-on services held (internet customers)"); axes[1].set_xlabel("Number of add-ons"); axes[1].legend()
savefig("v10_discrete_distributions")
print("Binomial: E[churners in 20] =", round(n*p,2), " SD =", round(np.sqrt(n*p*(1-p)),2))
print("P(at least 8 churners among 20) = %.4f" % stats.binom.sf(7, n, p))


# **Interpretation:** the number of churners in a random group of 20 follows the Binomial(20, 0.265) distribution closely (expected about 5.3, SD about 1.97), so the churn process is well described as independent Bernoulli trials at the population level. The add-on count does **not** follow a binomial: the observed distribution is more spread out (12.6% of internet customers hold no add-on and 5.1% hold all six, versus about 3% and under 1% expected under a binomial). This over-dispersion means that add-on purchases are not independent: some customers buy bundles, others buy none. It is a case where a distribution assumption fails, so add-ons should be modelled jointly, which the MLP does.
# --- Numeric vs churn: Mann-Whitney U (non-parametric, since features are not normal) ---
rows = []
for c in numeric_cols:
    a, b = df.loc[df.ChurnFlag==0, c], df.loc[df.ChurnFlag==1, c]
    u = stats.mannwhitneyu(a, b); rbc = 1 - 2*u.statistic/(len(a)*len(b))   # rank-biserial effect size
    rows.append([c, u.pvalue, round(rbc, 3)])
print(pd.DataFrame(rows, columns=["feature","MWU_p_value","rank_biserial_effect"]).to_string(index=False))

# --- Categorical vs churn: chi-square test of independence + Cramer's V effect size ---
def cramers_v(ct):
    chi2 = stats.chi2_contingency(ct)[0]; n = ct.values.sum()
    return np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
rows = []
for c in cat_cols + ["SeniorCitizen"]:
    ct = pd.crosstab(df[c], df["Churn"]); chi2, p_, dof, _ = stats.chi2_contingency(ct)
    rows.append([c, round(chi2,1), p_, round(cramers_v(ct),3)])
chi_tbl = pd.DataFrame(rows, columns=["feature","chi2","p_value","cramers_V"]).sort_values("cramers_V", ascending=False)
chi_tbl.round(4)

plt.figure(figsize=(8, 6))
sns.barplot(data=chi_tbl, x="cramers_V", y="feature", palette="viridis")
plt.title("Association strength with Churn (Cramer's V)"); plt.xlabel("Cramer's V")
savefig("v11_cramers_v")


# **Interpretation:** all numeric features differ significantly between churners and non-churners, with `tenure` giving the largest effect. Among categorical features, `Contract` has the strongest association (Cramer's V = 0.41), followed by `InternetService` (0.32) and `PaymentMethod` (0.30), then `PaperlessBilling`, `OnlineSecurity` and `TechSupport` (about 0.17-0.19). `gender` and `PhoneService` are statistically unrelated to churn (p > 0.05) and are expected to contribute little to the model.

# Stage 6: REFINE: Preprocessing for modelling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

model_df = df.drop(columns=["Churn", "TenureGroup", "n_addons"]).copy()
y = model_df.pop("ChurnFlag").values

# Binary Yes/No and gender -> 0/1 ; multi-class -> one-hot (drop_first avoids the dummy variable trap)
bin_cols = ["Partner","Dependents","PhoneService","PaperlessBilling","MultipleLines","OnlineSecurity","OnlineBackup",
            "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
for c in bin_cols: model_df[c] = (model_df[c] == "Yes").astype(int)
model_df["gender"] = (model_df["gender"] == "Male").astype(int)
X = pd.get_dummies(model_df, columns=["InternetService","Contract","PaymentMethod"], drop_first=True).astype(float)
print("Feature matrix:", X.shape); print(list(X.columns))

# 70 / 15 / 15 stratified split
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.15/0.85, stratify=y_tmp, random_state=SEED)

# Standardize numeric columns (fit on train only)
scaler = StandardScaler().fit(X_train[numeric_cols])
for part in (X_train, X_val, X_test):
    part[numeric_cols] = scaler.transform(part[numeric_cols])

print(f"Train {X_train.shape[0]} | Val {X_val.shape[0]} | Test {X_test.shape[0]}")
print("Churn rate  train/val/test:", [round(v.mean(),3) for v in (y_train, y_val, y_test)])


# Feature analysis (before modelling)
from sklearn.feature_selection import mutual_info_classif
mi = pd.Series(mutual_info_classif(X_train, y_train, random_state=SEED), index=X_train.columns).sort_values()
plt.figure(figsize=(8, 8)); mi.plot.barh(color="#66a3d2")
plt.title("Mutual information with churn (training set)"); plt.xlabel("Mutual information")
savefig("v12_mutual_information")
print(mi.sort_values(ascending=False).head(8).round(4).to_string())


# **Insight:** `tenure`, `Contract_Two year`, `PaymentMethod_Electronic check`, `InternetService_Fiber optic` and `MonthlyCharges` carry the most information about churn, matching the earlier statistical tests. `gender` and `PhoneService` are near zero, so they could be dropped without much loss.

# Stage 7: INTERACT: Interactive exploration
import plotly.express as px
fig = px.scatter(df.sample(2000, random_state=SEED), x="tenure", y="MonthlyCharges", color="Churn", symbol="Contract",
                 hover_data=["InternetService", "PaymentMethod", "TotalCharges"],
                 color_discrete_map={"No":"#66c2a5","Yes":"#fc8d62"}, title="Interactive: tenure vs monthly charges")
fig.write_html(f"{FIG_DIR}/interactive_scatter.html")
fig.show() if False else print("Interactive chart saved to figures/interactive_scatter.html (open in a browser)")


# Deep Learning Model: Multi-Layer Perceptron (MLP)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.keras.utils.set_random_seed(SEED)

# Class weights compensate for the 73.5 / 26.5 imbalance
cw = {0: len(y_train) / (2*np.sum(y_train==0)), 1: len(y_train) / (2*np.sum(y_train==1))}
print("Class weights:", {k: round(v,3) for k,v in cw.items()})

def build_mlp(input_dim, units=(64, 32), dropout=0.3, lr=1e-3, l2=1e-4):
    # Fully-connected network: [Dense -> BatchNorm -> ReLU -> Dropout] x N  ->  Dense(1, sigmoid)
    inp = keras.Input(shape=(input_dim,))
    x = inp
    for u in units:
        x = layers.Dense(u, kernel_regularizer=keras.regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return m

build_mlp(X_train.shape[1]).summary()


# Hyperparameter tuning (basic grid search)
import itertools, time
grid = {"units": [(32,), (64, 32), (128, 64, 32)], "dropout": [0.2, 0.4], "lr": [1e-3, 3e-3]}
early = lambda: keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8, restore_best_weights=True)
results = []; t0 = time.time()
for units, dr, lr in itertools.product(grid["units"], grid["dropout"], grid["lr"]):
    tf.keras.utils.set_random_seed(SEED)
    m = build_mlp(X_train.shape[1], units, dr, lr)
    h = m.fit(X_train.values, y_train, validation_data=(X_val.values, y_val), epochs=60, batch_size=128,
              class_weight=cw, callbacks=[early()], verbose=0)
    results.append({"units": str(units), "dropout": dr, "lr": lr, "best_val_auc": max(h.history["val_auc"]),
                    "epochs_run": len(h.history["loss"])})
tune = pd.DataFrame(results).sort_values("best_val_auc", ascending=False).reset_index(drop=True)
print(f"Grid search finished in {time.time()-t0:.0f}s")
tune.round(4)

best = tune.iloc[0]; print("Best configuration:\n", best)
import ast
BEST_UNITS = ast.literal_eval(best["units"]); BEST_DROPOUT = float(best["dropout"]); BEST_LR = float(best["lr"])

plt.figure(figsize=(9, 4.5))
lab = tune.apply(lambda r: f"{r.units} d={r.dropout} lr={r.lr}", axis=1)
plt.barh(lab[::-1], tune["best_val_auc"][::-1], color="#8da0cb"); plt.xlim(tune.best_val_auc.min()-0.005, tune.best_val_auc.max()+0.003)
plt.xlabel("Best validation AUC"); plt.title("Hyperparameter search results")
savefig("v13_hyperparameter_search")


# Final model training
tf.keras.utils.set_random_seed(SEED)
model = build_mlp(X_train.shape[1], BEST_UNITS, BEST_DROPOUT, BEST_LR)
callbacks = [keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=10, restore_best_weights=True),
             keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5)]
history = model.fit(X_train.values, y_train, validation_data=(X_val.values, y_val), epochs=100, batch_size=128,
                    class_weight=cw, callbacks=callbacks, verbose=0)
print("Epochs trained:", len(history.history["loss"]))


# Model Evaluation and Result Visualization
hist = pd.DataFrame(history.history)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(hist["loss"], label="Train loss"); ax[0].plot(hist["val_loss"], label="Validation loss")
ax[0].set_title("Loss vs Epoch"); ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Binary cross-entropy"); ax[0].legend()
ax[1].plot(hist["accuracy"], label="Train accuracy"); ax[1].plot(hist["val_accuracy"], label="Validation accuracy")
ax[1].set_title("Accuracy vs Epoch"); ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Accuracy"); ax[1].legend()
savefig("r1_loss_accuracy_curves")


# **Interpretation:** training loss keeps falling, while validation loss flattens at about 0.52-0.53 after the first few epochs. The widening gap shows that mild over-fitting starts early, which is why dropout, L2 regularization, learning-rate reduction and early stopping (best-AUC weights restored after 18 epochs) are used. Accuracy is measured at a 0.5 threshold on a class-weighted model, so it sits below the 73.5% majority-class figure on some epochs; this is expected, because the model deliberately trades accuracy for recall on the minority class. AUC, recall and F1 are the more informative metrics.
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score, classification_report, ConfusionMatrixDisplay,
                             accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score)
p_test = model.predict(X_test.values, verbose=0).ravel()
p_val  = model.predict(X_val.values, verbose=0).ravel()

# Choose the decision threshold on the VALIDATION set (maximise F1), then apply once to the test set
ths = np.linspace(0.2, 0.8, 61)
best_th = ths[np.argmax([f1_score(y_val, p_val >= t) for t in ths])]
print(f"Threshold chosen on validation set: {best_th:.2f}")
pred_test = (p_test >= best_th).astype(int)

fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
ConfusionMatrixDisplay(confusion_matrix(y_test, pred_test), display_labels=["Stay","Churn"]).plot(ax=ax[0], cmap="Blues", colorbar=False)
ax[0].set_title(f"Confusion matrix (test, threshold={best_th:.2f})")
fpr, tpr, _ = roc_curve(y_test, p_test); auc = roc_auc_score(y_test, p_test)
ax[1].plot(fpr, tpr, lw=2, label=f"MLP (AUC = {auc:.3f})"); ax[1].plot([0,1],[0,1],"k--", label="Chance")
ax[1].set_xlabel("False positive rate"); ax[1].set_ylabel("True positive rate"); ax[1].set_title("ROC curve (test)"); ax[1].legend()
pr, rc, _ = precision_recall_curve(y_test, p_test); ap = average_precision_score(y_test, p_test)
ax[2].plot(rc, pr, lw=2, label=f"AP = {ap:.3f}"); ax[2].axhline(y_test.mean(), color="k", ls="--", label="Baseline (prevalence)")
ax[2].set_xlabel("Recall"); ax[2].set_ylabel("Precision"); ax[2].set_title("Precision-Recall curve (test)"); ax[2].legend()
savefig("r2_confusion_roc_pr")
print(classification_report(y_test, pred_test, target_names=["Stay","Churn"], digits=3))


# **Interpretation:** the confusion matrix shows how many churners are caught (bottom-right) versus missed (bottom-left). The ROC curve sits well above the diagonal; the AUC measures the probability that a random churner gets a higher score than a random non-churner. The precision-recall curve is the more honest view for an imbalanced problem: precision is far above the 26.5% prevalence baseline across a wide range of recall.
from sklearn.calibration import calibration_curve
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
sns.histplot(x=p_test, hue=np.where(y_test==1,"Churn","Stay"), bins=30, stat="density", common_norm=False, element="step", ax=ax[0], palette={"Churn":"#fc8d62","Stay":"#66c2a5"}, hue_order=["Stay","Churn"])
ax[0].axvline(best_th, color="k", ls="--"); ax[0].set_title("Predicted churn probability by true class"); ax[0].set_xlabel("Predicted probability")
fp, mp = calibration_curve(y_test, p_test, n_bins=8, strategy="quantile")
ax[1].plot(mp, fp, "o-", label="MLP"); ax[1].plot([0,1],[0,1],"k--", label="Perfect"); ax[1].set_xlabel("Mean predicted probability")
ax[1].set_ylabel("Observed churn fraction"); ax[1].set_title("Calibration curve (class-weighted model)"); ax[1].legend()
savefig("r3_score_distribution_calibration")


# **Interpretation:** the two class score distributions are clearly separated but overlap in the middle (the hard-to-classify customers). Because we trained with class weights, predicted probabilities are *over-estimated* relative to true churn frequency (curve lies below the diagonal). The ranking is still good (high AUC), but if actual probabilities are needed, a calibration step (e.g. isotonic regression) should be applied.
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

models = {"Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
          "Random Forest": RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced", random_state=SEED, n_jobs=-1),
          "Gradient Boosting": GradientBoostingClassifier(random_state=SEED)}
rows = []
for name, m in models.items():
    m.fit(X_train, y_train); pr_ = m.predict_proba(X_test)[:, 1]
    th = ths[np.argmax([f1_score(y_val, m.predict_proba(X_val)[:,1] >= t) for t in ths])]
    pd_ = (pr_ >= th).astype(int)
    rows.append([name, accuracy_score(y_test,pd_), precision_score(y_test,pd_), recall_score(y_test,pd_), f1_score(y_test,pd_), roc_auc_score(y_test,pr_)])
rows.append(["MLP (ours)", accuracy_score(y_test,pred_test), precision_score(y_test,pred_test), recall_score(y_test,pred_test), f1_score(y_test,pred_test), auc])
bench = pd.DataFrame(rows, columns=["Model","Accuracy","Precision","Recall","F1","ROC-AUC"]).set_index("Model").round(3)
print(bench)
bench[["F1","ROC-AUC"]].plot.bar(figsize=(8,4), rot=15, color=["#8da0cb","#fc8d62"]); plt.ylim(0.4, 0.9)
plt.title("Model comparison on the test set"); plt.ylabel("Score")
savefig("r4_model_comparison")


# **Interpretation:** the MLP (AUC 0.845) lands within about 0.005 AUC of logistic regression (0.849), random forest (0.850) and gradient boosting (0.851); the MLP is marginally lower on AUC and highest on accuracy and precision. This is typical for small tabular datasets (about 5k training rows) and suggests the signal is largely additive rather than strongly non-linear. The deep model therefore adds no measurable gain over simple baselines here, which is itself a useful finding.
# Permutation importance: shuffle one feature at a time on the validation set and measure the AUC drop
rng = np.random.default_rng(SEED); base = roc_auc_score(y_val, p_val); imp = {}
for col in X_val.columns:
    drops = []
    for _ in range(10):
        Xp = X_val.copy(); Xp[col] = rng.permutation(Xp[col].values)
        drops.append(base - roc_auc_score(y_val, model.predict(Xp.values, verbose=0).ravel()))
    imp[col] = (np.mean(drops), np.std(drops))
imp = pd.DataFrame(imp, index=["mean","std"]).T.sort_values("mean")
top = imp.tail(15)
plt.figure(figsize=(8, 6)); plt.barh(top.index, top["mean"], xerr=top["std"], color="#66a3d2")
plt.xlabel("Drop in validation AUC when feature is shuffled"); plt.title("Top-15 permutation feature importances (MLP)")
savefig("r5_permutation_importance")
print(imp.sort_values("mean", ascending=False).head(8).round(4).to_string())


# **Interpretation:** the importance ranking confirms the EDA: `tenure`, contract type, monthly/total charges and internet service type matter most; demographics such as `gender` have near-zero importance. This agreement between statistical tests (Cramer's V), mutual information and model-based importance gives us confidence that the findings are robust.
