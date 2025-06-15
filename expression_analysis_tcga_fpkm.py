import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Load Files ---
data_dir = r"C:\Users\Jordy\Desktop\DATA\GSE288840_RAW"
file_pattern = os.path.join(data_dir, "*.csv.gz")
file_list = glob.glob(file_pattern)

mcf7_list = []
pr_list = []
dxr_list = []

for file_path in file_list:
    file_name = os.path.basename(file_path)
    try:
        df = pd.read_csv(file_path, compression="gzip")
    except Exception as e:
        print(f"Error loading {file_name}: {str(e)}")
        continue

    if "MCF7" in file_name:
        mcf7_list.append(df)
    elif "PR" in file_name:
        pr_list.append(df)
    elif "DXR" in file_name:
        dxr_list.append(df)
    else:
        print(f"File {file_name} did not match any expected group.")

# --- Step 2: Aggregate Replicates ---
def aggregate_replicates(df_list, sample_label):
    aggregated = pd.DataFrame()
    replicate_cols = []  # To store renamed expression column names

    for idx, df in enumerate(df_list):
        df = df.copy()
        expr_col = df.columns[2]  # Adjust if needed
        new_col = f"{sample_label}_{idx+1}_FPKM"
        df = df.rename(columns={expr_col: new_col})
        replicate_cols.append(new_col)

        if aggregated.empty:
            aggregated = df
        else:
            aggregated = pd.merge(aggregated, df, on=["Gene_ID", "Gene_Symbol"], how="inner")

    aggregated[replicate_cols] = aggregated[replicate_cols].apply(pd.to_numeric, errors='coerce')
    aggregated[f"avg_{sample_label}_FPKM"] = aggregated[replicate_cols].mean(axis=1)
    return aggregated[["Gene_ID", "Gene_Symbol", f"avg_{sample_label}_FPKM"]]

if mcf7_list:
    df_mcf7_agg = aggregate_replicates(mcf7_list, "MCF7")
    print("Aggregated MCF7 sample shape:", df_mcf7_agg.shape)
else:
    print("No MCF7 files found.")

if pr_list:
    df_pr_agg = aggregate_replicates(pr_list, "PR")
    print("Aggregated PR sample shape:", df_pr_agg.shape)
else:
    print("No PR files found.")

if dxr_list:
    df_dxr_agg = aggregate_replicates(dxr_list, "DXR")
    print("Aggregated DXR sample shape:", df_dxr_agg.shape)
else:
    df_dxr_agg = None
    print("No DXR files found.")

# --- Step 3: Merge Aggregated Data ---
df_merged = pd.merge(df_mcf7_agg, df_pr_agg, on=["Gene_ID", "Gene_Symbol"], how="inner")
print("Merged data shape (MCF7 and PR):", df_merged.shape)

# --- Step 4: Calculate Log2 Fold Change ---
df_merged["logFC"] = np.log2(df_merged["avg_PR_FPKM"] + 1) - np.log2(df_merged["avg_MCF7_FPKM"] + 1)
print("\n===== Aggregated Data Preview =====")
print(df_merged[["Gene_Symbol", "avg_MCF7_FPKM", "avg_PR_FPKM", "logFC"]].head())

# --- Step 5: Identify Significantly Altered Genes ---
deg_threshold = 2
deg_genes = df_merged[df_merged["logFC"].abs() > deg_threshold]
print("\n===== Significantly Altered Genes =====")
print(deg_genes[["Gene_Symbol", "logFC"]])

# --- Step 6: Visualization - Histogram ---
plt.figure(figsize=(8, 6))
sns.histplot(df_merged["logFC"], bins=50, color="blue")
plt.xlabel("Log2 Fold Change")
plt.ylabel("Gene Count")
plt.title("Histogram of Log2 Fold Change (PR vs. MCF7)")
plt.tight_layout()
plt.show()

# --- Step 7: Filter Out Low Expression Genes ---
expr_threshold = 1
df_filtered = df_merged[(df_merged["avg_MCF7_FPKM"] > expr_threshold) |
                        (df_merged["avg_PR_FPKM"] > expr_threshold)]
print("Filtered data shape (removing low expression genes):", df_filtered.shape)

# --- Step 8: Label Regulation Direction ---
def label_regulation(fc, threshold):
    if fc > threshold:
        return "Upregulated"
    elif fc < -threshold:
        return "Downregulated"
    else:
        return "Not significant"

# Make a copy of the filtered DataFrame to avoid SettingWithCopyWarning
df_filtered = df_filtered.copy()
df_filtered["Regulation"] = df_filtered["logFC"].apply(lambda x: label_regulation(x, deg_threshold))

# --- Step 9: Scatter Plot of Gene Expression ---
plt.figure(figsize=(8, 6))
plt.scatter(np.log2(df_filtered["avg_MCF7_FPKM"] + 1),
            np.log2(df_filtered["avg_PR_FPKM"] + 1),
            c=df_filtered["logFC"], cmap="RdBu_r", alpha=0.5)
plt.xlabel("log2(avg_MCF7_FPKM + 1)")
plt.ylabel("log2(avg_PR_FPKM + 1)")
plt.title("Scatter Plot of Gene Expression (MCF7 vs. PR)")
plt.colorbar(label="logFC")
plt.tight_layout()
plt.show()

# --- Step 10: Heatmap of Top 50 Differentially Expressed Genes ---
top50 = df_filtered.reindex(df_filtered["logFC"].abs().sort_values(ascending=False).index).head(50)
heatmap_data = top50[["avg_MCF7_FPKM", "avg_PR_FPKM"]].apply(lambda x: np.log2(x + 1))
plt.figure(figsize=(8, 10))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis", yticklabels=top50["Gene_Symbol"])
plt.title("Heatmap of Top 50 Differentially Expressed Genes")
plt.ylabel("Gene Symbol")
plt.xlabel("Condition")
plt.tight_layout()
plt.show()
