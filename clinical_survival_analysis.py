import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
import os


def load_clinical_data(file_path):
    try:
        df = pd.read_csv(file_path, sep="\t", comment="#")
        print("Clinical data loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading clinical data: {e}")
        return pd.DataFrame()


def clean_clinical_data(df):
    df = df.copy()

    # Convert survival status to binary indicator
    df["OS_STATUS"] = df["OS_STATUS"].apply(lambda x: 1 if "DECEASED" in str(x).upper() else 0)

    # Map metastasis column to descriptive labels
    df["METASTASIS_LABEL"] = df["METASTASIS"].map({"M0": "No Metastasis", "M1": "Metastasis"})

    # Ensure OS_MONTHS is numeric
    df["OS_MONTHS"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")

    # Drop rows missing key data
    df_clean = df.dropna(subset=["OS_MONTHS", "OS_STATUS", "METASTASIS"])

    print("Data cleaning complete. Missing values per column:")
    print(df_clean.isnull().sum())
    return df_clean


def plot_survival_histogram(df, save_fig=False):
    plt.figure(figsize=(8, 6))
    sns.histplot(df["OS_MONTHS"].dropna(), bins=20, kde=True, color="skyblue")
    plt.xlabel("Overall Survival (Months)")
    plt.ylabel("Patient Count")
    plt.title("Distribution of Overall Survival in Patients")
    plt.tight_layout()
    if save_fig:
        plt.savefig("survival_histogram.png", dpi=300)
    plt.show()


def plot_survival_boxplot(df, save_fig=False):
    plt.figure(figsize=(6, 6))
    sns.boxplot(x="METASTASIS_LABEL", y="OS_MONTHS", data=df, palette="Set2")
    plt.xlabel("Metastasis Status")
    plt.ylabel("Overall Survival (Months)")
    plt.title("Survival by Metastasis Status")
    plt.tight_layout()
    if save_fig:
        plt.savefig("survival_boxplot.png", dpi=300)
    plt.show()


def plot_km_survival(df, save_fig=False):
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8, 6))

    for status in df["METASTASIS_LABEL"].unique():
        mask = df["METASTASIS_LABEL"] == status
        durations = df.loc[mask, "OS_MONTHS"]
        events = df.loc[mask, "OS_STATUS"]

        if len(durations) > 0:
            kmf.fit(durations=durations, event_observed=events, label=status)
            kmf.plot_survival_function(ci_show=True)
        else:
            print(f"No data available for group: {status}")

    plt.xlabel("Time (Months)")
    plt.ylabel("Survival Probability")
    plt.title("Kaplan-Meier Survival Curves by Metastasis Status")
    plt.legend(title="Metastasis Status")
    plt.tight_layout()
    if save_fig:
        plt.savefig("km_survival_curves.png", dpi=300)
    plt.show()


# --------------------------
# Main Script Workflow
# --------------------------
def main():
    # File path to clinical data
    file_path = r"C:\Users\jordy\Desktop\DATA\brca_tcga_pub\data_clinical_patient.txt"

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # Load data
    df = load_clinical_data(file_path)
    print("First 5 rows of the clinical data:")
    print(df.head())

    # Clean data
    df_clean = clean_clinical_data(df)
    print("\nCleaned Data Overview (first 5 rows):")
    print(df_clean.head())

    # Visualizations
    plot_survival_histogram(df_clean)
    plot_survival_boxplot(df_clean)
    plot_km_survival(df_clean)


if __name__ == "__main__":
    main()
