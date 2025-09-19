# analysis.py
import os
import sys
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("summaries", exist_ok=True)

def create_csv_if_missing(path="data/iris.csv"):
    if os.path.exists(path):
        print(f"[info] Found existing CSV: {path}")
        return
    print("[info] CSV not found — creating iris CSV using sklearn.datasets...")
    try:
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
        df.to_csv(path, index=False)
        print(f"[info] Saved iris CSV to {path}")
    except Exception as e:
        print("[error] Could not create CSV via sklearn:", e)
        raise

def load_dataset(path="data/iris.csv"):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print(f"[error] File not found: {path}")
        raise
    except pd.errors.EmptyDataError:
        print("[error] CSV is empty")
        raise
    except Exception:
        print("[error] Unexpected error while reading CSV:")
        traceback.print_exc()
        raise

def inspect_and_clean(df):
    print("\n=== HEAD ===")
    print(df.head(), "\n")
    print("=== DTypes ===")
    print(df.dtypes, "\n")
    print("=== Missing values per column ===")
    print(df.isnull().sum(), "\n")

    total_missing = int(df.isnull().sum().sum())
    if total_missing > 0:
        print(f"[info] Found {total_missing} missing values. Filling numericals with median, categoricals with mode.")
        num_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(include=['object','category']).columns
        for c in num_cols:
            median = df[c].median()
            df[c].fillna(median, inplace=True)
        for c in cat_cols:
            mode = df[c].mode()
            if not mode.empty:
                df[c].fillna(mode[0], inplace=True)
        print("[info] Missing values filled.")
    else:
        print("[info] No missing values found.")
    return df

def basic_analysis(df):
    num_desc = df.describe()
    num_desc.to_csv("summaries/describe_numeric.csv")
    print("\n=== .describe() saved to summaries/describe_numeric.csv ===\n")
    # group by species (if exists)
    if 'species' in df.columns:
        group_means = df.groupby('species').mean()
        group_means.to_csv("summaries/group_means_by_species.csv")
        print("=== Group means by species saved to summaries/group_means_by_species.csv ===")
        # print a short highlight
        if 'petal length (cm)' in df.columns:
            petal_means = df.groupby('species')['petal length (cm)'].mean().sort_values()
            print("\nAverage petal length per species (ascending):")
            print(petal_means)
    else:
        print("[warn] 'species' column not found — skipping group by.")

def create_plots(df):
    # 1) Line chart: synthetic time index with rolling mean of sepal length
    if 'sepal length (cm)' in df.columns:
        df_time = df.copy()
        df_time['date'] = pd.date_range(start='2020-01-01', periods=len(df_time), freq='D')
        df_time['sepal_rolling_7'] = df_time['sepal length (cm)'].rolling(window=7, min_periods=1).mean()
        plt.figure(figsize=(10,4))
        plt.plot(df_time['date'], df_time['sepal_rolling_7'], marker='.', linewidth=1)
        plt.title('7-day Rolling Mean of Sepal Length (synthetic dates)')
        plt.xlabel('Date')
        plt.ylabel('Sepal length (cm) - 7-day rolling mean')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("figures/sepal_rolling.png")
        plt.close()
        print("[plot] Saved figures/sepal_rolling.png")
    else:
        print("[skip] sepal length column not found for line chart.")

    # 2) Bar chart: average petal length per species
    if 'species' in df.columns and 'petal length (cm)' in df.columns:
        petal_means = df.groupby('species')['petal length (cm)'].mean().sort_values()
        plt.figure(figsize=(6,4))
        plt.bar(petal_means.index, petal_means.values)
        plt.title('Average Petal Length per Species')
        plt.xlabel('Species')
        plt.ylabel('Average petal length (cm)')
        plt.tight_layout()
        plt.savefig("figures/petal_bar.png")
        plt.close()
        print("[plot] Saved figures/petal_bar.png")
    else:
        print("[skip] Bar chart columns missing.")

    # 3) Histogram of sepal length
    if 'sepal length (cm)' in df.columns:
        plt.figure(figsize=(6,4))
        plt.hist(df['sepal length (cm)'], bins=15)
        plt.title('Distribution of Sepal Length')
        plt.xlabel('Sepal length (cm)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig("figures/sepal_hist.png")
        plt.close()
        print("[plot] Saved figures/sepal_hist.png")
    else:
        print("[skip] Histogram column missing.")

    # 4) Scatter: sepal length vs petal length by species
    if 'sepal length (cm)' in df.columns and 'petal length (cm)' in df.columns and 'species' in df.columns:
        plt.figure(figsize=(7,5))
        for name, group in df.groupby('species'):
            plt.scatter(group['sepal length (cm)'], group['petal length (cm)'], label=name)
        plt.title('Sepal length vs Petal length (by species)')
        plt.xlabel('Sepal length (cm)')
        plt.ylabel('Petal length (cm)')
        plt.legend(title='Species')
        plt.tight_layout()
        plt.savefig("figures/sepal_petal_scatter.png")
        plt.close()
        print("[plot] Saved figures/sepal_petal_scatter.png")
    else:
        print("[skip] Scatter plot columns missing.")

def main():
    ensure_dirs()
    create_csv_if_missing("data/iris.csv")
    df = load_dataset("data/iris.csv")
    df = inspect_and_clean(df)
    basic_analysis(df)
    create_plots(df)
    print("\n[done] Analysis complete. Check the 'figures' and 'summaries' folders for outputs.")

if __name__ == "__main__":
    main()
