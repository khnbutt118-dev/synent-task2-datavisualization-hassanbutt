"""Synent Data Science Internship - Task 2: Data Visualization (Iris)
Author: Hassan Butt
Run: python main.py
"""

import sys, os, re, json, shutil, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


DATASETS = {'iris': {'builtin': True,
          'files': ['Iris.csv', 'iris.csv'],
          'kaggle': 'uciml/iris',
          'name': 'Iris flower dataset',
          'page': 'https://www.kaggle.com/datasets/uciml/iris'}}



def _outdir(repo_dir):
    p = Path(repo_dir) / "outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_fig(fig, repo_dir, name):
    fig.savefig(_outdir(repo_dir) / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return name


def save_json(obj, repo_dir, name):
    (_outdir(repo_dir) / name).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _read_csv(path, spec):
    kw = {"encoding": spec.get("encoding", "utf-8")}
    if spec.get("no_header_file") and Path(path).name == spec["no_header_file"]:
        kw["header"] = None
        kw["names"] = spec["names"]
    try:
        return pd.read_csv(path, **kw)
    except UnicodeDecodeError:
        kw["encoding"] = "latin-1"
        return pd.read_csv(path, **kw)


def _builtin(key):
    if key == "iris":
        from sklearn.datasets import load_iris
        d = load_iris(as_frame=True)
        df = d.frame.drop(columns=["target"])
        df["species"] = [d.target_names[i] for i in d.target]
        return df
    raise KeyError(key)


def get_data(key, repo_dir):
    """Load a dataset: local data/ folder -> built-in -> kagglehub -> public URL."""
    spec = DATASETS[key]
    data_dir = Path(repo_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in spec["files"]:
        p = data_dir / name
        if p.exists():
            print(f"  [data] using local file {p}")
            return _read_csv(p, spec)
    if spec.get("builtin"):
        print("  [data] using built-in copy")
        return _builtin(key)
    if spec.get("kaggle"):
        try:
            import kagglehub
            folder = Path(kagglehub.dataset_download(spec["kaggle"]))
            cands = list(folder.rglob("*.csv"))
            pick = None
            for want in spec["files"]:
                for c in cands:
                    if c.name.lower() == want.lower():
                        pick = c
                        break
                if pick:
                    break
            if pick is None and cands:
                pick = cands[0]
            if pick is not None:
                dest = data_dir / pick.name
                shutil.copy(pick, dest)
                print(f"  [data] downloaded from Kaggle: {dest}")
                return _read_csv(dest, spec)
        except Exception as e:
            print(f"  [data] kagglehub download not available ({type(e).__name__}: {e})")
    if spec.get("url"):
        try:
            import urllib.request
            dest = data_dir / (spec.get("no_header_file") or spec["files"][0])
            urllib.request.urlretrieve(spec["url"], dest)
            print(f"  [data] downloaded from public URL: {dest}")
            return _read_csv(dest, spec)
        except Exception as e:
            print(f"  [data] public URL download failed ({type(e).__name__}: {e})")
    raise FileNotFoundError(
        f"Dataset '{spec['name']}' not found.\n"
        f"  -> Download it from {spec['page']}\n"
        f"  -> Save the CSV as one of {spec['files']} inside: {data_dir.resolve()}\n"
        f"  -> Then run the script again.")


def _to_datetime(series):
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        return pd.to_datetime(series, errors="coerce")


def _snake(col):
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(col).strip()).strip("_").lower()


def task2_run(repo_dir):
    df = get_data("iris", repo_dir)
    ren = {}
    for c in df.columns:
        lc = str(c).lower()
        if "sepal" in lc and "length" in lc:
            ren[c] = "sepal_length"
        elif "sepal" in lc and "width" in lc:
            ren[c] = "sepal_width"
        elif "petal" in lc and "length" in lc:
            ren[c] = "petal_length"
        elif "petal" in lc and "width" in lc:
            ren[c] = "petal_width"
        elif lc in ("species", "variety", "class"):
            ren[c] = "species"
    df = df.rename(columns=ren)
    df["species"] = df["species"].astype(str).str.replace("Iris-", "", regex=False)
    feats = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    df = df[feats + ["species"]]
    shape = [int(df.shape[0]), int(df.shape[1])]

    # Bar chart: average of each feature per species
    means = df.groupby("species")[feats].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    means.T.plot(kind="bar", ax=ax, colormap="viridis")
    ax.set_title("Bar chart: average measurement of each feature by species")
    ax.set_ylabel("cm")
    ax.tick_params(axis="x", rotation=0)
    save_fig(fig, repo_dir, "bar_chart_feature_means.png")

    # Histograms
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for a, f in zip(axes.ravel(), feats):
        sns.histplot(data=df, x=f, hue="species", kde=True, ax=a, bins=20)
        a.set_title(f"Histogram of {f}")
    fig.tight_layout()
    save_fig(fig, repo_dir, "histograms.png")

    # Scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(data=df, x="sepal_length", y="sepal_width", hue="species", s=60, ax=axes[0])
    axes[0].set_title("Scatter: sepal length vs sepal width")
    sns.scatterplot(data=df, x="petal_length", y="petal_width", hue="species", s=60, ax=axes[1])
    axes[1].set_title("Scatter: petal length vs petal width")
    save_fig(fig, repo_dir, "scatter_plots.png")

    # Compare features: box plots, correlation heatmap, pair plot
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for a, f in zip(axes.ravel(), feats):
        sns.boxplot(data=df, x="species", y=f, ax=a)
        a.set_title(f"{f} by species")
    fig.tight_layout()
    save_fig(fig, repo_dir, "boxplots_compare_features.png")
    corr = df[feats].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation between features")
    save_fig(fig, repo_dir, "correlation_heatmap.png")
    pg = sns.pairplot(df, hue="species", corner=True)
    pg.fig.suptitle("Pair plot of all features", y=1.02)
    pg.fig.savefig(_outdir(repo_dir) / "pairplot.png", dpi=130, bbox_inches="tight")
    plt.close(pg.fig)

    from sklearn.feature_selection import f_classif
    F, _ = f_classif(df[feats], df["species"])
    ranking = sorted(zip(feats, F), key=lambda t: -t[1])
    best = ranking[0][0]
    setosa_sep = None
    if "setosa" in set(df["species"]):
        others_min = df[df["species"] != "setosa"]["petal_length"].min()
        setosa_max = df[df["species"] == "setosa"]["petal_length"].max()
        setosa_sep = bool(setosa_max < others_min)
    insights = [
        f"{best} separates the three species best (ANOVA F = {ranking[0][1]:.0f}); {ranking[-1][0]} separates them least (F = {ranking[-1][1]:.0f}).",
        f"Petal length and petal width are strongly correlated (r = {corr.loc['petal_length', 'petal_width']:.2f}).",
        ("Setosa is completely separable from the other two species using petal length alone."
         if setosa_sep else "Versicolor and virginica overlap on most features, so they are harder to separate."),
        f"Average petal length: " + ", ".join(f"{s} {v:.1f} cm" for s, v in means["petal_length"].items()) + ".",
    ]
    return {"headline": {"Samples": shape[0], "Features compared": len(feats),
                         "Best separating feature": best,
                         "Petal length-width correlation": round(float(corr.loc['petal_length', 'petal_width']), 2)},
            "insights": insights, "dataset_shape": shape,
            "files": ["bar_chart_feature_means.png", "histograms.png", "scatter_plots.png",
                      "boxplots_compare_features.png", "correlation_heatmap.png", "pairplot.png"]}



if __name__ == "__main__":
    res = task2_run(Path(__file__).parent)
    print(json.dumps(res, indent=2, default=str))
