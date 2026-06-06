import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_wine
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

df = pd.read_csv("moscow.csv")

#bai 1
def scatter_plot():
    plt.figure(figsize=(8,6))
    plt.scatter(df['secondary_price_per_sqm'], df['secondary_mom_change_pct'], alpha=0.5)
    plt.title("Scatter Plot")
    plt.xlabel("Secondary Price per sqm")
    plt.ylabel("Secondary Mom Change Pct")
    plt.grid()
    plt.show()
scatter_plot()

#bai 2
def scatter_matrix():
    iris = load_iris()
    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )
    df["species"] = iris.target

    sns.pairplot(
    df,
    hue="species",
    diag_kind="kde",
    )

    plt.suptitle("Scatter Matrix", y=1.02)
    plt.show()
scatter_matrix()

#bai 3
def Parallel_Coordinates():
    wine = load_wine()
    df = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
    )
    df["class"] = wine.target
    df.head()

    features = ["alcohol", "malic_acid", "ash", "flavanoids", "proline"]
    df_plot = df[features + ["class"]]

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df_plot[features])
    df_scaled = pd.DataFrame(
    scaled_features,
    columns=features
    )
    df_scaled["class"] = df_plot["class"]
    plt.figure(figsize=(10,6))
    parallel_coordinates(
    df_scaled,
    "class",
    colormap=plt.cm.Set1,
    linewidth=1
    )
    plt.title("Parallel Coordinates Plot - Wine Dataset")
    plt.xlabel("Features")
    plt.ylabel("Normalized Value")
    plt.show()
Parallel_Coordinates()

#bai 4
def pca():
    mnist = fetch_openml("mnist_784")
    X = mnist.data
    y = mnist.target
    X = X[:2000]
    y = y[:2000]

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(7,6))

    plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=y.astype(int),
    cmap="tab10",
    s=10
    )

    plt.title("PCA Visualization of MNIST")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    plt.show()
pca()

def tSNE():
    mnist = fetch_openml("mnist_784")
    X = mnist.data
    y = mnist.target
    X = X[:2000]
    y = y[:2000]

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)

    X_tsne = tsne.fit_transform(X)
    plt.figure(figsize=(7,6))

    plt.scatter(
    X_tsne[:,0],
    X_tsne[:,1],
    c=y.astype(int),
    cmap="tab10",
    s=10
    )

    plt.title("t-SNE Visualization of MNIST")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.colorbar()
    plt.show()
tSNE()
