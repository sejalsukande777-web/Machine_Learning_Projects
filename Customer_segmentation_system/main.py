import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def load_customer_data(path: str) -> pd.DataFrame:
    """Load and clean the customer dataset."""
    df = pd.read_csv(path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select features for clustering and scale them."""
    features = df[['AnnualIncome', 'SpendingScore']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features


def find_optimal_clusters(X_scaled, max_clusters: int = 10) -> list:
    """Compute WCSS for a range of cluster counts using the elbow method."""
    wcss = []
    for i in range(1, max_clusters + 1):
        model = KMeans(n_clusters=i, random_state=42, n_init=10)
        model.fit(X_scaled)
        wcss.append(model.inertia_)
    return wcss


def plot_elbow(wcss: list):
    """Plot the elbow curve to help choose the number of clusters."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(wcss) + 1), wcss, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method for K-Means Clustering')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def train_kmeans(X_scaled, n_clusters: int = 4) -> KMeans:
    """Train a K-Means model on the scaled features."""
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    model.fit(X_scaled)
    return model


def assign_clusters(df: pd.DataFrame, clusters: list) -> pd.DataFrame:
    """Add cluster labels to the original dataframe."""
    result_df = df.copy()
    result_df['Cluster'] = clusters
    return result_df


def summarize_clusters(result_df: pd.DataFrame):
    """Print cluster counts and representative customer segments."""
    print("\nCluster distribution for marketing segments:")
    print(result_df['Cluster'].value_counts().sort_index())
    print("\nSample records by cluster:")
    print(result_df.sort_values(['Cluster', 'AnnualIncome']).head(15))


def main():
    print('Applied K-Means Clustering to group customers by behavior patterns for marketing analysis')
    print('-' * 80)

    data_path = 'Mall_Customers.csv'
    df = load_customer_data(data_path)

    print('Dataset loaded:')
    print(df.head())
    print('\nDataset shape:', df.shape)
    print('\nMissing values:')
    print(df.isnull().sum())

    X_scaled = prepare_features(df)

    print('\nComputing elbow method...')
    wcss = find_optimal_clusters(X_scaled, max_clusters=10)
    plot_elbow(wcss)

    n_clusters = 4
    print(f'\nTraining K-Means with n_clusters={n_clusters}')
    model = train_kmeans(X_scaled, n_clusters=n_clusters)
    clusters = model.predict(X_scaled)

    result_df = assign_clusters(df, clusters)
    summarize_clusters(result_df)

    print('\nCluster centers (scaled feature space):')
    print(model.cluster_centers_)

    output_path = 'Mall_Customers_with_clusters.csv'
    result_df.to_csv(output_path, index=False)
    print(f'\nClustered dataset saved to: {output_path}')


if __name__ == '__main__':
    main()
