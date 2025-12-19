import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from datetime import datetime

csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'housing.csv')
df = pd.read_csv(csv_path)

data = df[["longitude","latitude","median_income"]]

def prepare_data():
    data2 = data.dropna().copy()
    data2 = data2[~data2.isin([np.inf, -np.inf]).any(axis=1)]
    data2 = data2.reset_index(drop=True)
    
    return data2


def train_gmm(data, n_clusters=5):
    income_data = data[['median_income']].values
    
    gmm = GaussianMixture(n_components=n_clusters, random_state=12, n_init=10)
    clusters = gmm.fit_predict(income_data)
    
    cluster_incomes = {i: gmm.means_[i][0] for i in range(n_clusters)}
    sorted_clusters = sorted(cluster_incomes.keys(), key=lambda x: cluster_incomes[x], reverse=True)
    
    return clusters, gmm, sorted_clusters


def train_kmeans(data, n_clusters=5):
    income_data = data[['median_income']].values
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=12, n_init=10)
    clusters = kmeans.fit_predict(income_data)
    
    cluster_incomes = {i: kmeans.cluster_centers_[i][0] for i in range(n_clusters)}
    sorted_clusters = sorted(cluster_incomes.keys(), key=lambda x: cluster_incomes[x], reverse=True)
    
    return clusters, kmeans, sorted_clusters


INCOME_LABELS = ['Süper Zengin', 'Zengin', 'Orta', 'Fakir', 'Çok Fakir']
COLORS = ['#e74c3c', '#f39c12', '#27ae60', '#3498db', '#9b59b6']


def plot_combined_clustering_comparison(data, clusters_gmm, clusters_kmeans, gmm, kmeans, sorted_clusters_gmm, sorted_clusters_kmeans, save_path):                                     
    income_data = data[['median_income']].values
    
    metrics = {
        'GMM': {
            'Silhouette': silhouette_score(income_data, clusters_gmm),
            'Calinski-Harabasz': calinski_harabasz_score(income_data, clusters_gmm),
            'Davies-Bouldin': davies_bouldin_score(income_data, clusters_gmm)
        },
        'KMeans': {
            'Silhouette': silhouette_score(income_data, clusters_kmeans),
            'Calinski-Harabasz': calinski_harabasz_score(income_data, clusters_kmeans),
            'Davies-Bouldin': davies_bouldin_score(income_data, clusters_kmeans)
        }
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    for i, cluster_id in enumerate(sorted_clusters_gmm):
        mask = clusters_gmm == cluster_id
        if i < len(INCOME_LABELS):
            label = INCOME_LABELS[i]
            color = COLORS[i]
        else:
            label = f"Küme {cluster_id}"
            color = COLORS[i % len(COLORS)]
        
        axes[0, 0].scatter(
            data.loc[mask, 'longitude'],
            data.loc[mask, 'latitude'],
            c=color, label=label, alpha=0.6, s=8
        )
    
    axes[0, 0].set_xlabel('Boylam (Longitude)', fontsize=11)
    axes[0, 0].set_ylabel('Enlem (Latitude)', fontsize=11)
    axes[0, 0].set_title('GMM - California Gelir Haritası', fontsize=14, fontweight='bold')
    axes[0, 0].legend(loc='upper right', fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(-125, -114)
    axes[0, 0].set_ylim(32, 42)
    
    for i, cluster_id in enumerate(sorted_clusters_kmeans):
        mask = clusters_kmeans == cluster_id
        if i < len(INCOME_LABELS):
            label = INCOME_LABELS[i]
            color = COLORS[i]
        else:
            label = f"Küme {cluster_id}"
            color = COLORS[i % len(COLORS)]
        
        axes[0, 1].scatter(
            data.loc[mask, 'longitude'],
            data.loc[mask, 'latitude'],
            c=color, label=label, alpha=0.6, s=8
        )
    
    axes[0, 1].set_xlabel('Boylam (Longitude)', fontsize=11)
    axes[0, 1].set_ylabel('Enlem (Latitude)', fontsize=11)
    axes[0, 1].set_title('KMeans - California Gelir Haritası', fontsize=14, fontweight='bold')
    axes[0, 1].legend(loc='upper right', fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(-125, -114)
    axes[0, 1].set_ylim(32, 42)
    
    models = ['GMM', 'KMeans']
    silhouette_values = [metrics['GMM']['Silhouette'], metrics['KMeans']['Silhouette']]
    colors_bar = ['#3498db', '#27ae60']
    
    bars1 = axes[1, 0].bar(models, silhouette_values, color=colors_bar, edgecolor='black', width=0.5)
    axes[1, 0].set_ylabel('Silhouette Score', fontsize=12)
    axes[1, 0].set_title('Silhouette Score Karşılaştırması\n(Yüksek = Daha İyi)', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylim(0, max(silhouette_values) * 1.3)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        axes[1, 0].annotate(f'{height:.4f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 5),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    x = np.arange(2)
    width = 0.35
    
    ch_gmm = metrics['GMM']['Calinski-Harabasz']
    ch_kmeans = metrics['KMeans']['Calinski-Harabasz']
    db_gmm = metrics['GMM']['Davies-Bouldin']
    db_kmeans = metrics['KMeans']['Davies-Bouldin']
    
    ax2 = axes[1, 1]
    ax2_twin = ax2.twinx()
    
    bars2 = ax2.bar(x[0] - width/2, ch_gmm, width, label='GMM (CH)', color='#3498db', edgecolor='black')
    bars3 = ax2.bar(x[0] + width/2, ch_kmeans, width, label='KMeans (CH)', color='#27ae60', edgecolor='black')
    
    bars4 = ax2_twin.bar(x[1] - width/2, db_gmm, width, label='GMM (DB)', color='#3498db', edgecolor='black', alpha=0.6)
    bars5 = ax2_twin.bar(x[1] + width/2, db_kmeans, width, label='KMeans (DB)', color='#27ae60', edgecolor='black', alpha=0.6)
    
    ax2.set_ylabel('Calinski-Harabasz (↑ iyi)', fontsize=11, color='#2c3e50')
    ax2_twin.set_ylabel('Davies-Bouldin (↓ iyi)', fontsize=11, color='#7f8c8d')
    ax2.set_title('Kümeleme Kalite Metrikleri', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Calinski-Harabasz', 'Davies-Bouldin'], fontsize=11)
    
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(['GMM', 'KMeans'], loc='upper right', fontsize=10)
    
    ax2.grid(axis='y', alpha=0.3)
    
    ax2.annotate(f'{ch_gmm:.0f}', xy=(x[0] - width/2, ch_gmm), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=9)
    ax2.annotate(f'{ch_kmeans:.0f}', xy=(x[0] + width/2, ch_kmeans), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=9)
    ax2_twin.annotate(f'{db_gmm:.3f}', xy=(x[1] - width/2, db_gmm), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=9)
    ax2_twin.annotate(f'{db_kmeans:.3f}', xy=(x[1] + width/2, db_kmeans), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    fig.suptitle('🔬 GMM vs KMeans - Kümeleme Karşılaştırması', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    plt.gcf().text(0.98, 0.02, timestamp, ha='right', va='bottom', fontsize=9, color='gray')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return metrics


if __name__ == "__main__":
    data = prepare_data()
    
    clusters_gmm, gmm, sorted_clusters_gmm = train_gmm(data, n_clusters=5)
    
    clusters_kmeans, kmeans, sorted_clusters_kmeans = train_kmeans(data, n_clusters=5)
    
    print("Grafik Oluşturuluyor")
    
    images_dir = os.path.join(os.path.dirname(__file__), 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    print('Grafik oluşturuldu')

    metrics = plot_combined_clustering_comparison(
        data, clusters_gmm, clusters_kmeans, gmm, kmeans,
        sorted_clusters_gmm, sorted_clusters_kmeans,
        os.path.join(images_dir, 'gmm_kmeans_comparison.png')
    )
    
    print(f"\n GMM Metrikleri:")
    print(f"   Silhouette Score: {metrics['GMM']['Silhouette']:.4f}")
    print(f"   Calinski-Harabasz: {metrics['GMM']['Calinski-Harabasz']:.2f}")
    print(f"   Davies-Bouldin: {metrics['GMM']['Davies-Bouldin']:.4f}")
    
    print(f"\n KMeans Metrikleri:")
    print(f"   Silhouette Score: {metrics['KMeans']['Silhouette']:.4f}")
    print(f"   Calinski-Harabasz: {metrics['KMeans']['Calinski-Harabasz']:.2f}")
    print(f"   Davies-Bouldin: {metrics['KMeans']['Davies-Bouldin']:.4f}")
    
    winner = "GMM" if metrics['GMM']['Silhouette'] > metrics['KMeans']['Silhouette'] else "KMeans"
    print(f"\n🏆 En iyi model (Silhouette Score): {winner}")
