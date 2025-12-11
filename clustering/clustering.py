import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import warnings
import sys
import os

# Uyarıları kapat
warnings.filterwarnings('ignore')

# Üst dizini path'e ekle (data.py'ye erişim için)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import dataForClustering

class IncomeClusteringModel:
    """California gelir bazlı GMM (Gaussian Mixture Model) kümeleme modeli"""
    
    # 5 gelir seviyesi için etiketler (en yüksekten en düşüğe)
    INCOME_LABELS = ['Süper Zengin', 'Zengin', 'Orta', 'Fakir', 'Çok Fakir']
    
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=10)
        self.data = None
        self.clusters = None
        self.cluster_centers = None
        self.income_by_cluster = None
        self.sorted_clusters = None  # Gelire göre sıralanmış küme indeksleri
    
    def prepare_data(self):
        """Veriyi temizle ve hazırla"""
        self.data = dataForClustering.dropna().copy()
        self.data = self.data[~self.data.isin([np.inf, -np.inf]).any(axis=1)]
        self.data = self.data.reset_index(drop=True)
        return self.data
    
    def train(self):
        """Modeli eğit"""
        if self.data is None:
            self.prepare_data()
        
        # Sadece gelir bazlı kümeleme
        income_data = self.data[['median_income']].values
        self.gmm.fit(income_data)
        self.clusters = self.gmm.predict(income_data)
        
        # Küme merkezlerini hesapla (GMM'de means_ kullanılır)
        self.cluster_centers = self.gmm.means_
        self.income_by_cluster = {i: self.cluster_centers[i][0] for i in range(self.n_clusters)}
        
        # Kümeleri gelire göre sırala (yüksekten düşüğe)
        self.sorted_clusters = sorted(
            self.income_by_cluster.keys(), 
            key=lambda x: self.income_by_cluster[x], 
            reverse=True
        )
        
        return self.clusters
    
    def get_cluster_label(self, cluster_id):
        """Küme ID'sine göre etiket döndür"""
        if self.sorted_clusters is None:
            return f"Küme {cluster_id}"
        
        rank = self.sorted_clusters.index(cluster_id)
        if rank < len(self.INCOME_LABELS):
            return self.INCOME_LABELS[rank]
        return f"Küme {cluster_id}"
    
    def get_cluster_labels(self):
        """Tüm veri noktaları için küme etiketlerini döndür"""
        labels = np.array([self.get_cluster_label(c) for c in self.clusters])
        return labels
    
    def get_cluster_stats(self):
        """Küme istatistiklerini döndür"""
        stats = {}
        for cluster_id in range(self.n_clusters):
            label = self.get_cluster_label(cluster_id)
            stats[label] = {
                'cluster_id': cluster_id,
                'count': int(np.sum(self.clusters == cluster_id)),
                'avg_income': self.income_by_cluster[cluster_id]
            }
        return stats
    
    def print_stats(self):
        """İstatistikleri yazdır"""
        stats = self.get_cluster_stats()
        print("=" * 50)
        print("GMM KÜMELEME SONUÇLARI (5 GELİR SEVİYESİ)")
        print("=" * 50)
        
        # Gelire göre sıralı yazdır
        for cluster_id in self.sorted_clusters:
            label = self.get_cluster_label(cluster_id)
            cluster_stats = stats[label]
            print(f"\n{label} Bölgeler (Küme {cluster_id}):")
            print(f"  - Toplam nokta sayısı: {cluster_stats['count']}")
            print(f"  - Ortalama gelir: ${cluster_stats['avg_income']*10000:.0f}")
        
        print("\n" + "=" * 50)


# Ana çalıştırma
if __name__ == "__main__":
    model = IncomeClusteringModel(n_clusters=5)
    model.prepare_data()
    model.train()
    model.print_stats()
