import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import warnings
import sys
import os

# Uyarıları kapat
warnings.filterwarnings('ignore')

# Üst dizini path'e ekle (data.py'ye erişim için)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import dataForClustering


def haversine_distance(coord1, coord2):
    """
    İki coğrafi koordinat arasındaki Haversine mesafesini hesapla (km cinsinden).
    coord1, coord2: (latitude, longitude) formatında
    """
    R = 6371  # Dünya'nın yarıçapı (km)
    
    lat1, lon1 = np.radians(coord1[0]), np.radians(coord1[1])
    lat2, lon2 = np.radians(coord2[0]), np.radians(coord2[1])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def haversine_distance_matrix(coords):
    """
    Koordinat matrisi için Haversine mesafe matrisi oluştur.
    coords: (n_samples, 2) şeklinde [lat, lon] array
    """
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            dist = haversine_distance(coords[i], coords[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    return dist_matrix


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
        print("==" * 25)
        
        # Gelire göre sıralı yazdır
        for cluster_id in self.sorted_clusters:
            label = self.get_cluster_label(cluster_id)
            cluster_stats = stats[label]
            print(f"\n{label} Bölgeler (Küme {cluster_id}):")
            print(f"  - Toplam nokta sayısı: {cluster_stats['count']}")
            print(f"  - Ortalama gelir: ${cluster_stats['avg_income']*10000:.0f}")
        
        print("\n" + "=" * 50)


class GeoClusteringModel:
    """Haversine distance kullanan coğrafi kümeleme modeli"""
    
    REGION_LABELS = ['Kuzey', 'Güney', 'Doğu', 'Batı', 'Merkez']
    
    def __init__(self, n_clusters=5, sample_size=2000):
        self.n_clusters = n_clusters
        self.sample_size = sample_size  # Büyük veri için örnekleme
        self.data = None
        self.coords = None
        self.clusters = None
        self.cluster_centers = None
    
    def prepare_data(self):
        """Veriyi temizle ve hazırla"""
        self.data = dataForClustering.dropna().copy()
        self.data = self.data[~self.data.isin([np.inf, -np.inf]).any(axis=1)]
        self.data = self.data.reset_index(drop=True)
        
        # Koordinatları al (lat, lon formatında)
        self.coords = self.data[['latitude', 'longitude']].values
        
        return self.data
    
    def train(self):
        """Haversine distance ile kümeleme yap"""
        if self.data is None:
            self.prepare_data()
        
        n_samples = len(self.coords)
        
        # Büyük veri setleri için örnekleme yap
        if n_samples > self.sample_size:
            print(f"📍 Büyük veri seti ({n_samples} nokta), {self.sample_size} örnekle Haversine hesaplanıyor...")
            sample_idx = np.random.choice(n_samples, self.sample_size, replace=False)
            sample_coords = self.coords[sample_idx]
            
            # Örnek için Haversine mesafe matrisi
            dist_matrix = haversine_distance_matrix(sample_coords)
            
            # AgglomerativeClustering ile kümeleme (precomputed distance)
            clustering = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                metric='precomputed',
                linkage='average'
            )
            sample_clusters = clustering.fit_predict(dist_matrix)
            
            # Küme merkezlerini hesapla
            centers = []
            for i in range(self.n_clusters):
                mask = sample_clusters == i
                if np.sum(mask) > 0:
                    center = sample_coords[mask].mean(axis=0)
                    centers.append(center)
            self.cluster_centers = np.array(centers)
            
            # Tüm noktaları en yakın merkeze ata
            self.clusters = np.zeros(n_samples, dtype=int)
            for i, coord in enumerate(self.coords):
                min_dist = float('inf')
                for j, center in enumerate(self.cluster_centers):
                    dist = haversine_distance(coord, center)
                    if dist < min_dist:
                        min_dist = dist
                        self.clusters[i] = j
        else:
            # Küçük veri seti için direkt hesaplama
            dist_matrix = haversine_distance_matrix(self.coords)
            
            clustering = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                metric='precomputed',
                linkage='average'
            )
            self.clusters = clustering.fit_predict(dist_matrix)
            
            # Küme merkezlerini hesapla
            centers = []
            for i in range(self.n_clusters):
                mask = self.clusters == i
                if np.sum(mask) > 0:
                    center = self.coords[mask].mean(axis=0)
                    centers.append(center)
            self.cluster_centers = np.array(centers)
        
        return self.clusters
    
    def get_cluster_stats(self):
        """Küme istatistiklerini döndür"""
        stats = {}
        for cluster_id in range(self.n_clusters):
            mask = self.clusters == cluster_id
            cluster_data = self.data[mask]
            
            stats[f"Bölge {cluster_id}"] = {
                'cluster_id': cluster_id,
                'count': int(np.sum(mask)),
                'avg_lat': cluster_data['latitude'].mean(),
                'avg_lon': cluster_data['longitude'].mean(),
                'avg_income': cluster_data['median_income'].mean()
            }
        return stats
    
    def print_stats(self):
        """İstatistikleri yazdır"""
        stats = self.get_cluster_stats()
        print("=" * 60)
        print("🌍 COĞRAFİ KÜMELEME SONUÇLARI (HAVERSINE DISTANCE)")
        print("=" * 60)
        
        for cluster_id in range(self.n_clusters):
            key = f"Bölge {cluster_id}"
            s = stats[key]
            print(f"\n📍 Bölge {cluster_id}:")
            print(f"  - Nokta sayısı: {s['count']}")
            print(f"  - Merkez: ({s['avg_lat']:.4f}, {s['avg_lon']:.4f})")
            print(f"  - Ortalama gelir: ${s['avg_income']*10000:.0f}")
        
        print("\n" + "=" * 60)


# Ana çalıştırma
if __name__ == "__main__":
    print("\n" + "🔢 " + "="*46)
    print("1️⃣  GELİR BAZLI KÜMELEME (GMM)")
    income_model = IncomeClusteringModel(n_clusters=5)
    income_model.prepare_data()
    income_model.train()
    income_model.print_stats()
    
    print("\n" + "🗺️  " + "="*46)
    print("2️⃣  COĞRAFİ KÜMELEME (HAVERSINE)")
    geo_model = GeoClusteringModel(n_clusters=5, sample_size=2000)
    geo_model.prepare_data()
    geo_model.train()
    geo_model.print_stats()

