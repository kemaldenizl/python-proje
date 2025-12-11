"""
California Gelir ve Coğrafi Kümeleme Modeli
GMM (Gaussian Mixture Model) ve Haversine mesafe ile kümeleme
"""

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
import warnings
import sys
import os

# Uyarıları kapat
warnings.filterwarnings('ignore')

# Üst dizini path'e ekle (data.py'ye erişim için)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import dataForClustering


# ============================================================
# VERİ HAZIRLAMA
# ============================================================

def prepare_data():
    """
    Veriyi temizle ve hazırla.
    
    Returns:
        data: Temizlenmiş DataFrame
    """
    data = dataForClustering.dropna().copy()
    data = data[~data.isin([np.inf, -np.inf]).any(axis=1)]
    data = data.reset_index(drop=True)
    
    print(f"✅ Veri hazırlandı! Toplam {len(data)} kayıt")
    return data


# ============================================================
# HAVERSİNE MESAFE (Coğrafi mesafe)
# ============================================================

def haversine_distance(coord1, coord2):
    """
    İki coğrafi koordinat arasındaki mesafeyi km cinsinden hesapla.
    coord: (latitude, longitude)
    """
    R = 6371  # Dünya yarıçapı (km)
    
    lat1, lon1 = np.radians(coord1[0]), np.radians(coord1[1])
    lat2, lon2 = np.radians(coord2[0]), np.radians(coord2[1])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def haversine_distance_matrix(coords):
    """Koordinat matrisi için Haversine mesafe matrisi"""
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            dist = haversine_distance(coords[i], coords[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    return dist_matrix


# ============================================================
# GMM KÜMELEME (Gelir bazlı)
# ============================================================

def train_gmm(data, n_clusters=5):
    """
    GMM ile gelir bazlı kümeleme.
    
    Args:
        data: DataFrame
        n_clusters: Küme sayısı
    
    Returns:
        clusters: Küme etiketleri
        gmm: GMM modeli
        sorted_clusters: Gelire göre sıralı küme indeksleri
    """
    income_data = data[['median_income']].values
    
    gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=10)
    clusters = gmm.fit_predict(income_data)
    
    # Kümeleri gelire göre sırala (yüksekten düşüğe)
    cluster_incomes = {i: gmm.means_[i][0] for i in range(n_clusters)}
    sorted_clusters = sorted(cluster_incomes.keys(), key=lambda x: cluster_incomes[x], reverse=True)
    
    print(f"✅ GMM modeli eğitildi! ({n_clusters} küme)")
    
    return clusters, gmm, sorted_clusters


# ============================================================
# COĞRAFİ KÜMELEME (Haversine mesafe)
# ============================================================

def train_geo(data, n_clusters=5, sample_size=2000):
    """
    Haversine mesafe ile coğrafi kümeleme.
    
    Args:
        data: DataFrame
        n_clusters: Küme sayısı
        sample_size: Örnekleme boyutu (büyük veri için)
    
    Returns:
        clusters: Küme etiketleri
        centers: Küme merkezleri
    """
    coords = data[['latitude', 'longitude']].values
    n_samples = len(coords)
    
    # Büyük veri için örnekleme
    if n_samples > sample_size:
        print(f"📍 Büyük veri ({n_samples} nokta), {sample_size} örnekle hesaplanıyor...")
        sample_idx = np.random.choice(n_samples, sample_size, replace=False)
        sample_coords = coords[sample_idx]
        
        # Örnek için mesafe matrisi
        dist_matrix = haversine_distance_matrix(sample_coords)
        
        # Kümeleme
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, metric='precomputed', linkage='average'
        )
        sample_clusters = clustering.fit_predict(dist_matrix)
        
        # Küme merkezlerini hesapla
        centers = []
        for i in range(n_clusters):
            mask = sample_clusters == i
            if np.sum(mask) > 0:
                centers.append(sample_coords[mask].mean(axis=0))
        centers = np.array(centers)
        
        # Tüm noktaları en yakın merkeze ata
        clusters = np.zeros(n_samples, dtype=int)
        for i, coord in enumerate(coords):
            min_dist = float('inf')
            for j, center in enumerate(centers):
                dist = haversine_distance(coord, center)
                if dist < min_dist:
                    min_dist = dist
                    clusters[i] = j
    else:
        dist_matrix = haversine_distance_matrix(coords)
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, metric='precomputed', linkage='average'
        )
        clusters = clustering.fit_predict(dist_matrix)
        
        centers = []
        for i in range(n_clusters):
            mask = clusters == i
            if np.sum(mask) > 0:
                centers.append(coords[mask].mean(axis=0))
        centers = np.array(centers)
    
    print(f"✅ Coğrafi kümeleme tamamlandı! ({n_clusters} bölge)")
    
    return clusters, centers


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

# Gelir seviyesi etiketleri
INCOME_LABELS = ['Süper Zengin', 'Zengin', 'Orta', 'Fakir', 'Çok Fakir']

def get_income_label(cluster_id, sorted_clusters):
    """Küme ID'si için gelir etiketi döndür"""
    rank = sorted_clusters.index(cluster_id)
    if rank < len(INCOME_LABELS):
        return INCOME_LABELS[rank]
    return f"Küme {cluster_id}"


# ============================================================
# ANA ÇALIŞTIRMA
# ============================================================

if __name__ == "__main__":
    # 1. Veriyi hazırla
    data = prepare_data()
    
    # 2. GMM Kümeleme (Gelir bazlı)
    print("\n🔢 GELİR BAZLI KÜMELEME (GMM)")
    clusters_gmm, gmm, sorted_clusters = train_gmm(data, n_clusters=5)
    
    # Basit özet
    for i, cluster_id in enumerate(sorted_clusters):
        count = np.sum(clusters_gmm == cluster_id)
        avg_income = gmm.means_[cluster_id][0] * 10000
        print(f"   {INCOME_LABELS[i]}: {count} bölge, ${avg_income:,.0f}")
    
    # 3. Coğrafi Kümeleme
    print("\n🗺️  COĞRAFİ KÜMELEME (HAVERSINE)")
    clusters_geo, centers = train_geo(data, n_clusters=5)
    
    for i in range(5):
        count = np.sum(clusters_geo == i)
        print(f"   Bölge {i}: {count} nokta")
