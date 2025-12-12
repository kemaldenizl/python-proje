"""
California Kümeleme - Grafik ve İstatistikler
Tüm görselleştirme ve metrik hesaplamaları bu dosyada
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
from datetime import datetime

# Clustering modülünden fonksiyonları import et
from clustering import prepare_data, train_gmm, train_geo, INCOME_LABELS, get_income_label


# Renk paletleri
CLUSTER_COLORS = {
    'Süper Zengin': '#1a5f1a',
    'Zengin': '#4CAF50',
    'Orta': '#FFC107',
    'Fakir': '#FF9800',
    'Çok Fakir': '#f44336'
}

GEO_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

CLUSTER_EMOJIS = {
    'Süper Zengin': '💎',
    'Zengin': '🟢',
    'Orta': '🟡',
    'Fakir': '🟠',
    'Çok Fakir': '🔴'
}


# ============================================================
# İSTATİSTİK YAZDIRMA
# ============================================================

def print_gmm_stats(data, clusters, gmm, sorted_clusters):
    """GMM kümeleme istatistiklerini yazdır"""
    
    print("\n" + "=" * 60)
    print("📊 DETAYLI KÜMELEME ANALİZİ (5 GELİR SEVİYESİ)")
    print("=" * 60)
    
    total = len(clusters)
    
    for cluster_id in sorted_clusters:
        label = get_income_label(cluster_id, sorted_clusters)
        count = int(np.sum(clusters == cluster_id))
        avg_income = gmm.means_[cluster_id][0] * 10000
        pct = (count / total) * 100
        emoji = CLUSTER_EMOJIS.get(label, '⚪')
        
        print(f"\n{emoji} {label.upper()} BÖLGELER:")
        print(f"   ├─ Küme ID: {cluster_id}")
        print(f"   ├─ Bölge Sayısı: {count:,}")
        print(f"   ├─ Oran: %{pct:.1f}")
        print(f"   └─ Ortalama Gelir: ${avg_income:,.0f}")
    
    print("\n" + "=" * 60)
    print("📈 ÖZET DAĞILIM:")
    print("=" * 60)
    for cluster_id in sorted_clusters:
        label = get_income_label(cluster_id, sorted_clusters)
        count = int(np.sum(clusters == cluster_id))
        pct = (count / total) * 100
        bar = "█" * int(pct / 2)
        print(f"   {label:15} │ {bar} {pct:.1f}%")
    print("=" * 60)


def print_geo_stats(data, clusters, centers):
    """Coğrafi kümeleme istatistiklerini yazdır"""
    
    print("=" * 60)
    print("🌍 COĞRAFİ KÜMELEME SONUÇLARI (HAVERSINE DISTANCE)")
    print("=" * 60)
    
    for i in range(len(centers)):
        mask = clusters == i
        cluster_data = data[mask]
        count = int(np.sum(mask))
        avg_income = cluster_data['median_income'].mean() * 10000
        
        print(f"\n📍 Bölge {i}:")
        print(f"  - Nokta sayısı: {count}")
        print(f"  - Merkez: ({centers[i][0]:.4f}, {centers[i][1]:.4f})")
        print(f"  - Ortalama gelir: ${avg_income:.0f}")
    
    print("\n" + "=" * 60)


# ============================================================
# GRAFİK FONKSİYONLARI
# ============================================================

def create_income_map(data, clusters, gmm, sorted_clusters, save_path=None):
    """California gelir haritasını oluştur"""
    
    plt.close('all')
    
    # Her nokta için renk
    colors = []
    for c in clusters:
        label = get_income_label(c, sorted_clusters)
        colors.append(CLUSTER_COLORS.get(label, 'gray'))
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    ax.scatter(data['longitude'], data['latitude'], c=colors, alpha=0.6, s=20, edgecolors='none')
    
    ax.set_title('California Gelir Haritası - GMM Kümeleme\n(5 Gelir Seviyesi)', 
              fontsize=14, fontweight='bold')
    ax.set_xlabel('Boylam (Longitude)', fontsize=12)
    ax.set_ylabel('Enlem (Latitude)', fontsize=12)
    
    # Legend
    legend_elements = []
    for cluster_id in sorted_clusters:
        label = get_income_label(cluster_id, sorted_clusters)
        avg_income = gmm.means_[cluster_id][0] * 10000
        legend_elements.append(
            Patch(facecolor=CLUSTER_COLORS[label], alpha=0.6, 
                  label=f"{label} (Ort: ${avg_income:,.0f})")
        )
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-125, -114)
    ax.set_ylim(32, 42)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'images', 'california_income_map.png')
    
    # Sağ alt köşeye tarih/saat ekle
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    fig.text(0.98, 0.02, timestamp, ha='right', va='bottom', fontsize=9, color='gray')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✅ Gelir haritası kaydedildi: {save_path}")


def create_geo_map(data, clusters, centers, save_path=None):
    """Coğrafi kümeleme haritasını oluştur"""
    
    plt.close('all')
    
    n_clusters = len(centers)
    colors = [GEO_COLORS[c % len(GEO_COLORS)] for c in clusters]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    ax.scatter(data['longitude'], data['latitude'], c=colors, alpha=0.6, s=20, edgecolors='none')
    
    # Küme merkezlerini işaretle
    for i, center in enumerate(centers):
        ax.scatter(center[1], center[0], c='black', s=200, marker='X', 
                  edgecolors='white', linewidths=2, zorder=5)
        ax.annotate(f'Bölge {i}', (center[1], center[0]), 
                   textcoords="offset points", xytext=(10, 10),
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title('California Coğrafi Kümeleme - Haversine Distance\n(5 Bölge)', 
              fontsize=14, fontweight='bold')
    ax.set_xlabel('Boylam (Longitude)', fontsize=12)
    ax.set_ylabel('Enlem (Latitude)', fontsize=12)
    
    # Legend
    legend_elements = []
    for i in range(n_clusters):
        count = int(np.sum(clusters == i))
        avg_income = data[clusters == i]['median_income'].mean() * 10000
        legend_elements.append(
            Patch(facecolor=GEO_COLORS[i], alpha=0.6, 
                  label=f"Bölge {i} ({count:,} nokta, ${avg_income:,.0f})")
        )
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-125, -114)
    ax.set_ylim(32, 42)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'images', 'california_geo_map.png')
    
    # Sağ alt köşeye tarih/saat ekle
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    fig.text(0.98, 0.02, timestamp, ha='right', va='bottom', fontsize=9, color='gray')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✅ Coğrafi harita kaydedildi: {save_path}")


# ============================================================
# ANA ÇALIŞTIRMA
# ============================================================

if __name__ == "__main__":
    # 1. Veriyi hazırla
    data = prepare_data()
    
    # 2. GMM Kümeleme
    print("\n🔢 GELİR BAZLI KÜMELEME (GMM)")
    clusters_gmm, gmm, sorted_clusters = train_gmm(data, n_clusters=5)
    print_gmm_stats(data, clusters_gmm, gmm, sorted_clusters)
    create_income_map(data, clusters_gmm, gmm, sorted_clusters)
    
    # 3. Coğrafi Kümeleme
    print("\n🗺️  COĞRAFİ KÜMELEME (HAVERSINE)")
    clusters_geo, centers = train_geo(data, n_clusters=5)
    print_geo_stats(data, clusters_geo, centers)
    create_geo_map(data, clusters_geo, centers)
    
    print("\n✅ Tüm grafikler başarıyla oluşturuldu!")
