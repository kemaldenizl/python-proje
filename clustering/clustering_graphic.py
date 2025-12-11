import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
from datetime import datetime

# Clustering modelini import et
from clustering import IncomeClusteringModel, GeoClusteringModel


# 5 gelir seviyesi için renkler (en zengin -> en fakir)
CLUSTER_COLORS = {
    'Süper Zengin': '#1a5f1a',  # Koyu yeşil
    'Zengin': '#4CAF50',         # Yeşil
    'Orta': '#FFC107',           # Sarı/Amber
    'Fakir': '#FF9800',          # Turuncu
    'Çok Fakir': '#f44336'       # Kırmızı
}

GEO_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

CLUSTER_EMOJIS = {
    'Süper Zengin': '💎',
    'Zengin': '🟢',
    'Orta': '🟡',
    'Fakir': '🟠',
    'Çok Fakir': '🔴'
}


def create_income_map(model, save_path=None):
    """California gelir haritasını oluştur ve görselleştir"""
    
    # Önceki figürleri temizle
    plt.close('all')
    
    data = model.data
    clusters = model.clusters
    stats = model.get_cluster_stats()
    
    # Her nokta için renk oluştur
    colors = []
    for c in clusters:
        label = model.get_cluster_label(c)
        colors.append(CLUSTER_COLORS.get(label, 'gray'))
    
    # Harita görselleştirmesi
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Scatter plot - California haritası üzerinde noktalar
    scatter = ax.scatter(
        data['longitude'], 
        data['latitude'], 
        c=colors, 
        alpha=0.6, 
        s=20,
        edgecolors='none'
    )
    
    # Harita başlığı ve etiketler
    ax.set_title('California Gelir Haritası - GMM Kümeleme\n(5 Gelir Seviyesi)', 
              fontsize=14, fontweight='bold')
    ax.set_xlabel('Boylam (Longitude)', fontsize=12)
    ax.set_ylabel('Enlem (Latitude)', fontsize=12)
    
    # Legend oluştur
    legend_elements = []
    for cluster_id in model.sorted_clusters:
        label = model.get_cluster_label(cluster_id)
        cluster_stats = stats[label]
        legend_elements.append(
            Patch(facecolor=CLUSTER_COLORS[label], alpha=0.6, 
                  label=f"{label} (Ort. Gelir: ${cluster_stats['avg_income']*10000:,.0f})")
        )
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Grid ekle
    ax.grid(True, alpha=0.3)
    
    # California sınırlarını yaklaşık olarak ayarla
    ax.set_xlim(-125, -114)
    ax.set_ylim(32, 42)
    
    # Sağ alt köşeye tarih/saat ekle
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    ax.text(0.98, 0.02, timestamp, transform=ax.transAxes, 
            fontsize=9, color='gray', alpha=0.7,
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.tight_layout()
    
    # Kaydet
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'california_income_map.png')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)  # Belleği temizle
    print(f"\n✅ Gelir haritası '{save_path}' olarak kaydedildi.")
    
    return save_path


def create_geo_map(model, save_path=None):
    """Coğrafi kümeleme haritasını oluştur"""
    
    # Önceki figürleri temizle
    plt.close('all')
    
    data = model.data
    clusters = model.clusters
    stats = model.get_cluster_stats()
    
    # Her nokta için renk oluştur
    colors = [GEO_COLORS[c % len(GEO_COLORS)] for c in clusters]
    
    # Harita görselleştirmesi
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Scatter plot
    scatter = ax.scatter(
        data['longitude'], 
        data['latitude'], 
        c=colors, 
        alpha=0.6, 
        s=20,
        edgecolors='none'
    )
    
    # Küme merkezlerini işaretle
    if model.cluster_centers is not None:
        for i, center in enumerate(model.cluster_centers):
            ax.scatter(center[1], center[0], c='black', s=200, marker='X', 
                      edgecolors='white', linewidths=2, zorder=5)
            ax.annotate(f'Bölge {i}', (center[1], center[0]), 
                       textcoords="offset points", xytext=(10, 10),
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Başlık ve etiketler
    ax.set_title('California Coğrafi Kümeleme - Haversine Distance\n(5 Bölge)', 
              fontsize=14, fontweight='bold')
    ax.set_xlabel('Boylam (Longitude)', fontsize=12)
    ax.set_ylabel('Enlem (Latitude)', fontsize=12)
    
    # Legend oluştur
    legend_elements = []
    for i in range(model.n_clusters):
        key = f"Bölge {i}"
        s = stats[key]
        legend_elements.append(
            Patch(facecolor=GEO_COLORS[i % len(GEO_COLORS)], alpha=0.6, 
                  label=f"Bölge {i} ({s['count']:,} nokta, ${s['avg_income']*10000:,.0f})")
        )
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Grid ekle
    ax.grid(True, alpha=0.3)
    
    # California sınırları
    ax.set_xlim(-125, -114)
    ax.set_ylim(32, 42)
    
    # Sağ alt köşeye tarih/saat ekle
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    ax.text(0.98, 0.02, timestamp, transform=ax.transAxes, 
            fontsize=9, color='gray', alpha=0.7,
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.tight_layout()
    
    # Kaydet
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'california_geo_map.png')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✅ Coğrafi harita '{save_path}' olarak kaydedildi.")
    
    return save_path


def print_detailed_stats(model):
    """Detaylı istatistikleri yazdır"""
    stats = model.get_cluster_stats()
    
    print("\n" + "=" * 60)
    print("📊 DETAYLI KÜMELEME ANALİZİ (5 GELİR SEVİYESİ)")
    print("=" * 60)
    
    total = sum(s['count'] for s in stats.values())
    
    # Her küme için istatistikleri yazdır
    for cluster_id in model.sorted_clusters:
        label = model.get_cluster_label(cluster_id)
        cluster_stats = stats[label]
        emoji = CLUSTER_EMOJIS.get(label, '⚪')
        pct = (cluster_stats['count'] / total) * 100
        
        print(f"\n{emoji} {label.upper()} BÖLGELER:")
        print(f"   ├─ Küme ID: {cluster_stats['cluster_id']}")
        print(f"   ├─ Bölge Sayısı: {cluster_stats['count']:,}")
        print(f"   ├─ Oran: %{pct:.1f}")
        print(f"   └─ Ortalama Gelir: ${cluster_stats['avg_income']*10000:,.0f}")
    
    print("\n" + "=" * 60)
    print("📈 ÖZET DAĞILIM:")
    print("=" * 60)
    for cluster_id in model.sorted_clusters:
        label = model.get_cluster_label(cluster_id)
        cluster_stats = stats[label]
        pct = (cluster_stats['count'] / total) * 100
        bar = "█" * int(pct / 2)
        print(f"   {label:15} │ {bar} {pct:.1f}%")
    print("=" * 60)


# Ana çalıştırma
if __name__ == "__main__":
    # Gelir modeli
    print("\n🔢 GELİR BAZLI KÜMELEME (GMM)")
    income_model = IncomeClusteringModel(n_clusters=5)
    income_model.prepare_data()
    income_model.train()
    print_detailed_stats(income_model)
    create_income_map(income_model)
    
    # Coğrafi model
    print("\n🗺️  COĞRAFİ KÜMELEME (HAVERSINE)")
    geo_model = GeoClusteringModel(n_clusters=5, sample_size=2000)
    geo_model.prepare_data()
    geo_model.train()
    geo_model.print_stats()
    create_geo_map(geo_model)


