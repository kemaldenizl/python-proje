import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

# Clustering modelini import et
from clustering import IncomeClusteringModel


# 5 gelir seviyesi için renkler (en zengin -> en fakir)
CLUSTER_COLORS = {
    'Süper Zengin': '#1a5f1a',  # Koyu yeşil
    'Zengin': '#4CAF50',         # Yeşil
    'Orta': '#FFC107',           # Sarı/Amber
    'Fakir': '#FF9800',          # Turuncu
    'Çok Fakir': '#f44336'       # Kırmızı
}

CLUSTER_EMOJIS = {
    'Süper Zengin': '💎',
    'Zengin': '🟢',
    'Orta': '🟡',
    'Fakir': '🟠',
    'Çok Fakir': '🔴'
}


def create_income_map(model, save_path=None):
    """California gelir haritasını oluştur ve görselleştir"""
    
    data = model.data
    clusters = model.clusters
    stats = model.get_cluster_stats()
    
    # Her nokta için renk oluştur
    colors = []
    for c in clusters:
        label = model.get_cluster_label(c)
        colors.append(CLUSTER_COLORS.get(label, 'gray'))
    
    # Harita görselleştirmesi
    plt.figure(figsize=(14, 10))
    
    # Scatter plot - California haritası üzerinde noktalar
    scatter = plt.scatter(
        data['longitude'], 
        data['latitude'], 
        c=colors, 
        alpha=0.6, 
        s=20,
        edgecolors='none'
    )
    
    # Harita başlığı ve etiketler
    plt.title('California Gelir Haritası - K-Means Kümeleme\n(5 Gelir Seviyesi)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Boylam (Longitude)', fontsize=12)
    plt.ylabel('Enlem (Latitude)', fontsize=12)
    
    # Legend oluştur
    legend_elements = []
    for cluster_id in model.sorted_clusters:
        label = model.get_cluster_label(cluster_id)
        cluster_stats = stats[label]
        legend_elements.append(
            Patch(facecolor=CLUSTER_COLORS[label], alpha=0.6, 
                  label=f"{label} (Ort. Gelir: ${cluster_stats['avg_income']*10000:,.0f})")
        )
    
    plt.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Grid ekle
    plt.grid(True, alpha=0.3)
    
    # California sınırlarını yaklaşık olarak ayarla
    plt.xlim(-125, -114)
    plt.ylim(32, 42)
    
    plt.tight_layout()
    
    # Kaydet
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'california_income_map.png')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nHarita '{save_path}' olarak kaydedildi.")
    
    # Göster
    plt.show()
    
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
    # Modeli oluştur ve eğit
    model = IncomeClusteringModel(n_clusters=5)
    model.prepare_data()
    model.train()
    
    # Detaylı istatistikleri yazdır
    print_detailed_stats(model)
    
    # Haritayı oluştur ve göster
    create_income_map(model)

