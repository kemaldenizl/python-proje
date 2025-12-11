import numpy as np
import matplotlib.pyplot as plt
import os

# Regression modelini import et
from regression import HousePriceRegressionModel


def create_prediction_plots(model, save_path=None):
    """Tahmin grafikleri oluştur"""
    
    metrics = model.get_metrics()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Gerçek vs Tahmin scatter plot
    ax1 = axes[0, 0]
    ax1.scatter(model.y_test, model.y_pred, alpha=0.3, s=10, c='#4CAF50')
    
    # Mükemmel tahmin çizgisi
    min_val = min(model.y_test.min(), model.y_pred.min())
    max_val = max(model.y_test.max(), model.y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Mükemmel Tahmin')
    
    ax1.set_xlabel('Gerçek Değer ($)', fontsize=11)
    ax1.set_ylabel('Tahmin Değeri ($)', fontsize=11)
    ax1.set_title('Gerçek vs Tahmin Değerleri', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Hata dağılımı (residuals)
    ax2 = axes[0, 1]
    residuals = model.y_pred - model.y_test
    ax2.hist(residuals, bins=50, color='#2196F3', alpha=0.7, edgecolor='white')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Tahmin Hatası ($)', fontsize=11)
    ax2.set_ylabel('Frekans', fontsize=11)
    ax2.set_title('Hata Dağılımı (Residuals)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Feature importance bar chart
    ax3 = axes[1, 0]
    importance = model.get_feature_importance()
    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    colors = ['#FF9800', '#4CAF50', '#2196F3']
    
    bars = ax3.barh(names, values, color=colors[:len(names)], alpha=0.8)
    ax3.set_xlabel('Önem Skoru', fontsize=11)
    ax3.set_title('Özellik Önemleri', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Bar üzerine değer yaz
    for bar, val in zip(bars, values):
        ax3.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.2%}', va='center', fontsize=10)
    
    # 4. Metrikler özet kutusu
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    metrics_text = f"""
    ╔══════════════════════════════════════════╗
    ║     🏠 MODEL PERFORMANS ÖZETİ            ║
    ╠══════════════════════════════════════════╣
    ║                                          ║
    ║   R² Skoru:     {metrics['r2']:.4f}                  ║
    ║                                          ║
    ║   RMSE:         ${metrics['rmse']:>12,.0f}          ║
    ║                                          ║
    ║   MAE:          ${metrics['mae']:>12,.0f}          ║
    ║                                          ║
    ║   MSE:          {metrics['mse']:>15,.0f}     ║
    ║                                          ║
    ╚══════════════════════════════════════════╝
    
    📊 Eğitim Seti: {len(model.X_train):,} örnek
    📊 Test Seti:   {len(model.X_test):,} örnek
    🌲 Ağaç Sayısı: {model.n_estimators}
    """
    
    ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8))
    
    plt.suptitle('California Ev Fiyatı Tahmini - Random Forest Regresyon', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Kaydet
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'regression_results.png')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nGrafik '{save_path}' olarak kaydedildi.")
    
    plt.show()
    
    return save_path


def create_price_map(model, save_path=None):
    """California fiyat tahmin haritası oluştur"""
    
    data = model.data.copy()
    
    # Tüm veri için tahmin yap
    X_all = data[model.feature_names].values
    predictions = model.predict(X_all)
    
    plt.figure(figsize=(14, 10))
    
    # Scatter plot - tahmin edilen fiyatlara göre renklendirme
    scatter = plt.scatter(
        data['longitude'], 
        data['latitude'], 
        c=predictions, 
        cmap='RdYlGn',  # Kırmızı (düşük) -> Yeşil (yüksek)
        alpha=0.6, 
        s=15,
        edgecolors='none'
    )
    
    # Colorbar ekle
    cbar = plt.colorbar(scatter, shrink=0.8)
    cbar.set_label('Tahmin Edilen Ev Fiyatı ($)', fontsize=11)
    
    plt.title('California Ev Fiyatı Tahmin Haritası\n(Random Forest Regresyon)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Boylam (Longitude)', fontsize=12)
    plt.ylabel('Enlem (Latitude)', fontsize=12)
    
    plt.grid(True, alpha=0.3)
    plt.xlim(-125, -114)
    plt.ylim(32, 42)
    
    plt.tight_layout()
    
    # Kaydet
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'california_price_map.png')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nHarita '{save_path}' olarak kaydedildi.")
    
    plt.show()
    
    return save_path


def print_detailed_stats(model):
    """Detaylı istatistikleri yazdır"""
    metrics = model.get_metrics()
    importance = model.get_feature_importance()
    
    print("\n" + "=" * 60)
    print("🏠 RANDOM FOREST REGRESYON - DETAYLI ANALİZ")
    print("=" * 60)
    
    print("\n📊 VERİ SETİ BİLGİLERİ:")
    print(f"   ├─ Toplam Veri: {len(model.data):,} örnek")
    print(f"   ├─ Eğitim Seti: {len(model.X_train):,} örnek (%80)")
    print(f"   └─ Test Seti: {len(model.X_test):,} örnek (%20)")
    
    print("\n🌲 MODEL PARAMETRELERİ:")
    print(f"   ├─ Algoritma: Random Forest Regressor")
    print(f"   ├─ Ağaç Sayısı: {model.n_estimators}")
    print(f"   └─ Maks Derinlik: {model.max_depth or 'Sınırsız'}")
    
    print("\n📈 PERFORMANS METRİKLERİ:")
    print(f"   ├─ R² Skoru: {metrics['r2']:.4f} ({metrics['r2']*100:.1f}% varyans açıklanıyor)")
    print(f"   ├─ RMSE: ${metrics['rmse']:,.0f}")
    print(f"   ├─ MAE: ${metrics['mae']:,.0f}")
    print(f"   └─ MSE: {metrics['mse']:,.0f}")
    
    print("\n🎯 ÖZELLİK ÖNEMLERİ:")
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for i, (name, imp) in enumerate(sorted_importance):
        bar = "█" * int(imp * 40)
        prefix = "   ├─" if i < len(sorted_importance) - 1 else "   └─"
        print(f"{prefix} {name:15} │ {bar} {imp:.1%}")
    
    print("\n" + "=" * 60)


# Ana çalıştırma
if __name__ == "__main__":
    # Modeli oluştur ve eğit
    model = HousePriceRegressionModel(n_estimators=100, max_depth=15)
    model.prepare_data()
    model.train()
    
    # Detaylı istatistikleri yazdır
    print_detailed_stats(model)
    
    # Grafikleri oluştur
    create_prediction_plots(model)
    
    # Fiyat haritasını oluştur
    create_price_map(model)
