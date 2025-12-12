"""
California Ev Fiyatı Tahmini - Grafik ve İstatistikler
Tüm görselleştirme ve metrik hesaplamaları bu dosyada
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from datetime import datetime

# Regression modülünden fonksiyonları import et
from regression import prepare_data, train_model


# ============================================================
# METRİK HESAPLAMA
# ============================================================

def calculate_metrics(y_test, y_pred):
    """Tüm performans metriklerini hesapla"""
    mse = mean_squared_error(y_test, y_pred)
    return {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }


# ============================================================
# İSTATİSTİK YAZDIRMA
# ============================================================

def print_stats(y_train, y_test, y_pred, feature_names, model):
    """Detaylı istatistikleri yazdır"""
    
    metrics = calculate_metrics(y_test, y_pred)
    importance = dict(zip(feature_names, model.feature_importances_))
    
    print("\n" + "=" * 60)
    print("🏠 RANDOM FOREST REGRESYON - DETAYLI ANALİZ")
    print("=" * 60)
    
    print("\n📊 VERİ SETİ BİLGİLERİ:")
    print(f"   ├─ Eğitim Seti: {len(y_train)} örnek")
    print(f"   └─ Test Seti: {len(y_test)} örnek")
    
    print("\n📈 PERFORMANS METRİKLERİ:")
    print(f"   ├─ R² Skoru: {metrics['r2']:.4f} ({metrics['r2']*100:.1f}% varyans açıklanıyor)")
    print(f"   ├─ RMSE: ${metrics['rmse']:,.0f}")
    print(f"   ├─ MAE: ${metrics['mae']:,.0f}")
    print(f"   └─ MSE: {metrics['mse']:,.0f}")
    
    print("\n🎯 ÖZELLİK ÖNEMLERİ:")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for i, (name, imp) in enumerate(sorted_imp):
        prefix = "├─" if i < len(sorted_imp)-1 else "└─"
        bar = "█" * int(imp * 40)
        print(f"   {prefix} {name:15} │ {bar} {imp:.1%}")
    
    print("\n📉 TAHMİN ÖRNEKLERİ (İlk 5):")
    print("   Gerçek Değer     │ Tahmin          │ Fark")
    print("   " + "─" * 50)
    for i in range(min(5, len(y_test))):
        actual = y_test[i]
        pred = y_pred[i]
        diff = pred - actual
        sign = "+" if diff > 0 else ""
        print(f"   ${actual:>12,.0f}    │ ${pred:>12,.0f}   │ {sign}${diff:,.0f}")
    
    print("\n" + "=" * 60)


# ============================================================
# GRAFİK FONKSİYONLARI
# ============================================================

def create_prediction_plots(y_train, y_test, y_pred, feature_names, model, save_path=None):
    """Tahmin grafikleri oluştur"""
    
    metrics = calculate_metrics(y_test, y_pred)
    importance = dict(zip(feature_names, model.feature_importances_))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Gerçek vs Tahmin scatter plot
    ax1 = axes[0, 0]
    ax1.scatter(y_test, y_pred, alpha=0.3, s=10, c='#4CAF50')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Mükemmel Tahmin')
    ax1.set_xlabel('Gerçek Değer ($)', fontsize=11)
    ax1.set_ylabel('Tahmin Değeri ($)', fontsize=11)
    ax1.set_title('Gerçek vs Tahmin Değerleri', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Hata dağılımı
    ax2 = axes[0, 1]
    residuals = y_pred - y_test
    ax2.hist(residuals, bins=50, color='#2196F3', alpha=0.7, edgecolor='white')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Tahmin Hatası ($)', fontsize=11)
    ax2.set_ylabel('Frekans', fontsize=11)
    ax2.set_title('Hata Dağılımı (Residuals)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Feature importance
    ax3 = axes[1, 0]
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_imp]
    values = [x[1] for x in sorted_imp]
    ax3.barh(names, values, color=['#FF9800', '#4CAF50', '#2196F3'], alpha=0.8)
    ax3.set_xlabel('Önem Skoru', fontsize=11)
    ax3.set_title('Özellik Önemleri', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
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
    ╚══════════════════════════════════════════╝
    
    📊 Eğitim Seti: {len(y_train):,} örnek
    📊 Test Seti:   {len(y_test):,} örnek
    """
    
    ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8))
    
    plt.suptitle('California Ev Fiyatı Tahmini - Random Forest Regresyon', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Sağ alt köşeye tarih/saat ekle
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    fig.text(0.98, 0.02, timestamp, ha='right', va='bottom', fontsize=9, color='gray')
    
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'images', 'regression_results.png')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Performans grafikleri kaydedildi: {save_path}")
    plt.close()


def create_price_map(data, model, feature_names, save_path=None):
    """California fiyat tahmin haritası oluştur"""
    
    X_all = data[feature_names].values
    predictions = model.predict(X_all)
    
    plt.figure(figsize=(14, 10))
    
    scatter = plt.scatter(
        data['longitude'], data['latitude'], 
        c=predictions, cmap='RdYlGn', alpha=0.6, s=15, edgecolors='none'
    )
    
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
    
    # Sağ alt köşeye tarih/saat ekle
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    plt.gcf().text(0.98, 0.02, timestamp, ha='right', va='bottom', fontsize=9, color='gray')
    
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'images', 'california_price_map.png')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Fiyat haritası kaydedildi: {save_path}")
    plt.close()


# ============================================================
# ANA ÇALIŞTIRMA
# ============================================================

if __name__ == "__main__":
    # 1. Veriyi hazırla
    X_train, X_test, y_train, y_test, data, feature_names = prepare_data()
    
    # 2. Modeli eğit
    model = train_model(X_train, y_train)
    
    # 3. Tahminler
    y_pred = model.predict(X_test)
    
    # 4. İstatistikleri yazdır
    print_stats(y_train, y_test, y_pred, feature_names, model)
    
    # 5. Grafikleri oluştur
    print("\n📊 Grafikler oluşturuluyor...")
    create_prediction_plots(y_train, y_test, y_pred, feature_names, model)
    create_price_map(data, model, feature_names)
    
    print("\n✅ Tüm grafikler başarıyla oluşturuldu!")
