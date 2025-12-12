"""
California Ocean Proximity Sınıflandırma - Grafik ve İstatistikler
Tüm görselleştirme ve metrik hesaplamaları bu dosyada
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from matplotlib.patches import Patch
import os
from datetime import datetime

# Classification modülünden fonksiyonları import et
from classification import prepare_data, train_model


# Renk paleti
COLORS = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63', '#9C27B0']


# ============================================================
# METRİK HESAPLAMA
# ============================================================

def calculate_metrics(y_test, y_pred):
    """Tüm performans metriklerini hesapla"""
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }


def get_classification_report(y_test, y_pred, class_names):
    """Sınıflandırma raporu oluştur"""
    return classification_report(
        y_test, y_pred, 
        target_names=class_names, 
        output_dict=True,
        zero_division=0
    )


# ============================================================
# İSTATİSTİK YAZDIRMA
# ============================================================

def print_stats(y_test, y_pred, y_train, y_train_pred, class_names, feature_names, model):
    """Detaylı istatistikleri yazdır"""
    
    metrics = calculate_metrics(y_test, y_pred)
    train_acc = accuracy_score(y_train, y_train_pred)
    report = get_classification_report(y_test, y_pred, class_names)
    importance = dict(zip(feature_names, model.feature_importances_))
    
    print("\n" + "=" * 65)
    print("🌊 OCEAN PROXIMITY SINIFLANDIRMA - DETAYLI ANALİZ")
    print("=" * 65)
    
    print("\n📊 VERİ BİLGİLERİ:")
    print(f"   ├─ Eğitim Seti: {len(y_train)} örnek")
    print(f"   ├─ Test Seti: {len(y_test)} örnek")
    print(f"   └─ Sınıf Sayısı: {len(class_names)}")
    
    print("\n🎯 SINIFLAR:")
    for i, name in enumerate(class_names):
        prefix = "├─" if i < len(class_names)-1 else "└─"
        print(f"   {prefix} {name}")
    
    # Overfitting kontrolü
    overfit_diff = train_acc - metrics['accuracy']
    print("\n🔍 OVERFITTİNG KONTROLÜ:")
    print(f"   ├─ Train Accuracy: {train_acc:.2%}")
    print(f"   ├─ Test Accuracy:  {metrics['accuracy']:.2%}")
    print(f"   └─ Fark: {overfit_diff:.2%}")
    
    print("\n📈 GENEL PERFORMANS:")
    print(f"   ├─ Accuracy:  {metrics['accuracy']:.2%}")
    print(f"   ├─ Precision: {metrics['precision']:.2%}")
    print(f"   ├─ Recall:    {metrics['recall']:.2%}")
    print(f"   └─ F1-Score:  {metrics['f1']:.2%}")
    
    print("\n🎯 ÖZELLİK ÖNEMLERİ:")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for i, (name, imp) in enumerate(sorted_imp):
        prefix = "├─" if i < len(sorted_imp)-1 else "└─"
        bar = "█" * int(imp * 40)
        print(f"   {prefix} {name:20} │ {bar} {imp:.1%}")
    
    print("\n" + "=" * 65)


# ============================================================
# GRAFİK FONKSİYONLARI
# ============================================================

def create_confusion_matrix_plot(y_test, y_pred, class_names, save_path=None):
    """Confusion Matrix görselleştirmesi"""
    
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        cm_normalized, 
        annot=True, fmt='.2%', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        annot_kws={'size': 11}
    )
    
    # Gerçek değerleri de ekle
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j + 0.5, i + 0.7, f'({cm[i][j]})', 
                    ha='center', va='center', fontsize=9, color='gray')
    
    plt.title('Confusion Matrix\n(Oranlar ve Gerçek Değerler)', fontsize=14, fontweight='bold')
    plt.xlabel('Tahmin Edilen Sınıf', fontsize=12)
    plt.ylabel('Gerçek Sınıf', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'images', 'confusion_matrix.png')
    
    # Sağ alt köşeye tarih/saat ekle
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    plt.gcf().text(0.98, 0.02, timestamp, ha='right', va='bottom', fontsize=9, color='gray')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Confusion Matrix kaydedildi: {save_path}")
    plt.close()


def create_performance_plots(y_test, y_pred, class_names, feature_names, model, save_path=None):
    """Performans grafikleri oluştur"""
    
    metrics = calculate_metrics(y_test, y_pred)
    report = get_classification_report(y_test, y_pred, class_names)
    importance = dict(zip(feature_names, model.feature_importances_))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Genel metrikler
    ax1 = axes[0, 0]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
    bars = ax1.bar(metric_names, metric_values, color=COLORS[:4], alpha=0.8)
    ax1.set_ylim(0, 1.1)
    ax1.set_title('Genel Performans Metrikleri', fontsize=12, fontweight='bold')
    ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, metric_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.2%}', ha='center', fontweight='bold')
    
    # 2. Sınıf bazlı F1-Score
    ax2 = axes[0, 1]
    class_f1 = [(name, report[name]['f1-score']) for name in class_names if name in report]
    class_f1.sort(key=lambda x: x[1], reverse=True)
    names = [x[0] for x in class_f1]
    f1_scores = [x[1] for x in class_f1]
    ax2.barh(names, f1_scores, color=COLORS[:len(names)], alpha=0.8)
    ax2.set_xlim(0, 1.1)
    ax2.set_title('Sınıf Bazlı F1-Score', fontsize=12, fontweight='bold')
    
    # 3. Feature importance
    ax3 = axes[1, 0]
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    feat_names = [x[0] for x in sorted_imp]
    feat_values = [x[1] for x in sorted_imp]
    ax3.barh(feat_names, feat_values, color='#FF9800', alpha=0.8)
    ax3.set_title('Özellik Önemleri', fontsize=12, fontweight='bold')
    
    # 4. Sınıf dağılımı
    ax4 = axes[1, 1]
    unique, counts = np.unique(y_test, return_counts=True)
    labels = [class_names[i] for i in unique]
    ax4.pie(counts, labels=labels, autopct='%1.1f%%', colors=COLORS[:len(labels)], shadow=True)
    ax4.set_title('Test Seti Sınıf Dağılımı', fontsize=12, fontweight='bold')
    
    plt.suptitle('🌊 Ocean Proximity Sınıflandırma - Model Değerlendirmesi', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'images', 'classification_performance.png')
    
    # Sağ alt köşeye tarih/saat ekle
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    fig.text(0.98, 0.02, timestamp, ha='right', va='bottom', fontsize=9, color='gray')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Performans grafikleri kaydedildi: {save_path}")
    plt.close()


def create_geographic_map(data, model, label_encoder, feature_names, class_names, save_path=None):
    """Coğrafi sınıflandırma haritası"""
    
    X_all = data[feature_names].values
    predictions = model.predict(X_all)
    predicted_labels = label_encoder.inverse_transform(predictions)
    
    color_map = {name: COLORS[i % len(COLORS)] for i, name in enumerate(class_names)}
    colors = [color_map[label] for label in predicted_labels]
    
    plt.figure(figsize=(14, 10))
    
    plt.scatter(data['longitude'], data['latitude'], c=colors, alpha=0.6, s=15)
    
    legend_elements = [Patch(facecolor=color_map[name], alpha=0.6, label=name) 
                      for name in class_names]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.title('California Okyanus Yakınlığı Tahmin Haritası\n(Random Forest Classification)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Boylam (Longitude)', fontsize=12)
    plt.ylabel('Enlem (Latitude)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(-125, -114)
    plt.ylim(32, 42)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'images', 'california_ocean_proximity_map.png')
    
    # Sağ alt köşeye tarih/saat ekle
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    plt.gcf().text(0.98, 0.02, timestamp, ha='right', va='bottom', fontsize=9, color='gray')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Coğrafi harita kaydedildi: {save_path}")
    plt.close()


# ============================================================
# ANA ÇALIŞTIRMA
# ============================================================

if __name__ == "__main__":
    # 1. Veriyi hazırla
    X_train, X_test, y_train, y_test, data, label_encoder, feature_names, class_names = prepare_data()
    
    # 2. Modeli eğit
    model = train_model(X_train, y_train)
    
    # 3. Tahminler
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    
    # 4. İstatistikleri yazdır
    print_stats(y_test, y_pred, y_train, y_train_pred, class_names, feature_names, model)
    
    # 5. Grafikleri oluştur
    print("\n📊 Grafikler oluşturuluyor...")
    create_confusion_matrix_plot(y_test, y_pred, class_names)
    create_performance_plots(y_test, y_pred, class_names, feature_names, model)
    create_geographic_map(data, model, label_encoder, feature_names, class_names)
    
    print("\n✅ Tüm grafikler başarıyla oluşturuldu!")
