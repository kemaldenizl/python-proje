import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
import os
from datetime import datetime

# Classification modelini import et
from classification import OceanProximityClassifier


# Renk paleti
COLORS = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63', '#9C27B0']


def create_confusion_matrix_plot(model, save_path=None):
    """Confusion matrix görselleştirmesi"""
    
    cm = model.get_confusion_matrix()
    class_names = model.class_names
    
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Heatmap
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.2%', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Oran'},
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
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'images', 'confusion_matrix.png')
    
    # Sağ alt köşeye tarih/saat ekle
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    plt.gcf().text(0.98, 0.02, timestamp, fontsize=9, color='gray', alpha=0.7,
                   ha='right', va='bottom', transform=plt.gcf().transFigure)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nConfusion Matrix '{save_path}' olarak kaydedildi.")
    
    plt.show()
    return save_path


def create_performance_plots(model, save_path=None):
    """Performans grafikleri oluştur"""
    
    metrics = model.get_metrics()
    report = model.get_classification_report()
    importance = model.get_feature_importance()
    class_dist = model.get_class_distribution()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Genel metrikler bar chart
    ax1 = axes[0, 0]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
    bars = ax1.bar(metric_names, metric_values, color=COLORS[:4], alpha=0.8, edgecolor='white', linewidth=2)
    
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel('Skor', fontsize=11)
    ax1.set_title('Genel Performans Metrikleri', fontsize=12, fontweight='bold')
    ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='İyi Eşik (0.8)')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Bar üzerine değer yaz
    for bar, val in zip(bars, metric_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.2%}', ha='center', fontsize=11, fontweight='bold')
    
    # 2. Sınıf bazlı F1-Score
    ax2 = axes[0, 1]
    class_f1 = [(name, report[name]['f1-score'] if name in report else 0) for name in model.class_names]
    class_f1.sort(key=lambda x: x[1], reverse=True)
    names = [x[0] for x in class_f1]
    f1_scores = [x[1] for x in class_f1]
    
    bars2 = ax2.barh(names, f1_scores, color=COLORS[:len(names)], alpha=0.8)
    ax2.set_xlim(0, 1.1)
    ax2.set_xlabel('F1-Score', fontsize=11)
    ax2.set_title('Sınıf Bazlı F1-Score', fontsize=12, fontweight='bold')
    ax2.axvline(x=0.8, color='red', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='x')
    
    for bar, val in zip(bars2, f1_scores):
        ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{val:.2%}', va='center', fontsize=10)
    
    # 3. Feature importance
    ax3 = axes[1, 0]
    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    feat_names = [x[0] for x in sorted_items]
    feat_values = [x[1] for x in sorted_items]
    
    bars3 = ax3.barh(feat_names, feat_values, color=['#FF9800', '#4CAF50', '#2196F3', '#E91E63'], alpha=0.8)
    ax3.set_xlabel('Önem Skoru', fontsize=11)
    ax3.set_title('Özellik Önemleri', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    for bar, val in zip(bars3, feat_values):
        ax3.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.1%}', va='center', fontsize=10)
    
    # 4. Sınıf dağılımı (pie chart)
    ax4 = axes[1, 1]
    test_dist = class_dist['test']
    labels = list(test_dist.keys())
    sizes = list(test_dist.values())
    
    wedges, texts, autotexts = ax4.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        colors=COLORS[:len(labels)], 
        explode=[0.02] * len(labels),
        shadow=True,
        startangle=90
    )
    ax4.set_title('Test Seti Sınıf Dağılımı', fontsize=12, fontweight='bold')
    
    # Autotexts daha okunabilir yap
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
    
    plt.suptitle('🌊 Ocean Proximity Sınıflandırma - Model Değerlendirmesi', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'images', 'classification_performance.png')
    
    # Sağ alt köşeye tarih/saat ekle
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    fig.text(0.98, 0.02, timestamp, fontsize=9, color='gray', alpha=0.7,
             ha='right', va='bottom', transform=fig.transFigure)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPerformans grafikleri '{save_path}' olarak kaydedildi.")
    
    plt.show()
    return save_path


def create_precision_recall_heatmap(model, save_path=None):
    """Precision-Recall heatmap sınıf bazlı"""
    
    report = model.get_classification_report()
    class_names = model.class_names
    
    # Veriyi hazırla
    metrics_data = []
    for name in class_names:
        if name in report:
            metrics_data.append([
                report[name]['precision'],
                report[name]['recall'],
                report[name]['f1-score']
            ])
        else:
            metrics_data.append([0, 0, 0])
    
    metrics_array = np.array(metrics_data)
    
    plt.figure(figsize=(10, 6))
    
    sns.heatmap(
        metrics_array,
        annot=True,
        fmt='.2%',
        cmap='RdYlGn',
        xticklabels=['Precision', 'Recall', 'F1-Score'],
        yticklabels=class_names,
        vmin=0, vmax=1,
        cbar_kws={'label': 'Skor'},
        annot_kws={'size': 12, 'weight': 'bold'}
    )
    
    plt.title('Sınıf Bazlı Performans Metrikleri\n(Precision, Recall, F1-Score)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Metrik', fontsize=12)
    plt.ylabel('Sınıf', fontsize=12)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'images', 'precision_recall_heatmap.png')
    
    # Sağ alt köşeye tarih/saat ekle
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    plt.gcf().text(0.98, 0.02, timestamp, fontsize=9, color='gray', alpha=0.7,
                   ha='right', va='bottom', transform=plt.gcf().transFigure)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPrecision-Recall Heatmap '{save_path}' olarak kaydedildi.")
    
    plt.show()
    return save_path


def create_geographic_classification_map(model, save_path=None):
    """Coğrafi sınıflandırma haritası"""
    
    data = model.data.copy()
    
    # Tüm veri için tahmin yap
    X_all = data[model.feature_names].values
    predictions = model.predict(X_all)
    predicted_labels = model.label_encoder.inverse_transform(predictions)
    
    # Renk haritası
    color_map = {name: COLORS[i % len(COLORS)] for i, name in enumerate(model.class_names)}
    colors = [color_map[label] for label in predicted_labels]
    
    plt.figure(figsize=(14, 10))
    
    scatter = plt.scatter(
        data['longitude'], 
        data['latitude'], 
        c=colors, 
        alpha=0.6, 
        s=15,
        edgecolors='none'
    )
    
    # Legend oluştur
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[name], alpha=0.6, label=name) 
                      for name in model.class_names]
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
    plt.gcf().text(0.98, 0.02, timestamp, fontsize=9, color='gray', alpha=0.7,
                   ha='right', va='bottom', transform=plt.gcf().transFigure)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nCoğrafi harita '{save_path}' olarak kaydedildi.")
    
    plt.show()
    return save_path


def print_detailed_stats(model):
    """Detaylı istatistikleri yazdır"""
    metrics = model.get_metrics()
    
    print("\n" + "=" * 65)
    print("🌊 OCEAN PROXIMITY SINIFLANDIRMA - DETAYLI ANALİZ")
    print("=" * 65)
    
    print("\n📊 MODEL BİLGİLERİ:")
    print(f"   ├─ Algoritma: Random Forest Classifier")
    print(f"   ├─ Toplam Veri: {len(model.data):,} örnek")
    print(f"   ├─ Eğitim Seti: {len(model.X_train):,} örnek (%80)")
    print(f"   ├─ Test Seti: {len(model.X_test):,} örnek (%20)")
    print(f"   └─ Sınıf Sayısı: {len(model.class_names)}")
    
    print("\n🎯 SINIFLAR:")
    for i, name in enumerate(model.class_names):
        print(f"   {'├─' if i < len(model.class_names)-1 else '└─'} {name}")
    
    print("\n📈 GENEL PERFORMANS:")
    print(f"   ├─ Accuracy:  {metrics['accuracy']:.2%}")
    print(f"   ├─ Precision: {metrics['precision']:.2%}")
    print(f"   ├─ Recall:    {metrics['recall']:.2%}")
    print(f"   └─ F1-Score:  {metrics['f1']:.2%}")
    
    print("\n" + "=" * 65)


# Ana çalıştırma
if __name__ == "__main__":
    # Modeli oluştur ve eğit
    model = OceanProximityClassifier(n_estimators=100, max_depth=15)
    model.prepare_data()
    model.train()
    
    # Detaylı istatistikleri yazdır
    print_detailed_stats(model)
    
    # Tüm grafikleri oluştur
    print("\n📊 Grafikler oluşturuluyor...")
    
    # 1. Confusion Matrix
    create_confusion_matrix_plot(model)
    
    # 2. Performans grafikleri
    create_performance_plots(model)
    
    # 3. Precision-Recall Heatmap
    create_precision_recall_heatmap(model)
    
    # 4. Coğrafi harita
    create_geographic_classification_map(model)
    
    print("\n✅ Tüm grafikler başarıyla oluşturuldu!")
