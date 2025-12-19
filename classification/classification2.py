import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,precision_score, recall_score, f1_score)
from datetime import datetime

csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'housing.csv')
df = pd.read_csv(csv_path)

data = df[["longitude","latitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income","median_house_value","ocean_proximity"]]

def prepare_data(test_size=0.2, random_state=12):
    data2 = data.dropna().copy()
    data2 = data2[~data2.isin([np.inf, -np.inf]).any(axis=1)]
    data2 = data2.reset_index(drop=True)
    
    feature_names = [
        'longitude', 'latitude', 'housing_median_age', 
        'total_rooms', 'total_bedrooms', 'population', 
        'households', 'median_income', 'median_house_value'
    ]
    target_name = 'ocean_proximity'
    
    X = data2[feature_names].values
    y_raw = data2[target_name].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = label_encoder.classes_

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, data, label_encoder, feature_names, class_names, scaler


def train_svm_model(X_train_scaled, y_train, kernel='rbf', C=1.0, gamma='scale'):
    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        class_weight='balanced',
        random_state=12
    )
    
    model.fit(X_train_scaled, y_train)
    
    return model


def train_rf_model(X_train, y_train, n_estimators=100, max_depth=15):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight='balanced',
        random_state=12,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    return model


def plot_combined_comparison(y_test, y_pred_svm, y_pred_rf, class_names, save_path):
    metrics = {
        'SVM': {
            'Accuracy': accuracy_score(y_test, y_pred_svm),
            'Precision': precision_score(y_test, y_pred_svm, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, y_pred_svm, average='weighted', zero_division=0),
            'F1-Score': f1_score(y_test, y_pred_svm, average='weighted', zero_division=0)
        },
        'Random Forest': {
            'Accuracy': accuracy_score(y_test, y_pred_rf),
            'Precision': precision_score(y_test, y_pred_rf, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, y_pred_rf, average='weighted', zero_division=0),
            'F1-Score': f1_score(y_test, y_pred_rf, average='weighted', zero_division=0)
        }
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0, 0])
    axes[0, 0].set_title('SVM - Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Tahmin Edilen', fontsize=12)
    axes[0, 0].set_ylabel('Gerçek', fontsize=12)
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].tick_params(axis='y', rotation=0)
    
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0, 1])
    axes[0, 1].set_title('Random Forest - Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Tahmin Edilen', fontsize=12)
    axes[0, 1].set_ylabel('Gerçek', fontsize=12)
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].tick_params(axis='y', rotation=0)
    
    metric_names = list(metrics['SVM'].keys())
    svm_values = list(metrics['SVM'].values())
    rf_values = list(metrics['Random Forest'].values())
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    bars1 = axes[1, 0].bar(x - width/2, svm_values, width, label='SVM', color='#3498db', edgecolor='black')
    bars2 = axes[1, 0].bar(x + width/2, rf_values, width, label='Random Forest', color='#27ae60', edgecolor='black')
    
    axes[1, 0].set_ylabel('Skor', fontsize=12)
    axes[1, 0].set_title('Model Performans Karşılaştırması', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(metric_names, fontsize=11)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].set_ylim(0, 1.15)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        axes[1, 0].annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        axes[1, 0].annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    f1_svm_per_class = f1_score(y_test, y_pred_svm, average=None, zero_division=0)
    f1_rf_per_class = f1_score(y_test, y_pred_rf, average=None, zero_division=0)
    
    x_classes = np.arange(len(class_names))
    
    bars3 = axes[1, 1].bar(x_classes - width/2, f1_svm_per_class, width, label='SVM', color='#3498db', edgecolor='black')
    bars4 = axes[1, 1].bar(x_classes + width/2, f1_rf_per_class, width, label='Random Forest', color='#27ae60', edgecolor='black')
    
    axes[1, 1].set_ylabel('F1-Score', fontsize=12)
    axes[1, 1].set_title('Sınıf Bazlı F1-Score Karşılaştırması', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x_classes)
    axes[1, 1].set_xticklabels(class_names, fontsize=10, rotation=45, ha='right')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].set_ylim(0, 1.15)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    for bar in bars3:
        height = bar.get_height()
        axes[1, 1].annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    for bar in bars4:
        height = bar.get_height()
        axes[1, 1].annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    fig.suptitle('🔬 SVM vs Random Forest - Model Karşılaştırması', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    plt.gcf().text(0.98, 0.02, timestamp, ha='right', va='bottom', fontsize=9, color='gray')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Birleşik Karşılaştırma Grafiği kaydedildi: {save_path}")
    
    return metrics


if __name__ == "__main__":
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, data, label_encoder, feature_names, class_names, scaler = prepare_data()
    
    svm_model = train_svm_model(X_train_scaled, y_train)
    y_pred_svm = svm_model.predict(X_test_scaled)
    
    rf_model = train_rf_model(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    svm_accuracy = accuracy_score(y_test, y_pred_svm)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    
    print(f"\n SVM Test Doğruluğu: {svm_accuracy:.2%}")
    print(f" Random Forest Test Doğruluğu: {rf_accuracy:.2%}")
    
    print("SVM Classification Report:")
    print(classification_report(y_test, y_pred_svm, target_names=class_names))
    
    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred_rf, target_names=class_names))
    

    print('Grafik oluşturuluyor')
    images_dir = os.path.join(os.path.dirname(__file__), 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    print('Grafik oluşturuldu')

    metrics = plot_combined_comparison(
        y_test, y_pred_svm, y_pred_rf, class_names,
        os.path.join(images_dir, 'svm_rf_comparison.png')
    )
    
    winner = "SVM" if svm_accuracy > rf_accuracy else "Random Forest"
    diff = abs(svm_accuracy - rf_accuracy)
    print(f"\n🏆 En iyi model: {winner} (+{diff:.2%} fark)")
