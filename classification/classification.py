import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import warnings
import sys
import os

# Uyarıları kapat
warnings.filterwarnings('ignore')

# Üst dizini path'e ekle (data.py'ye erişim için)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import dataForClassification


class OceanProximityClassifier:
    """California okyanus yakınlığı sınıflandırma modeli (Random Forest)"""
    
    def __init__(self, n_estimators=200, max_depth=8, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=10,   # Daha güçlü regularizasyon
            min_samples_leaf=4,     # Daha güçlü regularizasyon
            class_weight='balanced', # Dengesiz sınıfları dengele
            random_state=random_state,
            n_jobs=-1
        )
        self.label_encoder = LabelEncoder()
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.y_train_pred = None  # Overfitting kontrolü için
        # Lat/lon dahil - ocean proximity için gerekli
        self.feature_names = [
            'longitude', 'latitude', 'housing_median_age', 
            'total_rooms', 'total_bedrooms', 'population', 
            'households', 'median_income', 'median_house_value'
        ]
        self.target_name = 'ocean_proximity'
        self.class_names = None
    
    def prepare_data(self, test_size=0.2, use_spatial_split=False, grid_size=0.5):
        """
        Veriyi temizle ve eğitim/test setlerine ayır.
        
        Args:
            test_size: Test seti oranı (0-1 arası)
            use_spatial_split: True ise grid-based spatial split kullan (daha zor test)
            grid_size: Grid hücre boyutu (derece cinsinden, örn: 0.5 derece ≈ 55km)
        """
        self.data = dataForClassification.dropna().copy()
        self.data = self.data[~self.data.isin([np.inf, -np.inf]).any(axis=1)]
        self.data = self.data.reset_index(drop=True)
        
        # Feature ve target ayır
        X = self.data[self.feature_names].values
        y_raw = self.data[self.target_name].values
        
        # Label encoding
        y = self.label_encoder.fit_transform(y_raw)
        self.class_names = self.label_encoder.classes_
        
        if use_spatial_split:
            # Spatial (Grid-based) Train/Test Split
            # Koordinatları doğrudan veriden al (feature'lardan ayrı)
            lon = self.data['longitude'].values
            lat = self.data['latitude'].values
            
            # Grid hücre ID'lerini hesapla
            grid_x = np.floor(lon / grid_size).astype(int)
            grid_y = np.floor(lat / grid_size).astype(int)
            grid_ids = grid_x * 10000 + grid_y  # Benzersiz grid ID
            
            # Benzersiz grid hücrelerini al
            unique_grids = np.unique(grid_ids)
            np.random.seed(self.random_state)
            np.random.shuffle(unique_grids)
            
            # Grid hücrelerini test/train olarak ayır
            n_test_grids = max(1, int(len(unique_grids) * test_size))
            test_grids = set(unique_grids[:n_test_grids])
            
            # Her noktayı train veya test'e ata
            test_mask = np.isin(grid_ids, list(test_grids))
            train_mask = ~test_mask
            
            self.X_train = X[train_mask]
            self.X_test = X[test_mask]
            self.y_train = y[train_mask]
            self.y_test = y[test_mask]
            
            print(f"📍 Spatial Split: {len(unique_grids)} grid hücresi, {n_test_grids} test gridi")
            print(f"   Train: {len(self.X_train)} | Test: {len(self.X_test)}")
        else:
            # Normal random split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
        
        return self.data
    
    def train(self):
        """Modeli eğit"""
        if self.X_train is None:
            self.prepare_data()
        
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        self.y_train_pred = self.model.predict(self.X_train)  # Overfitting kontrolü
        
        return self.model
    
    def predict(self, X):
        """Yeni veriler için tahmin yap"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Olasılık tahminleri"""
        return self.model.predict_proba(X)
    
    def get_metrics(self):
        """Model performans metriklerini döndür"""
        if self.y_pred is None:
            self.train()
        
        accuracy = accuracy_score(self.y_test, self.y_pred)
        train_accuracy = accuracy_score(self.y_train, self.y_train_pred)
        precision = precision_score(self.y_test, self.y_pred, average='weighted')
        recall = recall_score(self.y_test, self.y_pred, average='weighted')
        f1 = f1_score(self.y_test, self.y_pred, average='weighted')
        
        return {
            'accuracy': accuracy,
            'train_accuracy': train_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def get_confusion_matrix(self):
        """Confusion matrix döndür"""
        # Tüm sınıflar için labels belirt (spatial split'te eksik olabilir)
        all_labels = list(range(len(self.class_names)))
        return confusion_matrix(self.y_test, self.y_pred, labels=all_labels)
    
    def get_classification_report(self):
        """Sınıflandırma raporu döndür"""
        # Test setinde bulunan sınıfları al
        unique_labels = np.unique(np.concatenate([self.y_test, self.y_pred]))
        label_names = [self.class_names[i] for i in unique_labels]
        
        return classification_report(
            self.y_test, self.y_pred, 
            labels=unique_labels,
            target_names=label_names, 
            output_dict=True,
            zero_division=0
        )
    
    def get_feature_importance(self):
        """Özellik önemlerini döndür"""
        importance = self.model.feature_importances_
        return {name: imp for name, imp in zip(self.feature_names, importance)}
    
    def get_class_distribution(self):
        """Sınıf dağılımlarını döndür"""
        train_dist = {}
        test_dist = {}
        
        for i, name in enumerate(self.class_names):
            train_dist[name] = int(np.sum(self.y_train == i))
            test_dist[name] = int(np.sum(self.y_test == i))
        
        return {'train': train_dist, 'test': test_dist}
    
    def print_stats(self):
        """İstatistikleri yazdır"""
        metrics = self.get_metrics()
        importance = self.get_feature_importance()
        cm = self.get_confusion_matrix()
        class_dist = self.get_class_distribution()
        
        print("=" * 65)
        print("🌊 RANDOM FOREST SINIFLANDIRMA SONUÇLARI")
        print("   (Ocean Proximity Tahmini)")
        print("=" * 65)
        
        print("\n📊 MODEL PARAMETRELERİ:")
        print(f"   ├─ Algoritma: Random Forest Classifier")
        print(f"   ├─ Ağaç Sayısı: {self.n_estimators}")
        print(f"   ├─ Maksimum Derinlik: {self.max_depth}")
        print(f"   └─ Eğitim/Test Oranı: {len(self.X_train)}/{len(self.X_test)}")
        
        # Overfitting Kontrolü
        overfit_diff = metrics['train_accuracy'] - metrics['accuracy']
        print("\n🔍 OVERFITTİNG KONTROLÜ:")
        print(f"   ├─ Train Accuracy: {metrics['train_accuracy']:.4f} ({metrics['train_accuracy']*100:.1f}%)")
        print(f"   ├─ Test Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
        print(f"   └─ Fark:           {overfit_diff:.4f} ({overfit_diff*100:.1f}%)")
        if overfit_diff > 0.05:
            print("   ⚠️  UYARI: Train-Test farkı >5%, overfitting olabilir!")
        else:
            print("   ✅ Model dengeli görünüyor (fark ≤5%)")
        
        print("\n📈 GENEL PERFORMANS METRİKLERİ:")
        print(f"   ├─ Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
        print(f"   ├─ Precision: {metrics['precision']:.4f}")
        print(f"   ├─ Recall:    {metrics['recall']:.4f}")
        print(f"   └─ F1-Score:  {metrics['f1']:.4f}")
        
        print("\n🎯 SINIF BAZLI PERFORMANS:")
        report = self.get_classification_report()
        print(f"   {'Sınıf':20} │ Precision │ Recall │ F1-Score │ Destek")
        print("   " + "─" * 60)
        for class_name in self.class_names:
            if class_name in report:
                r = report[class_name]
                print(f"   {class_name:20} │   {r['precision']:.2f}    │  {r['recall']:.2f}  │   {r['f1-score']:.2f}   │  {int(r['support'])}")
            else:
                print(f"   {class_name:20} │   N/A     │  N/A   │   N/A    │  0")
        
        print("\n🎯 ÖZELLİK ÖNEMLERİ:")
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for i, (name, imp) in enumerate(sorted_importance):
            bar = "█" * int(imp * 40)
            prefix = "   ├─" if i < len(sorted_importance) - 1 else "   └─"
            print(f"{prefix} {name:20} │ {bar} {imp:.1%}")
        
        print("\n📊 SINIF DAĞILIMI (Test Seti):")
        for class_name in self.class_names:
            count = class_dist['test'][class_name]
            pct = count / len(self.y_test) * 100
            bar = "█" * int(pct / 2)
            print(f"   {class_name:20} │ {bar} {count} ({pct:.1f}%)")
        
        print("\n📉 CONFUSION MATRIX:")
        print(f"   {'':20}", end="")
        for name in self.class_names:
            print(f" {name[:8]:>8}", end="")
        print()
        print("   " + "─" * (20 + 9 * len(self.class_names)))
        for i, name in enumerate(self.class_names):
            print(f"   {name:20}", end="")
            for j in range(len(self.class_names)):
                print(f" {cm[i][j]:>8}", end="")
            print()
        
        print("\n" + "=" * 65)


# Ana çalıştırma
if __name__ == "__main__":
    model = OceanProximityClassifier(n_estimators=100, max_depth=10)
    model.prepare_data()
    model.train()
    model.print_stats()
