import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import sys
import os

# Uyarıları kapat
warnings.filterwarnings('ignore')

# Üst dizini path'e ekle (data.py'ye erişim için)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import dataForRegression


class HousePriceRegressionModel:
    """California ev fiyatı tahmini için Random Forest Regresyon modeli"""
    
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1  # Tüm CPU çekirdeklerini kullan
        )
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.feature_names = ['longitude', 'latitude', 'median_income']
        self.target_name = 'median_house_value'
    
    def prepare_data(self, test_size=0.2):
        """Veriyi temizle ve eğitim/test setlerine ayır"""
        self.data = dataForRegression.dropna().copy()
        self.data = self.data[~self.data.isin([np.inf, -np.inf]).any(axis=1)]
        self.data = self.data.reset_index(drop=True)
        
        # Feature ve target ayır
        X = self.data[self.feature_names].values
        y = self.data[self.target_name].values
        
        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        return self.data
    
    def train(self):
        """Modeli eğit"""
        if self.X_train is None:
            self.prepare_data()
        
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        
        return self.model
    
    def predict(self, X):
        """Yeni veriler için tahmin yap"""
        return self.model.predict(X)
    
    def get_metrics(self):
        """Model performans metriklerini döndür"""
        if self.y_pred is None:
            self.train()
        
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def get_feature_importance(self):
        """Özellik önemlerini döndür"""
        importance = self.model.feature_importances_
        return {name: imp for name, imp in zip(self.feature_names, importance)}
    
    def print_stats(self):
        """İstatistikleri yazdır"""
        metrics = self.get_metrics()
        importance = self.get_feature_importance()
        
        print("=" * 60)
        print("🏠 RANDOM FOREST REGRESYON SONUÇLARI")
        print("=" * 60)
        
        print("\n📊 MODEL PARAMETRELERİ:")
        print(f"   ├─ Ağaç Sayısı (n_estimators): {self.n_estimators}")
        print(f"   ├─ Maksimum Derinlik: {self.max_depth or 'Sınırsız'}")
        print(f"   └─ Eğitim/Test Oranı: {len(self.X_train)}/{len(self.X_test)}")
        
        print("\n📈 PERFORMANS METRİKLERİ:")
        print(f"   ├─ R² Skoru: {metrics['r2']:.4f}")
        print(f"   ├─ RMSE: ${metrics['rmse']:,.0f}")
        print(f"   ├─ MAE: ${metrics['mae']:,.0f}")
        print(f"   └─ MSE: {metrics['mse']:,.0f}")
        
        print("\n🎯 ÖZELLİK ÖNEMLERİ:")
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for i, (name, imp) in enumerate(sorted_importance):
            bar = "█" * int(imp * 50)
            prefix = "   ├─" if i < len(sorted_importance) - 1 else "   └─"
            print(f"{prefix} {name}: {bar} {imp:.2%}")
        
        print("\n📉 TAHMİN ÖRNEKLERİ (İlk 5):")
        print("   Gerçek Değer     │ Tahmin          │ Fark")
        print("   " + "─" * 50)
        for i in range(min(5, len(self.y_test))):
            actual = self.y_test[i]
            pred = self.y_pred[i]
            diff = pred - actual
            sign = "+" if diff > 0 else ""
            print(f"   ${actual:>12,.0f}    │ ${pred:>12,.0f}   │ {sign}${diff:,.0f}")
        
        print("\n" + "=" * 60)


# Ana çalıştırma
if __name__ == "__main__":
    model = HousePriceRegressionModel(n_estimators=100, max_depth=15)
    model.prepare_data()
    model.train()
    model.print_stats()
