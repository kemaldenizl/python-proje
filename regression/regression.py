"""
California Ev Fiyatı Tahmin Modeli
Random Forest Regresyon kullanarak fiyat tahmini
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
import sys
import os

# Uyarıları kapat
warnings.filterwarnings('ignore')

# Üst dizini path'e ekle (data.py'ye erişim için)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import dataForRegression


# ============================================================
# VERİ HAZIRLAMA
# ============================================================

def prepare_data(test_size=0.2, random_state=42):
    """
    Veriyi temizle ve eğitim/test setlerine ayır.
    
    Returns:
        X_train, X_test, y_train, y_test, data, feature_names
    """
    # Veriyi yükle ve temizle
    data = dataForRegression.dropna().copy()
    data = data[~data.isin([np.inf, -np.inf]).any(axis=1)]
    data = data.reset_index(drop=True)
    
    # Özellik ve hedef değişkenler
    feature_names = ['longitude', 'latitude', 'median_income']
    target_name = 'median_house_value'
    
    X = data[feature_names].values
    y = data[target_name].values
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"✅ Veri hazırlandı!")
    print(f"   Eğitim seti: {len(X_train)} örnek")
    print(f"   Test seti: {len(X_test)} örnek")
    
    return X_train, X_test, y_train, y_test, data, feature_names


# ============================================================
# MODEL EĞİTİMİ
# ============================================================

def train_model(X_train, y_train, n_estimators=100, max_depth=15):
    """
    Random Forest regresyon modelini eğit.
    
    Args:
        X_train: Eğitim özellikleri
        y_train: Eğitim hedefleri
        n_estimators: Ağaç sayısı
        max_depth: Maksimum derinlik
    
    Returns:
        Eğitilmiş model
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print(f"✅ Model eğitildi!")
    print(f"   Ağaç sayısı: {n_estimators}")
    print(f"   Maksimum derinlik: {max_depth}")
    
    return model


# ============================================================
# ANA ÇALIŞTIRMA
# ============================================================

if __name__ == "__main__":
    # 1. Veriyi hazırla
    X_train, X_test, y_train, y_test, data, feature_names = prepare_data()
    
    # 2. Modeli eğit
    model = train_model(X_train, y_train)
    
    # 3. Basit değerlendirme
    y_pred = model.predict(X_test)
    r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2)
    
    print(f"\n📊 R² Skoru: {r2:.4f}")
