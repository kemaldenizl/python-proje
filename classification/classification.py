"""
California Ocean Proximity Sınıflandırma Modeli
Random Forest Classifier kullanarak ocean_proximity tahmini
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
import sys
import os

# Uyarıları kapat
warnings.filterwarnings('ignore')

# Üst dizini path'e ekle (data.py'ye erişim için)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import dataForClassification


# ============================================================
# VERİ HAZIRLAMA
# ============================================================

def prepare_data(test_size=0.2, random_state=42):
    """
    Veriyi temizle ve eğitim/test setlerine ayır.
    
    Returns:
        X_train, X_test, y_train, y_test, data, label_encoder, feature_names, class_names
    """
    # Veriyi yükle ve temizle
    data = dataForClassification.dropna().copy()
    data = data[~data.isin([np.inf, -np.inf]).any(axis=1)]
    data = data.reset_index(drop=True)
    
    # Özellik ve hedef değişkenler
    feature_names = [
        'longitude', 'latitude', 'housing_median_age', 
        'total_rooms', 'total_bedrooms', 'population', 
        'households', 'median_income', 'median_house_value'
    ]
    target_name = 'ocean_proximity'
    
    X = data[feature_names].values
    y_raw = data[target_name].values
    
    # Label encoding (kategorik -> sayısal)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = label_encoder.classes_
    
    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✅ Veri hazırlandı!")
    print(f"   Eğitim seti: {len(X_train)} örnek")
    print(f"   Test seti: {len(X_test)} örnek")
    print(f"   Sınıflar: {list(class_names)}")
    
    return X_train, X_test, y_train, y_test, data, label_encoder, feature_names, class_names


# ============================================================
# MODEL EĞİTİMİ
# ============================================================

def train_model(X_train, y_train, n_estimators=100, max_depth=15):
    """
    Random Forest sınıflandırma modelini eğit.
    
    Args:
        X_train: Eğitim özellikleri
        y_train: Eğitim hedefleri
        n_estimators: Ağaç sayısı
        max_depth: Maksimum derinlik
    
    Returns:
        Eğitilmiş model
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight='balanced',  # Dengesiz sınıfları dengele
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
    X_train, X_test, y_train, y_test, data, label_encoder, feature_names, class_names = prepare_data()
    
    # 2. Modeli eğit
    model = train_model(X_train, y_train)
    
    # 3. Basit doğruluk testi
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"\n📊 Test Doğruluğu: {accuracy:.2%}")
