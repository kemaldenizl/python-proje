import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'housing.csv')
df = pd.read_csv(csv_path)

data = df[["longitude","latitude","median_income","total_rooms","total_bedrooms","population","median_house_value"]]

def prepare_data(test_size=0.2, random_state=12):

    data2 = data.dropna().copy()
    data2 = data2[~data2.isin([np.inf, -np.inf]).any(axis=1)]
    data2 = data2.reset_index(drop=True)

    feature_names = ['longitude', 'latitude', 'median_income','total_rooms','total_bedrooms','population']
    target_name = 'median_house_value'
    
    X = data[feature_names].values
    y = data[target_name].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"✅ Veri hazırlandı!")
    print(f"   Eğitim seti: {len(X_train)} örnek")
    print(f"   Test seti: {len(X_test)} örnek")
    
    return X_train, X_test, y_train, y_test, data, feature_names

def train_model(X_train, y_train, n_estimators=100, max_depth=30):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=12,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print(f"✅ Model eğitildi!")
    print(f"   Ağaç sayısı: {n_estimators}")
    print(f"   Maksimum derinlik: {max_depth}")
    
    return model

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, data, feature_names = prepare_data()

    model = train_model(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2)
    
    print(f"\n📊 R² Skoru: {r2:.4f}")
