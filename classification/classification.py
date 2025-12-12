import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


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
    
    print(f"✅ Veri hazırlandı!")
    print(f"   Eğitim seti: {len(X_train)} örnek")
    print(f"   Test seti: {len(X_test)} örnek")
    print(f"   Sınıflar: {list(class_names)}")
    
    return X_train, X_test, y_train, y_test, data, label_encoder, feature_names, class_names

def train_model(X_train, y_train, n_estimators=100, max_depth=15):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight='balanced',
        random_state=12,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print(f"✅ Model eğitildi!")
    print(f"   Ağaç sayısı: {n_estimators}")
    print(f"   Maksimum derinlik: {max_depth}")
    
    return model

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, data, label_encoder, feature_names, class_names = prepare_data()
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"\n📊 Test Doğruluğu: {accuracy:.2%}")
