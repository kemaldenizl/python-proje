import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime

csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'housing.csv')
df = pd.read_csv(csv_path)

data = df[["longitude","latitude","median_income","total_rooms","total_bedrooms","population","median_house_value"]]

def prepare_data(test_size=0.2, random_state=12):
    data2 = data.dropna().copy()
    data2 = data2[~data2.isin([np.inf, -np.inf]).any(axis=1)]
    data2 = data2.reset_index(drop=True)

    feature_names = ['longitude', 'latitude', 'median_income','total_rooms','total_bedrooms','population']
    target_name = 'median_house_value'
    
    X = data2[feature_names].values
    y = data2[target_name].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, y_train_scaled, data2, feature_names, scaler_X, scaler_y


def train_svr_model(X_train_scaled, y_train_scaled, kernel='rbf', C=1.0, epsilon=0.1):
    model = SVR(
        kernel=kernel,
        C=C,
        epsilon=epsilon,
        gamma='scale'
    )
    
    model.fit(X_train_scaled, y_train_scaled)
    
    return model


def train_rf_model(X_train, y_train, n_estimators=100, max_depth=30):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=12,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    return model


def plot_combined_regression_comparison(y_test, y_pred_svr, y_pred_rf, save_path):
    metrics = {
        'SVR': {
            'R²': r2_score(y_test, y_pred_svr),
            'MAE': mean_absolute_error(y_test, y_pred_svr),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_svr))
        },
        'Random Forest': {
            'R²': r2_score(y_test, y_pred_rf),
            'MAE': mean_absolute_error(y_test, y_pred_rf),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf))
        }
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    axes[0, 0].scatter(y_test, y_pred_svr, alpha=0.5, color='#3498db', s=10)
    min_val = min(y_test.min(), y_pred_svr.min())
    max_val = max(y_test.max(), y_pred_svr.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='İdeal')
    axes[0, 0].set_xlabel('Gerçek Değer', fontsize=12)
    axes[0, 0].set_ylabel('Tahmin Değer', fontsize=12)
    axes[0, 0].set_title(f'SVR - Gerçek vs Tahmin\n(R² = {metrics["SVR"]["R²"]:.4f})', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].scatter(y_test, y_pred_rf, alpha=0.5, color='#27ae60', s=10)
    min_val = min(y_test.min(), y_pred_rf.min())
    max_val = max(y_test.max(), y_pred_rf.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='İdeal')
    axes[0, 1].set_xlabel('Gerçek Değer', fontsize=12)
    axes[0, 1].set_ylabel('Tahmin Değer', fontsize=12)
    axes[0, 1].set_title(f'Random Forest - Gerçek vs Tahmin\n(R² = {metrics["Random Forest"]["R²"]:.4f})', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    models = ['SVR', 'Random Forest']
    r2_values = [metrics['SVR']['R²'], metrics['Random Forest']['R²']]
    colors = ['#3498db', '#27ae60']
    
    bars1 = axes[1, 0].bar(models, r2_values, color=colors, edgecolor='black', width=0.5)
    axes[1, 0].set_ylabel('R² Score', fontsize=12)
    axes[1, 0].set_title('R² Score Karşılaştırması', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylim(0, 1.1)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        axes[1, 0].annotate(f'{height:.4f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 5),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=12, fontweight='bold')

    x = np.arange(2)
    width = 0.35
    
    mae_values = [metrics['SVR']['MAE'], metrics['Random Forest']['MAE']]
    rmse_values = [metrics['SVR']['RMSE'], metrics['Random Forest']['RMSE']]

    mae_values_k = [v/1000 for v in mae_values]
    rmse_values_k = [v/1000 for v in rmse_values]
    
    bars2 = axes[1, 1].bar(x - width/2, [mae_values_k[0], rmse_values_k[0]], width, label='SVR', color='#3498db', edgecolor='black')
    bars3 = axes[1, 1].bar(x + width/2, [mae_values_k[1], rmse_values_k[1]], width, label='Random Forest', color='#27ae60', edgecolor='black')
    
    axes[1, 1].set_ylabel('Hata (bin $)', fontsize=12)
    axes[1, 1].set_title('Hata Metriklerinin Karşılaştırması', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(['MAE', 'RMSE'], fontsize=12)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(axis='y', alpha=0.3)

    for bar in bars2:
        height = bar.get_height()
        axes[1, 1].annotate(f'{height:.1f}K',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)
    
    for bar in bars3:
        height = bar.get_height()
        axes[1, 1].annotate(f'{height:.1f}K',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)
    
    fig.suptitle('🔬 SVR vs Random Forest Regressor - Model Karşılaştırması', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    plt.gcf().text(0.98, 0.02, timestamp, ha='right', va='bottom', fontsize=9, color='gray')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Birleşik Karşılaştırma Grafiği kaydedildi: {save_path}")
    
    return metrics


if __name__ == "__main__":
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, y_train_scaled, data2, feature_names, scaler_X, scaler_y = prepare_data()

    svr_model = train_svr_model(X_train_scaled, y_train_scaled)
    y_pred_svr_scaled = svr_model.predict(X_test_scaled)
    y_pred_svr = scaler_y.inverse_transform(y_pred_svr_scaled.reshape(-1, 1)).ravel()
    
    rf_model = train_rf_model(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    svr_r2 = r2_score(y_test, y_pred_svr)
    svr_mae = mean_absolute_error(y_test, y_pred_svr)
    svr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_svr))
    
    rf_r2 = r2_score(y_test, y_pred_rf)
    rf_mae = mean_absolute_error(y_test, y_pred_rf)
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    
    print(f"\n SVR Performansı:")
    print(f"   R² Score: {svr_r2:.4f}")
    print(f"   MAE: ${svr_mae:,.0f}")
    print(f"   RMSE: ${svr_rmse:,.0f}")
    
    print(f"\n Random Forest Performansı:")
    print(f"   R² Score: {rf_r2:.4f}")
    print(f"   MAE: ${rf_mae:,.0f}")
    print(f"   RMSE: ${rf_rmse:,.0f}")
    
    print("Grafik oluşturuluyor")
    
    images_dir = os.path.join(os.path.dirname(__file__), 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    print('Grafik oluştu')

    metrics = plot_combined_regression_comparison(
        y_test, y_pred_svr, y_pred_rf,
        os.path.join(images_dir, 'svr_rf_comparison.png')
    )
    
    winner = "SVR" if svr_r2 > rf_r2 else "Random Forest"
    diff = abs(svr_r2 - rf_r2)
    print(f"\n🏆 En iyi model: {winner} (R² farkı: {diff:.4f})")
