"""
Создание и обучение модели линейной регрессии для предсказания максимальной частоты сердечных сокращений
на основе датасета Heart Disease Cleveland
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json
from pathlib import Path

# Создание папки для моделей если её нет
Path('models').mkdir(exist_ok=True)

# Загрузка и обработка данных
def load_and_prepare_data(filepath='cleve.mod'):
    """Загружает и подготавливает данные для регрессии"""
    
    # Чтение данных, пропуская строки с комментариями (начинаются на %)
    df = pd.read_csv(filepath, sep=r'\s+', engine='python', header=None, 
                     comment='%', skip_blank_lines=True)
    
    # Названия столбцов согласно описанию
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'class', 'extra']
    df.columns = columns
    
    # Целевая переменная - максимальная частота сердечных сокращений (thalach)
    # Это хороший признак для регрессии (непрерывный, числовой)
    target_col = 'thalach'
    
    # Выбираем независимые числовые признаки для регрессии (4+)
    numeric_features = ['age', 'trestbps', 'chol', 'oldpeak']
    
    # Категориальные признаки
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    
    # Обработка пропусков - замена '?' на NaN и далее на медиану
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].median(), inplace=True)
    
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df[target_col].fillna(df[target_col].median(), inplace=True)
    
    # Обработка категориальных признаков
    label_encoders = {}
    for cat_col in categorical_features:
        le = LabelEncoder()
        df[cat_col] = df[cat_col].astype(str)
        # Обработка пропусков ('?')
        df[cat_col] = df[cat_col].replace('?', df[cat_col].mode()[0] if len(df[cat_col].mode()) > 0 else 'unknown')
        df[f'{cat_col}_encoded'] = le.fit_transform(df[cat_col])
        label_encoders[cat_col] = le
    
    # Удаление выбросов (IQR метод)
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[target_col] >= lower_bound) & (df[target_col] <= upper_bound)]
    
    # Удаление дубликатов
    df = df.drop_duplicates()
    
    # Формирование матрицы признаков
    feature_cols = numeric_features + [f'{cat}_encoded' for cat in categorical_features]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    return X, y, feature_cols, label_encoders, numeric_features, categorical_features

# Обучение модели
def train_model():
    """Обучает модель линейной регрессии"""
    
    # Загрузка и подготовка данных
    X, y, feature_cols, label_encoders, numeric_features, categorical_features = load_and_prepare_data()
    
    print(f"Размер датасета: {X.shape[0]} примеров, {X.shape[1]} признаков")
    print(f"Целевая переменная (thalach): min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # Масштабирование признаков
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Обучение модели
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Прогнозы
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Метрики на тренировочном наборе
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    # Метрики на тестовом наборе
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print(f"\n--- TRAIN SET ---")
    print(f"R2 Score: {train_r2:.4f}")
    print(f"RMSE: {train_rmse:.4f}")
    print(f"MAE: {train_mae:.4f}")
    
    print(f"\n--- TEST SET ---")
    print(f"R2 Score: {test_r2:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"MAE: {test_mae:.4f}")
    
    # Коэффициенты модели
    print(f"\n--- MODEL COEFFICIENTS ---")
    print(f"Intercept: {model.intercept_:.4f}")
    for feat, coef in zip(feature_cols, model.coef_):
        print(f"{feat}: {coef:.4f}")
    
    # Найти наиболее важный признак (по абсолютному значению коэффициента)
    feature_importance = np.abs(model.coef_)
    most_important_idx = np.argmax(feature_importance)
    most_important_feature = feature_cols[most_important_idx]
    
    print(f"\nMost important feature: {most_important_feature} (|coef|={feature_importance[most_important_idx]:.4f})")
    
    # Сохранение моделей и конфигурации
    joblib.dump(model, 'models/regression_model.joblib')
    joblib.dump(scaler, 'models/regression_scaler.joblib')
    
    # Сохранение конфигурации
    config = {
        'feature_columns': feature_cols,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'target_variable': 'thalach',
        'model_type': 'LinearRegression',
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'train_rmse': float(train_rmse),
        'test_rmse': float(test_rmse),
        'train_mae': float(train_mae),
        'test_mae': float(test_mae),
        'intercept': float(model.intercept_),
        'coefficients': {feat: float(coef) for feat, coef in zip(feature_cols, model.coef_)},
        'most_important_feature': most_important_feature,
        'train_size': int(X_train.shape[0]),
        'test_size': int(X_test.shape[0]),
        'feature_means': [float(x) for x in scaler.mean_],
        'feature_stds': [float(x) for x in scaler.scale_]
    }
    
    with open('models/regression_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print("\n✅ Model saved to models/regression_model.joblib")
    print("✅ Scaler saved to models/regression_scaler.joblib")
    print("✅ Config saved to models/regression_config.json")
    
    return model, scaler, config, X, y, feature_cols, X_test, y_test, y_test_pred

if __name__ == '__main__':
    train_model()
