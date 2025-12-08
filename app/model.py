# -*- coding: utf-8 -*-
"""
Модуль загрузки и предсказания модели для линейной регрессии
Загружает предварительно обученную модель и scaler для предсказаний
"""

import joblib
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional


class RegressionModel:
    """Обёртка для модели линейной регрессии"""
    
    def __init__(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None, config_path: Optional[str] = None):
        """
        Инициализировать модель
        
        Args:
            model_path: Путь к модели регрессии (.joblib)
            scaler_path: Путь к scaler (.joblib)
            config_path: Путь к конфигурации (.json)
        """
        base_path = Path(__file__).parent.parent / 'models'
        
        self.model_path = model_path or str(base_path / 'regression_model.joblib')
        self.scaler_path = scaler_path or str(base_path / 'regression_scaler.joblib')
        self.config_path = config_path or str(base_path / 'regression_config.json')
        
        # Загрузить модель, scaler и конфигурацию
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.feature_columns = self.config['feature_columns']
        self.numeric_features = self.config['numeric_features']
        self.categorical_features = self.config['categorical_features']
        self.target_variable = self.config['target_variable']
        
        # Загрузить информацию об энкодерах из конфигурации, если доступно
        self._init_encoders()
    
    def _init_encoders(self):
        """Инициализировать категориальные энкодеры на основе конфигурации"""
        # Примечание: это простой плейсхолдер
        # В продакшене вы захотите сохранить и загрузить реальные энкодеры
        self.categorical_mapping = {}
    
    def predict(self, age: float, trestbps: float, chol: float, oldpeak: float,
                sex: str = 'male', cp: str = 'asympt', fbs: str = 'false',
                restecg: str = 'norm', exang: str = 'false', 
                slope: str = 'flat', thal: str = 'norm') -> Dict:
        """
        Сделать предсказание
        
        Args:
            age: Возраст пациента (числовой)
            trestbps: Артериальное давление в покое (числовой)
            chol: Холестерин (числовой)
            oldpeak: ST депрессия (числовой)
            sex: Пол ('male' или 'female')
            cp: Тип боли в груди ('asympt', 'angina', 'notang', 'abnang')
            fbs: Глюкоза натощак ('true' или 'false')
            restecg: ЭКГ в покое ('norm', 'hyp', 'abn')
            exang: Стенокардия при нагрузке ('true' или 'false')
            slope: Наклон ('up', 'flat', 'down')
            thal: Тип таллассемии ('norm', 'fix', 'rev')
        
        Returns:
            Словарь с предсказанием и информацией о модели
        """
        
        # Подготовить признаки
        features = self._prepare_features(
            age, trestbps, chol, oldpeak,
            sex, cp, fbs, restecg, exang, slope, thal
        )
        
        # Масштабировать признаки - создаём DataFrame с правильными названиями колонок
        features_df = pd.DataFrame([features], columns=self.feature_columns)
        features_scaled = self.scaler.transform(features_df)
        
        # Предсказать
        prediction = self.model.predict(features_scaled)[0]
        
        return {
            'prediction': float(prediction),
            'features_dict': {
                'age': age,
                'trestbps': trestbps,
                'chol': chol,
                'oldpeak': oldpeak,
                'sex': sex,
                'cp': cp,
                'fbs': fbs,
                'restecg': restecg,
                'exang': exang,
                'slope': slope,
                'thal': thal
            }
        }
    
    def _prepare_features(self, age, trestbps, chol, oldpeak,
                         sex, cp, fbs, restecg, exang, slope, thal) -> List[float]:
        """Подготовить и закодировать признаки для ввода в модель"""
        
        # Простое категориальное кодирование на основе общих паттернов
        sex_map = {'male': 1, 'female': 0}
        cp_map = {'asympt': 3, 'angina': 1, 'notang': 2, 'abnang': 0}
        fbs_map = {'true': 1, 'false': 0}
        restecg_map = {'norm': 0, 'hyp': 2, 'abn': 1}
        exang_map = {'true': 1, 'false': 0}
        slope_map = {'up': 1, 'flat': 2, 'down': 3}
        thal_map = {'norm': 3, 'fix': 1, 'rev': 2}
        
        features = [
            age,
            trestbps,
            chol,
            oldpeak,
            sex_map.get(sex.lower(), 1),
            cp_map.get(cp.lower(), 3),
            fbs_map.get(fbs.lower(), 0),
            restecg_map.get(restecg.lower(), 0),
            exang_map.get(exang.lower(), 0),
            slope_map.get(slope.lower(), 2),
            thal_map.get(thal.lower(), 3)
        ]
        
        return features
    
    def get_model_info(self) -> Dict:
        """Получить информацию о модели и метрики"""
        return {
            'model_type': self.config['model_type'],
            'target_variable': self.target_variable,
            'intercept': self.config['intercept'],
            'coefficients': self.config['coefficients'],
            'test_r2': self.config['test_r2'],
            'test_rmse': self.config['test_rmse'],
            'test_mae': self.config['test_mae'],
            'most_important_feature': self.config['most_important_feature'],
            'feature_columns': self.feature_columns,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features
        }
    
    def get_equation_latex(self) -> str:
        """Получить уравнение регрессии в формате LaTeX"""
        intercept = self.config['intercept']
        coefficients = self.config['coefficients']
        
        # Начать уравнение
        equation = f"$y = {intercept:.2f}"
        
        # Добавить коэффициенты
        for feat, coef in coefficients.items():
            sign = "+" if coef >= 0 else ""
            equation += f" {sign} {coef:.2f} \\cdot {feat}"
        
        equation += "$"
        return equation
    
    def get_r2_score(self) -> float:
        """Получить R2 на тестовом наборе"""
        return self.config['test_r2']


# Загрузить модель при импорте модуля
def load_model():
    """Загрузить модель и вернуть экземпляр"""
    return RegressionModel()
