# -*- coding: utf-8 -*-
"""
Веб-приложение Gradio для предсказания линейной регрессии
Датасет болезней сердца - предсказание максимального пульса (thalach)
"""

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import json
from app.model import load_model

# Загрузить модель при запуске
model = load_model()

# Загрузить данные обучения для диаграммы рассеяния
def load_training_data():
    """Загрузить и подготовить данные обучения для визуализации"""
    df = pd.read_csv('cleve.mod', sep=r'\s+', engine='python', header=None, 
                     comment='%', skip_blank_lines=True)
    
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'class', 'extra']
    df.columns = columns
    
    # Преобразовать в числовой формат
    for col in ['age', 'trestbps', 'chol', 'oldpeak', 'thalach']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
    
    return df

# Загрузить данные
training_data = load_training_data()

# Получить информацию о модели для отображения
model_info = model.get_model_info()
r2_score = model.get_r2_score()
equation_latex = model.get_equation_latex()


def predict_and_plot(age, trestbps, chol, oldpeak,
                    sex, cp, fbs, restecg, exang, slope, thal):
    """
    Сделать предсказание и создать диаграмму рассеяния
    
    Returns:
        prediction: float - предсказанный пульс
        plot: matplotlib figure - диаграмма рассеяния
        metrics_text: str - метрики и уравнение модели
    """
    
    # Сделать предсказание
    result = model.predict(age, trestbps, chol, oldpeak,
                          sex, cp, fbs, restecg, exang, slope, thal)
    prediction = result['prediction']
    
    # Создать диаграмму рассеяния
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Самый важный признак - 'age'
    most_important_feature = 'age'
    target_variable = 'thalach'
    
    # Построить данные обучения
    ax.scatter(training_data[most_important_feature], 
              training_data[target_variable],
              alpha=0.5, s=50, label='Данные обучения', color='blue')
    
    # Построить точку пользователя
    ax.scatter([age], [prediction], 
              color='red', s=200, marker='*', 
              label='Наше предсказание', zorder=5, edgecolors='darkred', linewidths=2)
    
    # Добавить линию регрессии
    x_range = np.linspace(training_data[most_important_feature].min(), 
                         training_data[most_important_feature].max(), 100)
    
    # Создать вектор признаков для линии регрессии
    y_line = []
    for x_val in x_range:
        # Использовать медианные значения для других признаков
        test_input = model._prepare_features(
            age=x_val,
            trestbps=training_data['trestbps'].median(),
            chol=training_data['chol'].median(),
            oldpeak=training_data['oldpeak'].median(),
            sex='male',
            cp='asympt',
            fbs='false',
            restecg='norm',
            exang='false',
            slope='flat',
            thal='norm'
        )
        test_input_scaled = model.scaler.transform([test_input])
        y_val = model.model.predict(test_input_scaled)[0]
        y_line.append(y_val)
    
    ax.plot(x_range, y_line, color='green', linewidth=2, label='Линия регрессии')
    
    # Форматирование
    ax.set_xlabel(f'{most_important_feature} (возраст)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{target_variable} (макс. пульс)', fontsize=12, fontweight='bold')
    ax.set_title('Линейная регрессия: Данные обучения + Наше предсказание', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Создать текст метрик
    metrics_text = f"""
    ## Информация о модели
    
    **Тип модели:** {model_info['model_type']}
    
    **R² (Тестовый набор):** {r2_score:.4f}
    
    ### Уравнение регрессии
    {equation_latex}
    
    ### Ваше предсказание
    - **Предсказанный максимальный пульс (thalach):** {prediction:.2f} уд/мин
    
    ### Производительность модели
    - Тестовая RMSE: {model_info['test_rmse']:.4f}
    - Тестовая MAE: {model_info['test_mae']:.4f}
    - Самый важный признак: {model_info['most_important_feature']}
    """
    
    return prediction, fig, metrics_text


# Создать интерфейс Gradio
def create_interface():
    """Создать и настроить интерфейс Gradio"""
    
    interface = gr.Interface(
        fn=predict_and_plot,
        inputs=[
            gr.Number(label="Возраст (лет)", value=55, minimum=20, maximum=80),
            gr.Number(label="Артериальное давление в покое (мм рт. ст.)", value=120, minimum=80, maximum=200),
            gr.Number(label="Холестерин (мг/дл)", value=200, minimum=100, maximum=400),
            gr.Number(label="ST депрессия", value=1.5, minimum=0, maximum=10),
            gr.Dropdown(label="Пол", choices=["male", "female"], value="male"),
            gr.Dropdown(label="Тип боли в груди", 
                       choices=["asympt", "angina", "notang", "abnang"], 
                       value="asympt"),
            gr.Dropdown(label="Глюкоза натощак > 120", 
                       choices=["true", "false"], 
                       value="false"),
            gr.Dropdown(label="ЭКГ в покое", 
                       choices=["norm", "hyp", "abn"], 
                       value="norm"),
            gr.Dropdown(label="Стенокардия при нагрузке", 
                       choices=["true", "false"], 
                       value="false"),
            gr.Dropdown(label="Наклон ST", 
                       choices=["up", "flat", "down"], 
                       value="flat"),
            gr.Dropdown(label="Тип таллассемии", 
                       choices=["norm", "fix", "rev"], 
                       value="norm"),
        ],
        outputs=[
            gr.Number(label="Предсказанный пульс (уд/мин)", precision=2),
            gr.Plot(label="Диаграмма: Возраст vs Максимальный пульс"),
            gr.Markdown(label="Метрики модели")
        ],
        title="Датасет болезней сердца - Предсказание максимального пульса",
        description="Предсказание максимального пульса (thalach) с использованием линейной регрессии на основе медицинских показателей пациента",
        article=f"""
        ## Об этой модели
        
        Приложение использует модель **линейной регрессии**, обученную на датасете Heart Disease Cleveland
        для предсказания максимального пульса, достигнутого при нагрузочном тесте.
        
        ### Информация о датасете
        - **Всего записей:** 301 пациент
        - **Тренировочный набор:** 240 примеров
        - **Тестовый набор:** 61 пример
        - **Целевая переменная:** thalach (максимальный пульс)
        
        ### Производительность модели
        - **R² (Тестовый набор):** {r2_score:.4f}
        - **RMSE:** {model_info['test_rmse']:.4f}
        - **MAE:** {model_info['test_mae']:.4f}
        
        ### Используемые признаки
        - **Числовые:** age, trestbps, chol, oldpeak
        - **Категориальные:** sex, cp, fbs, restecg, exang, slope, thal
        
        """,
        examples=[
            [55, 120, 200, 1.5, "male", "asympt", "false", "norm", "false", "flat", "norm"],
            [45, 110, 180, 0.8, "female", "angina", "false", "norm", "true", "up", "fix"],
            [65, 140, 250, 2.5, "male", "abnang", "true", "hyp", "false", "flat", "rev"],
        ]
    )
    
    return interface


if __name__ == "__main__":
    # Создать и запустить интерфейс
    interface = create_interface()
    interface.launch(server_name="127.0.0.1", server_port=7861, share=False)
