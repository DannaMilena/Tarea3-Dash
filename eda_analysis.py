"""
eda_analysis.py
----------------
Análisis exploratorio de datos (EDA) y un modelo de Machine Learning
supervisado (regresión) sobre el dataset Air Quality (UCI).

Genera un resumen impreso en consola y guarda los resultados clave en
data/eda_summary.json para que puedan citarse en el informe.
"""

import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_processing import NUMERIC_COLS, clean, load_raw

OUT_PATH = "data/eda_summary.json"


def correlation_with_co(df: pd.DataFrame) -> dict:
    corr = df[NUMERIC_COLS].corr(numeric_only=True)["CO(GT)"].drop("CO(GT)")
    return corr.round(3).sort_values(ascending=False).to_dict()


def train_regression_model(df: pd.DataFrame) -> dict:
    """
    Modelo supervisado: predice CO(GT) (concentración real de monóxido de
    carbono) a partir de las respuestas de los sensores de bajo costo.
    Esto reproduce el caso de uso real del dataset (calibración de sensores).
    """
    features = [
        "PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)",
        "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH",
    ]
    target = "CO(GT)"

    X = df[features].values
    y = df[target].values

    # Dividimos el dataset en conjunto de entrenamiento y prueba (75%/25%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    scaler = StandardScaler().fit(X_train)
    # aqui se estandarizan los datos de entrenamiento y prueba
    # al estadarizar estamos transformando los datos para que tengan media 0 y desviación estándar 1

    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)


    models = {
        "Regresión Lineal": LinearRegression(),
        #los estimadores son los arboles de decisión que se van a usar para el random forest
        #el random state es para que los resultados sean reproducibles
        #son reproducibles porque el random state es una semilla para el generador de números aleatorios
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    }

    results = {}
    for name, model in models.items():
        # Entrenamiento y evaluación del modelo

        #aqui se entrena el modelo, x es el conjunto de entrenamiento y y es el target
        model.fit(X_train_s, y_train)
        #aqui se predice sobre el conjunto de prueba
        preds = model.predict(X_test_s)
        results[name] = {
            "R2": round(float(r2_score(y_test, preds)), 3),
            "MAE": round(float(mean_absolute_error(y_test, preds)), 3),
        }
    return {"features": features, "target": target, "resultados": results}


def main():
    raw = load_raw("data/AirQualityUCI.csv")
    df, _ = clean(raw)

    summary = {
        "n_observaciones": int(len(df)),
        "rango_fechas": [
            str(df["timestamp"].min()),
            str(df["timestamp"].max()),
        ],
        "estadisticos_descriptivos": json.loads(
            df[NUMERIC_COLS].describe().round(3).to_json()
        ),
        "correlacion_con_CO_GT": correlation_with_co(df),
        "modelo_supervisado_regresion": train_regression_model(df),
    }

    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Observaciones: {summary['n_observaciones']}")
    print(f"Rango de fechas: {summary['rango_fechas']}")
    print("\nCorrelación de variables con CO(GT):")
    for var, c in summary["correlacion_con_CO_GT"].items():
        print(f"  {var:15s} {c:+.3f}")
    print("\nModelo supervisado (predicción de CO(GT) a partir de sensores):")
    for name, res in summary["modelo_supervisado_regresion"]["resultados"].items():
        #MAE = mean absolute error, R2 = coeficiente de determinación
        print(f"  {name:18s} R2={res['R2']:.3f}  MAE={res['MAE']:.3f}")
    print(f"\nResumen completo guardado en {OUT_PATH}")


if __name__ == "__main__":
    main()
