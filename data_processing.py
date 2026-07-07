"""
data_processing.py
-------------------
Carga y limpieza del dataset "Air Quality" (UCI Machine Learning Repository,
https://archive.ics.uci.edu/dataset/360/air+quality).

Funciones principales:
    load_raw(path)        -> DataFrame crudo, tal como viene el CSV original
    clean(df)              -> DataFrame limpio (fechas parseadas, -200 -> NaN,
                               columnas vacías eliminadas, imputación de
                               valores faltantes)
    missing_summary(df)    -> tabla con el % de valores faltantes por columna
                               (antes de imputar)
"""

import numpy as np
import pandas as pd

# Columnas de interés (ground truth de referencia + sensores)
NUMERIC_COLS = [
    "CO(GT)", 
    "PT08.S1(CO)", 
    "NMHC(GT)", 
    "C6H6(GT)", 
    "PT08.S2(NMHC)",
    "NOx(GT)", 
    "PT08.S3(NOx)", 
    "NO2(GT)", 
    "PT08.S4(NO2)", 
    "PT08.S5(O3)",
    "T", 
    "RH", 
    "AH",
]

POLLUTANT_LABELS = {
    "CO(GT)": "CO de referencia (mg/m³)",
    "PT08.S1(CO)": "Sensor PT08.S1 - CO (resp. sensor)",
    "NMHC(GT)": "NMHC de referencia (µg/m³)",
    "C6H6(GT)": "Benceno C6H6 de referencia (µg/m³)",
    "PT08.S2(NMHC)": "Sensor PT08.S2 - NMHC (resp. sensor)",
    "NOx(GT)": "NOx de referencia (ppb)",
    "PT08.S3(NOx)": "Sensor PT08.S3 - NOx (resp. sensor)",
    "NO2(GT)": "NO2 de referencia (µg/m³)",
    "PT08.S4(NO2)": "Sensor PT08.S4 - NO2 (resp. sensor)",
    "PT08.S5(O3)": "Sensor PT08.S5 - O3 (resp. sensor)",
    "T": "Temperatura (°C)",
    "RH": "Humedad relativa (%)",
    "AH": "Humedad absoluta",
}


def load_raw(path: str) -> pd.DataFrame:
    """Lee el CSV original (separador ';', decimales con coma)."""
    df = pd.read_csv(path, sep=";", decimal=",")
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Limpia el dataset crudo:
      1. Elimina columnas/filas completamente vacías (artefacto del CSV original).
      2. Construye una columna datetime a partir de Date + Time.
      3. Reemplaza el código de faltante -200 por NaN.
      4. Convierte columnas numéricas a float.
      5. Imputa los faltantes por interpolación temporal lineal
         (apropiado para una serie de tiempo horaria) y, si quedan huecos
         en los extremos, por la mediana de la columna.

    Devuelve (df_limpio, df_con_nan_sin_imputar) para poder reportar el
    tratamiento de datos realizado.
    """
    df = df.copy()

    # 1. quitar columnas "Unnamed" vacías y filas totalmente vacías
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.dropna(how="all")
    df = df.dropna(subset=["Date"])

    # 2. timestamp
    # Date y Time vienen como string, pero Time tiene formato "HH.MM.SS" (puntos en vez de dos puntos)
    # Se reemplaza el punto por dos puntos para poder parsear correctamente.
    df["Time"] = df["Time"].astype(str).str.replace(".", ":", regex=False)
    df["timestamp"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )
    # Se eliminan filas con timestamp inválido (poco probable, pero por si acaso)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # 3. -200 -> NaN (código de dato faltante documentado por UCI)
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] == -200, col] = np.nan

    df_with_nan = df.copy()  # versión sin imputar, para reportar % de faltantes

    # 4. NMHC(GT) tiene >90% de datos faltantes -> se descarta
    #    de la imputación masiva y se documenta aparte; el resto se imputa.
    cols_to_impute = [c for c in NUMERIC_COLS if c != "NMHC(GT)"]


    # 5. Imputación de faltantes por interpolación temporal lineal

    # Se hace un set_index temporal para poder interpolar por tiempo
    # si no se hace así, la interpolación es por índice (no por tiempo) y no es correcta.
    df_clean = df.set_index("timestamp")
    df_clean[cols_to_impute] = df_clean[cols_to_impute].interpolate(
        method="time", limit_direction="both"
    )
    # cualquier hueco remanente (poco probable) se llena con la mediana
    df_clean[cols_to_impute] = df_clean[cols_to_impute].fillna(
        df_clean[cols_to_impute].median()
    )

    # Se resetea el índice para que timestamp vuelva a ser columna normal
    df_clean = df_clean.reset_index()

    # variables auxiliares de calendario, útiles para el EDA / dashboard
    df_clean["mes"] = df_clean["timestamp"].dt.month
    df_clean["hora"] = df_clean["timestamp"].dt.hour
    df_clean["dia_semana"] = df_clean["timestamp"].dt.day_name()

    return df_clean, df_with_nan
# el df clean tiene 


def missing_summary(df_with_nan: pd.DataFrame) -> pd.DataFrame:
    """% de valores faltantes (código -200) por columna numérica."""
    pct = df_with_nan[NUMERIC_COLS].isna().mean().mul(100).round(2)
    out = pct.reset_index()
    out.columns = ["variable", "pct_faltante"]
    return out.sort_values("pct_faltante", ascending=False)


if __name__ == "__main__":
    raw = load_raw("data/AirQualityUCI.csv")
    clean_df, raw_nan_df = clean(raw)
    print("Filas crudo:", len(raw))
    print("Filas limpio:", len(clean_df))
    print("\nResumen de faltantes (%):")
    print(missing_summary(raw_nan_df).to_string(index=False))
    clean_df.to_csv("data/AirQualityUCI_clean.csv", index=False)
    print("\nGuardado data/AirQualityUCI_clean.csv")
