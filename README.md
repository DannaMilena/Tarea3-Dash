# Air Quality Dashboard — Tarea #3

**Curso:** Análisis de Datos y Toma de Decisiones en Computación
**Grupo:** 1IL-133 — 1er Semestre 2026
**Dataset:** [Air Quality — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/360/air+quality)
(De Vito, S. et al., 2008. DOI: 10.24432/C59K5F)

## Descripción

Dashboard interactivo construido con **Flask + Dash + Plotly** para explorar el
dataset *Air Quality* del UCI ML Repository: 9357 lecturas horarias (marzo 2004
– abril 2005) de un dispositivo multisensor de gases desplegado en una ciudad
italiana, junto con sus concentraciones de referencia certificadas
(CO, NOx, NO2, Benceno).

## Estructura del repositorio

```
.
├── app.py                  # Dashboard Flask + Dash + Plotly (frontend)
├── data_processing.py      # Carga y limpieza de datos (backend)
├── eda_analysis.py         # EDA, pruebas de normalidad y modelo de ML
├── requirements.txt
├── README.md
└── data/
    ├── AirQualityUCI.csv       # Dataset original (UCI)
    ├── AirQualityUCI_clean.csv # Dataset limpio (generado)
    └── eda_summary.json        # Resumen del EDA (generado)
```

## Instalación

```bash
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

1. **(Opcional) Regenerar el dataset limpio y el resumen del EDA:**
   ```bash
   python data_processing.py
   python eda_analysis.py
   ```
2. **Ejecutar el dashboard:**
   ```bash
   python app.py
   ```
3. Abrir el navegador en `http://127.0.0.1:8050`

## Tratamiento de datos

- Los valores faltantes están codificados en el dataset original como `-200`;
  se convierten a `NaN`.
- La variable `NMHC(GT)` tiene ~90% de datos faltantes (sensor saturado durante
  gran parte del despliegue) y se documenta aparte, sin imputar masivamente.
- El resto de variables numéricas (`CO(GT)`, `NOx(GT)`, `NO2(GT)`,
  `PT08.S1..S5`, `T`, `RH`, `AH`) tienen entre 3.9% y 18% de faltantes, que se
  imputan mediante **interpolación temporal lineal**, adecuada para una serie
  de tiempo horaria.

## Contenido del dashboard

**Gráficas:**
1. Serie de tiempo de la variable seleccionada (agregación horaria/diaria/semanal)
2. Histograma con curva normal de referencia
3. Boxplot mensual
4. Matriz de correlación (heatmap)
5. Porcentaje de valores faltantes por variable

**Controladores / callbacks :** dropdown de variable, selector de rango de
fechas, agregación de la serie de tiempo, y los callbacks de las graficas

## Autores

_(Completar con nombre y cédula de cada integrante del grupo)_
