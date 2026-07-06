"""
app.py
------
Dashboard de análisis del dataset Air Quality (UCI Machine Learning
Repository) construido con Flask + Dash + Plotly.

Ejecutar:
    python app.py
Luego abrir http://127.0.0.1:8050 en el navegador.

Gráficas incluidas (>= 4 requeridas):
    1. Serie de tiempo de la variable seleccionada           [reactiva]
    2. Histograma / distribución con curva normal de referencia [reactiva]
    3. Boxplot por mes de la variable seleccionada            [reactiva]
    4. Mapa de calor de correlación entre todas las variables [estático]
    5. Porcentaje de valores faltantes (-200) por variable    [estático]

Controladores / callbacks (>= 2 requeridos): se implementan 4 callbacks.
"""

from flask import Flask
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy import stats

from data_processing import NUMERIC_COLS, POLLUTANT_LABELS, clean, load_raw, missing_summary

# ----------------------------------------------------------------------
# 1. Carga y limpieza de datos (se hace una sola vez al iniciar el server)
# ----------------------------------------------------------------------
raw = load_raw("data/AirQualityUCI.csv")
df, df_with_nan = clean(raw)
missing_df = missing_summary(df_with_nan)

MESES_ES = {
    1: "Ene", 
    2: "Feb", 
    3: "Mar", 
    4: "Abr", 
    5: "May", 
    6: "Jun",
    7: "Jul", 
    8: "Ago", 
    9: "Sep", 
    10: "Oct", 
    11: "Nov", 
    12: "Dic",
}
df["mes_nombre"] = df["mes"].map(MESES_ES)

VARIABLE_OPTIONS = [
    {"label": POLLUTANT_LABELS[c], 
     "value": c}
    for c in [
        "CO(GT)", 
        "NOx(GT)", 
        "NO2(GT)", 
        "C6H6(GT)",
        "T", 
        "RH", 
        "AH"]
]

# ----------------------------------------------------------------------
# 2. App Flask + Dash
# ----------------------------------------------------------------------
server = Flask(__name__)
app = dash.Dash(__name__, server=server, title="Air Quality Dashboard")

CARD_STYLE = {
    "backgroundColor": "#ffffff",
    "borderRadius": "10px",
    "padding": "16px",
    "boxShadow": "0 1px 4px rgba(0,0,0,0.12)",
    "marginBottom": "20px",
}

app.layout = html.Div(
    style={"backgroundColor": "#d487f06a", "fontFamily": "Segoe UI, Arial, sans-serif",
           "padding": "24px"},
    children=[
        html.H1("Air Quality – Dashboard de Análisis Exploratorio",
                style={"marginBottom": "4px"}),
        html.P(
            "Dataset: UCI Machine Learning Repository — Air Quality "
            "(De Vito et al., 2008). Sensores de bajo costo desplegados en una "
            "ciudad italiana, marzo 2004 – abril 2005.",
            style={"color": "#555", "marginTop": "0"},
        ),

        # ---- Controles ----
        html.Div(
            style={**CARD_STYLE, "display": "flex", "gap": "32px", "flexWrap": "wrap"},
            children=[
                html.Div([
                    html.Label("Variable de análisis", style={"fontWeight": "bold"}),
                    dcc.Dropdown(
                        id="variable-dropdown",
                        options=VARIABLE_OPTIONS,
                        value="CO(GT)",
                        clearable=False,
                        style={"width": "320px"},
                    ),
                ]),
                html.Div([
                    html.Label("Rango de fechas", style={"fontWeight": "bold"}),
                    dcc.DatePickerRange(
                        id="date-range",
                        min_date_allowed=df["timestamp"].min().date(),
                        max_date_allowed=df["timestamp"].max().date(),
                        start_date=df["timestamp"].min().date(),
                        end_date=df["timestamp"].max().date(),
                        display_format="DD/MM/YYYY",
                    ),
                ]),
                html.Div([
                    html.Label("Agregación serie de tiempo", style={"fontWeight": "bold"}),
                    dcc.RadioItems(
                        id="agg-radio",
                        options=[
                            {"label": " Horaria", "value": "H"},
                            {"label": " Diaria (promedio)", "value": "D"},
                            {"label": " Semanal (promedio)", "value": "W"},
                        ],
                        value="D",
                        labelStyle={"display": "inline-block", "marginRight": "12px"},
                    ),
                ]),
            ],
        ),

        # ---- KPIs ----
        html.Div(id="kpi-row", style={"display": "flex", "gap": "16px",
                                       "flexWrap": "wrap", "marginBottom": "8px"}),

        # ---- Fila 1: serie de tiempo ----
        html.Div(style=CARD_STYLE, children=[dcc.Graph(id="timeseries-graph")]),

        # ---- Fila 2: histograma + boxplot mensual ----
        html.Div(
            style={"display": "flex", "gap": "20px", "flexWrap": "wrap"},
            children=[
                html.Div(style={**CARD_STYLE, "flex": "1", "minWidth": "420px"},
                         children=[dcc.Graph(id="hist-graph")]),
                html.Div(style={**CARD_STYLE, "flex": "1", "minWidth": "420px"},
                         children=[dcc.Graph(id="box-month-graph")]),
            ],
        ),

        # ---- Fila 3: Correlación ----
        html.Div(
            style={"display": "flex", "gap": "20px", "flexWrap": "wrap"},
            children=[
                html.Div(style={**CARD_STYLE, "flex": "1", "minWidth": "420px"},
                         children=[dcc.Graph(id="corr-heatmap", figure=px.imshow(
                             df[NUMERIC_COLS].corr(numeric_only=True).round(2),
                             text_auto=True, color_continuous_scale="RdBu_r",
                             title="Matriz de correlación entre variables",
                             aspect="auto"))]),
            ],
        ),

        # ---- Fila 4: faltantes ----
        html.Div(style=CARD_STYLE, children=[
            dcc.Graph(
                id="missing-graph",
                figure=px.bar(
                    missing_df, x="variable", y="pct_faltante",
                    title="Porcentaje de valores faltantes (código -200) por variable",
                    labels={"pct_faltante": "% faltante", "variable": "Variable"},
                    color="pct_faltante", color_continuous_scale="Oranges",
                ),
            )
        ]),

        html.Footer(
            "Tarea #3 – Dashboards en Python | Análisis de Datos y Toma de "
            "Decisiones en Computación | Grupo 1IL-133",
            style={"textAlign": "center", "color": "#888", "marginTop": "12px"},
        ),
    ],
)


# ----------------------------------------------------------------------
# 3. Callbacks (controladores reactivos)
# ----------------------------------------------------------------------

def filter_df(start_date, end_date):
    mask = (df["timestamp"] >= pd.to_datetime(start_date)) & \
           (df["timestamp"] <= pd.to_datetime(end_date) + pd.Timedelta(days=1))
    return df.loc[mask]


@app.callback(
    Output("kpi-row", "children"),
    Input("variable-dropdown", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
)
def update_kpis(variable, start_date, end_date):
    sub = filter_df(start_date, end_date)
    serie = sub[variable].dropna()
    kpis = [
        ("Promedio", f"{serie.mean():.2f}"),
        ("Mediana", f"{serie.median():.2f}"),
        ("Desv. estándar", f"{serie.std():.2f}"),
        ("Máximo", f"{serie.max():.2f}"),
        ("Observaciones", f"{len(serie):,}"),
    ]
    cards = []
    for label, value in kpis:
        cards.append(html.Div(
            style={**CARD_STYLE, "flex": "1", "minWidth": "140px", "textAlign": "center"},
            children=[
                html.Div(label, style={"color": "#666", "fontSize": "13px"}),
                html.Div(value, style={"fontSize": "22px", "fontWeight": "bold"}),
            ],
        ))
    return cards


@app.callback(
    Output("timeseries-graph", "figure"),
    Input("variable-dropdown", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("agg-radio", "value"),
)
def update_timeseries(variable, start_date, end_date, freq):
    sub = filter_df(start_date, end_date).set_index("timestamp")[variable]
    if freq != "H":
        sub = sub.resample(freq).mean()
    fig = px.line(
        sub, title=f"Serie de tiempo — {POLLUTANT_LABELS[variable]}",
        labels={"value": POLLUTANT_LABELS[variable], "timestamp": "Fecha"},
    )
    fig.update_layout(showlegend=False)
    return fig


@app.callback(
    Output("hist-graph", "figure"),
    Input("variable-dropdown", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
)
def update_histogram(variable, start_date, end_date):
    sub = filter_df(start_date, end_date)
    serie = sub[variable].dropna()
    fig = px.histogram(
        serie, nbins=40, histnorm="probability density",
        title=f"Distribución — {POLLUTANT_LABELS[variable]} "
              f"(asimetría={stats.skew(serie):.2f})",
        labels={"value": POLLUTANT_LABELS[variable]},
    )
    x = np.linspace(serie.min(), serie.max(), 200)
    normal_pdf = stats.norm.pdf(x, serie.mean(), serie.std())
    fig.add_trace(go.Scatter(x=x, y=normal_pdf, mode="lines",
                              name="Normal teórica", line=dict(color="red", dash="dash")))
    fig.update_layout(showlegend=False, bargap=0.02)
    return fig


@app.callback(
    Output("box-month-graph", "figure"),
    Input("variable-dropdown", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
)
def update_boxplot(variable, start_date, end_date):
    sub = filter_df(start_date, end_date)
    order = [m for m in MESES_ES.values() if m in sub["mes_nombre"].unique()]
    fig = px.box(
        sub, x="mes_nombre", y=variable, category_orders={"mes_nombre": order},
        title=f"Distribución mensual — {POLLUTANT_LABELS[variable]}",
        labels={variable: POLLUTANT_LABELS[variable], "mes_nombre": "Mes"},
        color_discrete_sequence=["#2c7fb8"],
    )
    return fig


if __name__ == "__main__":
    app.run(debug=True)
