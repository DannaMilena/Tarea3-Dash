"""Genera imágenes estáticas (PNG) con matplotlib para incluir en el informe
Word. (Se usa matplotlib en vez de la exportación nativa de Plotly porque el
entorno de generación no cuenta con Chrome/Kaleido; el dashboard interactivo
en app.py sí usa Plotly/Dash normalmente)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from data_processing import NUMERIC_COLS, clean, load_raw, missing_summary

os.makedirs("report_images", exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

raw = load_raw("data/AirQualityUCI.csv")
df, df_nan = clean(raw)
missing_df = missing_summary(df_nan)

MESES_ES = {1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"}
df["mes_nombre"] = df["mes"].map(MESES_ES)

# 1. Serie de tiempo diaria de CO(GT)
sub = df.set_index("timestamp")["CO(GT)"].resample("D").mean()
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(sub.index, sub.values, color="#2c7fb8", linewidth=1)
ax.set_title("Serie de tiempo diaria — CO de referencia (mg/m³)")
ax.set_xlabel("Fecha"); ax.set_ylabel("CO (mg/m³)")
fig.tight_layout(); fig.savefig("report_images/01_timeseries_CO.png"); plt.close(fig)

# 2. Histograma + normal
serie = df["CO(GT)"].dropna()
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(serie, bins=40, density=True, color="#74a9cf", edgecolor="white", alpha=0.85)
x = np.linspace(serie.min(), serie.max(), 200)
ax.plot(x, stats.norm.pdf(x, serie.mean(), serie.std()), "r--", label="Normal teórica")
ax.set_title(f"Distribución de CO(GT) (asimetría={stats.skew(serie):.2f})")
ax.set_xlabel("CO (mg/m³)"); ax.set_ylabel("Densidad"); ax.legend()
fig.tight_layout(); fig.savefig("report_images/02_histograma_CO.png"); plt.close(fig)

# 3. Boxplot mensual
order = [m for m in MESES_ES.values() if m in df["mes_nombre"].unique()]
data_by_month = [df.loc[df["mes_nombre"] == m, "CO(GT)"].dropna().values for m in order]
fig, ax = plt.subplots(figsize=(9, 4))
ax.boxplot(data_by_month, labels=order, patch_artist=True,
           boxprops=dict(facecolor="#a6bddb"))
ax.set_title("Distribución mensual de CO(GT)")
ax.set_xlabel("Mes"); ax.set_ylabel("CO (mg/m³)")
fig.tight_layout(); fig.savefig("report_images/03_boxplot_mensual_CO.png"); plt.close(fig)

# 4. Heatmap de correlación
corr = df[NUMERIC_COLS].corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticks(range(len(corr.columns))); ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
ax.set_title("Matriz de correlación entre variables")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout(); fig.savefig("report_images/04_correlacion.png"); plt.close(fig)

# 5. Dispersión sensor vs referencia (calibración) con regresión lineal
sample = df.sample(2000, random_state=1)
x_vals, y_vals = sample["PT08.S1(CO)"], sample["CO(GT)"]
slope, intercept, r, p, se = stats.linregress(x_vals, y_vals)
fig, ax = plt.subplots(figsize=(9, 4))
ax.scatter(x_vals, y_vals, alpha=0.25, s=10, color="#3182bd")
xs = np.linspace(x_vals.min(), x_vals.max(), 100)
ax.plot(xs, slope * xs + intercept, color="red", linewidth=2,
        label=f"OLS (R²={r**2:.2f})")
ax.set_title("Calibración: sensor PT08.S1(CO) vs. CO de referencia")
ax.set_xlabel("Respuesta sensor PT08.S1(CO)"); ax.set_ylabel("CO (mg/m³)")
ax.legend()
fig.tight_layout(); fig.savefig("report_images/05_scatter_calibracion.png"); plt.close(fig)

# 6. Faltantes
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(missing_df["variable"], missing_df["pct_faltante"], color="#fd8d3c")
ax.set_title("Porcentaje de valores faltantes (código -200) por variable")
ax.set_xlabel("Variable"); ax.set_ylabel("% faltante")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
fig.tight_layout(); fig.savefig("report_images/06_faltantes.png"); plt.close(fig)

print("Imágenes generadas en report_images/")
