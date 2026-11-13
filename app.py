"""Dashboard interactivo: clasificacion binaria de campañas publicitarias (venta alta / venta baja)."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# 1. Carga de datos en memoria
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
NOMBRE_CSV = "semana5_visualizacion_datos.csv"
CANDIDATOS = [os.environ.get("DATA_PATH", ""), ROOT / "data" / NOMBRE_CSV, ROOT / NOMBRE_CSV]
RUTA = next(Path(c) for c in CANDIDATOS if c and Path(c).exists())

DATA = pd.read_csv(RUTA)
ORDEN_MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio"]
DATA["mes"] = pd.Categorical(DATA["mes"], categories=ORDEN_MESES, ordered=True)

NUM = ["inversion_publicidad_cop", "visitas", "tiempo_sitio_min", "edad_publico"]
CAT = ["canal", "region", "dispositivo", "mes"]
OBJETIVO = "ventas_cop"
# Variable objetivo binaria: 1 = venta alta (ventas >= mediana), 0 = venta baja
UMBRAL = float(DATA[OBJETIVO].median())
DATA["venta_alta"] = (DATA[OBJETIVO] >= UMBRAL).astype(int)
ETIQUETA = {1: "Alta", 0: "Baja"}

COLORS = {"navy": "#12304A", "blue": "#1976D2", "cyan": "#26A6A1", "gold": "#F4B942", "red": "#D95D5D", "paper": "#F4F7FA"}
PALETA_CANAL = [COLORS["blue"], COLORS["cyan"], COLORS["gold"], COLORS["red"]]


# ---------------------------------------------------------------------------
# 2. Modelos de clasificacion binaria: probabilidades fuera de muestra (validacion cruzada)
# ---------------------------------------------------------------------------
def prep():
    return ColumnTransformer(
        [("num", StandardScaler(), NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)]
    )


MODELOS = {
    "v1 Regresion logistica (inversion)": Pipeline(
        [("sel", ColumnTransformer([("num", StandardScaler(), ["inversion_publicidad_cop"])])),
         ("m", LogisticRegression(max_iter=1000))]
    ),
    "v2 Regresion logistica multiple": Pipeline([("prep", prep()), ("m", LogisticRegression(max_iter=1000))]),
    "v3 Random Forest": Pipeline(
        [("prep", prep()), ("m", RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1))]
    ),
}
MODELO_DEFECTO = "v3 Random Forest"

X, y = DATA[NUM + CAT], DATA["venta_alta"]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
PROB = pd.DataFrame(
    {nombre: cross_val_predict(m, X, y, cv=skf, method="predict_proba")[:, 1] for nombre, m in MODELOS.items()},
    index=DATA.index)


def metricas(real, prob):
    """Metricas de clasificacion con umbral 0.5 sobre la probabilidad de venta alta."""
    real = np.asarray(real)
    pred = (np.asarray(prob) >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(real, pred),
        "precision": precision_score(real, pred, zero_division=0),
        "recall": recall_score(real, pred, zero_division=0),
        "f1": f1_score(real, pred, zero_division=0),
        "auc": roc_auc_score(real, prob) if len(np.unique(real)) == 2 else np.nan,
    }


COMPARACION = pd.DataFrame([{"modelo": n, **metricas(y, PROB[n])} for n in MODELOS])

# Importancia por permutacion del Random Forest (ajustado en 80 %, evaluado en 20 %) con AUC
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
_rf = MODELOS[MODELO_DEFECTO].fit(X_tr, y_tr)
_imp = permutation_importance(_rf, X_te, y_te, n_repeats=15, random_state=42, scoring="roc_auc")
IMPORTANCIA = (
    pd.DataFrame({"variable": X.columns, "importancia": _imp.importances_mean}).sort_values("importancia", ascending=False)
)

# ---------------------------------------------------------------------------
# 3. Aplicacion e interfaz
# ---------------------------------------------------------------------------
app = Dash(__name__, title="Clasificacion de campañas publicitarias", suppress_callback_exceptions=True)
server = app.server

INV_MIN = int(DATA.inversion_publicidad_cop.min() // 100_000 * 100_000)
INV_MAX = int(-(-DATA.inversion_publicidad_cop.max() // 100_000) * 100_000)


def card(title, value_id, note):
    return html.Div([html.P(title, className="kpi-title"), html.H2(id=value_id), html.P(note, className="kpi-note")], className="kpi-card")


app.layout = html.Div([
    html.Header([
        html.Div([
            html.P("CASO DE ESTUDIO SINTETICO", className="eyebrow"),
            html.H1("Clasificacion de campañas: ¿venta alta o baja?"),
            html.P(f"Modelos de clasificacion binaria: venta alta = ventas de {UMBRAL / 1e6:,.1f} millones COP o mas (mediana). "
                   "Los datos son ficticios y su uso es exclusivamente pedagogico.", className="subtitle"),
        ]),
        html.Div("DASH + PLOTLY", className="badge"),
    ], className="hero"),
    html.Div([
        html.Div([
            html.H3("Filtros interactivos"),
            html.Label("Canal"),
            dcc.Dropdown(sorted(DATA.canal.unique()), multi=True, id="canal", placeholder="Todos los canales"),
            html.Label("Region"),
            dcc.Dropdown(sorted(DATA.region.unique()), multi=True, id="region", placeholder="Todas las regiones"),
            html.Label("Dispositivo"),
            dcc.Dropdown(sorted(DATA.dispositivo.unique()), multi=True, id="dispositivo", placeholder="Todos los dispositivos"),
            html.Label("Mes"),
            dcc.Dropdown(ORDEN_MESES, multi=True, id="mes", placeholder="Todos los meses"),
            html.Label("Inversion publicitaria (millones COP)"),
            dcc.RangeSlider(INV_MIN, INV_MAX, 100_000, value=[INV_MIN, INV_MAX], id="inversion",
                            marks={v: f"{v / 1e6:.1f}" for v in range(INV_MIN, INV_MAX + 1, 500_000)}),
            html.Hr(),
            html.Label("Modelo a evaluar"),
            dcc.Dropdown(list(MODELOS), value=MODELO_DEFECTO, clearable=False, id="modelo"),
            html.P("Probabilidad de venta alta con validacion cruzada estratificada (5 pliegues): cada campaña se clasifica con un modelo que no la vio al entrenar. Umbral de decision: 0.5.", className="help"),
            html.Button("Restablecer filtros", id="reset", n_clicks=0),
        ], className="sidebar"),
        html.Main([
            html.Div([
                card("Campañas", "kpi-n", "poblacion filtrada"),
                card("Ventas totales", "kpi-ventas", "millones de COP"),
                card("Inversion total", "kpi-inv", "millones de COP"),
                card("Retorno (ventas / inversion)", "kpi-roi", "veces la inversion"),
                card("Satisfaccion media", "kpi-sat", "escala de 1 a 5"),
            ], className="kpi-grid"),
            dcc.Tabs([
                dcc.Tab(label="Panorama", children=[
                    html.Div([dcc.Graph(id="canal-chart"), dcc.Graph(id="mes-chart")], className="grid-2"),
                    html.Div([dcc.Graph(id="scatter-chart"), dcc.Graph(id="hist-chart")], className="grid-2"),
                ]),
                dcc.Tab(label="Rendimiento del modelo", children=[
                    html.Div(id="metrics-strip", className="metric-strip"),
                    html.Div([dcc.Graph(id="real-pred-chart"), dcc.Graph(id="residual-chart")], className="grid-2"),
                    html.Div([dcc.Graph(id="model-chart"), dcc.Graph(id="importance-chart")], className="grid-2"),
                    html.Div(id="narrative", className="narrative"),
                ]),
                dcc.Tab(label="Errores de clasificacion", children=[
                    html.P("Campañas que el modelo seleccionado clasifica peor (errores primero, ordenados por la distancia entre la probabilidad estimada y la clase real). Registros sinteticos, sin datos personales."),
                    dash_table.DataTable(
                        id="tabla", page_size=12, sort_action="native", filter_action="native",
                        style_table={"overflowX": "auto"}, style_cell={"fontFamily": "Arial", "fontSize": 13, "padding": "8px"},
                        style_header={"backgroundColor": COLORS["navy"], "color": "white", "fontWeight": "bold"},
                        style_data_conditional=[
                            {"if": {"filter_query": '{resultado} = "Error"', "column_id": "resultado"}, "color": COLORS["red"], "fontWeight": "bold"},
                            {"if": {"filter_query": '{resultado} = "Acierto"', "column_id": "resultado"}, "color": COLORS["cyan"]},
                        ]),
                ]),
            ]),
            html.Footer("Fuente: datos sinteticos de campañas digitales. Clasificacion fuera de muestra con validacion cruzada estratificada de 5 pliegues."),
        ], className="content"),
    ], className="shell"),
], className="app")


# ---------------------------------------------------------------------------
# 4. Callbacks
# ---------------------------------------------------------------------------
@app.callback(
    Output("canal", "value"), Output("region", "value"), Output("dispositivo", "value"), Output("mes", "value"),
    Output("inversion", "value"), Output("modelo", "value"),
    Input("reset", "n_clicks"), prevent_initial_call=True)
def reset_filters(_):
    return [], [], [], [], [INV_MIN, INV_MAX], MODELO_DEFECTO


@app.callback(
    Output("kpi-n", "children"), Output("kpi-ventas", "children"), Output("kpi-inv", "children"),
    Output("kpi-roi", "children"), Output("kpi-sat", "children"),
    Output("canal-chart", "figure"), Output("mes-chart", "figure"), Output("scatter-chart", "figure"), Output("hist-chart", "figure"),
    Output("metrics-strip", "children"), Output("real-pred-chart", "figure"), Output("residual-chart", "figure"),
    Output("model-chart", "figure"), Output("importance-chart", "figure"), Output("narrative", "children"),
    Output("tabla", "data"), Output("tabla", "columns"),
    Input("canal", "value"), Input("region", "value"), Input("dispositivo", "value"), Input("mes", "value"),
    Input("inversion", "value"), Input("modelo", "value"))
def update(canales, regiones, dispositivos, meses, inversion, modelo):
    d = DATA.copy()
    if canales: d = d[d.canal.isin(canales)]
    if regiones: d = d[d.region.isin(regiones)]
    if dispositivos: d = d[d.dispositivo.isin(dispositivos)]
    if meses: d = d[d.mes.isin(meses)]
    d = d[d.inversion_publicidad_cop.between(inversion[0], inversion[1])]
    template = "plotly_white"
    hay = len(d) > 0

    # --- Panorama -----------------------------------------------------------
    por_canal = d.groupby("canal", observed=True, as_index=False).agg(
        ventas=(OBJETIVO, "sum"), inversion=("inversion_publicidad_cop", "sum"), campanas=("id_campana", "count"))
    por_canal["roi"] = por_canal.ventas / por_canal.inversion
    long = por_canal.melt(id_vars=["canal", "roi"], value_vars=["inversion", "ventas"], var_name="metrica", value_name="cop")
    long["metrica"] = long.metrica.map({"inversion": "Inversion", "ventas": "Ventas"})
    fig_canal = px.bar(long, x="canal", y="cop", color="metrica", barmode="group", title="Inversion vs. ventas por canal",
                       color_discrete_map={"Inversion": COLORS["gold"], "Ventas": COLORS["blue"]})
    fig_canal.update_layout(template=template, yaxis_title="COP", xaxis_title="", legend_title="")

    por_mes = d.groupby("mes", observed=True, as_index=False).agg(ventas=(OBJETIVO, "sum"), inversion=("inversion_publicidad_cop", "sum"))
    fig_mes = px.line(por_mes.melt(id_vars="mes", var_name="metrica", value_name="cop"), x="mes", y="cop", color="metrica", markers=True,
                      title="Evolucion mensual", color_discrete_map={"ventas": COLORS["blue"], "inversion": COLORS["gold"]})
    fig_mes.update_layout(template=template, yaxis_title="COP", xaxis_title="", legend_title="")

    muestra = d.sample(min(600, len(d)), random_state=7) if hay else d
    fig_scatter = px.scatter(muestra, x="inversion_publicidad_cop", y=OBJETIVO, color="canal", size="conversiones", size_max=16,
                             opacity=0.72, title="Inversion, ventas y canal (tamaño = conversiones)", color_discrete_sequence=PALETA_CANAL)
    fig_scatter.update_layout(template=template, xaxis_title="Inversion (COP)", yaxis_title="Ventas (COP)")

    fig_hist = px.histogram(d, x=OBJETIVO, nbins=30, title="Distribucion de las ventas", color_discrete_sequence=[COLORS["cyan"]])
    fig_hist.add_vline(x=UMBRAL, line_dash="dash", line_color=COLORS["gold"], annotation_text="umbral venta alta")
    fig_hist.update_layout(template=template, xaxis_title="Ventas (COP)", yaxis_title="Frecuencia")

    # --- Modelo -------------------------------------------------------------
    p = d.assign(prob_alta=PROB.loc[d.index, modelo])
    p["pred"] = (p.prob_alta >= 0.5).astype(int)
    if len(p) >= 3:
        m = metricas(p.venta_alta, p.prob_alta)
        auc_txt = "--" if np.isnan(m["auc"]) else f"{m['auc']:.3f}"
        strip = [html.Div([html.Span("Exactitud (accuracy)"), html.Strong(f"{m['accuracy']:.1%}")]),
                 html.Div([html.Span("Precision"), html.Strong(f"{m['precision']:.1%}")]),
                 html.Div([html.Span("Sensibilidad (recall)"), html.Strong(f"{m['recall']:.1%}")]),
                 html.Div([html.Span("F1"), html.Strong(f"{m['f1']:.3f}")]),
                 html.Div([html.Span("ROC-AUC"), html.Strong(auc_txt)])]
    else:
        m = {k: np.nan for k in ("accuracy", "precision", "recall", "f1", "auc")}
        strip = [html.Div([html.Span("Sin muestra suficiente"), html.Strong("--")])]

    # Matriz de confusion
    cm = confusion_matrix(p.venta_alta, p.pred, labels=[0, 1]) if len(p) else np.zeros((2, 2), dtype=int)
    fig_cm = px.imshow(cm, text_auto=True, x=["Baja", "Alta"], y=["Baja", "Alta"],
                       labels=dict(x="Clase predicha", y="Clase real", color="Campañas"),
                       color_continuous_scale=[[0, "#EAF2FB"], [1, COLORS["blue"]]],
                       title=f"Matriz de confusion ({modelo.split(' ')[0]})")
    fig_cm.update_layout(template=template, coloraxis_showscale=False)

    # Curva ROC
    fig_roc = go.Figure()
    if p.venta_alta.nunique() == 2:
        fpr, tpr, _ = roc_curve(p.venta_alta, p.prob_alta)
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{modelo.split(' ')[0]} (AUC {m['auc']:.2f})",
                                     line=dict(color=COLORS["blue"], width=3)))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar (AUC 0.50)", line=dict(dash="dash", color=COLORS["navy"])))
    fig_roc.update_layout(template=template, title="Curva ROC", xaxis_title="Tasa de falsos positivos", yaxis_title="Tasa de verdaderos positivos")

    comp = COMPARACION.copy()
    comp["seleccionado"] = np.where(comp.modelo == modelo, "Seleccionado", "Otros")
    fig_models = px.bar(comp, x="modelo", y="auc", color="seleccionado", text=comp.auc.map(lambda v: f"{v:.2f}"),
                        title="Comparacion de iteraciones (ROC-AUC, todas las campañas)",
                        color_discrete_map={"Seleccionado": COLORS["blue"], "Otros": "#B9C7D4"})
    fig_models.add_hline(y=0.5, line_dash="dash", line_color=COLORS["gold"], annotation_text="azar")
    fig_models.update_layout(template=template, xaxis_title="", yaxis_title="ROC-AUC", legend_title="")

    imp = IMPORTANCIA.sort_values("importancia")
    fig_imp = px.bar(imp, x="importancia", y="variable", orientation="h", title="Importancia por permutacion (Random Forest)",
                     color_discrete_sequence=[COLORS["cyan"]])
    fig_imp.update_layout(template=template, xaxis_title="Caida media en ROC-AUC", yaxis_title="")

    top_var = IMPORTANCIA.iloc[0].variable
    pct_alta = d.venta_alta.mean() if hay else np.nan
    narrativa = [
        html.H3("Lectura ejecutiva"),
        html.P(f"La vista contiene {len(d):,} campañas; {pct_alta:.0%} son de venta alta (≥ {UMBRAL / 1e6:,.1f} millones COP). Ventas totales: "
               f"{d[OBJETIVO].sum() / 1e6:,.1f} millones COP con una inversion de {d.inversion_publicidad_cop.sum() / 1e6:,.1f} millones." if hay else "No hay registros con estos filtros."),
        html.P(f"El modelo {modelo} clasifica bien {m['accuracy']:.1%} de las campañas (ROC-AUC: {'--' if np.isnan(m['auc']) else format(m['auc'], '.2f')}). "
               "Un AUC cercano a 0.50 equivale a adivinar al azar." if not np.isnan(m["accuracy"]) else "La seleccion es muy pequeña para calcular metricas."),
        html.P(f"La variable con mayor peso predictivo en el Random Forest es «{top_var}». Si la importancia es cercana a cero, las variables disponibles aportan poca informacion para separar ventas altas de bajas."),
        html.P("Los factores muestran asociacion predictiva, no causalidad. Al ser datos sinteticos, estos resultados no describen a ninguna empresa real."),
    ]

    # --- Tabla --------------------------------------------------------------
    p["clase_real"] = p.venta_alta.map(ETIQUETA)
    p["clase_predicha"] = p.pred.map(ETIQUETA)
    p["resultado"] = np.where(p.pred == p.venta_alta, "Acierto", "Error")
    p["distancia"] = (p.venta_alta - p.prob_alta).abs()
    p["prob_alta"] = p.prob_alta.round(3)
    cols = ["id_campana", "mes", "canal", "region", "dispositivo", "inversion_publicidad_cop", "visitas", OBJETIVO,
            "clase_real", "clase_predicha", "prob_alta", "resultado"]
    t = p.sort_values("distancia", ascending=False)[cols].head(100).copy()
    t["mes"] = t.mes.astype(str)
    columnas = [{"name": c.replace("_cop", " (COP)").replace("_", " ").title(), "id": c} for c in cols]

    return (f"{len(d):,}",
            f"{d[OBJETIVO].sum() / 1e6:,.1f}" if hay else "--",
            f"{d.inversion_publicidad_cop.sum() / 1e6:,.1f}" if hay else "--",
            f"{d[OBJETIVO].sum() / d.inversion_publicidad_cop.sum():.2f}x" if hay else "--",
            f"{d.satisfaccion.mean():.2f}" if hay else "--",
            fig_canal, fig_mes, fig_scatter, fig_hist,
            strip, fig_cm, fig_roc, fig_models, fig_imp, narrativa,
            t.to_dict("records"), columnas)


# ---------------------------------------------------------------------------
# 5. Inicio del servidor
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
