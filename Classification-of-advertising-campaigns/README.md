# Clasificación de campañas publicitarias: ¿venta alta o baja?

Dashboard interactivo hecho con **Dash + Plotly** que explora campañas publicitarias digitales y evalúa modelos de **clasificación binaria** para predecir si una campaña tendrá una **venta alta** o una **venta baja**.

> **Aviso: los datos son sintéticos.** El conjunto de datos es ficticio, generado con fines pedagógicos. No describe a ninguna empresa, persona o campaña real y no contiene datos personales. Los resultados del modelo **no deben usarse para tomar decisiones reales de negocio**.

## Estructura del proyecto

```
proyecto_campanas/
├── app.py                 # Aplicación completa (datos, modelos, interfaz, callbacks, servidor)
├── requirements.txt       # Dependencias
├── assets/
│   └── style.css          # Estilos (Dash lo carga automáticamente)
└── data/
    └── semana5_visualizacion_datos.csv
```

`app.py` sigue un flujo lineal en cinco secciones: importar librerías → cargar los datos en memoria → crear la aplicación y definir la interfaz → registrar los callbacks → iniciar el servidor.

## Cómo ejecutarlo

1. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

2. Coloca el CSV en `data/semana5_visualizacion_datos.csv` (también se acepta junto a `app.py`, o en cualquier ruta indicada con la variable de entorno `DATA_PATH`).

3. Desde la carpeta del proyecto ejecuta:

   ```bash
   python app.py
   ```

4. Abre <http://127.0.0.1:8050> en el navegador.

Al arrancar, la app entrena los modelos en memoria. Puede tardar entre 30 segundos y un par de minutos antes de mostrar el mensaje de que el servidor está corriendo.

## Datos

Cada fila es una campaña publicitaria. Columnas del CSV:

| Columna | Descripción |
|---|---|
| `id_campana` | Identificador de la campaña |
| `mes` | Mes de la campaña (Enero a Junio) |
| `canal` | Redes sociales, Buscadores, Email, Referidos |
| `region` | Bogotá, Medellín, Cali, Barranquilla |
| `dispositivo` | Móvil, Escritorio, Tablet |
| `edad_publico` | Edad del público objetivo |
| `visitas` | Visitas generadas |
| `tiempo_sitio_min` | Tiempo en el sitio (minutos) |
| `inversion_publicidad_cop` | Inversión publicitaria (COP) |
| `conversiones` | Conversiones logradas |
| `tasa_conversion` | Conversiones sobre visitas |
| `ventas_cop` | Ventas generadas (COP) |
| `satisfaccion` | Satisfacción media (escala de 1 a 5) |

## Variable objetivo

Se crea la variable binaria `venta_alta`:

- **1 (Alta):** `ventas_cop` es mayor o igual a la **mediana** de las ventas.
- **0 (Baja):** `ventas_cop` es menor que la mediana.

Se usa la mediana como umbral porque deja las dos clases balanceadas (aprox. 50 % y 50 %), lo que hace que métricas como la exactitud sean fáciles de interpretar.

## Variables usadas en el modelo

**Numéricas** (se estandarizan con `StandardScaler`):

- `inversion_publicidad_cop`
- `visitas`
- `tiempo_sitio_min`
- `edad_publico`

**Categóricas** (se codifican con `OneHotEncoder`):

- `canal`
- `region`
- `dispositivo`
- `mes`

**Variables excluidas a propósito:**

- `ventas_cop`: es la variable de la que se deriva el objetivo (sería fuga de información).
- `conversiones`, `tasa_conversion` y `satisfaccion`: se conocen **después** de que la campaña ya produjo resultados, y están muy ligadas a las ventas. Incluirlas haría que el modelo "adivinara" con información que no estaría disponible al planear una campaña.
- `id_campana`: es solo un identificador.

## Modelos y por qué se eligieron

El dashboard compara tres iteraciones, de la más simple a la más flexible:

| Modelo | Qué usa | Para qué sirve |
|---|---|---|
| **v1 Regresión logística (inversión)** | Solo `inversion_publicidad_cop` | Línea base: responde si la inversión por sí sola alcanza para anticipar una venta alta |
| **v2 Regresión logística múltiple** | Las 8 variables | Modelo lineal e interpretable: muestra el aporte de cada variable a la probabilidad de venta alta |
| **v3 Random Forest** | Las 8 variables | Captura relaciones no lineales e interacciones entre variables (por ejemplo, canal × región) |

**¿Por qué clasificación binaria y regresión logística?**

- El análisis original usaba una regresión lineal simple de la inversión sobre las ventas, que predecía mal (R² cercano a 0). Plantear el problema como "venta alta o baja" es más directo para una decisión de negocio.
- La regresión logística es la versión de clasificación de la regresión lineal: mantiene la interpretabilidad y funciona de forma estable con pocos datos (del orden de cientos de filas).
- El Random Forest se incluye para comprobar si relaciones más complejas mejoran el resultado frente a los modelos lineales. Es el modelo seleccionado por defecto en el dashboard, pero **no se asume que sea el mejor**: la pestaña de rendimiento permite compararlo con los otros dos.

## Evaluación

- **Validación cruzada estratificada de 5 pliegues:** cada campaña se clasifica con un modelo que no la vio durante el entrenamiento, así las métricas no se inflan y siguen siendo útiles al aplicar filtros.
- **Umbral de decisión:** 0.5 sobre la probabilidad de venta alta.
- **Métricas:** exactitud (*accuracy*), precisión, sensibilidad (*recall*), F1 y ROC-AUC.
- **Importancia de variables:** importancia por permutación del Random Forest (ajustado con 80 % de los datos y evaluado con el 20 % restante), medida como la caída del ROC-AUC.

**Cómo leer el ROC-AUC:** 0.50 equivale a adivinar al azar; cuanto más cerca de 1.0, mejor separa el modelo las ventas altas de las bajas. Con datos sintéticos en los que las variables no se relacionan fuertemente con las ventas, es normal obtener valores cercanos a 0.50. Eso es un resultado válido, no un error de la aplicación.

## Qué muestra el dashboard

- **Filtros:** canal, región, dispositivo, mes, rango de inversión y modelo a evaluar.
- **KPIs:** número de campañas, ventas totales, inversión total, retorno (ventas / inversión) y satisfacción media.
- **Panorama:** inversión vs. ventas por canal, evolución mensual, dispersión inversión-ventas y distribución de las ventas con el umbral de venta alta.
- **Rendimiento del modelo:** métricas, matriz de confusión, curva ROC, comparación de modelos por ROC-AUC, importancia de variables y una lectura ejecutiva.
- **Errores de clasificación:** las campañas que el modelo seleccionado clasifica peor.

## Limitaciones

- Los datos son sintéticos y el conjunto es pequeño; los resultados no se pueden generalizar.
- Los factores del modelo muestran **asociación predictiva, no causalidad**.
- El umbral de "venta alta" (la mediana) es una decisión de diseño; otro umbral produciría otros resultados.
- No se hace ajuste de hiperparámetros: los modelos usan valores razonables fijos (Random Forest con 300 árboles y `min_samples_leaf=3`).

## Tecnologías

Python · Dash · Plotly · pandas · NumPy · scikit-learn

## Licencia y uso

Proyecto con fines académicos. Antes de publicarlo en GitHub, añade el archivo de licencia que prefieras (por ejemplo, MIT).
