"""
AquaMind - App local en Streamlit para visualizar consumos y disponibilidad hídrica por región.

Características:
- Subir archivos CSV
- Seleccionar región
- Gráficos comparativos (agua disponible vs agua potable)
- Calcular impacto hídrico estimado de un centro de datos:
    * Consumo promedio: 25.000.000 L/año ≈ 68.000 L/día
- Indicador visual de estrés hídrico
- Tabla con datos del CSV
- Modular y comentado para extender fácilmente
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import xml.etree.ElementTree as ET
from typing import Optional, Tuple

# -----------------------
# Constantes y parámetros
# -----------------------
CD_CONS_ANUAL_L = 25_000_000        # litros/año (referencia)
CD_CONS_DIARIO_L = 68_000          # litros/día (aprox. redondeado)
# Umbrales de estrés hídrico (porcentaje del consumo del CD respecto al agua disponible)
STRESS_THRESHOLDS = {
    "Bajo": 5,       # <5% -> Bajo
    "Moderado": 15,  # 5-15% -> Moderado
    "Alto": 30,      # 15-30% -> Alto
    "Crítico": 100   # >30% -> Crítico (o uso mayoritario)
}

# -----------------------
# Funciones utilitarias
# -----------------------

def read_csv_bytes(uploaded_file) -> pd.DataFrame:
    """Lee un CSV subido (bytes) usando pandas. Detecta separador comúnmente usado."""
    try:
        uploaded_file.seek(0)
        # Intentar auto-detección simple: si tiene ';' o ','.
        content = uploaded_file.read().decode('utf-8', errors='replace')
        # prueba separador
        sep = ',' if content.count(',') >= content.count(';') else ';'
        df = pd.read_csv(io.StringIO(content), sep=sep)
        return df
    except Exception as e:
        st.error(f"Error leyendo CSV: {e}")
        return pd.DataFrame()

def read_xml_bytes(uploaded_file) -> pd.DataFrame:
    """Si el usuario sube XML, intenta parsearlo con ElementTree y convertirlo a DataFrame.
    Se asume estructura simple <root><row><col1>..</col1>..</row>..</root>"""
    try:
        uploaded_file.seek(0)
        tree = ET.parse(io.BytesIO(uploaded_file.read()))
        root = tree.getroot()
        rows = []
        # tomar el primer nivel de hijos como filas
        for child in root:
            row = {}
            for elem in child:
                row[elem.tag] = elem.text
            rows.append(row)
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Error leyendo XML: {e}")
        return pd.DataFrame()

def detect_column_names(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Intenta detectar columnas de región, agua disponible y agua potable (insensible a mayúsculas).
       Devuelve (col_region, col_disponible, col_potable) o None si no encuentra."""
    lower_cols = {c.lower(): c for c in df.columns}
    # Posibles nombres
    region_keys = ['region', 'provincia', 'zona', 'departamento']
    disponible_keys = ['agua_disponible', 'disponible', 'reservas', 'reserva', 'agua_reservas', 'volumen_disponible']
    potable_keys = ['agua_potable', 'potable', 'potabilidad', 'potable_l', 'agua_tratable']
    def find_key(possible):
        for k in possible:
            if k in lower_cols:
                return lower_cols[k]
        # también soportar cuando columna contiene la palabra
        for lk, orig in lower_cols.items():
            for k in possible:
                if k in lk:
                    return orig
        return None
    return find_key(region_keys), find_key(disponible_keys), find_key(potable_keys)

def normalize_liters(value) -> Optional[float]:
    """Convierte valores a float (litros). Si el valor parece estar en m3 (unidad en nombre), convertir fuera.
       Aquí asumimos que los datos ya están en litros; si detectas 'm3' en una columna, ajustar afuera."""
    try:
        if pd.isna(value):
            return None
        # limpiar separadores de miles y comas decimales
        s = str(value).strip().replace('.', '').replace(',', '.')
        return float(s)
    except:
        return None

def compute_impact(row_value_available_l: float, cd_daily_l: float = CD_CONS_DIARIO_L) -> dict:
    """Calcula impacto del centro de datos en esa región basado en agua disponible (litros).
       Retorna diccionario con porcentaje diario/anual y nivel de estrés."""
    result = {}
    if row_value_available_l is None or row_value_available_l <= 0:
        result.update({"pct_diario": None, "pct_anual": None, "stress_level": "Sin datos"})
        return result
    # porcentajes
    pct_diario = (cd_daily_l / row_value_available_l) * 100
    pct_anual = (CD_CONS_ANUAL_L / row_value_available_l) * 100
    # determinar nivel de estrés por umbrales (usar pct_anual o pct_diario según preferencia)
    pct_for_threshold = pct_diario  # usamos diario para indicar impacto inmediato
    if pct_for_threshold < STRESS_THRESHOLDS["Bajo"]:
        level = "Bajo"
    elif pct_for_threshold < STRESS_THRESHOLDS["Moderado"]:
        level = "Moderado"
    elif pct_for_threshold < STRESS_THRESHOLDS["Alto"]:
        level = "Alto"
    else:
        level = "Crítico"
    result.update({
        "pct_diario": pct_diario,
        "pct_anual": pct_anual,
        "stress_level": level
    })
    return result

def plot_available_vs_potable(region_name: str, available_l: float, potable_l: float):
    """Grafico de barras simple comparando agua disponible vs potable"""
    labels = ['Agua disponible (L)', 'Agua potable (L)']
    values = [available_l if available_l is not None else 0,
              potable_l if potable_l is not None else 0]
    fig, ax = plt.subplots(figsize=(6,4))
    bars = ax.bar(labels, values)
    ax.set_title(f'Comparación - {region_name}')
    ax.set_ylabel('Litros')
    # etiqueta de valores encima de las barras
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height):,}', xy=(bar.get_x()+bar.get_width()/2, height),
                    xytext=(0,3), textcoords="offset points", ha='center', va='bottom')
    plt.tight_layout()
    return fig

# -----------------------
# Interfaz Streamlit
# -----------------------

st.set_page_config(page_title="AquaMind - Impacto Hídrico", layout="centered", page_icon="💧")

st.title("AquaMind — Visualización de consumo e impacto hídrico")
st.markdown("Carga archivos **CSV** con datos por región. Interfaz en *español*.")

# panel lateral con instrucciones y parámetros
with st.sidebar:
    st.header("Instrucciones")
    st.write("""
    1. Sube uno o más archivos CSV (o XML) con columnas que incluyan:
       - **region** (nombre de la región / provincia)
       - **agua_disponible** (en litros preferiblemente)
       - **agua_potable** (en litros preferiblemente)
    2. Selecciona la región desde el menú desplegable.
    3. Usa los botones para ver gráficos, tabla o calcular impacto del centro de datos.
    """)
    st.markdown("**Parámetros del centro de datos**")
    # permitir al usuario ajustar el consumo del centro de datos si quisiera
    cd_anual_user = st.number_input("Consumo anual CD (litros)", value=CD_CONS_ANUAL_L, step=1_000_000, format="%d")
    cd_diario_user = st.number_input("Consumo diario CD (litros)", value=CD_CONS_DIARIO_L, step=1_000, format="%d")
    st.markdown("---")
    st.write("Colores/íconos: los botones tienen emojis para facilitar la lectura.")

# Carga de archivos
uploaded_files = st.file_uploader("Subir archivos (.csv o .xml)", type=['csv','xml'], accept_multiple_files=True)

if not uploaded_files:
    st.info("Sube al menos un archivo CSV para comenzar. Puedes usar los archivos de ejemplo proporcionados antes.")
    st.stop()

# Leemos y concatenamos dataframes
df_list = []
for f in uploaded_files:
    if f.name.lower().endswith('.xml'):
        df_tmp = read_xml_bytes(f)
    else:
        df_tmp = read_csv_bytes(f)
    if not df_tmp.empty:
        # añadir columna fuente para trazabilidad
        df_tmp['_source_file'] = f.name
        df_list.append(df_tmp)

if len(df_list) == 0:
    st.error("No se pudo leer ningún archivo con datos válidos.")
    st.stop()

df = pd.concat(df_list, ignore_index=True)

# Intentar detectar columnas principales
col_region, col_disponible, col_potable = detect_column_names(df)

if col_region is None:
    st.error("No se detectó la columna de 'región' en los archivos. Asegúrate de que exista una columna llamada 'region', 'provincia' o similar.")
    st.write("Columnas encontradas:", list(df.columns))
    st.stop()

if col_disponible is None or col_potable is None:
    st.warning("No se detectaron automáticamente columnas para 'agua disponible' o 'agua potable'. Revisa los nombres de columnas.")
    st.write("Columnas encontradas:", list(df.columns))
    # permitimos al usuario elegir manualmente las columnas si la detección falló
    cols = list(df.columns)
    col_disponible = st.selectbox("Seleccionar columna para 'Agua disponible'", cols, index=0)
    col_potable = st.selectbox("Seleccionar columna para 'Agua potable'", cols, index=min(1, len(cols)-1))

# Mostrar preview de datos
st.subheader("Vista previa de los datos cargados")
st.dataframe(df.head(50))

# Normalizamos valores numéricos de las columnas seleccionadas
df['_disp_l'] = df[col_disponible].apply(normalize_liters)
df['_pot_l'] = df[col_potable].apply(normalize_liters)
df['_region_norm'] = df[col_region].astype(str).str.strip()

# Crear lista de regiones ordenadas
regions = sorted(df['_region_norm'].unique())

# Selector de región
region_sel = st.selectbox("Seleccionar región", regions)

# Filtrar dataframe por región seleccionada
df_region = df[df['_region_norm'] == region_sel].copy()

# Si hay múltiples filas por región, agrupar sumando o tomando promedio según sentido
# Suponemos que la columna representa reservas totales; agrupamos usando sum si tiene múltiples entradas
agg_available = df_region['_disp_l'].sum(skipna=True)
agg_potable = df_region['_pot_l'].sum(skipna=True)

st.markdown("---")
st.markdown(f"### Datos para: **{region_sel}**")
st.write(f"Archivos fuente: {', '.join(sorted(df_region['_source_file'].unique()))}")
st.write(f"Agua disponible (sumatoria): **{int(agg_available):,} L**" if agg_available>0 else "Agua disponible: sin datos")
st.write(f"Agua potable (sumatoria): **{int(agg_potable):,} L**" if agg_potable>0 else "Agua potable: sin datos")

# Botones para visualizar contenido
col1, col2, col3 = st.columns(3)
with col1:
    btn_ver_disponible = st.button("💧 Ver comparación")
with col2:
    btn_ver_tabla = st.button("📋 Ver tabla")
with col3:
    btn_calcular_impacto = st.button("⚠️ Calcular impacto CD")

# Mostrar gráfico si pide
if btn_ver_disponible:
    st.subheader("Comparación: agua disponible vs agua potable")
    fig = plot_available_vs_potable(region_sel, agg_available, agg_potable)
    st.pyplot(fig)

# Mostrar tabla
if btn_ver_tabla:
    st.subheader("Tabla de datos (filtrada por región)")
    # presentar columnas relevantes y normalizadas
    display_cols = [col_region, col_disponible, col_potable, '_source_file']
    # Algunas pueden no existir si selección manual
    # mostrar además las columnas normalizadas en litros
    df_show = df_region.copy()
    df_show = df_show.assign(Agua_disponible_L = df_show['_disp_l'],
                             Agua_potable_L = df_show['_pot_l'])
    st.dataframe(df_show[[col_region, 'Agua_disponible_L', 'Agua_potable_L', '_source_file']].fillna(""))

# Calcular impacto hídrico y mostrar indicador
if btn_calcular_impacto:
    st.subheader("Impacto hídrico estimado si se instala un centro de datos")
    # usar parámetros del sidebar (puede ser modificado por el usuario)
    cd_anual = float(cd_anual_user)
    cd_diario = float(cd_diario_user)
    # Tomamos como referencia agua disponible agregada (podría cambiarse a promedio o a otra métrica)
    impact = compute_impact(agg_available, cd_daily_l=cd_diario)
    if impact['pct_diario'] is None:
        st.warning("No hay datos de agua disponible para calcular impacto.")
    else:
        st.metric("Consumo diario del CD (L)", f"{int(cd_diario):,}")
        st.write(f"- Porcentaje del consumo diario del CD respecto al agua disponible: **{impact['pct_diario']:.2f}%**")
        st.write(f"- Porcentaje anual estimado: **{impact['pct_anual']:.2f}%**")
        # Indicador visual: usar st.progress para mostrar el % (acotado a 100)
        pct_vis = min(impact['pct_diario'], 100) / 100.0
        st.progress(pct_vis)
        # Color/Texto del nivel de estrés
        level = impact['stress_level']
        if level == "Bajo":
            st.success(f"Nivel de estrés hídrico: {level} ✅")
        elif level == "Moderado":
            st.info(f"Nivel de estrés hídrico: {level} ⚠️")
        elif level == "Alto":
            st.warning(f"Nivel de estrés hídrico: {level} ⚠️⚠️")
        else:
            st.error(f"Nivel de estrés hídrico: {level} 🔥")
        # Mensaje interpretativo
        if impact['pct_diario'] < 5:
            st.write("Interpretación: impacto local bajo. Sin embargo, considerar variaciones estacionales.")
        elif impact['pct_diario'] < 15:
            st.write("Interpretación: impacto moderado — conviene estudiar medidas de reutilización y ahorro.")
        elif impact['pct_diario'] < 30:
            st.write("Interpretación: impacto alto — necesaria planificación de mitigación y alternativas.")
        else:
            st.write("Interpretación: impacto crítico — no recomendado instalar sin estudio profundo y medidas de mitigación.")

# -----------------------
# COMPARADOR DE TIPOS DE CENTROS DE DATOS
# -----------------------
st.markdown("---")
st.subheader("🏭 Comparador de impacto por tipo de Centro de Datos")

# Cargar CSV de tipos de centros (si el usuario lo subió)
tipos_file = st.file_uploader("Subir archivo con tipos de centros de datos", type=['csv'], key="tipos_cd")

if tipos_file:
    df_tipos = pd.read_csv(tipos_file, header=1)
    st.write("Vista previa de los tipos de CD:")
    st.dataframe(df_tipos)

    # -----------------------
    # NORMALIZACIÓN Y DETECCIÓN DE COLUMNAS
    # -----------------------
    df_tipos.columns = [c.strip().lower() for c in df_tipos.columns]
    st.write("Columnas detectadas en el archivo:", list(df_tipos.columns))

    # Buscar columnas por palabras clave
    def detectar_columna(df, posibles):
        for p in posibles:
            for c in df.columns:
                if p in c:
                    return c
        return None

    # Detección según tu estructura real
    col_tipo = detectar_columna(df_tipos, ['tipo'])
    col_consumo = detectar_columna(df_tipos, ['consumo anual', 'anual'])
    col_fuente = detectar_columna(df_tipos, ['fuente', 'ejemplo'])

    if not all([col_tipo, col_consumo]):
        st.error("⚠️ No se detectaron las columnas esperadas ('Tipo de Centro de Datos' y 'Consumo Anual Aproximado').")
        st.stop()

    # -----------------------
    # INTERFAZ DE COMPARACIÓN
    # -----------------------
    tipos = sorted(df_tipos[col_tipo].dropna().unique())
    regiones_sel = st.multiselect("🌍 Seleccionar regiones para comparar", regions, default=regions[:3])
    tipos_sel = st.multiselect("🏭 Seleccionar tipos de Centro de Datos", tipos, default=tipos[:2])

    if st.button("📊 Comparar impacto"):
        comparacion = []

        for region in regiones_sel:
            agua_disp = df[df['_region_norm'] == region]['_disp_l'].sum()
            for tipo in tipos_sel:
                fila = df_tipos[df_tipos[col_tipo] == tipo].iloc[0]
                consumo_anual = fila[col_consumo]

                # Intentar convertir texto como “3 – 7 millones” o “~342.000” a número
                def convertir_a_num(valor):
                    if pd.isna(valor):
                        return None
                    texto = str(valor).replace('~', '').replace('.', '').replace(',', '').lower()
                    if 'mill' in texto:
                        # Si dice “3 – 7 millones”, tomar promedio
                        nums = [float(x) for x in texto.replace('millones', '').replace('millón', '').split('–') if x.strip().isdigit()]
                        if len(nums) == 2:
                            return sum(nums) / 2 * 1_000_000
                        elif len(nums) == 1:
                            return nums[0] * 1_000_000
                        else:
                            return None
                    # Si solo es número
                    try:
                        return float(texto)
                    except:
                        return None

                consumo_real = convertir_a_num(consumo_anual)
                if not consumo_real:
                    continue

                pct_impacto = (consumo_real / agua_disp) * 100 if agua_disp > 0 else None

                comparacion.append({
                    "Región": region,
                    "Tipo de CD": tipo,
                    "Agua disponible (L)": agua_disp,
                    "Consumo anual estimado (L)": consumo_real,
                    "Impacto (%)": pct_impacto,
                    "Fuente": fila[col_fuente] if col_fuente else ""
                })

        # -----------------------
        # RESULTADOS
        # -----------------------
        if comparacion:
            df_comp = pd.DataFrame(comparacion)

            st.subheader("📋 Tabla comparativa")
            st.dataframe(df_comp.style.format({
                "Agua disponible (L)": "{:,.0f}",
                "Consumo anual estimado (L)": "{:,.0f}",
                "Impacto (%)": "{:.2f}"
            }))

            # Gráfico de barras comparativas
            fig, ax = plt.subplots(figsize=(8,4))
            for tipo in tipos_sel:
                df_temp = df_comp[df_comp["Tipo de CD"] == tipo]
                ax.bar(df_temp["Región"], df_temp["Impacto (%)"], label=tipo, alpha=0.7)
            ax.set_title("Impacto hídrico anual por tipo de Centro de Datos")
            ax.set_ylabel("Impacto (%) sobre agua disponible")
            ax.legend()
            st.pyplot(fig)

            # -----------------------
            # INTERPRETACIÓN AUTOMÁTICA
            # -----------------------
            st.markdown("### 💬 Interpretación del impacto")
            for _, row in df_comp.iterrows():
                region = row["Región"]
                tipo = row["Tipo de CD"]
                impacto = row["Impacto (%)"]
                if impacto < 1:
                    nivel = "muy bajo"
                elif impacto < 5:
                    nivel = "moderado"
                elif impacto < 15:
                    nivel = "alto"
                else:
                    nivel = "crítico"

                st.write(f"- En **{region}**, un centro de tipo **{tipo}** representaría un impacto **{nivel}** ({impacto:.2f}% del agua disponible).")

        else:
            st.warning("No se pudieron calcular los impactos; revisa los valores de consumo en el archivo.")

else:
    st.info("💾 Subí el archivo con los tipos de Centro de Datos (por ejemplo: 'refrigeración - Hoja 1.csv') para habilitar la comparación.")





# Área para exportar o simular (placeholder para futuras funcionalidades)
st.markdown("---")
st.subheader("Herramientas adicionales (futuro)")
st.write("Aquí se podrán añadir simulaciones, exportación de resultados (CSV/PDF), escenarios con múltiples centros de datos, etc.")


# -----------------------
# Panel de herramientas futuras
# -----------------------
st.markdown("---")
st.subheader("🔧 Herramientas adicionales (en desarrollo)")

st.markdown("""
💡 **AquaMind** evolucionará con nuevas funcionalidades que potenciarán el análisis hídrico regional.  
Estas herramientas permitirán tomar decisiones basadas en datos reales, simulaciones y escenarios comparativos.
""")

# Crear una interfaz tipo “tarjetas” con tres columnas
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📈 Simulaciones hídricas")

    st.write("Ajusta los parámetros y observa cómo varía la disponibilidad de agua.")

    # Selección de región
    region_sim = st.selectbox("Seleccionar región para simular", regions)

    # Parámetros del escenario
    num_centros = st.slider("Cantidad de centros de datos", 0, 10, 1)
    cambio_lluvia = st.slider("Cambio climático (% de lluvia anual)", -50, 50, 0)
    ahorro_agua = st.slider("Eficiencia de uso o ahorro (%)", 0, 50, 10)

    if st.button("🔮 Ejecutar simulación"):
        df_sel = df[df['_region_norm'] == region_sim].iloc[0]
        agua_disp = df_sel['_disp_l']

        # Cálculo de impacto
        consumo_cd_total = num_centros * CD_CONS_ANUAL_L
        ajuste_lluvia = 1 + (cambio_lluvia / 100)
        ahorro_factor = 1 - (ahorro_agua / 100)

        agua_final = (agua_disp * ajuste_lluvia - consumo_cd_total) * ahorro_factor
        variacion_pct = ((agua_final - agua_disp) / agua_disp) * 100

        st.metric("Variación de agua disponible", f"{variacion_pct:.2f}%")
        st.write(f"**Agua disponible original:** {agua_disp:,.0f} L")
        st.write(f"**Agua simulada final:** {agua_final:,.0f} L")

        # Gráfico de comparación
        fig, ax = plt.subplots()
        ax.bar(["Actual", "Simulado"], [agua_disp, max(0, agua_final)], color=["#1f77b4", "#2ca02c"])
        ax.set_ylabel("Litros disponibles")
        ax.set_title(f"Escenario simulado en {region_sim}")
        st.pyplot(fig)

with col2:
    st.markdown("### 💾 Exportación de resultados")
    st.write("""
    Guarda los resultados y visualizaciones en:
    - Archivos **CSV** para análisis externos  
    - Reportes **PDF** automáticos con gráficos  
    - Integración con hojas de cálculo colaborativas  
    """)
    st.button("📤 Exportar resultados (próximamente)", disabled=True)

with col3:
    st.markdown("### 🧠 Escenarios de planificación")
    st.write("""
    Compara el impacto de múltiples regiones:
    - Diferentes provincias o cuencas  
    - Capacidad de infraestructura hídrica  
    - Estrés hídrico promedio y total  
    """)
    st.button("🌍 Comparar regiones (próximamente)", disabled=True)

st.markdown("---")
st.info("💬 Sugerencia: puedes proponer nuevas herramientas para tu caso de estudio, como análisis temporal, visualización 3D o conexión con bases de datos de recursos naturales.")

# Footer con créditos
st.markdown("---")
st.markdown("Desarrollado con ❤️ por **AquaMind (prototipo)** — Código modular y comentado. Puedes adaptar fácilmente columnas, umbrales y cálculos.")

