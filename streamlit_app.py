
"""
Streamlit Agritech Dashboard - Prototipo
Archivo: streamlit_agritech_dashboard.py

Instrucciones de uso:
1) Crear un entorno virtual (recomendado) y activar.
2) Instalar dependencias:
   pip install streamlit pandas geopandas pydeck plotly openpyxl
3) Ejecutar:
   streamlit run streamlit_agritech_dashboard.py

Descripción:
- Centro: mapa del predio (pydeck)
- Inferior: condiciones climáticas actuales y serie temporal
- Lado derecho: panel con índices seleccionables (NDVI, NDWI, SAVI, NDRE)
- Top: selector de mapas guardados + alertas / recomendaciones desplegables
- Selector de fechas por índice, gráficos interactivos ampliables (plotly)
- Exportar reportes (.xlsx) con indices seleccionados y tablas
- Placeholder para carga de GeoJSON/parcelas y series temporales

Nota: Este prototipo usa datos de ejemplo generados aleatoriamente cuando no se
proveen archivos. Reemplaza las funciones `load_parcels()` y `load_timeseries()`
por conexiones a tus fuentes (GEE, AgroDataCube, S3, PostGIS, etc.).
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import pydeck as pdk
import plotly.express as px
import io
import geopandas as gpd
from shapely.geometry import Point, Polygon

# ---------------------- Helpers / Data loaders ----------------------

def generate_sample_parcels(n=5):
    # Genera polígonos cuadrados ficticios georreferenciados cerca de Rancagua
    base_lat, base_lon = -34.170, -70.740
    parcels = []
    for i in range(n):
        lon = base_lon + (i % 4) * 0.002
        lat = base_lat + (i // 4) * 0.002
        poly = Polygon([
            (lon, lat), (lon + 0.0015, lat), (lon + 0.0015, lat + 0.0015), (lon, lat + 0.0015)
        ])
        parcels.append({'id': f'P{i+1}', 'geometry': poly})
    gdf = gpd.GeoDataFrame(parcels, crs='EPSG:4326')
    return gdf


def generate_sample_timeseries(parcels_gdf, start_date, end_date):
    dates = pd.date_range(start_date, end_date, freq='7D')
    rows = []
    for _, r in parcels_gdf.iterrows():
        for d in dates:
            ndvi = np.clip(0.4 + 0.4 * np.sin((d.timetuple().tm_yday / 365.0) * 2 * np.pi) + np.random.normal(0, 0.03), 0, 1)
            ndwi = np.clip(0.2 + np.random.normal(0, 0.05), -1, 1)
            savi = np.clip(0.35 + np.random.normal(0, 0.03), 0, 1)
            ndre = np.clip(0.3 + np.random.normal(0, 0.04), 0, 1)
            tmax = 25 + 6 * np.sin((d.timetuple().tm_yday / 365.0) * 2 * np.pi) + np.random.normal(0,1)
            tmin = 8 + 4 * np.sin((d.timetuple().tm_yday / 365.0) * 2 * np.pi) + np.random.normal(0,1)
            precip = max(0, np.random.exponential(2.0) - 1.0)
            rows.append({
                'parcel_id': r['id'], 'date': d, 'NDVI': ndvi, 'NDWI': ndwi, 'SAVI': savi, 'NDRE': ndre,
                'Tmax': tmax, 'Tmin': tmin, 'Precip': precip
            })
    return pd.DataFrame(rows)


def load_parcels(uploaded_file):
    # Si el usuario sube un GeoJSON/GeoPackage, leerla. Si no, generar MUESTRA
    if uploaded_file is None:
        return generate_sample_parcels(6)
    try:
        gdf = gpd.read_file(uploaded_file)
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        return gdf
    except Exception as e:
        st.error(f"Error leyendo el archivo de parcelas: {e}")
        return generate_sample_parcels(6)


def load_timeseries(uploaded_csv, parcels_gdf, start, end):
    if uploaded_csv is None:
        return generate_sample_timeseries(parcels_gdf, start, end)
    try:
        df = pd.read_csv(uploaded_csv, parse_dates=['date'])
        return df
    except Exception as e:
        st.error(f"Error leyendo CSV de series temporales: {e}")
        return generate_sample_timeseries(parcels_gdf, start, end)


# ---------------------- Utility calculations ----------------------

def compute_gdd(df, tbase=10.0):
    # df must have 'Tmax' and 'Tmin' and 'date' and 'parcel_id'
    df = df.copy()
    df['Tmean'] = (df['Tmax'] + df['Tmin']) / 2.0
    df['GDD'] = (df['Tmean'] - tbase).clip(lower=0)
    return df


def aggregate_indices(df, parcel_id, start_date, end_date):
    sub = df[(df['parcel_id'] == parcel_id) & (df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    if sub.empty:
        return None
    # iNDVI by trapezoid (approx): sum of mean * delta days
    sub = sub.sort_values('date')
    sub['ndvi_next'] = sub['NDVI'].shift(-1)
    sub['date_next'] = sub['date'].shift(-1)
    sub = sub.dropna()
    sub['delta_days'] = (sub['date_next'] - sub['date']).dt.days
    sub['trap_area'] = ((sub['NDVI'] + sub['ndvi_next']) / 2.0) * sub['delta_days']
    iNDVI = sub['trap_area'].sum()
    stats = {
        'NDVI_mean': sub['NDVI'].mean(), 'NDVI_max': sub['NDVI'].max(), 'iNDVI': iNDVI,
        'NDWI_mean': sub['NDWI'].mean(), 'SAVI_mean': sub['SAVI'].mean(), 'NDRE_mean': sub['NDRE'].mean(),
        'GDD_sum': sub['GDD'].sum(), 'Precip_sum': sub['Precip'].sum()
    }
    return stats


def estimate_yield_simple(iNDVI, a=-5.0, b=0.5):
    # Modelo genérico de ejemplo; calibrar con datos locales
    return a + b * iNDVI


def chill_hours(df, thresh=7.0):
    return (df['Tmin'] < thresh).groupby(df['parcel_id']).sum()


# ---------------------- Streamlit layout ----------------------

st.set_page_config(layout='wide', page_title='Agritech Dashboard MVP')
st.title('Agritech - Prototipo Dashboard')

# Top: guardar / cargar mapas y alertas
with st.sidebar:
    st.header('Controles')
    uploaded_geo = st.file_uploader('Sube GeoJSON/GeoPackage de parcelas (opcional)', type=['geojson','gpkg','zip','shp'])
    uploaded_ts = st.file_uploader('Sube CSV series (cols: parcel_id,date,NDVI,NDWI,SAVI,NDRE,Tmax,Tmin,Precip) (opcional)', type=['csv'])
    map_saved = st.selectbox('Mapas guardados', ['Predio_actual', 'Mapa_historico_2024'])
    st.markdown('---')
    st.checkbox('Mostrar etiquetas en mapa', value=True, key='map_labels')
    st.checkbox('Interactividad mapas (pan/zoom)', value=True, key='map_interact')
    st.markdown('---')
    st.header('Exportar')
    report_name = st.text_input('Nombre de reporte', 'reporte_preds')
    if st.button('Generar reporte (.xlsx)'):
        st.session_state['generate_report'] = True

# cargar datos
parcels_gdf = load_parcels(uploaded_geo)
# centro del mapa: bbox
bbox = parcels_gdf.total_bounds if not parcels_gdf.empty else (-70.74,-34.17,-70.72,-34.168)
center_lon = (bbox[0] + bbox[2]) / 2.0
center_lat = (bbox[1] + bbox[3]) / 2.0

# Fecha rango global
col1, col2 = st.columns([1,3])
with col1:
    st.subheader('Fechas')
    today = datetime.date.today()
    default_start = today - datetime.timedelta(days=120)
    dt_start = st.date_input('Fecha inicio', default_start)
    dt_end = st.date_input('Fecha fin', today)

# cargar series
timeseries = load_timeseries(uploaded_ts, parcels_gdf, pd.Timestamp(dt_start), pd.Timestamp(dt_end))
# add GDD
timeseries = compute_gdd(timeseries, tbase=10.0)

# panel principal: mapa
with col2:
    st.subheader('Mapa del predio')
    # Convert parcels to GeoJSON for pydeck
    parcels_geojson = parcels_gdf.__geo_interface__
    # Generate random values per parcel for choropleth (placeholder: NDVI mean)
    ndvi_means = {}
    for pid in parcels_gdf['id']:
        sub = timeseries[timeseries['parcel_id'] == pid]
        ndvi_means[pid] = round(sub['NDVI'].mean() if not sub.empty else np.nan, 3)
    parcels_gdf['ndvi_mean'] = parcels_gdf['id'].map(ndvi_means)

    # pydeck layer
    polygon_layer = pdk.Layer(
        "GeoJsonLayer",
        parcels_gdf.__geo_interface__,
        pickable=True,
        stroked=True,
        filled=True,
        extruded=False,
        get_fill_color='[255*(1 - properties.ndvi_mean), 120, 50 + 200*properties.ndvi_mean, 150]',
        get_line_color=[0,0,0],
        auto_highlight=True
    )
    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=14, pitch=0)
    r = pdk.Deck(layers=[polygon_layer], initial_view_state=view_state, tooltip={"text":"{properties.id} - NDVI: {properties.ndvi_mean}"})
    st.pydeck_chart(r)

# Right column: indices y controles
right_col = st.columns([1])[0]
with right_col:
    st.subheader('Índices por parcela')
    parcel_selected = st.selectbox('Seleccionar parcela', parcels_gdf['id'].tolist())
    indices = st.multiselect('Seleccionar índices', ['NDVI','NDWI','SAVI','NDRE'], default=['NDVI','NDWI'])
    date_range = st.date_input('Rango de fechas (índices)', [dt_start, dt_end])

    # calcular agregados
    stats = aggregate_indices(timeseries, parcel_selected, pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]))
    if stats is not None:
        st.metric('iNDVI', f"{stats['iNDVI']:.2f}")
        st.metric('NDVI max', f"{stats['NDVI_max']:.2f}")
        st.metric('GDD (sum)', f"{stats['GDD_sum']:.1f}")
        st.metric('Precip (sum mm)', f"{stats['Precip_sum']:.1f}")

    # Time series chart (plotly)
    subts = timeseries[(timeseries['parcel_id'] == parcel_selected) & (timeseries['date'] >= pd.Timestamp(date_range[0])) & (timeseries['date'] <= pd.Timestamp(date_range[1]))]
    if not subts.empty:
        fig = px.line(subts, x='date', y=indices, title=f"Series indices - {parcel_selected}")
        fig.update_layout(autosize=True)
        st.plotly_chart(fig, use_container_width=True)
        # full screen (abrir en nueva pestaña): Streamlit no tiene fullscreen nativo, pero se puede abrir gráfico en nueva pestaña con link o permitir expandir el chart
        if st.button('Ver gráfico a pantalla completa'):
            st.write('Expande el gráfico usando el icono de expandir en la esquina del gráfico')

    # boton para exportar tabla de indices de la parcela
    if st.button('Exportar indices (.xlsx)'):
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine='openpyxl') as writer:
            subts.to_excel(writer, index=False, sheet_name=f'{parcel_selected}_indices')
        st.download_button(label='Descargar Excel', data=out.getvalue(), file_name=f'{parcel_selected}_indices.xlsx')

# Bottom: clima y condiciones
st.markdown('---')
col_a, col_b, col_c = st.columns([1,1,1])
with col_a:
    st.subheader('Condiciones climáticas (resumen)')
    # Agregar resumen global
    g = timeseries.groupby('date').agg({'Tmax':'mean','Tmin':'mean','Precip':'sum'}).reset_index()
    figc = px.line(g, x='date', y=['Tmax','Tmin'], title='Temperaturas medias')
    st.plotly_chart(figc, use_container_width=True)
with col_b:
    st.subheader('Horas frío y GDD')
    # horas frio approximated as count Tmin < 7
    hf = timeseries.groupby('parcel_id').apply(lambda df: (df['Tmin']<7).sum()).rename('ColdHours').reset_index()
    gddsum = timeseries.groupby('parcel_id')['GDD'].sum().reset_index()
    dfsum = hf.merge(gddsum, left_on='parcel_id', right_on='parcel_id')
    st.dataframe(dfsum)
with col_c:
    st.subheader('Estado enológico (estimado)')
    # Ejemplo simple: map GDD to estado
    parcel_gdd = timeseries.groupby('parcel_id')['GDD'].sum().reset_index()
    selected = parcel_gdd[parcel_gdd['parcel_id'] == parcel_selected]['GDD'].values[0]
    if selected < 200:
        phen = 'Brote - desarrollo' 
    elif selected < 500:
        phen = 'Floración - cuajado'
    elif selected < 900:
        phen = 'Llenado de baya'
    else:
        phen = 'Madurez'
    st.markdown(f"**Parcela {parcel_selected}:** {phen} (GDD={selected:.1f})")

# Alertas y recomendaciones desplegables
with st.expander('Alertas y Recomendaciones'):
    st.subheader('Alertas')
    # regla simple: NDVI bajo o NDWI alto
    alert_list = []
    for pid in parcels_gdf['id']:
        recent = timeseries[(timeseries['parcel_id']==pid) & (timeseries['date'] >= pd.Timestamp(dt_end) - pd.Timedelta(days=30))]
        if recent.empty:
            continue
        if recent['NDVI'].mean() < 0.45:
            alert_list.append((pid, 'NDVI bajo (posible estrés)'))
        if recent['NDWI'].mean() > 0.35:
            alert_list.append((pid, 'NDWI alto (agua superficial)'))
    if alert_list:
        for a in alert_list:
            st.warning(f"Parcela {a[0]}: {a[1]}")
    else:
        st.success('No hay alertas')

    st.subheader('Recomendaciones')
    # recomendaciones simples
    for pid in parcels_gdf['id']:
        st.write(f"Parcela {pid}:")
        st.write('- Revisar estado de riego si NDVI < 0.5')
        st.write('- Si NDWI alto, verificar en terreno en próximas 48 h')

# Report generation trigger
if st.session_state.get('generate_report', False):
    st.info('Generando reporte...')
    # Construir un reporte simple xlsx que contenga agregados por parcela
    out = io.BytesIO()
    summary_rows = []
    for pid in parcels_gdf['id']:
        s = aggregate_indices(timeseries, pid, pd.Timestamp(dt_start), pd.Timestamp(dt_end))
        if s is None:
            continue
        summary_rows.append({
            'parcel_id': pid, 'iNDVI': s['iNDVI'], 'NDVI_max': s['NDVI_max'], 'GDD_sum': s['GDD_sum'], 'Precip_sum': s['Precip_sum']
        })
    summary_df = pd.DataFrame(summary_rows)
    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        summary_df.to_excel(writer, index=False, sheet_name='Resumen')
        timeseries.to_excel(writer, index=False, sheet_name='Series')
    st.download_button('Descargar reporte .xlsx', data=out.getvalue(), file_name=f'{report_name}.xlsx')
    st.session_state['generate_report'] = False

st.markdown('---')
st.caption('Prototipo Agritech Dashboard — Reemplaza datos de ejemplo por tus datos reales (GeoJSON / CSV)')
