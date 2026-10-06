# -*- coding: utf-8 -*-
"""
OSM Point & Hex Analytics — Streamlit app
-----------------------------------------
1. Пользователь загружает файл (xlsx/csv) со списком адресов.
2. Адреса геокодируются (Nominatim) или берутся готовые координаты.
3. Точки отрисовываются на OSM-карте.
4. Включается режим гексов (H3, разрешения 7–9).
5. Радиокнопки выбирают метрику гексов:
   - Плотность населения (оценка по жилому фонду OSM)
   - Объём жилого фонда (м² надземной площади, OSM)
   - Индекс спроса (платёжеспособность, композитный индекс)
   - Трафик (пеший/автомобильный, индекс по дорожной сети OSM)
   - Мед. учреждения (без аптек и стоматологий)

Запуск:  streamlit run app.py
"""

import time

import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from shapely.geometry import Polygon

import h3
import osmnx as ox

# --------------------------------------------------------------------------- #
# Настройки страницы и константы
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="OSM Hex Analytics", layout="wide")

ox.settings.use_cache = True
ox.settings.timeout = 180

NOMINATIM_USER_AGENT = "osm_hex_analytics_app"

LAT_COLS = ["широта", "lat", "latitude", "y"]
LON_COLS = ["долгота", "lon", "lng", "longitude", "long", "x"]
ADDR_COLS = ["адрес", "address", "addr", "адрес объекта"]
CITY_COLS = ["город", "city", "town"]

RESIDENTIAL_BUILDINGS = {
    "residential", "house", "apartments", "detached", "terrace",
    "dormitory", "bungalow", "static_caravan", "houseboat",
}
MEDICAL_AMENITY = {"hospital", "clinic", "doctors", "nursing_home", "midwife"}
MEDICAL_HEALTHCARE = {"hospital", "clinic", "centre", "health_center",
                      "rehabilitation", "hospice", "blood_donation"}
MEDICAL_EXCLUDE_HEALTHCARE = {"pharmacy", "dentist", "dental", "alternative",
                              "physiotherapist", "optometrist", "podiatrist"}

CAR_HIGHWAYS = {"motorway", "trunk", "primary", "secondary", "tertiary",
                "unclassified", "residential", "service", "living_street",
                "motorway_link", "trunk_link", "primary_link",
                "secondary_link", "tertiary_link"}
PED_HIGHWAYS = {"footway", "pedestrian", "path", "steps", "track", "cycleway"}

METRICS = {
    "population": "Плотность населения (оценка OSM)",
    "housing": "Объём жилого фонда, м²",
    "demand": "Индекс спроса (платёжеспособность)",
    "traffic": "Трафик (пеший/автомобильный)",
    "medical": "Мед. учреждения (без аптек и стоматологий)",
}
MAX_HEXES = 80  # защита от перегрузки Overpass API


# --------------------------------------------------------------------------- #
# Утилиты загрузки и распознавания колонок
# --------------------------------------------------------------------------- #
def load_uploaded(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    if name.endswith(".csv"):
        return pd.read_csv(file)
    raise ValueError("Поддерживаются файлы .xlsx, .xls, .csv")


def norm(s: str) -> str:
    return str(s).strip().lower().replace("\ufeff", "")


def find_col(df_cols, candidates):
    lowered = {norm(c): c for c in df_cols}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


# --------------------------------------------------------------------------- #
# Геокодирование (с кэшем Streamlit)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, ttl=7 * 24 * 3600)
def geocode_one(query: str):
    """Возвращает (lat, lon) или None. 1.1 c задержка — лимит Nominatim."""
    time.sleep(1.1)
    geolocator = Nominatim(user_agent=NOMINATIM_USER_AGENT, timeout=15)
    try:
        loc = geolocator.geocode(query, exactly_one=True, language="ru")
    except (GeocoderTimedOut, GeocoderUnavailable):
        return None
    if loc is None:
        return None
    return (loc.latitude, loc.longitude)


# --------------------------------------------------------------------------- #
# OSM-данные по гексу (Overpass, с кэшем)
# --------------------------------------------------------------------------- #
def _col(gdf, name):
    return gdf[name] if name in gdf.columns else pd.Series(
        index=gdf.index, dtype=object)


@st.cache_data(show_spinner=False, ttl=7 * 24 * 3600)
def fetch_hex_features(cell: str) -> dict:
    """Собирает сырые признаки по полигону гекса из OSM."""
    empty = {"housing_area": 0.0, "population_est": 0.0,
             "traffic_index": 0.0, "car_km": 0.0, "ped_km": 0.0,
             "medical_count": 0, "poi_count": 0}
    boundary = h3.cell_to_boundary(cell)  # [(lat, lng), ...]
    poly = Polygon(boundary)

    tags = {"building": True, "highway": True, "amenity": True,
            "shop": True, "office": True, "tourism": True,
            "leisure": True, "healthcare": True}
    try:
        gdf = ox.features.features_from_polygon(poly, tags)
    except Exception:
        return empty
    if gdf is None or gdf.empty:
        return empty

    gdf = gdf.reset_index()
    out = dict(empty)

    # --- проекция в метры для площадей/длин -------------------------------
    try:
        gdf_m = gdf.to_crs(gdf.estimate_utm_crs())
    except Exception:
        gdf_m = gdf

    # --- жилой фон ----------------------------------------------------------
    b_mask = _col(gdf, "building").notna() & gdf.geometry.geom_type.isin(
        ["Polygon", "MultiPolygon"])
    if b_mask.any():
        b = gdf[b_mask]
        bm = gdf_m[b_mask]
        is_res = _col(b, "building").astype(str).str.lower().isin(
            RESIDENTIAL_BUILDINGS)
        levels = pd.to_numeric(_col(b, "building:levels"),
                               errors="coerce").fillna(1).clip(1, 40)
        area_m2 = bm.geometry.area.to_numpy() * levels.to_numpy()
        out["housing_area"] = float(area_m2[is_res.to_numpy()].sum())
        # грубая оценка населения: 25 м² жилой площади на человека
        out["population_est"] = out["housing_area"] / 25.0

    # --- дорожная сеть / трафик --------------------------------------------
    h_mask = _col(gdf, "highway").notna() & gdf.geometry.geom_type.isin(
        ["LineString", "MultiLineString"])
    if h_mask.any():
        hw = _col(gdf, "highway").astype(str).str.lower()
        car = h_mask & hw.isin(CAR_HIGHWAYS)
        ped = h_mask & hw.isin(PED_HIGHWAYS)
        car_km = float(gdf_m[car].geometry.length.sum()) / 1000.0
        ped_km = float(gdf_m[ped].geometry.length.sum()) / 1000.0
        out["car_km"], out["ped_km"] = car_km, ped_km
        out["traffic_index"] = 3.0 * car_km + 1.5 * ped_km

    # --- POI для индекса спроса ---------------------------------------------
    poi_cols = ["shop", "office", "amenity", "tourism", "leisure"]
    poi_mask = pd.Series(False, index=gdf.index)
    for c in poi_cols:
        poi_mask |= _col(gdf, c).notna()
    # исключаем сами здания из подсчёта POI
    poi_mask &= ~_col(gdf, "building").notna()
    out["poi_count"] = int(poi_mask.sum())

    # --- медицина (без аптек и стоматологий) --------------------------------
    med_mask = pd.Series(False, index=gdf.index)
    med_mask |= _col(gdf, "amenity").astype(str).str.lower().isin(
        MEDICAL_AMENITY)
    hc = _col(gdf, "healthcare").astype(str).str.lower()
    med_mask |= hc.isin(MEDICAL_HEALTHCARE)
    med_mask &= ~hc.isin(MEDICAL_EXCLUDE_HEALTHCARE)
    out["medical_count"] = int(med_mask.sum())

    return out


def minmax(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    rng = s.max() - s.min()
    if rng <= 0:
        return pd.Series(0.5, index=s.index)
    return (s - s.min()) / rng


def metric_value(df: pd.DataFrame, key: str) -> pd.Series:
    if key == "population":
        return df["population_est"]
    if key == "housing":
        return df["housing_area"]
    if key == "traffic":
        return df["traffic_index"]
    if key == "medical":
        return df["medical_count"].astype(float)
    if key == "demand":
        return (0.45 * minmax(df["housing_area"])
                + 0.35 * minmax(df["poi_count"])
                + 0.20 * minmax(df["medical_count"]))
    raise ValueError(key)


# --------------------------------------------------------------------------- #
# Отрисовка карты
# --------------------------------------------------------------------------- #
def lerp_color(t: float) -> str:
    """t in [0,1] -> цвет от красного к зелёному."""
    t = max(0.0, min(1.0, t))
    r = int(255 * (1 - t))
    g = int(200 * t)
    return f"#{r:02x}{g:02x}40"


def build_map(points: pd.DataFrame, hex_df: pd.DataFrame | None,
              metric_key: str | None, show_points: bool,
              center, zoom: int):
    fmap = folium.Map(location=center, zoom_start=zoom,
                      tiles="OpenStreetMap", control_scale=True)
    # CARTO basemaps (positron/dark_matter) больше недоступны без API-ключа
    # (https://carto.com/basemaps/apikey) — используем тайлы без ключа.
    folium.TileLayer("OpenTopoMap", name="Топографическая",
                     show=False).add_to(fmap)

    if show_points and not points.empty:
        cluster = MarkerCluster(name="Точки").add_to(fmap)
        for _, row in points.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=6, color="#1f77b4", fill=True, fill_opacity=0.8,
                popup=folium.Popup(str(row.get("label", ""))[:300],
                                   max_width=300),
            ).add_to(cluster)

    if hex_df is not None and not hex_df.empty:
        vmin, vmax = hex_df["value"].min(), hex_df["value"].max()
        for _, row in hex_df.iterrows():
            boundary = h3.cell_to_boundary(row["cell"])
            coords = [(lat, lng) for lat, lng in boundary]
            t = (row["value"] - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            color = lerp_color(t)
            tip = (f"Гекс {row['cell']}<br>"
                   f"{METRICS[metric_key]}:<br><b>{row['value']:.1f}</b><br>"
                   f"Точек в гексе: {int(row['n_points'])}")
            folium.Polygon(locations=coords, color="#333333", weight=1,
                           fill_color=color, fill_opacity=0.55,
                           tooltip=folium.Tooltip(tip, sticky=True)
                           ).add_to(fmap)
        # легенда
        legend = f"""<div style="position: fixed; bottom: 30px; left: 30px;
        z-index: 9999; background: rgba(255,255,255,.92); padding: 10px;
        border: 1px solid #999; border-radius: 6px; font-size: 13px;
        font-family: sans-serif;">
        <b>{METRICS[metric_key]}</b><br>
        <span style="color:#ff2840">■</span> {vmin:,.0f}
        &nbsp;—&nbsp;
        <span style="color:#32c840">■</span> {vmax:,.0f}</div>"""
        fmap.get_root().html.add_child(folium.Element(legend))

    folium.LayerControl().add_to(fmap)
    return fmap


# --------------------------------------------------------------------------- #
# Интерфейс
# --------------------------------------------------------------------------- #
st.title("📍 OSM Point & Hex Analytics")
st.caption("Загрузите список адресов → точки на OSM-карте → гексы H3 "
           "с аналитикой по данным OpenStreetMap.")

with st.sidebar:
    st.header("1. Данные")
    uploaded = st.file_uploader("Файл со списком адресов (.xlsx / .csv)",
                                type=["xlsx", "xls", "csv"])
    st.header("2. Гексы")
    hex_mode = st.toggle("Режим гексов", value=False)
    res = st.slider("Разрешение H3", min_value=7, max_value=9, value=8,
                    help="H7 ≈ 52 км², H8 ≈ 7,4 км², H9 ≈ 1 км² (средняя площадь)")
    metric_key = st.radio("Метрика гексов", list(METRICS.keys()),
                          format_func=lambda k: METRICS[k])
    show_points = st.checkbox("Показывать точки поверх гексов", value=True)
    st.divider()
    if st.button("🗑 Очистить кэш OSM / геокодирования"):
        st.cache_data.clear()
        st.success("Кэш очищен")

if uploaded is None:
    st.info("Загрузите файл. Ожидаются колонки вроде: "
            "**Адрес**, **Город**, **Широта**, **Долгота** "
            "(координаты не обязательны — адрес можно геокодировать).")
    st.stop()

try:
    raw = load_uploaded(uploaded)
except Exception as e:
    st.error(f"Не удалось прочитать файл: {e}")
    st.stop()

df = raw.copy()
df.columns = [norm(c) for c in df.columns]
c_lat = find_col(df.columns, LAT_COLS)
c_lon = find_col(df.columns, LON_COLS)
c_addr = find_col(df.columns, ADDR_COLS)
c_city = find_col(df.columns, CITY_COLS)

st.subheader("Распознанные колонки")
st.write(f"**Адрес:** {c_addr or '—'} | **Город:** {c_city or '—'} | "
         f"**Широта:** {c_lat or '—'} | **Долгота:** {c_lon or '—'}")

# --- координаты из файла или геокодирование адресов -------------------------
points = pd.DataFrame(columns=["lat", "lon", "label"])

if c_lat and c_lon:
    rows = []
    for i, r in df.iterrows():
        try:
            lat, lon = float(r[c_lat]), float(r[c_lon])
        except (TypeError, ValueError):
            continue
        rows.append({"lat": lat, "lon": lon,
                     "label": r[c_addr] if c_addr else f"Строка {i}"})
    points = pd.DataFrame(rows)
    st.info("Координаты взяты из колонок файла.")
elif c_addr:
    rows = []
    st.warning(f"Геокодируем {len(df)} адресов через Nominatim "
               f"(≈ {int(len(df) * 1.2)} сек). Пожалуйста, подождите…")
    bar = st.progress(0)
    for k, (_, r) in enumerate(df.iterrows()):
        parts = []
        if pd.notna(r[c_addr]):
            parts.append(str(r[c_addr]))
        if c_city and pd.notna(r[c_city]):
            parts.append(str(r[c_city]))
        q = ", ".join(parts)
        geo = geocode_one(q) if q else None
        if geo:
            rows.append({"lat": geo[0], "lon": geo[1], "label": q})
        bar.progress((k + 1) / len(df))
    points = pd.DataFrame(rows)
if points.empty:
    st.error("Не удалось получить ни одной точки: проверьте колонки "
             "адреса/координат в файле.")
    st.stop()

points = points.drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)
st.success(f"Точек на карте: **{len(points)}** "
           f"(строк в файле: {len(df)})")

center = [points["lat"].mean(), points["lon"].mean()]
zoom = 11 if (points["lat"].max() - points["lat"].min()) > 0.5 else 13

# --- режим гексов -------------------------------------------------------------
hex_df = None
if hex_mode:
    cells = sorted({h3.latlng_to_cell(lat, lon, res)
                    for lat, lon in points[["lat", "lon"]].itertuples(index=False)})
    n_points = {c: 0 for c in cells}
    for lat, lon in points[["lat", "lon"]].itertuples(index=False):
        n_points[h3.latlng_to_cell(lat, lon, res)] += 1

    if len(cells) > MAX_HEXES:
        st.error(f"Слишком много гексов ({len(cells)} > {MAX_HEXES}) — "
                 "уменьшите разброс точек или понижайте разрешение.")
        st.stop()

    st.info(f"Гексов: **{len(cells)}** (H{res}). Запрашиваем данные OSM "
            "через Overpass — первый расчёт может занять несколько минут.")
    feats = {}
    bar = st.progress(0)
    for k, cell in enumerate(cells):
        feats[cell] = fetch_hex_features(cell)
        bar.progress((k + 1) / len(cells))

    fdf = pd.DataFrame(feats).T
    fdf.index.name = "cell"
    fdf = fdf.reset_index()
    fdf["n_points"] = fdf["cell"].map(n_points).fillna(0).astype(int)
    fdf["value"] = metric_value(fdf, metric_key)
    hex_df = fdf

fmap = build_map(points, hex_df, metric_key if hex_mode else None,
                 show_points, center, zoom)
st_folium(fmap, width=None, height=650, returned_objects=[])

if hex_mode and hex_df is not None:
    st.subheader("Данные по гексам")
    st.dataframe(hex_df.sort_values("value", ascending=False)
                 .style.format({"value": "{:,.1f}", "housing_area": "{:,.0f}",
                                "population_est": "{:,.0f}",
                                "traffic_index": "{:,.1f}"}),
                 use_container_width=True)
    csv = hex_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇ Скачать таблицу гексов (CSV)", csv,
                       "hexes.csv", "text/csv")
