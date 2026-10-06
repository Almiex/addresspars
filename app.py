# -*- coding: utf-8 -*-
"""
OSM Point & Hex Analytics — Streamlit app
-----------------------------------------
1. Пользователь загружает файл (xlsx/csv) со списком адресов.
2. Точки строятся на OSM-карте (координаты из файла или геокодирование).
3. Режим гексов: полигон города из Nominatim → покрытие ВСЕГО города сеткой H3
   (разрешения 7–9), метрики считаются по данным OSM (Overpass), ОДИН запрос
   на город с пространственным распределением по гексам.
4. Метрики (радиокнопки):
   - Плотность населения (жилфонд OSM / м² на человека / км² гекса)
   - Объём жилого фонда (м² надземной площади, OSM)
   - Индекс спроса (платёжеспособность, композитный)
   - Трафик (пеший/автомобильный, индекс по дорожной сети OSM)
   - Мед. учреждения (без аптек и стоматологий)

Запуск:  streamlit run app.py
"""

import json
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import urllib.parse
import urllib.request
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from shapely.geometry import Polygon, shape

import h3
import osmnx as ox

# --------------------------------------------------------------------------- #
# Настройки
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="OSM Hex Analytics", layout="wide")

ox.settings.use_cache = True
ox.settings.timeout = 300
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
MEDICAL_EXCLUDE = {"pharmacy", "dentist", "dental", "alternative",
                   "physiotherapist", "optometrist", "podiatrist"}

CAR_HIGHWAYS = {"motorway", "trunk", "primary", "secondary", "tertiary",
                "unclassified", "residential", "service", "living_street",
                "motorway_link", "trunk_link", "primary_link",
                "secondary_link", "tertiary_link"}
PED_HIGHWAYS = {"footway", "pedestrian", "path", "steps", "track", "cycleway"}
POI_COLS = ["shop", "office", "amenity", "tourism", "leisure"]

METRICS = {
    "population": "Плотность населения, чел/км²",
    "housing": "Объём жилого фонда, м²",
    "demand": "Индекс спроса (платёжеспособность)",
    "traffic": "Трафик (пеший/автомобильный)",
    "medical": "Мед. учреждения (без аптек и стоматологий)",
}
METRIC_FMT = {
    "population": "{:,.1f}", "housing": "{:,.0f}", "demand": "{:,.2f}",
    "traffic": "{:,.1f}", "medical": "{:,.0f}",
}
MAX_HEXES = 3000  # защита от перегрузки


# --------------------------------------------------------------------------- #
# Загрузка файла и распознавание колонок
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
# Геокодирование точек (Nominatim)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, ttl=7 * 24 * 3600)
def geocode_one(query: str):
    time.sleep(1.1)  # лимит Nominatim: max 1 запр/с
    geolocator = Nominatim(user_agent=NOMINATIM_USER_AGENT, timeout=15)
    try:
        loc = geolocator.geocode(query, exactly_one=True, language="ru")
    except (GeocoderTimedOut, GeocoderUnavailable):
        return None
    return None if loc is None else (loc.latitude, loc.longitude)


# --------------------------------------------------------------------------- #
# Полигон города (Nominatim, polygon_geojson)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, ttl=30 * 24 * 3600)
def get_city_polygon(city_query: str):
    params = urllib.parse.urlencode(
        {"q": city_query, "format": "jsonv2", "polygon_geojson": 1,
         "limit": 1, "accept-language": "ru"})
    req = urllib.request.Request(
        f"https://nominatim.openstreetmap.org/search?{params}",
        headers={"User-Agent": NOMINATIM_USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
    if not data:
        return None
    geom = shape(data[0]["geojson"])
    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        return None
    return geom, data[0].get("display_name", city_query)


# --------------------------------------------------------------------------- #
# OSM-данные по ВСЕМУ городу (один запрос Overpass)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, ttl=7 * 24 * 3600)
def fetch_city_osm(city_query: str):
    """Возвращает (polygon, gdf_wgs84, gdf_utm). Ошибки пробрасываются наверх."""
    poly_info = get_city_polygon(city_query)
    if poly_info is None:
        raise RuntimeError(f"Город «{city_query}» не найден в Nominatim.")
    geom, _ = poly_info
    tags = {"building": True, "highway": True, "amenity": True,
            "shop": True, "office": True, "tourism": True,
            "leisure": True, "healthcare": True}
    gdf = ox.features.features_from_polygon(geom, tags)
    gdf = gdf.reset_index()
    gdf_m = gdf.to_crs(gdf.estimate_utm_crs())
    return geom, gdf, gdf_m


def _col(gdf, name):
    return gdf[name] if name in gdf.columns else pd.Series(
        index=gdf.index, dtype=object)


def h3_cells_for_polygon(geom, res: int) -> list:
    """Покрытие ВСЕГО полигона сеткой H3."""
    cells = set()
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    for p in polys:
        coords = [[lng, lat] for lng, lat in p.exterior.coords]
        geojson = {"type": "Polygon", "coordinates": [coords]}
        cells.update(h3.polygon_to_cells(geojson, res))
    return sorted(cells)


# --------------------------------------------------------------------------- #
# Метрики по гексам (пространственное распределение городских данных)
# --------------------------------------------------------------------------- #
def compute_hex_metrics(gdf, gdf_m, cells) -> pd.DataFrame:
    utm_crs = gdf_m.crs
    b_mask = _col(gdf, "building").notna() & gdf.geometry.geom_type.isin(
        ["Polygon", "MultiPolygon"])
    buildings, buildings_m = gdf[b_mask], gdf_m[b_mask]
    is_res = _col(buildings, "building").astype(str).str.lower().isin(
        RESIDENTIAL_BUILDINGS)
    levels = pd.to_numeric(_col(buildings, "building:levels"),
                           errors="coerce").fillna(1).clip(1, 40)

    h_mask = _col(gdf, "highway").notna() & gdf.geometry.geom_type.isin(
        ["LineString", "MultiLineString"])
    roads, roads_m = gdf[h_mask], gdf_m[h_mask]
    hw = _col(roads, "highway").astype(str).str.lower()
    car_mask = hw.isin(CAR_HIGHWAYS)
    ped_mask = hw.isin(PED_HIGHWAYS)

    med_mask = pd.Series(False, index=gdf.index)
    med_mask |= _col(gdf, "amenity").astype(str).str.lower().isin(
        MEDICAL_AMENITY)
    hc = _col(gdf, "healthcare").astype(str).str.lower()
    med_mask |= hc.isin(MEDICAL_HEALTHCARE)
    med_mask &= ~hc.isin(MEDICAL_EXCLUDE)
    medical_pts = gdf[med_mask]

    poi_mask = pd.Series(False, index=gdf.index)
    for c in POI_COLS:
        poi_mask |= _col(gdf, c).notna()
    poi_mask &= ~_col(gdf, "building").notna()
    pois = gdf[poi_mask]
    pois = pois[~_col(pois, "highway").notna()]  # дороги не POI

    rows = []
    for cell in cells:
        boundary = h3.cell_to_boundary(cell)
        hpoly_wgs = Polygon(boundary)
        hm = gpd.GeoSeries([hpoly_wgs], crs="EPSG:4326").to_crs(
            utm_crs).iloc[0]

        # здания, пересекающие гекс
        bi = buildings_m[buildings_m.geometry.intersects(hm)]
        if len(bi):
            inter = bi.geometry.intersection(hm)
            area = inter.area.to_numpy() * levels.loc[bi.index].to_numpy()
            housing_area = float(area[is_res.loc[bi.index].to_numpy()].sum())
        else:
            housing_area = 0.0

        ri = roads_m[roads_m.geometry.intersects(hm)]
        car_km = ped_km = 0.0
        if len(ri):
            inter_len = ri.geometry.intersection(hm).length
            car_km = float(inter_len[car_mask.loc[ri.index]].sum()) / 1000
            ped_km = float(inter_len[ped_mask.loc[ri.index]].sum()) / 1000

        mi = medical_pts[medical_pts.geometry.intersects(hm)]
        pi = pois[pois.geometry.intersects(hm)]

        rows.append({
            "cell": cell,
            "housing_area": housing_area,
            "car_km": car_km, "ped_km": ped_km,
            "medical_count": int(len(mi)),
            "poi_count": int(len(pi)),
        })
    return pd.DataFrame(rows)


def minmax(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    rng = s.max() - s.min()
    return pd.Series(0.5, index=s.index) if rng <= 0 else (s - s.min()) / rng


def metric_value(df: pd.DataFrame, key: str, m2_per_person: int) -> pd.Series:
    if key == "population":
        return df["housing_area"] / m2_per_person / df["area_km2"]
    if key == "housing":
        return df["housing_area"]
    if key == "traffic":
        return 3.0 * df["car_km"] + 1.5 * df["ped_km"]
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
    t = max(0.0, min(1.0, t))
    return f"#{int(255 * (1 - t)):02x}{int(200 * t):02x}40"


def build_map(points, hex_df, metric_key, show_points, center, zoom):
    fmap = folium.Map(location=center, zoom_start=zoom,
                      tiles="OpenStreetMap", control_scale=True)
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
        fmt = METRIC_FMT[metric_key]
        vmin, vmax = hex_df["value"].min(), hex_df["value"].max()
        for _, row in hex_df.iterrows():
            coords = [(lat, lng) for lat, lng in h3.cell_to_boundary(row["cell"])]
            t = ((row["value"] - vmin) / (vmax - vmin)) if vmax > vmin else 0.5
            tip = (f"{METRICS[metric_key]}:<br><b>{fmt.format(row['value'])}</b>"
                   f"<br>Точек в гексе: {int(row['n_points'])}")
            folium.Polygon(locations=coords, color="#333333", weight=1,
                           fill_color=lerp_color(t), fill_opacity=0.55,
                           tooltip=folium.Tooltip(tip, sticky=True)
                           ).add_to(fmap)
        legend = f"""<div style="position: fixed; bottom: 30px; left: 30px;
        z-index: 9999; background: rgba(255,255,255,.92); padding: 10px;
        border: 1px solid #999; border-radius: 6px; font-size: 13px;
        font-family: sans-serif;">
        <b>{METRICS[metric_key]}</b><br>
        <span style="color:#ff2840">■</span> {fmt.format(vmin)}
        &nbsp;—&nbsp;
        <span style="color:#32c840">■</span> {fmt.format(vmax)}</div>"""
        fmap.get_root().html.add_child(folium.Element(legend))

    folium.LayerControl().add_to(fmap)
    return fmap


# --------------------------------------------------------------------------- #
# Интерфейс
# --------------------------------------------------------------------------- #
st.title("📍 OSM Point & Hex Analytics")
st.caption("Загрузите список адресов → точки на OSM-карте → покрытие ВСЕГО "
           "города гексами H3 с аналитикой по данным OpenStreetMap.")

with st.sidebar:
    st.header("1. Данные")
    uploaded = st.file_uploader("Файл со списком адресов (.xlsx / .csv)",
                                type=["xlsx", "xls", "csv"])
    st.header("2. Гексы")
    hex_mode = st.toggle("Режим гексов", value=False)
    res = st.slider("Разрешение H3", 7, 9, 8,
                    help="H7 ≈ 52 км², H8 ≈ 7,4 км², H9 ≈ 1 км² (средняя площадь)")
    metric_key = st.radio("Метрика гексов", list(METRICS.keys()),
                          format_func=lambda k: METRICS[k])
    m2_per_person = st.slider("м² жилой площади на 1 человека",
                              min_value=10, max_value=60, value=25,
                              help="Используется для оценки плотности населения")
    show_points = st.checkbox("Показывать точки поверх гексов", value=True)
    st.divider()
    if st.button("🗑 Очистить кэш OSM / геокодирования"):
        st.cache_data.clear()
        st.success("Кэш очищен")

if uploaded is None:
    st.info("Загрузите файл. Ожидаются колонки вроде: **Адрес**, **Город**, "
            "**Широта**, **Долгота** (координаты не обязательны).")
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

# --- определение города -------------------------------------------------------
cities = []
if c_city:
    cities = sorted({str(x).strip() for x in df[c_city].dropna().unique()
                     if str(x).strip()})
city_query = ", ".join(cities) if cities else ""

st.subheader("Распознанные колонки")
st.write(f"**Адрес:** {c_addr or '—'} | **Город:** {c_city or '—'} | "
         f"**Широта:** {c_lat or '—'} | **Долгота:** {c_lon or '—'}")

# --- точки --------------------------------------------------------------------
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
        geo = geocode_one(", ".join(parts))
        if geo:
            rows.append({"lat": geo[0], "lon": geo[1],
                         "label": ", ".join(parts)})
        bar.progress((k + 1) / len(df))
    points = pd.DataFrame(rows)
    if not points.empty and cities == [] and c_city is None:
        pass

points = points.dropna().drop_duplicates(subset=["lat", "lon"])
if points.empty:
    st.error("Не удалось получить ни одной точки: проверьте колонки "
             "адреса/координат в файле.")
    st.stop()

if not cities:
    manual = st.text_input("Не удалось определить город из файла. "
                           "Введите город для анализа гексов:", city_query)
    if manual.strip():
        cities = [manual.strip()]

st.success(f"Точек на карте: **{len(points)}** | Город анализа: "
           f"**{', '.join(cities) if cities else 'не определён'}**")

center = [points["lat"].mean(), points["lon"].mean()]
zoom = 11 if (points["lat"].max() - points["lat"].min()) > 0.5 else 13

# --- режим гексов -------------------------------------------------------------
hex_df = None
if hex_mode:
    if not cities:
        st.error("Не определён город — укажите его в поле выше.")
        st.stop()

    city_query = ", ".join(cities)
    with st.spinner(f"Получаем полигон города «{city_query}» и данные OSM "
                    "на весь город (один запрос Overpass, может занять "
                    "несколько минут)…"):
        try:
            geom, gdf, gdf_m = fetch_city_osm(city_query)
        except Exception as e:
            st.error(f"Ошибка загрузки данных OSM: `{type(e).__name__}: {e}`\n\n"
                     "Если это InsufficientResponseError — Overpass вернул пустой "
                     "ответ (перегруз/таймаут), попробуйте ещё раз или позже.")
            st.stop()

    with st.spinner(f"Строим сетку H3{res} на весь город…"):
        cells = h3_cells_for_polygon(geom, res)
    if len(cells) > MAX_HEXES:
        st.error(f"Слишком много гексов ({len(cells)} > {MAX_HEXES}). "
                 "Понизьте разрешение.")
        st.stop()
    st.info(f"Город: **{city_query}** | Гексов: **{len(cells)}** (H{res})")

    with st.spinner("Распределяем данные OSM по гексам…"):
        hex_df = compute_hex_metrics(gdf, gdf_m, cells)
    hex_df["area_km2"] = hex_df["cell"].map(
        lambda c: h3.cell_area(c, unit="km^2"))
    from collections import Counter
    pt_cells = Counter(h3.latlng_to_cell(r.lat, r.lon, res)
                       for r in points.itertuples(index=False))
    hex_df["n_points"] = hex_df["cell"].map(pt_cells).fillna(0).astype(int)
    hex_df["value"] = metric_value(hex_df, metric_key, m2_per_person)

fmap = build_map(points, hex_df, metric_key if hex_mode else None,
                 show_points, center, zoom)
st_folium(fmap, width=None, height=650, returned_objects=[])

if hex_mode and hex_df is not None:
    st.subheader(f"Данные по гексам — {city_query}")
    show = hex_df.copy()
    show["value"] = show["value"].map(lambda v: round(v, 2))
    st.dataframe(show.sort_values("value", ascending=False),
                 use_container_width=True)
    csv = hex_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇ Скачать таблицу гексов (CSV)", csv,
                       "hexes.csv", "text/csv")
