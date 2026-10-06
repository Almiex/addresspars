# -*- coding: utf-8 -*-
"""
OSM Point & Hex Analytics — Streamlit app (light)
-------------------------------------------------
1. Пользователь загружает файл (xlsx/csv) со списком адресов.
2. На OSM-карте появляются «капельки» — метки с адресами
   (координаты из файла либо геокодирование Nominatim).
3. По тумблеру включается режим гексов: ВЕСЬ город покрывается сеткой H3
   (res 7–9), метрики считаются из OSM (Overpass, одна выгрузка на город).

Метрики (радиокнопки):
  - Плотность населения (жилфонд OSM / м² на чел / км² гекса)
  - Объём жилого фонда (м² надземной площади)
  - Индекс спроса (платёжеспособность, 0–100)
  - Трафик (пеший/автомобильный, индекс 0–100)
  - Мед. учреждения (без аптек и стоматологий)

Запуск:  streamlit run app.py
"""

import re
import time

import numpy as np
import pandas as pd
import h3
import folium
import streamlit as st
import geopandas as gpd
from shapely.geometry import Polygon
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from branca.colormap import LinearColormap
from jinja2 import Template as _Jinja2Template

# --------------------------------------------------------------------------- #
#  Константы
# --------------------------------------------------------------------------- #
OVERPASS_ENDPOINTS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.nchc.org.tw/api/interpreter",
    "https://overpass.maps.mail.ru/api/interpreter",
]
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
PHOTON_URL = "https://photon.komoot.io/api/"
HEADERS = {"User-Agent": "OSMHexAnalytics/1.0 (OSM data)"}
NOMINATIM_MIN_INTERVAL = 1.2
_last_nominatim_ts = [0.0]

RES_SPACING_KM = {7: 2.4, 8: 0.92, 9: 0.35}
MAX_GRID_CELLS = 20000

LAT_COLS = ["широта", "lat", "latitude", "y"]
LON_COLS = ["долгота", "lon", "lng", "longitude", "long", "x"]
ADDR_COLS = ["адрес", "address", "addr", "адрес объекта"]
CITY_COLS = ["город", "city", "town"]

METRICS = [
    "1. Плотность населения",
    "2. Объём жилого фонда",
    "3. Индекс спроса (платёжеспособность)",
    "4. Трафик (индекс, пеший/авто)",
    "5. Мед. учреждения (без аптек и стоматологий)",
]
TRAFFIC_MODES = ["Автомобильный (primary/secondary/tertiary и выше)",
                 "Пешеходный (footway/pedestrian/path и т.п.)"]

# жилые здания: НЕ только apartments/residential — во многих городах РФ
# жильё размечено как house/detached и т.п.
RESIDENTIAL_BUILDINGS = {"apartments", "residential", "house", "detached",
                         "semidetached_house", "terrace", "bungalow",
                         "dormitory"}
AUTO_HIGHWAYS = {"motorway", "motorway_link", "trunk", "trunk_link", "primary",
                 "primary_link", "secondary", "secondary_link", "tertiary",
                 "tertiary_link"}
PED_HIGHWAYS = {"footway", "pedestrian", "path", "steps", "cycleway",
                "living_street"}

# мед. объекты БЕЗ аптек и стоматологий
MEDICAL_TYPES = {
    "Больницы": ("#d62728", lambda t: t.get("amenity") == "hospital"),
    "Клиники и медцентры": ("#2ca02c",
                            lambda t: t.get("amenity") == "clinic"
                            or t.get("emergency") == "trauma_centre"
                            or "травмпункт" in t.get("name", "").lower()),
    "Врачебные кабинеты": ("#ff7f0e",
                           lambda t: t.get("amenity") == "doctors"),
}

# палитра с «растянутой» жёлто-оранжевой серединой: красный уходит к верху шкалы
COLORS = ["#2c7fb8", "#41b6c4", "#7fcdbb", "#ffffb2", "#ffeda0",
          "#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#f03b20", "#bd0026"]
GAMMA = 0.6  # показатель нелинейности раскраски: t = (v/vmax)**GAMMA


# --------------------------------------------------------------------------- #
#  Геокодинг города: Photon -> Nominatim (с полигоном границы)
# --------------------------------------------------------------------------- #
def _nominatim_throttle():
    now = time.monotonic()
    gap = now - _last_nominatim_ts[0]
    if gap < NOMINATIM_MIN_INTERVAL:
        time.sleep(NOMINATIM_MIN_INTERVAL - gap)
    _last_nominatim_ts[0] = time.monotonic()


def _nominatim_geocode(city: str):
    _nominatim_throttle()
    try:
        import requests
        r = requests.get(NOMINATIM_URL, params={
            "q": city, "format": "json", "limit": 1, "accept-language": "ru",
            "polygon_geojson": 1, "polygon_threshold": 0.0005,
        }, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception:  # noqa: BLE001
        return None
    if not data:
        return None
    d = data[0]
    return {"lat": float(d["lat"]), "lon": float(d["lon"]),
            "display": d["display_name"],
            "bbox": [float(x) for x in d["boundingbox"]],
            "geojson": d.get("geojson")}


def _photon_geocode(city: str):
    """Резервный геокодер Photon (komoot). Границы не отдаёт — только bbox."""
    try:
        import requests
        r = requests.get(PHOTON_URL, params={"q": city, "limit": 1, "lang": "ru"},
                         headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        feats = r.json().get("features", [])
        if not feats:
            return None
        f0 = feats[0]
        lon, lat = f0["geometry"]["coordinates"]
        p = f0["properties"]
        display = ", ".join(str(x) for x in (
            p.get("name"), p.get("city") or p.get("town") or p.get("state"),
            p.get("country")) if x) or city
        ext = p.get("extent")
        bbox = ([float(ext[1]), float(ext[3]), float(ext[0]), float(ext[2])]
                if ext and len(ext) == 4
                else [lat - 0.05, lat + 0.05, lon - 0.05, lon + 0.05])
        return {"lat": float(lat), "lon": float(lon), "display": display,
                "bbox": bbox, "geojson": None}
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def geocode_city(city: str):
    geo = _photon_geocode(city)          # Photon быстрее и не банит
    if geo is None:
        return None
    # догеокодинг ради ПОЛИГОНА границы: 1 throttled-запрос на город
    geo2 = _nominatim_geocode(city)
    if geo2 and geo2.get("geojson"):
        geo["geojson"] = geo2["geojson"]
        geo["bbox"] = geo2["bbox"]
        geo["display"] = geo2["display"]
    return geo


# --------------------------------------------------------------------------- #
#  Геокодинг адресов точек (только если в файле нет координат)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=86400, show_spinner=False)
def geocode_address(query: str):
    _nominatim_throttle()
    try:
        import requests
        r = requests.get(NOMINATIM_URL, params={
            "q": query, "format": "json", "limit": 1, "accept-language": "ru",
        }, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception:  # noqa: BLE001
        return None
    if not data:
        return None
    return {"lat": float(data[0]["lat"]), "lon": float(data[0]["lon"]),
            "display": data[0]["display_name"]}


# --------------------------------------------------------------------------- #
#  Загрузка файла и распознавание колонок
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
#  Overpass: одна выгрузка на город
# --------------------------------------------------------------------------- #
def _query_overpass(q: str, timeout: int = 600) -> dict:
    errors = []
    for url in OVERPASS_ENDPOINTS:
        try:
            import requests
            r = requests.post(url, data={"data": q}, headers=HEADERS,
                              timeout=timeout)
            if r.status_code == 200:
                return r.json()
            errors.append(f"{url}: HTTP {r.status_code}")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{url}: {type(e).__name__}")
        time.sleep(2)
    raise RuntimeError("Overpass недоступен: " + "; ".join(errors[-4:]))


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_city_data(city: str):
    geo = geocode_city(city)
    if geo is None:
        return None, None, None
    south, north, west, east = geo["bbox"]   # bbox Nominatim: [S, N, W, E]
    # запас на границах — чтобы объекты у края гексов не потерялись
    south, north, west, east = (south - 0.02, north + 0.02,
                                west - 0.02, east + 0.02)
    bb = f"{south},{west},{north},{east}"

    q_ways = f"""
[out:json][timeout:300];
(
  way["building"]({bb});
  way["highway"]({bb});
  way["amenity"]({bb});
  way["shop"]({bb});
  way["office"]({bb});
  way["leisure"]({bb});
  way["tourism"]({bb});
  way["healthcare"]({bb});
);
out geom;"""
    q_nodes = f"""
[out:json][timeout:300];
(
  node["amenity"]({bb});
  node["shop"]({bb});
  node["office"]({bb});
  node["craft"]({bb});
  node["leisure"]({bb});
  node["tourism"]({bb});
  node["healthcare"]({bb});
);
out body;"""

    ways_raw = _query_overpass(q_ways).get("elements", [])
    nodes_raw = _query_overpass(q_nodes).get("elements", [])
    if not ways_raw and not nodes_raw:
        raise RuntimeError("Overpass вернул пустой набор данных — зеркало "
                           "перегружено. Подождите минуту и повторите.")

    ways, nodes = [], []
    for el in ways_raw:
        if "geometry" not in el:
            continue
        ways.append({"id": el["id"],
                     "coords": [(p["lon"], p["lat"]) for p in el["geometry"]],
                     "tags": el.get("tags", {})})
    for el in nodes_raw:
        nodes.append({"id": el["id"], "lat": el["lat"], "lon": el["lon"],
                      "tags": el.get("tags", {})})
    return geo, pd.DataFrame(nodes, columns=["id", "lat", "lon", "tags"]), \
        pd.DataFrame(ways, columns=["id", "coords", "tags"])


# --------------------------------------------------------------------------- #
#  Подготовка слоёв
# --------------------------------------------------------------------------- #
def ways_centroids(ways_df):
    if ways_df.empty:
        return pd.DataFrame(columns=["lat", "lon", "tags"])
    return pd.DataFrame({
        "lat": ways_df["coords"].apply(lambda c: float(np.mean([p[1] for p in c]))),
        "lon": ways_df["coords"].apply(lambda c: float(np.mean([p[0] for p in c]))),
        "tags": ways_df["tags"],
    })


def _to_int(v, default):
    try:
        return max(1, int(float(v)))
    except (TypeError, ValueError):
        return default


def _safe_polygon(coords):
    pts = []
    for p in coords:
        if not pts or p != pts[-1]:
            pts.append(p)
    if len(pts) < 3:
        return None
    try:
        poly = Polygon(pts)
        return poly if poly.is_valid else poly.buffer(0)
    except Exception:  # noqa: BLE001
        return None


def buildings_gdf(ways_df):
    """Жилые здания с площадью застройки (м², метрическая проекция) и этажностью."""
    cols = ["levels", "area", "lat", "lon"]
    if ways_df.empty:
        return gpd.GeoDataFrame(columns=cols, geometry=[], crs="EPSG:4326")
    mask = ways_df["tags"].apply(lambda t: t.get("building") in RESIDENTIAL_BUILDINGS)
    sub = ways_df[mask]
    if sub.empty:
        return gpd.GeoDataFrame(columns=cols, geometry=[], crs="EPSG:4326")
    recs, geoms = [], []
    for coords, tags in zip(sub["coords"], sub["tags"]):
        geom = _safe_polygon(coords)
        if geom is None or geom.is_empty:
            continue
        recs.append({"levels": _to_int(tags.get("building:levels"), 1),
                     "lat": float(np.mean([p[1] for p in coords])),
                     "lon": float(np.mean([p[0] for p in coords]))})
        geoms.append(geom)
    if not geoms:
        return gpd.GeoDataFrame(columns=cols, geometry=[], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(recs, geometry=geoms, crs="EPSG:4326")
    gdf["area"] = gdf.to_crs(gdf.estimate_utm_crs()).area.values
    return gdf


def roads_gdf(ways_df):
    """Дороги с длиной (км, метрическая проекция)."""
    cols = ["highway", "len_km", "lat", "lon"]
    if ways_df.empty:
        return gpd.GeoDataFrame(columns=cols, geometry=[], crs="EPSG:4326")
    mask = ways_df["tags"].apply(lambda t: "highway" in t)
    sub = ways_df[mask]
    if sub.empty:
        return gpd.GeoDataFrame(columns=cols, geometry=[], crs="EPSG:4326")
    from shapely.geometry import LineString
    recs, geoms = [], []
    for coords, tags in zip(sub["coords"], sub["tags"]):
        pts = []
        for p in coords:
            if not pts or p != pts[-1]:
                pts.append(p)
        if len(pts) < 2:
            continue
        try:
            geom = LineString(pts)
        except Exception:  # noqa: BLE001
            continue
        recs.append({"highway": tags.get("highway", ""),
                     "lat": float(np.mean([p[1] for p in pts])),
                     "lon": float(np.mean([p[0] for p in pts]))})
        geoms.append(geom)
    if not geoms:
        return gpd.GeoDataFrame(columns=cols, geometry=[], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(recs, geometry=geoms, crs="EPSG:4326")
    gdf["len_km"] = gdf.to_crs(gdf.estimate_utm_crs()).length.values / 1000.0
    return gdf


# --------------------------------------------------------------------------- #
#  H3-сетка на ВЕСЬ город
# --------------------------------------------------------------------------- #
def _cells_from_polygon(coordinates, res):
    outer = [(lat, lng) for lng, lat in coordinates[0]]
    if outer[0] != outer[-1]:
        outer.append(outer[0])
    holes = []
    for hole in coordinates[1:]:
        h = [(lat, lng) for lng, lat in hole]
        if h[0] != h[-1]:
            h.append(h[0])
        holes.append(h)
    try:
        poly = h3.LatLngPoly(outer, *holes)      # h3 >= 4.1
    except AttributeError:
        poly = {"type": "Polygon",
                "coordinates": [[[lng, lat] for lat, lng in outer]] +
                               [[[lng, lat] for lat, lng in h_] for h_ in holes]}
    return h3.polygon_to_cells(poly, res)


def make_grid(geo, res):
    geom = geo.get("geojson")
    if geom and geom.get("type") in ("Polygon", "MultiPolygon"):
        polys = (geom["coordinates"] if geom["type"] == "MultiPolygon"
                 else [geom["coordinates"]])
        cells = set()
        for poly in polys:
            cells.update(_cells_from_polygon(poly, res))
        if cells:
            return sorted(cells), True
    # fallback: граница не получена — прямоугольник по bbox
    south, north, west, east = geo["bbox"]
    m = 0.02
    ring = [(south - m, west - m), (south - m, east + m),
            (north + m, east + m), (north + m, west - m),
            (south - m, west - m)]
    try:
        poly = h3.LatLngPoly(ring)
    except AttributeError:
        poly = {"type": "Polygon",
                "coordinates": [[[lng, lat] for lat, lng in ring]]}
    return sorted(h3.polygon_to_cells(poly, res)), False


def hex_area_km2(res):
    b = h3.cell_to_boundary(h3.latlng_to_cell(55.0, 83.0, res))
    gdf = gpd.GeoDataFrame(geometry=[Polygon([(lng, lat) for lat, lng in b])],
                           crs="EPSG:4326")
    return gdf.to_crs(gdf.estimate_utm_crs()).area.iloc[0] / 1e6


def hex_counts(points_df, res, mask=None):
    if points_df is None or points_df.empty:
        return pd.Series(dtype=float)
    d = points_df if mask is None else points_df[mask]
    if d.empty:
        return pd.Series(dtype=float)
    cells = [h3.latlng_to_cell(a, b, res) for a, b in zip(d["lat"], d["lon"])]
    return pd.Series(cells).value_counts().astype(float)


def _norm100(s):
    if s is None or s.empty:
        return pd.Series(dtype=float)
    mx = float(s.max())
    return s / mx * 100.0 if mx > 0 else s * 0.0


def populated_with_ring(grid, series):
    """Гексы с value>0 + кольцо соседей (k=1) вокруг них."""
    populated = set(series[series > 0].index)
    keep = set(populated)
    for cell in populated:
        keep.update(h3.grid_disk(cell, 1))
    return [c for c in grid if c in keep]


# --------------------------------------------------------------------------- #
#  Метрики
# --------------------------------------------------------------------------- #
def compute_series(map_type, sub_option, res, data, m2_per_person=25,
                   pt_cells=None):
    geo, nodes_df, ways_df = data
    wcent = ways_centroids(ways_df)

    if map_type.startswith("1.") or map_type.startswith("2."):
        b = buildings_gdf(ways_df)
        if b.empty:
            return pd.Series(dtype=float), ("чел./км²" if map_type.startswith("1.")
                                            else "м²"), None, None
        cells = [h3.latlng_to_cell(a, b_, res) for a, b_ in zip(b["lat"], b["lon"])]
        vol = (b.assign(cell=cells, vol=b["area"] * b["levels"])
                 .groupby("cell")["vol"].sum())
        if map_type.startswith("2."):
            return vol, "м² суммарной площади жилых зданий", None, None
        area = hex_area_km2(res)
        dens = vol / m2_per_person / area
        return dens, f"чел./км² (оценка, {m2_per_person} м²/чел)", None, None

    if map_type.startswith("3."):
        # Индекс спроса: 40% жилой фонд + 20% розница/общепит
        # + 15% банки/офисы + 15% остановки + 10% парковки/АЗС; всё в 0–100
        b = buildings_gdf(ways_df)
        parts = {}
        if not b.empty:
            cells = [h3.latlng_to_cell(a, b_, res) for a, b_ in zip(b["lat"], b["lon"])]
            parts["people"] = _norm100(
                b.assign(cell=cells, vol=b["area"] * b["levels"])
                 .groupby("cell")["vol"].sum())
        food = {"supermarket", "convenience", "greengrocer", "deli", "butcher",
                "bakery", "seafood"}
        retail_amen = {"restaurant", "cafe", "fast_food", "bar", "pub", "food_court"}
        mr = (nodes_df["tags"].apply(lambda t: "shop" in t and t.get("shop") not in food)
              | nodes_df["tags"].apply(lambda t: t.get("amenity") in retail_amen))
        wr = (wcent["tags"].apply(lambda t: "shop" in t and t.get("shop") not in food)
              | wcent["tags"].apply(lambda t: t.get("amenity") in retail_amen))
        parts["retail"] = _norm100(pd.concat(
            [hex_counts(nodes_df, res, mr), hex_counts(wcent, res, wr)]
        ).groupby(level=0).sum())
        mb = (nodes_df["tags"].apply(lambda t: t.get("amenity") in {"bank", "bureau_de_change"}
                                     or "office" in t))
        wb = (wcent["tags"].apply(lambda t: t.get("amenity") in {"bank", "bureau_de_change"}
                                  or "office" in t))
        parts["biz"] = _norm100(pd.concat(
            [hex_counts(nodes_df, res, mb), hex_counts(wcent, res, wb)]
        ).groupby(level=0).sum())
        stops = ("bus_stop", "bus_station", "tram_stop")
        mt = (nodes_df["tags"].apply(lambda t: t.get("highway") in stops or
                                               t.get("public_transport") in ("platform", "stop_position")))
        wt = (wcent["tags"].apply(lambda t: t.get("highway") in stops or
                                             t.get("public_transport") in ("platform", "stop_position")))
        parts["transit"] = _norm100(pd.concat(
            [hex_counts(nodes_df, res, mt), hex_counts(wcent, res, wt)]
        ).groupby(level=0).sum())
        ma = nodes_df["tags"].apply(lambda t: t.get("amenity") in {"parking", "fuel"})
        wa = wcent["tags"].apply(lambda t: t.get("amenity") in {"parking", "fuel"})
        parts["auto"] = _norm100(pd.concat(
            [hex_counts(nodes_df, res, ma), hex_counts(wcent, res, wa)]
        ).groupby(level=0).sum())
        if not parts:
            return pd.Series(dtype=float), "индекс 0-100", None, None
        idx = None
        weights = {"people": 0.40, "retail": 0.20, "biz": 0.15,
                   "transit": 0.15, "auto": 0.10}
        for name, w in weights.items():
            if name not in parts:
                continue
            idx = parts[name] * w if idx is None else idx.add(parts[name] * w, fill_value=0)
        idx = idx.fillna(0)
        extra = pd.DataFrame(parts).reindex(idx.index).fillna(0).round(0)
        extra.columns = [f"c_{c}" for c in extra.columns]
        return idx.round(1), "индекс 0-100", extra, None

    if map_type.startswith("4."):
        # Трафик: плотность сети (км/км²), индекс 0–100; тултип показывает оба
        r = roads_gdf(ways_df)
        if r.empty:
            return pd.Series(dtype=float), "индекс 0-100", None, None
        area = hex_area_km2(res)
        cells = [h3.latlng_to_cell(a, b, res) for a, b in zip(r["lat"], r["lon"])]
        by_cell = r.assign(cell=cells).groupby("cell")
        ped = by_cell.apply(lambda g: g["len_km"][g["highway"].isin(PED_HIGHWAYS)].sum(),
                            include_groups=False) / area
        auto = by_cell.apply(lambda g: g["len_km"][g["highway"].isin(AUTO_HIGHWAYS)].sum(),
                             include_groups=False) / area
        extra = pd.DataFrame({
            "ped": _norm100(ped).round(0), "auto": _norm100(auto).round(0),
        }).fillna(0)
        if sub_option.startswith("Пешеходный"):
            return extra["ped"], "индекс пешего трафика 0-100", extra, None
        return extra["auto"], "индекс автотрафика 0-100", extra, None

    if map_type.startswith("5."):
        # Мед. объекты (без аптек и стоматологий) + точки на карту
        types = list(MEDICAL_TYPES)
        type_series, points = {}, []
        for name in types:
            color, matcher = MEDICAL_TYPES[name]
            mn = nodes_df["tags"].apply(matcher)
            mw = wcent["tags"].apply(matcher)
            type_series[name] = pd.concat(
                [hex_counts(nodes_df, res, mn), hex_counts(wcent, res, mw)]
            ).groupby(level=0).sum()
            for d in (nodes_df[mn], wcent[mw]):
                for _, row in d.iterrows():
                    points.append({"lat": row["lat"], "lon": row["lon"],
                                   "color": color, "type": name,
                                   "label": row["tags"].get("name", name)})
        extra = pd.DataFrame(type_series).fillna(0).round(0)
        total = extra.sum(axis=1) if len(extra) else pd.Series(dtype=float)
        return total, "мед. объектов", extra, points

    return pd.Series(dtype=float), "", None, None


# --------------------------------------------------------------------------- #
#  Тултип без табличной вёрстки folium (компактные строки «подпись: значение»)
# --------------------------------------------------------------------------- #
class HexTooltip(folium.GeoJsonTooltip):
    base_template = """
    function(layer){
    let div = L.DomUtil.create('div');
    let fields = {{ this.fields | tojson | safe }};
    let aliases = {{ this.aliases | tojson | safe }};
    let props = layer.feature.properties;
    div.innerHTML = fields.map((v, i) => {
        let val = props[v];
        if (val === null || val === undefined) { val = ''; }
        else if (typeof val === 'object') { val = JSON.stringify(val); }
        {% if this.localize %}
        else if (typeof val === 'number') { val = val.toLocaleString(); }
        {% endif %}
        return '<div style="margin:1px 0;"><span style="color:#555;font-weight:600;">'
            + aliases[i] + '</span>&nbsp;&nbsp;' + val + '</div>';
    }).join('');
    return div
    }
    """
    def render(self, **kwargs):
        from branca.element import Element as _El
        if getattr(self, "style", None):
            self._parent.get_root().header.add_child(
                _El(f"<style>.{self.class_name}{{{self.style}}}</style>"),
                name=f"tooltip_style_{self.get_name()}",
            )
        _El.render(self, **kwargs)

    _template = _Jinja2Template(
        """
    {% macro script(this, kwargs) %}
    {{ this._parent.get_name() }}.bindTooltip("""
        + base_template + """,{{ this.tooltip_options | tojson }});
                     {% endmacro %}
                     """
    )


# --------------------------------------------------------------------------- #
#  Отрисовка карты
# --------------------------------------------------------------------------- #
def render_map(points, hex_df_grid, series, unit, center, map_type,
               marker_points=None, hex_extra=None, extra_aliases=None,
               legend=None, npoints=None):
    m = folium.Map(location=center, tiles="OpenStreetMap", control_scale=True,
                   prefer_canvas=True)
    folium.TileLayer("OpenTopoMap", name="Топографическая", show=False).add_to(m)
    m.get_root().header.add_child(folium.Element("""
<style>
.foliumtooltip { background: #fff; color: #222; border-radius: 4px;
  box-shadow: 0 1px 4px rgba(0,0,0,.35); padding: 8px 10px; font-size: 13px;
  line-height: 1.45; }
.foliumtooltip table { margin: 0 !important; width: auto !important;
  border-collapse: collapse; }
.foliumtooltip th, .foliumtooltip td { text-align: left !important;
  padding: 1px 10px 1px 0 !important; vertical-align: top; }
.foliumtooltip th { color: #555; font-weight: 600; white-space: nowrap; }
</style>
"""))

    # ── «капельки» — метки адресов ────────────────────────────────────────
    if marker_points is not None and not marker_points.empty:
        cluster = MarkerCluster(name="Адреса").add_to(m)
        for _, row in marker_points.iterrows():
            folium.Marker(
                location=[row["lat"], row["lon"]],
                tooltip=str(row.get("label", ""))[:150],
                popup=folium.Popup(str(row.get("label", ""))[:400],
                                   max_width=320),
                icon=folium.Icon(color="blue", icon="glyphicon-map-marker"),
            ).add_to(cluster)

    if hex_df_grid is None:
        folium.LayerControl().add_to(m)
        return m

    grid = hex_df_grid
    vals = series.reindex(grid).fillna(0.0)
    vmax = float(vals.max()) if len(vals) else 0.0
    if vmax <= 0:
        st.warning("Нет данных для выбранного слоя в этом городе.")
        vmax = 1.0
    cm = LinearColormap(COLORS, vmin=0, vmax=1)

    def _palette_at(t):
        pos = min(max(t, 0.0), 1.0) * (len(COLORS) - 1)
        i = min(int(pos), len(COLORS) - 2)
        frac = pos - i
        return "#" + "".join(
            f"{round(int(COLORS[i][k:k+2], 16) + (int(COLORS[i+1][k:k+2], 16)
                    - int(COLORS[i][k:k+2], 16)) * frac):02x}"
            for k in (1, 3, 5))

    cm_legend = LinearColormap([_palette_at((i / 24) ** GAMMA) for i in range(25)],
                               vmin=0, vmax=vmax)
    cm_legend.caption = f"{map_type} — {unit}"

    extra_cols = list(hex_extra.columns) if hex_extra is not None else []
    safe_cols = {c: f"x{i}" for i, c in enumerate(extra_cols)}

    feats = []
    for cell in grid:
        boundary = h3.cell_to_boundary(cell)
        ring = [[lng, lat] for lat, lng in boundary]
        ring.append(ring[0])
        v = float(vals.get(cell, 0.0))
        props = {"v": round(v, 2) if vmax < 100 else round(v),
                 "n": int((npoints or {}).get(cell, 0))}
        for c in extra_cols:
            props[safe_cols[c]] = (float(hex_extra.loc[cell, c])
                                   if cell in hex_extra.index else 0)
        feats.append({"type": "Feature", "properties": props,
                      "geometry": {"type": "Polygon",
                                   "coordinates": [ring]}})

    def _style(f):
        v = f["properties"]["v"]
        t = (v / vmax) ** GAMMA if vmax > 0 else 0.0
        return {"fillColor": cm(t), "color": "#555555", "weight": 0.6,
                "fillOpacity": 0.03 if v <= 0 else 0.20 + 0.40 * t}

    aliases = [f"{unit}: ", "Точек в гексе: "] + \
              [extra_aliases.get(c, c) for c in extra_cols]
    folium.GeoJson(
        {"type": "FeatureCollection", "features": feats},
        style_function=_style,
        tooltip=HexTooltip(fields=["v", "n"] + [safe_cols[c] for c in extra_cols],
                           aliases=aliases, localize=True),
    ).add_to(m)
    cm_legend.add_to(m)

    # легенда типов точек
    if legend:
        from branca.element import MacroElement, Template
        rows = "".join(
            f'<div><span style="color:{color}; font-size:1.15em;">&#9679;</span>'
            f"&nbsp;{name}</div>" for name, color in legend)
        macro = MacroElement()
        macro._template = Template(
            "{% macro html(this, kwargs) %}"
            '<div style="position: fixed; bottom: 55px; left: 55px; z-index: 9999; '
            'background: rgba(255,255,255,0.92); padding: 8px 12px; border-radius: 6px; '
            'border: 1px solid #999; font-size: 13px; line-height: 1.5;">'
            f"{rows}</div>"
            "{% endmacro %}")
        m.get_root().add_child(macro)

    # точки мед. объектов поверх гексов — ОДИН GeoJSON-слой
    if points:
        def _clean(s):
            return re.sub(r'["\'<>\n\r\\]', " ", str(s))[:120]
        pfeats = [{
            "type": "Feature",
            "properties": {"tip": f'{_clean(p["type"])}: {_clean(p["label"])}',
                           "c": p["color"]},
            "geometry": {"type": "Point",
                         "coordinates": [p["lon"], p["lat"]]},
        } for p in points]
        folium.GeoJson(
            {"type": "FeatureCollection", "features": pfeats},
            marker=folium.CircleMarker(radius=5, weight=1.5, fill=True,
                                       fill_opacity=0.9),
            style_function=lambda f: {"color": f["properties"]["c"],
                                      "fillColor": f["properties"]["c"]},
            tooltip=folium.GeoJsonTooltip(fields=["tip"], aliases=[""],
                                          localize=False),
            name="Мед. объекты",
        ).add_to(m)

    lats, lngs = [], []
    for c in grid:
        for lat_, lng_ in h3.cell_to_boundary(c):
            lats.append(lat_); lngs.append(lng_)
    m.fit_bounds([[min(lats), min(lngs)], [max(lats), max(lngs)]])
    folium.LayerControl().add_to(m)
    return m


# --------------------------------------------------------------------------- #
#  UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="OSM Hex Analytics", layout="wide")
st.title("🗺️ OSM Point & Hex Analytics")
st.caption("Загрузите список адресов → метки на карте → гексы H3 на весь город "
           "с аналитикой по OpenStreetMap (Overpass).")

with st.sidebar:
    st.header("1. Данные")
    uploaded = st.file_uploader("Файл со списком адресов (.xlsx / .csv)",
                                type=["xlsx", "xls", "csv"])
    st.header("2. Гексы")
    hex_mode = st.toggle("Режим гексов", value=False)
    res = st.select_slider("Размер гекса (resolution)", options=[7, 8, 9],
                           value=8, key="res_slider",
                           help="Res 7 ≈ 2,4 км · Res 8 ≈ 0,92 км · "
                                "Res 9 ≈ 0,35 км между центрами")
    map_type = st.radio("Метрика гексов", METRICS, index=0)
    sub_option = None
    if map_type.startswith("4."):
        sub_option = st.radio("Что раскрашиваем", TRAFFIC_MODES,
                              help="Тултип гекса показывает оба индекса")
    m2_per_person = 25
    if map_type.startswith("1."):
        m2_per_person = st.slider("Норма м² жилья на человека", 15, 60, 25)

if uploaded is None:
    st.info("Загрузите файл со списком адресов. Ожидаются колонки вроде: "
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

cities = []
if c_city:
    cities = sorted({str(x).strip() for x in df[c_city].dropna().unique()
                     if str(x).strip()})

st.subheader("Распознанные колонки")
st.write(f"**Адрес:** {c_addr or '—'} | **Город:** {c_city or '—'} | "
         f"**Широта:** {c_lat or '—'} | **Долгота:** {c_lon or '—'}")

# ── точки: координаты из файла или геокодирование ──────────────────────────
marker_points = pd.DataFrame(columns=["lat", "lon", "label"])
if c_lat and c_lon:
    rows = []
    for i, r in df.iterrows():
        try:
            lat, lon = float(r[c_lat]), float(r[c_lon])
        except (TypeError, ValueError):
            continue
        rows.append({"lat": lat, "lon": lon,
                     "label": r[c_addr] if c_addr else f"Строка {i}"})
    marker_points = pd.DataFrame(rows)
    st.info("Координаты взяты из колонок файла.")
elif c_addr:
    rows = []
    st.warning(f"Геокодируем {len(df)} адресов через Nominatim "
               f"(≈ {int(len(df) * 1.3)} сек). Пожалуйста, подождите…")
    bar = st.progress(0)
    for k, (_, r) in enumerate(df.iterrows()):
        parts = []
        if pd.notna(r[c_addr]):
            parts.append(str(r[c_addr]))
        if c_city and pd.notna(r[c_city]):
            parts.append(str(r[c_city]))
        geo_pt = geocode_address(", ".join(parts))
        if geo_pt:
            rows.append({"lat": geo_pt["lat"], "lon": geo_pt["lon"],
                         "label": geo_pt["display"]})
        bar.progress((k + 1) / len(df))
    marker_points = pd.DataFrame(rows)

marker_points = marker_points.dropna().drop_duplicates(subset=["lat", "lon"])
if marker_points.empty:
    st.error("Не удалось получить ни одной точки: проверьте колонки "
             "адреса/координат в файле.")
    st.stop()

if not cities:
    manual = st.text_input("Не удалось определить город из файла. "
                           "Введите город для анализа гексов:")
    if manual.strip():
        cities = [manual.strip()]

city_query = ", ".join(cities) if cities else ""
st.success(f"Точек на карте: **{len(marker_points)}** | Город анализа: "
           f"**{city_query or 'не определён — гексы недоступны'}**")

center = [marker_points["lat"].mean(), marker_points["lon"].mean()]

# ── режим гексов ────────────────────────────────────────────────────────────
grid, series, unit, hex_extra, points_layer = None, None, "", None, None
pt_cells = {}
legend = None
extra_aliases = None

if hex_mode:
    if not city_query:
        st.error("Не определён город — укажите его в поле выше.")
        st.stop()
    with st.spinner(f"Загружаю данные OpenStreetMap для «{city_query}» через "
                    "Overpass (1–5 минут; первый запуск дольше)…"):
        try:
            geo, nodes_df, ways_df = fetch_city_data(city_query)
        except Exception as e:  # noqa: BLE001
            st.error(f"Не удалось загрузить данные для «{city_query}»: {e}")
            st.stop()
    if geo is None:
        st.error(f"Город «{city_query}» не найден. Уточните название.")
        st.stop()

    st.success(f"📍 {geo['display']}")
    grid, in_boundary = make_grid(geo, res)
    if not in_boundary:
        st.info("Граница города не получена от Nominatim — сетка построена "
                "по прямоугольной области.")
    if len(grid) > MAX_GRID_CELLS:
        st.error(f"Сетка слишком велика ({len(grid)} гексов при res {res}). "
                 "Понизьте разрешение.")
        st.stop()

    with st.spinner("Считаю метрики по гексам…"):
        series, unit, hex_extra, points_layer = compute_series(
            map_type, sub_option, res, (geo, nodes_df, ways_df),
            m2_per_person=m2_per_person)

    # точки загруженного списка по гексам (для тултипа «Точек в гексе»)
    pt_cells = pd.Series(
        [h3.latlng_to_cell(a, b, res) for a, b in
         zip(marker_points["lat"], marker_points["lon"])]
    ).value_counts()
    # карты 1–2: только заселённые гексы + кольцо вокруг них
    if map_type.startswith(("1.", "2.")):
        grid = populated_with_ring(grid, series)
        if not grid:
            st.warning("Нет данных по жилым зданиям OSM в этом городе. "
                       "Попробуйте другую метрику.")
            st.stop()

    legend = [("Больницы", "#d62728"), ("Клиники и медцентры", "#2ca02c"),
              ("Врачебные кабинеты", "#ff7f0e")] if map_type.startswith("5.") \
        else None
    if map_type.startswith("3."):
        extra_aliases = {"c_people": "Жилой фонд: ", "c_retail": "Розница/общепит: ",
                         "c_biz": "Банки/офисы: ", "c_transit": "Остановки: ",
                         "c_auto": "Парковки/АЗС: "}
    elif map_type.startswith("4."):
        extra_aliases = {"ped": "Пешеходный трафик: ",
                         "auto": "Автомобильный трафик: "}
    elif map_type.startswith("5."):
        extra_aliases = {name: f"{name}: " for name in MEDICAL_TYPES}

    c1, c2, c3 = st.columns(3)
    c1.metric("Гексов в сетке", f"{len(grid):,}")
    c2.metric("Resolution", f"res {res} (~{RES_SPACING_KM[res]} км)")
    c3.metric("Максимум в ячейке",
              f"{series.max():,.0f} {unit}" if len(series) else "—")

fmap = render_map(points_layer, grid, series, unit, center, map_type,
                  marker_points=marker_points, hex_extra=hex_extra,
                  extra_aliases=extra_aliases,
                  legend=legend if hex_mode else None,
                  npoints=pt_cells.to_dict())
st_folium(fmap, width=None, height=680, returned_objects=[])

if hex_mode:
    st.subheader(f"Данные по гексам — {city_query}")
    tbl = pd.DataFrame({"cell": grid})
    tbl["value"] = tbl["cell"].map(series.reindex(grid)).fillna(0)
    tbl["точек_в_гексе"] = tbl["cell"].map(pt_cells).fillna(0).astype(int)
    if hex_extra is not None:
        tbl = tbl.merge(hex_extra, left_on="cell", right_index=True, how="left")
    st.dataframe(tbl.sort_values("value", ascending=False),
                 use_container_width=True)
    csv = tbl.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇ Скачать таблицу гексов (CSV)", csv, "hexes.csv",
                       "text/csv")
