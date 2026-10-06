# -*- coding: utf-8 -*-
"""
OSM Point & Hex Analytics
-------------------------
Сценарий:
  1. Пользователь загружает таблицу (xlsx/csv) со списком адресов.
  2. На карте города появляются «капельки» — метки с адресами
     (координаты из файла либо геокодирование Nominatim/Photon).
  3. По включении «Режима гексов» весь город покрывается сеткой H3
     и отрабатывает полная картовая логика: все 7 карт, POI-рои,
     социнфраструктура, медицина, трафик, индекс спроса.

Источники (все бесплатные):
  * Геокодинг           — Photon (komoot) / Nominatim (OSM)
  * Здания/дороги/POI   — Overpass API (OpenStreetMap)
  * Численность (опц.)  — Kontur Population (локальный файл, data.humdata.org)

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

st.set_page_config(page_title="OSM Point & Hex Analytics", layout="wide")

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
# резервный геокодер (komoot) — подхватывает город, пока Nominatim режет
# запросы с нашего IP (на shared-хостинге лимит часто срабатывает из-за
# чужого трафика). Полигона границы не отдаёт — сетка построится по bbox
PHOTON_URL = "https://photon.komoot.io/api/"
HEADERS = {"User-Agent": "OSMHexAnalytics/1.0 (OSM data)"}

NOMINATIM_MIN_INTERVAL = 1.2
_last_nominatim_ts = [0.0]


def _nominatim_throttle():
    """Гарантирует NOMINATIM_MIN_INTERVAL между запросами к Nominatim
    по ВСЕМУ приложению (адреса точек, догеокодинг полигона города)."""
    now = time.monotonic()
    gap = now - _last_nominatim_ts[0]
    if gap < NOMINATIM_MIN_INTERVAL:
        time.sleep(NOMINATIM_MIN_INTERVAL - gap)
    _last_nominatim_ts[0] = time.monotonic()

# расстояние между центрами соседних гексов, км (по стандарту Uber H3)
RES_SPACING_KM = {7: 2.4, 8: 0.92, 9: 0.35, 10: 0.13}
MAX_GRID_CELLS = 50000  # canvas-рендер держит десятки тысяч полигонов

MAP_TYPES = [
    "1. Плотность населения",
    "2. Объём жилого фонда",
    "3. Индекс спроса (платёжеспособность)",
    "4. Инфраструктура и POI",
    "5. Социальная инфраструктура",
    "6. Трафик (индекс, пеший/авто)",
    "7. Мед. учреждения и аптеки",
]

# ПВЗ в OSM не имеют единого тега — находим по названию/бренду
PVZ_RE = re.compile(
    r"ozon|озон|wildberries|вайлдберриз|сдэк|cdek|авито|avito|"
    r"яндекс[ -]?маркет|yandex[ -]?market", re.IGNORECASE)
PVZ_WB_RE = re.compile(r"(^|[^a-z])wb([^a-z]|$)|(^|\s)вб(\s|$)", re.IGNORECASE)


def is_pvz(tags: dict) -> bool:
    text = f"{tags.get('name', '')} {tags.get('brand', '')}"
    return bool(PVZ_RE.search(text) or PVZ_WB_RE.search(text))


def _name_of(tags: dict) -> str:
    return f"{tags.get('name', '')} {tags.get('brand', '')}".lower()


# категория -> (цвет точки, matcher по тегам)
POI_CATEGORIES = {
    "Торговые центры": ("#e377c2",
                        lambda t: t.get("shop") in {"mall", "department_store"}
                        or re.search(r"торгов(ый|ого) центр|(^|\s)тц(\s|$)|молл", _name_of(t))),
    "Бизнес-центры и офисы": ("#17becf",
                              lambda t: "office" in t
                              or re.search(r"бизнес[ -]?центр|business center|(^|\s)бц(\s|$)",
                                           _name_of(t))),
    "ПВЗ маркетплейсов (Ozon, WB, СДЭК, Авито, Яндекс)": ("#d62728", is_pvz),
    "Продуктовые магазины": ("#2ca02c",
                             lambda t: t.get("shop") in
                             {"supermarket", "convenience", "greengrocer", "deli",
                              "butcher", "bakery", "seafood"}),
    "Общепит": ("#ff7f0e",
                lambda t: t.get("amenity") in
                {"restaurant", "cafe", "fast_food", "bar", "pub",
                 "food_court", "biergarten"}),
    "Аптеки": ("#1f6fd6", lambda t: t.get("amenity") == "pharmacy"),
    "Салоны красоты и барбершопы": ("#9467bd",
                                    lambda t: t.get("shop") in
                                    {"hairdresser", "beauty", "massage", "tattoo"}),
    "Спорт и фитнес": ("#8c564b",
                       lambda t: t.get("leisure") in
                       {"fitness_centre", "sports_centre", "pitch", "stadium",
                        "track", "swimming_pool"}),
    "Банки и банкоматы": ("#bcbd22",
                          lambda t: t.get("amenity") in {"bank", "atm", "bureau_de_change"}),
    "Образование": ("#1f77b4",
                    lambda t: t.get("amenity") in
                    {"school", "kindergarten", "college", "university", "library",
                     "driving_school", "language_school", "music_school", "arts_centre"}),
}

TRAFFIC_MODES = [
    "Пешеходный (footway/pedestrian/path и т.п.)",
    "Автомобильный (primary/secondary/tertiary и выше)",
]

SOCIAL_CARE_RE = re.compile(
    r"престарел|соцзащит|социальн|реабилитац|инвалид", re.IGNORECASE)

# группа -> (цвет точки, matcher по тегам). None = особый случай (общежития-здания)
SOCIAL_GROUPS = {
    "Образование": ("#1f77b4",
                    lambda t: t.get("amenity") in
                    {"kindergarten", "school", "college", "university",
                     "arts_centre", "music_school"}),
    "Культура и досуг": ("#9467bd",
                         lambda t: t.get("amenity") in
                         {"theatre", "museum", "library", "cinema", "community_centre"}
                         or t.get("leisure") == "park"),
    "Физкультура и спорт": ("#2ca02c",
                            lambda t: t.get("leisure") in
                            {"stadium", "sports_centre", "fitness_centre", "pitch",
                             "swimming_pool", "track"}),
    "Социальная защита": ("#8c564b",
                          lambda t: t.get("amenity") == "social_facility"
                          or bool(SOCIAL_CARE_RE.search(t.get("name", "")))),
    "Жилищный фонд (общежития)": ("#7f7f7f", None),
}


# лаборатории/диагностика выделяем из amenity=doctors по названию
LAB_RE = re.compile(
    r"invitro|инвитро|гемотест|gemotest|хеликс|helix|кдл|лаборатор|диагност",
    re.IGNORECASE)


def is_med_lab(tags: dict) -> bool:
    text = f"{tags.get('name', '')} {tags.get('brand', '')}"
    return bool(LAB_RE.search(text))


# тип -> (цвет точки, matcher по тегам). Моноклиники в OSM тегируются как
# amenity=doctors — отделить их от кабинетов нельзя, честно пишем оба.
COMPETITOR_TYPES = {
    "Аптеки": ("#1f6fd6", lambda t: t.get("amenity") == "pharmacy"),
    "Больницы": ("#d62728", lambda t: t.get("amenity") == "hospital"),
    "Клиники и медцентры": ("#2ca02c",
                            lambda t: t.get("amenity") == "clinic"
                            or t.get("emergency") == "trauma_centre"
                            or "травмпункт" in t.get("name", "").lower()),
    "Диагностика и лаборатории": ("#9467bd",
                                  lambda t: t.get("amenity") == "doctors" and is_med_lab(t)),
    "Врачебные кабинеты и моноклиники": ("#ff7f0e",
                                         lambda t: t.get("amenity") == "doctors"
                                         and not is_med_lab(t)),
}


AUTO_HIGHWAYS = {"motorway", "motorway_link", "trunk", "trunk_link", "primary",
                 "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link"}
# жилые здания: НЕ только apartments/residential — во многих городах РФ
# жильё размечено как house/detached и т.п.; без них выгрузка по городу
# приходит пустой (Воронеж!). Должно совпадать с regex в fetch_city_data.
RESIDENTIAL_BUILDINGS = {"apartments", "residential", "house", "detached",
                         "semidetached_house", "terrace", "bungalow",
                         "dormitory"}
PED_HIGHWAYS = {"footway", "pedestrian", "path", "steps", "cycleway", "living_street"}

# колонки загружаемой таблицы адресов (рус/англ)
LAT_COLS = ["широта", "lat", "latitude", "y"]
LON_COLS = ["долгота", "lon", "lng", "longitude", "long", "x"]
ADDR_COLS = ["адрес", "address", "addr", "адрес объекта"]
CITY_COLS = ["город", "city", "town"]

# палитра с "растянутой" жёлто-оранжевой серединой: 11 стопов вместо 7,
# красные сдвинуты к самому верху шкалы (глубокий красный — лишь ~от 75% макс.)
COLORS = ["#2c7fb8", "#41b6c4", "#7fcdbb", "#ffffb2", "#ffeda0",
          "#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#f03b20", "#bd0026"]
# ══ ГАММА РАСКРАСКИ ШКАЛЫ ════════════════════════════════════════════════════
# t = (v / vmax) ** GAMMA: < 1 — жёлтый начинается раньше (низ шкалы сжат),
# > 1 — верх растянут, разница высоких значений различима. Легенда считается
# с той же гаммой — расхождения шкалы и заливки не будет.
GAMMA = 0.75

# --------------------------------------------------------------------------- #
#  Геокодинг
# --------------------------------------------------------------------------- #
def _nominatim_geocode(city: str):
    """Nominatim: ОДИН запрос (после throttle), без ретраев — при 429
    сразу None, вызывающий код уходит на резерв/фолбэк.
    Отдаёт полигон административной границы."""
    _nominatim_throttle()
    try:
        import requests
        r = requests.get(NOMINATIM_URL, params={
            "q": city, "format": "json", "limit": 1, "accept-language": "ru",
            "polygon_geojson": 1,          # полигон административной границы
            "polygon_threshold": 0.0005,   # упрощение границы (меньше трафик)
        }, headers=HEADERS, timeout=30)
    except Exception:  # noqa: BLE001
        return None
    if r.status_code != 200:   # 429 и прочее — не дожимаем сервер
        return None
    data = r.json()
    if not data:
        return None
    d = data[0]
    return {"lat": float(d["lat"]), "lon": float(d["lon"]),
            "display": d["display_name"],
            "bbox": [float(x) for x in d["boundingbox"]],  # [south, north, west, east]
            "geojson": d.get("geojson")}                   # граница города или None


def _photon_geocode(city: str):
    """Резервный геокодер Photon (komoot). Границы не отдаёт — в geo
    кладём bbox из extent либо небольшой квадрат вокруг точки; дальше
    сработает штатный прямоугольный фолбэк сетки."""
    try:
        import requests
        r = requests.get(PHOTON_URL,
                         params={"q": city, "limit": 1, "lang": "ru"},
                         headers=HEADERS, timeout=15)  # короткий: при недоступности быстрее уйдём на резерв
        if r.status_code != 200:
            return None
        feats = r.json().get("features", [])
        if not feats:
            return None
        f0 = feats[0]
        lon, lat = f0["geometry"]["coordinates"]
        p = f0["properties"]
        display = ", ".join(str(x) for x in (
            p.get("name"),
            p.get("city") or p.get("town") or p.get("village") or p.get("state"),
            p.get("country")) if x) or city
        ext = p.get("extent")
        if ext and len(ext) == 4:
            bbox = [float(ext[1]), float(ext[3]), float(ext[0]), float(ext[2])]
        else:
            bbox = [lat - 0.05, lat + 0.05, lon - 0.05, lon + 0.05]
        return {"lat": float(lat), "lon": float(lon), "display": display,
                "bbox": bbox, "geojson": None}
    except Exception:  # noqa: BLE001
        return None


def _overpass_geocode(city: str):
    """КРАЙНИЙ резерв: граница города через Overpass — те самые зеркала,
    через которые приложение уже грузит данные (значит, они ДОСТУПНЫ из
    этого деплоя). Ищем отношение-границу с точным именем; геометрию не
    тянем — только bbox (сетка умеет жить по прямоугольнику)."""
    try:
        name_re = "^" + re.escape(city.strip()) + "$"
        # ВАЖНО: regex по имени — полный перебор отношений, запрос ТЯЖЁЛЫЙ.
        # Поэтому timeout=40 и 1 попытка: медленное зеркало быстро пропускаем
        # (иначе вечная загрузка), пусть цепочка упадёт в понятную ошибку.
        q = f"""[out:json][timeout:35];
rel["name"~"{name_re}", i]["boundary"="administrative"];
out tags bb;"""
        els = _query_overpass(q, timeout=40, attempts=1).get("elements", [])
        if not els:
            return None
        # город (admin_level 6/8) предпочтительнее крупных регионов
        els.sort(key=lambda e: int(e.get("tags", {}).get("admin_level", "99")))
        b = els[0].get("bounds")
        if not b:
            return None
        bbox = [float(b["minlat"]), float(b["maxlat"]),
                float(b["minlon"]), float(b["maxlon"])]
        return {"lat": (bbox[0] + bbox[1]) / 2, "lon": (bbox[2] + bbox[3]) / 2,
                "display": f"{city} (граница OSM, bbox)",
                "bbox": bbox, "geojson": None}
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def geocode_city(city: str):
    """Цепочка: Photon -> Nominatim -> Overpass. None — город реально
    отсутствует во всех доступных источниках."""
    trace = []
    geo = _photon_geocode(city)
    trace.append(f"Photon: {'ok' if geo else 'fail'}")
    if geo is not None:
        # догеокодинг ради ПОЛИГОНА границы: 1 throttled-запрос на город,
        # результат кэшируется целиком. При 429 молча живём по bbox Photon.
        geo2 = _nominatim_geocode(city)
        trace.append(f"Nominatim (полигон границы): {'ok' if geo2 else 'fail'}")
        if geo2 and geo2.get("geojson"):
            geo["geojson"] = geo2["geojson"]
            geo["bbox"] = geo2["bbox"]
            geo["display"] = geo2["display"]   # у Nominatim адрес полнее
        st.session_state["geocode_trace"] = trace
        return geo
    # Photon не нашёл/упал — полный ответ из Nominatim
    geo2 = _nominatim_geocode(city)
    trace.append(f"Nominatim: {'ok' if geo2 else 'fail'}")
    if geo2 is not None:
        st.session_state["geocode_trace"] = trace
        return geo2
    # внешние геокодеры лежат/банят — геокодим через Overpass (bbox границы)
    geo3 = _overpass_geocode(city)
    trace.append(f"Overpass: {'ok' if geo3 else 'fail'}")
    st.session_state["geocode_trace"] = trace
    return geo3


# --------------------------------------------------------------------------- #
#  Геокодинг адресов точек (только если в файле нет координат)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=86400, show_spinner=False)
def geocode_address(address: str, bbox=None):
    """Геокодинг адреса: 1) Nominatim (с вариантами написания),
    2) fallback — поиск по addr-тегам OSM с КОРОТКИМ таймаутом."""
    # варианты написания: «7к1» -> «7 к1», убрать «1-я» и лишние пробелы
    variants = [address]
    v = re.sub(r"(\d)\s*к\s*(\d)", r"\1 к\2", address)
    v = re.sub(r"\b[1-5]-я\b", "", v)
    v = re.sub(r"\s+", " ", v).strip(" ,")
    if v != address:
        variants.append(v)

    for cand in variants:
        try:
            _nominatim_throttle()
            import requests
            r = requests.get(NOMINATIM_URL, params={
                "q": cand, "format": "json", "limit": 1, "accept-language": "ru",
                "addressdetails": 1,
            }, headers=HEADERS, timeout=15)
            data = r.json() if r.status_code == 200 else []  # 429 — без ретраев
        except Exception:  # noqa: BLE001
            data = []
        if data:
            d = data[0]
            addr = d.get("address", {})
            street = ", ".join(x for x in (addr.get("road"),
                                           addr.get("house_number")) if x)
            city = (addr.get("city") or addr.get("town") or addr.get("village")
                    or addr.get("municipality") or "")
            label = ", ".join(x for x in (street, city) if x) or d["display_name"]
            return {"lat": float(d["lat"]), "lon": float(d["lon"]),
                    "display": label}

    # fallback: ищем по тегам addr:* в OSM — только 2 зеркала, жёсткий таймаут
    if bbox is None:
        return None
    m = re.search(r"(\d+)\s*[кk]\s*(\d+)", address)
    plain = re.search(r"(\d+)", address)
    if not (m or plain):
        return None
    d1 = m.group(1) if m else plain.group(1)
    d2 = m.group(2) if m else None
    hvars = ([f"{d1}к{d2}", f"{d1} к{d2}", f"{d1}К{d2}", d1] if d2 else [d1])
    hre = "^(" + "|".join(re.escape(v2) for v2 in hvars) + ")$"
    street = address[: (m.start() if m else plain.start())]
    street = re.sub(r"\b[1-5]-я\b", "", street)
    street = re.sub(r"\b(улица|ул\.?|переулок|проспект|пр-кт|бульвар|б-р|"
                    r"шоссе|ш\.)\b", "", street, flags=re.IGNORECASE).strip(" ,.")
    if not street:
        return None
    south, north, west, east = bbox
    bb = f"{south - 0.05},{west - 0.05},{north + 0.05},{east + 0.05}"
    q = f"""[out:json][timeout:25];(
  way["addr:housenumber"~"{hre}"]["addr:street"~"{street}",i]({bb});
  node["addr:housenumber"~"{hre}"]["addr:street"~"{street}",i]({bb});
);out center 3;"""
    els = []
    for url in OVERPASS_ENDPOINTS[:2]:
        try:
            import requests
            r = requests.post(url, data={"data": q}, headers=HEADERS, timeout=35)
            if r.status_code == 200:
                els = r.json().get("elements", [])
                break
        except Exception:  # noqa: BLE001
            time.sleep(1)
    if not els:
        return None
    el = els[0]
    lat = el.get("lat") or (el.get("center") or {}).get("lat")
    lon = el.get("lon") or (el.get("center") or {}).get("lon")
    if lat is None or lon is None:
        return None
    tags = el.get("tags", {})
    label = ", ".join(x for x in (tags.get("addr:street"),
                                  tags.get("addr:housenumber")) if x) or address
    return {"lat": float(lat), "lon": float(lon), "display": label}


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
def _query_overpass(q: str, timeout: int = 600, attempts: int = 2) -> dict:
    """timeout/attempts: у больших выгрузок данных — щедрые значения (600/2),
    у маленьких служебных запросов (геокодинг) — малые: мёртвое/тупящее
    зеркало с timeout=600 иначе вешает приложение на многие минуты."""
    errors = []
    for url in OVERPASS_ENDPOINTS:
        for attempt in range(attempts):  # заходы на зеркало с нарастающей паузой
            try:
                import requests
                r = requests.post(url, data={"data": q}, headers=HEADERS,
                                  timeout=timeout)
                if r.status_code == 200:
                    return r.json()
                errors.append(f"{url}: HTTP {r.status_code}")
            except Exception as e:  # noqa: BLE001
                errors.append(f"{url}: {type(e).__name__}")
            time.sleep(3 + 3 * attempt)
    raise RuntimeError("Overpass недоступен на всех зеркалах: "
                       + "; ".join(errors[-4:]))


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_city_data(city: str):
    """Две выгрузки Overpass: (1) way-объекты, (2) node-объекты.
    Возвращает (geo, nodes_df, ways_df) или (None, None, None), если город
    не найден ни в одном геокодере."""
    geo = geocode_city(city)
    if geo is None:
        return None, None, None
    if geo.get("geojson") and geo["geojson"].get("type") in ("Polygon", "MultiPolygon"):
        south, north, west, east = _geojson_bounds(geo["geojson"])  # гексы только в городе
    else:
        south, north, west, east = geo["bbox"]
    # запас на границах — чтобы объекты у края гексов не потерялись
    south, north, west, east = south - 0.02, north + 0.02, west - 0.02, east + 0.02
    bb = f"{south},{west},{north},{east}"

    q_ways = f"""
[out:json][timeout:300];
(
  way["building"~"^(apartments|residential|house|detached|semidetached_house|terrace|bungalow|dormitory)$"]({bb});
  way["highway"]({bb});
  way["amenity"]({bb});
  way["shop"]({bb});
  way["name"~"Ozon|Озон|Wildberries|Вайлдберриз|СДЭК|CDEK|Авито|Avito|Яндекс Маркет",i]({bb});
  way["brand"~"Ozon|Озон|Wildberries|Вайлдберриз|СДЭК|CDEK|Авито|Avito|Яндекс Маркет",i]({bb});
  way["emergency"="trauma_centre"]({bb});
  way["building"="dormitory"]({bb});
);
out geom;"""

    q_nodes = f"""
[out:json][timeout:300];
(
  node["amenity"]({bb});
  node["shop"]({bb});
  node["craft"]({bb});
  node["office"]({bb});
  node["leisure"]({bb});
  node["name"~"Ozon|Озон|Wildberries|Вайлдберриз|СДЭК|CDEK|Авито|Avito|Яндекс Маркет",i]({bb});
  node["brand"~"Ozon|Озон|Wildberries|Вайлдберриз|СДЭК|CDEK|Авито|Avito|Яндекс Маркет",i]({bb});
  node["highway"~"^(bus_stop|bus_station|tram_stop)$"]({bb});
  node["public_transport"~"^(platform|stop_position)$"]({bb});
  node["emergency"="trauma_centre"]({bb});
);
out body;"""

    ways_raw = _query_overpass(q_ways).get("elements", [])
    nodes_raw = _query_overpass(q_nodes).get("elements", [])

    if not ways_raw and not nodes_raw:
        # пустой ответ зеркала НЕ кэшируем: иначе «пустой город» сидит
        # в кэше 30 минут и выглядит как «всё сломалось навсегда»
        raise RuntimeError(
            "Overpass вернул пустой набор данных (0 ways и 0 nodes) — "
            "зеркало отвечает пустым результатом (перегрузка/сбой). "
            "Подождите минуту и повторите; при устойчивом повторении "
            "проверьте статус Overpass API.")

    ways, nodes = [], []
    for el in ways_raw:
        if "geometry" not in el:
            continue
        ways.append({
            "id": el["id"],
            "coords": [(p["lon"], p["lat"]) for p in el["geometry"]],
            "tags": el.get("tags", {}),
        })
    for el in nodes_raw:
        nodes.append({"id": el["id"], "lat": el["lat"], "lon": el["lon"],
                      "tags": el.get("tags", {})})

    ways_df = pd.DataFrame(ways, columns=["id", "coords", "tags"])
    nodes_df = pd.DataFrame(nodes, columns=["id", "lat", "lon", "tags"])
    return geo, nodes_df, ways_df

# --------------------------------------------------------------------------- #
#  Подготовка слоёв
# --------------------------------------------------------------------------- #
def ways_centroids(ways_df):
    """Центроиды way-объектов (широта/долгота) — для подсчёта POI/остановок-ways."""
    if ways_df.empty:
        return pd.DataFrame(columns=["lat", "lon", "tags"])
    lat = ways_df["coords"].apply(lambda c: np.mean([p[1] for p in c]))
    lon = ways_df["coords"].apply(lambda c: np.mean([p[0] for p in c]))
    return pd.DataFrame({"lat": lat, "lon": lon, "tags": ways_df["tags"]})


def _ring_coords(coords, min_pts):
    """Чистим кольцо/линию от подряд идущих дублей; None если точек < min_pts."""
    pts = []
    for p in coords:
        if not pts or p != pts[-1]:
            pts.append(p)
    return pts if len(pts) >= min_pts else None


def _safe_polygon(coords):
    pts = _ring_coords(coords, 3)
    if pts is None:
        return None
    try:
        poly = Polygon(pts)
        return poly if poly.is_valid else poly.buffer(0)
    except Exception:  # noqa: BLE001
        return None


def buildings_gdf(ways_df):
    """Жилые здания с площадью застройки (м², в метрической проекции) и этажностью."""
    if ways_df.empty:
        return gpd.GeoDataFrame(columns=["levels", "area", "lat", "lon"], geometry=[], crs="EPSG:4326")
    mask = ways_df["tags"].apply(lambda t: t.get("building") in RESIDENTIAL_BUILDINGS)
    sub = ways_df[mask]
    if sub.empty:
        return gpd.GeoDataFrame(columns=["levels", "area", "lat", "lon"], geometry=[], crs="EPSG:4326")
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
        return gpd.GeoDataFrame(columns=["levels", "area", "lat", "lon"],
                                geometry=[], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(recs, geometry=geoms, crs="EPSG:4326")
    metric = gdf.to_crs(gdf.estimate_utm_crs())
    gdf["area"] = metric.area.values
    return gdf


def roads_gdf(ways_df):
    """Дороги с длиной (км, в метрической проекции)."""
    if ways_df.empty:
        return gpd.GeoDataFrame(columns=["highway", "len_km", "lat", "lon"], geometry=[], crs="EPSG:4326")
    mask = ways_df["tags"].apply(lambda t: "highway" in t)
    sub = ways_df[mask]
    if sub.empty:
        return gpd.GeoDataFrame(columns=["highway", "len_km", "lat", "lon"], geometry=[], crs="EPSG:4326")
    from shapely.geometry import LineString
    recs, geoms = [], []
    for coords, tags in zip(sub["coords"], sub["tags"]):
        pts = _ring_coords(coords, 2)
        if pts is None:
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
        return gpd.GeoDataFrame(columns=["highway", "len_km", "lat", "lon"],
                                geometry=[], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(recs, geometry=geoms, crs="EPSG:4326")
    metric = gdf.to_crs(gdf.estimate_utm_crs())
    gdf["len_km"] = metric.length.values / 1000.0
    return gdf


def dorm_centroids(ways_df):
    """Общежития (building=dormitory) — центроиды для точек и подсчёта по гексам."""
    empty = pd.DataFrame(columns=["lat", "lon"])
    if ways_df.empty:
        return empty
    mask = ways_df["tags"].apply(lambda t: t.get("building") == "dormitory")
    sub = ways_df[mask]
    if sub.empty:
        return empty
    return pd.DataFrame({
        "lat": sub["coords"].apply(lambda c: float(np.mean([p[1] for p in c]))),
        "lon": sub["coords"].apply(lambda c: float(np.mean([p[0] for p in c]))),
    })


def parking_gdf(ways_df):
    """Парковки-way: площадь и центроид."""
    if ways_df.empty:
        return gpd.GeoDataFrame(columns=["area", "lat", "lon"], geometry=[], crs="EPSG:4326")
    mask = ways_df["tags"].apply(lambda t: t.get("amenity") == "parking")
    sub = ways_df[mask]
    if sub.empty:
        return gpd.GeoDataFrame(columns=["area", "lat", "lon"], geometry=[], crs="EPSG:4326")
    recs, geoms = [], []
    for coords in sub["coords"]:
        geom = _safe_polygon(coords)
        if geom is None or geom.is_empty:
            continue
        recs.append({"lat": float(np.mean([p[1] for p in coords])),
                     "lon": float(np.mean([p[0] for p in coords]))})
        geoms.append(geom)
    if not geoms:
        return gpd.GeoDataFrame(columns=["area", "lat", "lon"], geometry=[], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(recs, geometry=geoms, crs="EPSG:4326")
    metric = gdf.to_crs(gdf.estimate_utm_crs())
    gdf["area"] = metric.area.values
    return gdf


def _to_int(v, default):
    try:
        return max(1, int(float(v)))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
#  H3-сетка
# --------------------------------------------------------------------------- #
def _cells_from_polygon(coordinates, res):
    """coordinates — GeoJSON Polygon: [outer, hole1, ...] в порядке (lng, lat)."""
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
        poly = h3.LatLngPoly(outer, *holes)          # h3 >= 4.1
    except AttributeError:
        poly = {"type": "Polygon",
                "coordinates": [[[lng, lat] for lat, lng in outer]] +
                                [[[lng, lat] for lat, lng in h] for h in holes]}
    return h3.polygon_to_cells(poly, res)


def _geojson_bounds(geom):
    """south, north, west, east из GeoJSON Polygon/MultiPolygon."""
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    pts = [p for poly in polys for p in poly[0]]
    lngs = [p[0] for p in pts]
    lats = [p[1] for p in pts]
    return min(lats), max(lats), min(lngs), max(lngs)


def make_grid(geo, res):
    geom = geo.get("geojson")
    if geom and geom.get("type") in ("Polygon", "MultiPolygon"):
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        cells = set()
        for poly in polys:
            cells.update(_cells_from_polygon(poly, res))
        if cells:
            return sorted(cells), True
    # fallback: граница не нашлась (лимит Nominatim 0,5 МБ) — старый прямоугольник
    south, north, west, east = geo["bbox"]
    m = 0.02
    ring = [(south - m, west - m), (south - m, east + m),
            (north + m, east + m), (north + m, west - m),
            (south - m, west - m)]
    try:
        poly = h3.LatLngPoly(ring)
    except AttributeError:
        poly = {"type": "Polygon", "coordinates": [[[lng, lat] for lat, lng in ring]]}
    return sorted(h3.polygon_to_cells(poly, res)), False


def hex_area_km2(res):
    b = h3.cell_to_boundary(h3.latlng_to_cell(55.0, 83.0, res))
    ll = [(lng, lat) for lat, lng in b]
    gdf = gpd.GeoDataFrame(geometry=[Polygon(ll)], crs="EPSG:4326")
    return gdf.to_crs(gdf.estimate_utm_crs()).area.iloc[0] / 1e6


def hex_counts(points_df, res, mask=None):
    """Подсчёт точек по гексам. points_df: lat/lon [+ mask]."""
    if points_df is None or points_df.empty:
        return pd.Series(dtype=float)
    df = points_df if mask is None else points_df[mask]
    if df.empty:
        return pd.Series(dtype=float)
    cells = [h3.latlng_to_cell(a, b, res) for a, b in zip(df["lat"], df["lon"])]
    return pd.Series(cells).value_counts().astype(float)


def _norm100(s):
    """Мин-макс нормировка в индекс 0-100 (доля от максимума по городу)."""
    if s is None or s.empty:
        return pd.Series(dtype=float)
    mx = float(s.max())
    return s / mx * 100.0 if mx > 0 else s * 0.0


def populated_with_ring(grid, series):
    """Гексы с value>0 + кольцо соседей (k=1) вокруг них. Нули дальше кольца отбрасываем."""
    populated = set(series[series > 0].index)
    keep = set(populated)
    for cell in populated:
        keep.update(h3.grid_disk(cell, 1))
    return [c for c in grid if c in keep]


# --------------------------------------------------------------------------- #
#  Метрики по гексам (все 7 карт — логика без изменений)
# --------------------------------------------------------------------------- #
def compute_series(map_type, sub_option, res, data, kontur_df=None, m2_per_person=30):
    geo, nodes_df, ways_df = data
    wcent = ways_centroids(ways_df)

    if map_type.startswith("1."):
        if kontur_df is not None:  # реальная численность Kontur
            cells = [h3.cell_to_parent(c, res) if h3.get_resolution(c) > res else c
                     for c in kontur_df["h3"]]
            s = kontur_df.assign(cell=cells).groupby("cell")["population"].sum()
            return s, "чел.", None, None
        b = buildings_gdf(ways_df)
        if b.empty:
            return pd.Series(dtype=float), "чел. (оценка)", None, None
        cells = [h3.latlng_to_cell(a, b_, res) for a, b_ in zip(b["lat"], b["lon"])]
        vol = (b.assign(cell=cells, vol=b["area"] * b["levels"])
                 .groupby("cell")["vol"].sum())
        return vol / m2_per_person, f"чел. (суррогат, {m2_per_person} м²/чел)", None, None

    if map_type.startswith("2."):
        b = buildings_gdf(ways_df)
        if b.empty:
            return pd.Series(dtype=float), "м²", None, None
        cells = [h3.latlng_to_cell(a, b_, res) for a, b_ in zip(b["lat"], b["lon"])]
        vol = (b.assign(cell=cells, vol=b["area"] * b["levels"])
                 .groupby("cell")["vol"].sum())
        return vol, "м² суммарной площади жилых зданий", None, None

    if map_type.startswith("3."):
        # ---- Индекс спроса (суррогат платёжеспособности), 0-100 ----
        # 40% жилой фонд (люди) + 20% розница/общепит + 15% банки/офисы
        # + 15% остановки + 10% парковки/АЗС; каждый компонент в индексе 0-100
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
        wb_ = (wcent["tags"].apply(lambda t: t.get("amenity") in {"bank", "bureau_de_change"}
                                   or "office" in t))
        parts["biz"] = _norm100(pd.concat(
            [hex_counts(nodes_df, res, mb), hex_counts(wcent, res, wb_)]
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
        # компоненты — в тултип для прозрачности
        extra = pd.DataFrame(parts).reindex(idx.index).fillna(0).round(0)
        extra.columns = [f"c_{c}" for c in extra.columns]
        return idx.round(1), "индекс 0-100", extra, None

    if map_type.startswith("4."):
        # пустой выбор / Select all = все подразделы; точка попадает строго
        # в первую подходящую категорию (без двойного счёта)
        cats = [c for c in POI_CATEGORIES if not sub_option or c in sub_option]
        matchers = [(name, POI_CATEGORIES[name][0], POI_CATEGORIES[name][1])
                    for name in cats]

        def _classify(tags):
            for name, color, m in matchers:
                if m(tags):
                    return name, color
            return None

        counts = {name: {} for name, _, _ in matchers}
        points = []
        for df_ in (nodes_df, wcent):
            for _, row in df_.iterrows():
                hit = _classify(row["tags"])
                if hit is None:
                    continue
                name, color = hit
                cell = h3.latlng_to_cell(row["lat"], row["lon"], res)
                counts[name][cell] = counts[name].get(cell, 0) + 1
                points.append({"lat": row["lat"], "lon": row["lon"],
                               "color": color, "type": name,
                               "label": row["tags"].get("name", name)})
        type_series = {name: pd.Series(c) for name, c in counts.items()}
        extra = pd.DataFrame(type_series).fillna(0).round(0)
        total = extra.sum(axis=1) if len(extra) else pd.Series(dtype=float)
        return total, "объектов", extra, points

    if map_type.startswith("5."):
        groups = sub_option or list(SOCIAL_GROUPS)
        type_series, points = {}, []
        for name in groups:
            color, matcher = SOCIAL_GROUPS[name]
            if matcher is None:  # общежития — отдельный источник
                d = dorm_centroids(ways_df)
                type_series[name] = hex_counts(d, res)
                for _, row in d.iterrows():
                    points.append({"lat": row["lat"], "lon": row["lon"],
                                   "color": color, "type": name,
                                   "label": "общежитие"})
                continue
            mn = nodes_df["tags"].apply(matcher)
            mw = wcent["tags"].apply(matcher)
            type_series[name] = pd.concat(
                [hex_counts(nodes_df, res, mn), hex_counts(wcent, res, mw)]
            ).groupby(level=0).sum()
            for df_ in (nodes_df[mn], wcent[mw]):
                for _, row in df_.iterrows():
                    points.append({"lat": row["lat"], "lon": row["lon"],
                                   "color": color, "type": name,
                                   "label": row["tags"].get("name", name)})
        extra = pd.DataFrame(type_series).fillna(0).round(0)
        total = extra.sum(axis=1) if len(extra) else pd.Series(dtype=float)
        return total, "объектов", extra, points

    if map_type.startswith("6."):
        # ---- Трафик: оба показателя нормируем в индекс 0-100 ----
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

    if map_type.startswith("7."):
        # ---- Мед. объекты: разбивка по типам + точки на карту ----
        types = sub_option or list(COMPETITOR_TYPES)
        type_series, points = {}, []
        for name in types:
            color, matcher = COMPETITOR_TYPES[name]
            mn = nodes_df["tags"].apply(matcher)
            mw = wcent["tags"].apply(matcher)
            type_series[name] = pd.concat(
                [hex_counts(nodes_df, res, mn), hex_counts(wcent, res, mw)]
            ).groupby(level=0).sum()
            for df_ in (nodes_df[mn], wcent[mw]):
                for _, row in df_.iterrows():
                    points.append({"lat": row["lat"], "lon": row["lon"],
                                   "color": color, "type": name,
                                   "label": row["tags"].get("name", name)})
        extra = pd.DataFrame(type_series).fillna(0).round(0)
        total = extra.sum(axis=1) if len(extra) else pd.Series(dtype=float)
        return total, "мед. объектов", extra, points

    return pd.Series(dtype=float), "", None, None


# --------------------------------------------------------------------------- #
#  Агрегаты с кэшем: данные города тянем из session по версии, а не хэшируем
#  многомегабайтные DataFrame при каждом rerun
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=3600, show_spinner=False)
def _compute_series_cached(map_type, sub_option, res_eff, data_ver,
                           kontur_loaded, m2_per_person):
    stored = st.session_state["data"]
    _kontur = st.session_state.get("_kontur") if kontur_loaded else None
    return compute_series(map_type, sub_option, res_eff,
                          (stored["geo"], stored["nodes"], stored["ways"]),
                          kontur_df=_kontur, m2_per_person=m2_per_person)

# --------------------------------------------------------------------------- #
#  Тултип без табличной вёрстки folium (компактные строки «подпись: значение»)
# --------------------------------------------------------------------------- #
class HexTooltip(folium.GeoJsonTooltip):
    """Тултип БЕЗ табличной вёрстки folium: каждая строка — 'подпись: значение'
    инлайном. Штатный шаблон рисует <table>, где колонка подписей шириной
    с самую длинную подпись — после коротких подписей получаются дыры."""
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
        """Стиль тултипа (как у штатного GeoJsonTooltip) + рендер БЕЗ
        assert'ов GeoJsonDetail.render: штатный рендер падает с
        AssertionError, если хотя бы одно поле fields отсутствует в
        properties хотя бы одной фичи. Наш шаблон безопасно показывает
        пустое значение для отсутствующего поля."""
        from branca.element import Element as _El
        if getattr(self, "style", None):
            self._parent.get_root().header.add_child(
                _El(f"<style>.{self.class_name}{{{self.style}}}</style>"),
                name=f"tooltip_style_{self.get_name()}",
            )
        _El.render(self, **kwargs)

    # NB: ТОЛЬКО встроенные фильтры jinja2 (tojson/safe). Кастомный фильтр
    # folium "tojavascript" живёт в его собственном окружении — в сыром
    # jinja2.Template его нет и компиляция падает с TemplateAssertionError.
    _template = _Jinja2Template(
        """
    {% macro script(this, kwargs) %}
    {{ this._parent.get_name() }}.bindTooltip("""
        + base_template + """,{{ this.tooltip_options | tojson }});
                     {% endmacro %}
                     """
    )


# --------------------------------------------------------------------------- #
#  Отрисовка карты: «капельки» адресов + (в режиме гексов) сетка H3
# --------------------------------------------------------------------------- #
def render_map(center, grid, series, unit, map_type, marker_points=None,
               points=None, hex_extra=None, extra_aliases=None, legend=None,
               source=None, yzoom=15, npoints=None):
    # prefer_canvas: векторы рисуются на canvas — сотни тысяч полигонов без лагов
    m = folium.Map(location=center, tiles="OpenStreetMap", control_scale=True,
                   prefer_canvas=True)
    folium.TileLayer("OpenTopoMap", name="Топографическая", show=False).add_to(m)

    # верстка тултипов: подпись и значение идут плотно, без разъезда
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

    # ── «капельки» — метки загруженных адресов ─────────────────────────────
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

    if grid:  # режим гексов
        vals = series.reindex(grid).fillna(0.0)
        vmax = float(vals.max()) if len(vals) else 0.0
        if vmax <= 0:
            st.warning("Нет данных для выбранного слоя в этом городе.")
            vmax = 1.0
        cm = LinearColormap(COLORS, vmin=0, vmax=1)  # цвет по доле максимума
        # гамма < 1 сжимает низ шкалы: жёлтый начинается заметно раньше.
        # Легенда сэмплирована с ТЕМ ЖЕ отображением — цвета на шкале совпадают
        # с реальной заливкой (прямое интерполирование между hex-стопами).
        def _palette_at(t):
            pos = min(max(t, 0.0), 1.0) * (len(COLORS) - 1)
            i = min(int(pos), len(COLORS) - 2)
            frac = pos - i
            out = []
            for k in (1, 3, 5):
                a, b = int(COLORS[i][k:k + 2], 16), int(COLORS[i + 1][k:k + 2], 16)
                out.append(f"{round(a + (b - a) * frac):02x}")
            return "#" + "".join(out)

        cm_legend = LinearColormap([_palette_at((i / 24) ** GAMMA) for i in range(25)],
                                   vmin=0, vmax=vmax)
        cm_legend.caption = f"{map_type} — {unit}"

        extra_cols = list(hex_extra.columns) if hex_extra is not None else []
        # ключи свойств — машинно-безопасные (x0, x1…): русские ключи с пробелами
        # вставляются folium в JS-тултип и могут уронить весь скрипт карты
        safe_cols = {c: f"x{i}" for i, c in enumerate(extra_cols)}
        # вся сетка — один GeoJSON FeatureCollection (быстро на тысячах гексов)
        feats = []
        for cell in grid:
            boundary = h3.cell_to_boundary(cell)          # [(lat,lng),...]
            ring = [[lng, lat] for lat, lng in boundary]
            ring.append(ring[0])
            v = float(vals.get(cell, 0.0))
            rv = round(v, 2) if vmax < 100 else round(v)  # большие числа — целыми
            props = {"v": rv,
                     "n": int((npoints or {}).get(cell, 0))}  # точек из файла
            if source:
                props["src"] = source
            # бабл гекса: ссылка на Яндекс.Карты по его центру
            _lat, _lon = h3.cell_to_latlng(cell)
            props["link"] = (f'<a href="https://yandex.ru/maps/?pt={_lon:.6f},{_lat:.6f}'
                             f'&z={yzoom}&l=map" target="_blank" rel="noopener">'
                             f'🗺 Открыть в Яндекс.Картах</a>')
            # доп. поля обязаны быть у КАЖДОГО гекса, иначе folium падает на тултипе
            for c in extra_cols:
                props[safe_cols[c]] = float(hex_extra.loc[cell, c]) \
                    if cell in hex_extra.index else 0
            feats.append({
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            })
        gj = {"type": "FeatureCollection", "features": feats}

        def _style(f):
            v = f["properties"]["v"]
            # гамма-раскрытие: жёлтое начинается раньше, красный уходит к вершине
            t = (v / vmax) ** GAMMA if vmax > 0 else 0.0
            opacity = 0.03 if v <= 0 else 0.20 + 0.40 * t
            return {"fillColor": cm(t), "color": "#555555", "weight": 0.6,
                    "fillOpacity": opacity}

        extra_aliases = extra_aliases or {}
        src_fields = ["src"] if source else []
        aliases = [f"{unit}: ", "Точек в гексе: "] + \
                  [extra_aliases.get(c, c) for c in extra_cols] + \
                  (["Источник: "] if source else [])
        folium.GeoJson(
            gj,
            style_function=_style,
            tooltip=HexTooltip(
                fields=["v", "n"] + [safe_cols[c] for c in extra_cols] + src_fields,
                aliases=aliases, localize=True,
            ),
            popup=folium.GeoJsonPopup(fields=["link"], aliases=[""], labels=False,
                                      localize=False, max_width=280),
            name="Гексы",
        ).add_to(m)

        cm_legend.add_to(m)

        # легенда типов точек — каноничный паттерн MacroElement
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

    # точки объектов поверх гексов — ОДИН GeoJSON-слой (на тысячах точек
    # отдельные CircleMarker с тултипами ломают/тормозят карту)
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
            marker=folium.CircleMarker(radius=5, weight=1.5,
                                       fill=True, fill_opacity=0.9),
            style_function=lambda f: {"color": f["properties"]["c"],
                                      "fillColor": f["properties"]["c"]},
            tooltip=folium.GeoJsonTooltip(fields=["tip"], aliases=[""],
                                          localize=False),
            name="Объекты",
        ).add_to(m)

    # границы: сетка гексов, иначе точки
    lats, lngs = [], []
    if grid:
        for c in grid:
            for lat_, lng_ in h3.cell_to_boundary(c):
                lats.append(lat_); lngs.append(lng_)
    elif marker_points is not None and not marker_points.empty:
        lats = marker_points["lat"].tolist()
        lngs = marker_points["lon"].tolist()
    if lats:
        m.fit_bounds([[min(lats), min(lngs)], [max(lats), max(lngs)]])
    folium.LayerControl().add_to(m)
    st_folium(m, width=None, height=680, returned_objects=[])


# --------------------------------------------------------------------------- #
#  Kontur Population (локальный файл)
# --------------------------------------------------------------------------- #
def load_kontur(file, res, bbox=None):
    """Файл Kontur Population (.gpkg/.geojson/.parquet) -> DataFrame[h3, population].
    bbox=[south, north, west, east] — обрезать под город."""
    name = file.name.lower()
    tmp_path = None
    if name.endswith(".gpkg"):
        # pyogrio читает только с пути — пишем во временный файл
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name
        src_path = tmp_path
    else:
        src_path = file

    try:
        try:
            import pyogrio
            info = pyogrio.read_info(src_path)
            fields = list(info["fields"])
            lower = {c.lower(): c for c in fields}
            pop_col = next((lower[k] for k in ("population", "pop", "count")
                            if k in lower), None)
            if pop_col is None:
                st.error(f"Колонка population не найдена. Колонки файла: {fields[:15]}")
                return None

            if "h3" in lower:
                # быстрый путь: только h3+population, без геометрии
                df = pyogrio.read_dataframe(src_path, read_geometry=False,
                                            columns=[lower["h3"], pop_col])
                df = df.rename(columns={lower["h3"]: "h3", pop_col: "population"})
                raw = df["h3"]
                cells = raw.astype(str) \
                    if (raw.dtype == object or str(raw.dtype).startswith("str")) \
                    else raw.apply(lambda x: format(int(x), "x"))
                out = pd.DataFrame({
                    "h3": cells,
                    "population": pd.to_numeric(df["population"],
                                                errors="coerce").fillna(0)})
                if bbox is not None:  # фильтр по центрам ячеек
                    south, north, west, east = bbox
                    ll = out["h3"].map(h3.cell_to_latlng)
                    keep = ll.map(lambda p: south - 0.1 <= p[0] <= north + 0.1
                                  and west - 0.1 <= p[1] <= east + 0.1)
                    out = out[keep]
            else:
                gdf = pyogrio.read_dataframe(src_path, columns=[pop_col])
                gdf = gdf.rename(columns={pop_col: "population"})
                if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
                    gdf = gdf.to_crs(4326)  # Kontur иногда в 3857
                if bbox is not None:
                    south, north, west, east = bbox
                    gdf = gdf.cx[west - 0.05:east + 0.05, south - 0.05:north + 0.05]
                if gdf.empty:
                    st.error("Kontur-файл не пересекается с городом. Проверьте, что "
                             "скачан датасет «400m H3 Hexagons», а не «Administrative "
                             "Division», и что город загружен.")
                    return None
                cent = gdf.geometry.centroid
                cells = [h3.latlng_to_cell(y, x, res)
                         for y, x in zip(cent.y, cent.x)]
                out = pd.DataFrame({
                    "h3": cells,
                    "population": pd.to_numeric(gdf["population"],
                                                errors="coerce").fillna(0)})
        except ImportError:
            gdf = gpd.read_file(src_path)
            if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(4326)
            if bbox is not None:
                south, north, west, east = bbox
                gdf = gdf.cx[west - 0.05:east + 0.05, south - 0.05:north + 0.05]
            if gdf.empty:
                st.error("Kontur-файл не пересекается с городом.")
                return None
            pop_col = next((c for c in gdf.columns if c.lower() in
                            ("population", "pop", "count")), None)
            if pop_col is None:
                st.error("В файле нет колонки population.")
                return None
            cent = gdf.geometry.centroid
            cells = [h3.latlng_to_cell(y, x, res)
                     for y, x in zip(cent.y, cent.x)]
            out = pd.DataFrame({"h3": cells,
                                "population": gdf[pop_col].astype(float).values})
    finally:
        if tmp_path:
            import os
            os.unlink(tmp_path)

    if out.empty:
        st.error("После обрезки под город данные пусты — файл точно датасет "
                 "«Population Density for 400m H3 Hexagons»?")
        return None
    out = out[out["population"] > 0]
    src_res = h3.get_resolution(out["h3"].iloc[0])
    if res > src_res:
        st.warning(f"Kontur идёт в res{src_res}: показ возможен только при "
                   f"res ≤ {src_res}. Понижаю детализацию до res{src_res}.")
    return out

# --------------------------------------------------------------------------- #
#  UI
# --------------------------------------------------------------------------- #
st.title("🗺️ OSM Point & Hex Analytics")
st.caption("Загрузите таблицу адресов → метки на карте → включите «Режим гексов»: "
           "весь город покроется сеткой H3 со всеми картами аналитики. "
           "Источники: OpenStreetMap (Overpass API), Photon/Nominatim, "
           "опционально Kontur Population. Все API бесплатные.")

# ------------------------------- сайдбар ---------------------------------- #
with st.sidebar:
    st.header("1. Данные")
    uploaded = st.file_uploader("Файл со списком адресов (.xlsx / .csv)",
                                type=["xlsx", "xls", "csv"])

    st.header("2. Гексы")
    hex_mode = st.toggle("Режим гексов", value=False)
    map_type = st.radio("Тип карты", MAP_TYPES, index=0)

    sub_option = None
    if map_type.startswith("4."):
        sub_option = st.multiselect(
            "Категории POI", list(POI_CATEGORIES.keys()),
            default=list(POI_CATEGORIES.keys()),
            help="Пустой выбор или Select all = все категории. Цвет гекса — "
                 "суммарное число объектов, точки окрашены по категориям")
    elif map_type.startswith("5."):
        sub_option = st.multiselect("Группы соц. инфраструктуры",
                                    list(SOCIAL_GROUPS.keys()),
                                    default=list(SOCIAL_GROUPS.keys()))
    elif map_type.startswith("6."):
        sub_option = st.radio("Что раскрашиваем", TRAFFIC_MODES,
                              help="Тултип гекса показывает оба индекса")
    elif map_type.startswith("7."):
        sub_option = st.multiselect(
            "Типы мед. объектов", list(COMPETITOR_TYPES.keys()),
            default=list(COMPETITOR_TYPES.keys()),
            help="Пустой выбор или Select all = все типы")

    m2_per_person = 30
    kontur_df = None
    if map_type.startswith("1."):
        src = st.radio("Источник численности",
                       ["Суррогатная оценка (OSM-здания)", "Kontur Population (файл)"],
                       key="pop_src")  # явный ключ — состояние переживает
        # переключение на другой тип карты
        if src.startswith("Суррогат"):
            m2_per_person = st.slider("Норма м² жилья на человека", 15, 60, 30)
        else:
            f = st.file_uploader("Файл Kontur Population (.gpkg / .geojson / .parquet)",
                                 type=["gpkg", "geojson", "json", "parquet"],
                                 key="kontur_file")
            # Распарсенные данные живут в session_state и ПЕРЕЖИВАЮТ уход
            # с карты 1: виджет uploader при возврате может быть пустым,
            # но повторно парсить мегабайты не нужно — подхватываем из кэша.
            kontur_df = st.session_state.get("_kontur")
            if f is not None:
                _kkey = (f.name, f.size)
                if st.session_state.get("_kontur_key") != _kkey or kontur_df is None:
                    with st.spinner("Загружаю Kontur Population…"):
                        _prev = (st.session_state.get("data") or {}).get("geo")
                        kontur_df = load_kontur(f, 8,
                                                _prev["bbox"] if _prev else None)
                    if kontur_df is not None:
                        st.session_state["_kontur"] = kontur_df
                        st.session_state["_kontur_key"] = _kkey
            elif kontur_df is not None:
                st.caption("Kontur Population загружен ранее — файл можно "
                           "не перезагружать.")

    # слайдер разрешения: если выбран Kontur, его res ограничивает максимум
    _max_res = 10
    if map_type.startswith("1.") and kontur_df is not None and not kontur_df.empty:
        _max_res = h3.get_resolution(kontur_df["h3"].iloc[0])
        if st.session_state.get("res_slider", 8) > _max_res:
            # слайдер ещё не отрисован в этом прогоне — менять значение можно
            st.session_state["res_slider"] = _max_res

    res = st.select_slider(
        "Размер гекса (resolution)",
        options=[7, 8, 9, 10], value=8, key="res_slider",
        help="Res 7 ≈ 2,4 км между центрами · Res 8 ≈ 0,92 км · "
             "Res 9 ≈ 0,35 км · Res 10 ≈ 0,13 км.",
    )
    if _max_res < 10:
        st.caption(f"⚠️ Kontur Population рассчитан в res{_max_res} — "
                   f"детализация выше res{_max_res} недоступна. "
                   f"Переключитесь на суррогатный источник для res 9–10.")

    st.divider()
    if st.button("🗑 Очистить кэш"):
        st.cache_data.clear()
        st.session_state.pop("data", None)
        st.success("Кэш очищен — загрузите данные заново")

# ------------------------------ точки ------------------------------------- #
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
    st.warning(f"Геокодируем {len(df)} адресов через Nominatim/Photon "
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

# город анализа: из колонки «Город», иначе — вручную
city_query = ", ".join(cities) if cities else ""
if not cities:
    manual = st.text_input("Не удалось определить город из файла. "
                           "Введите город для анализа гексов:")
    if manual.strip():
        city_query = manual.strip()

st.success(f"Точек на карте: **{len(marker_points)}** | Город анализа: "
           f"**{city_query or 'не определён — гексы недоступны'}**")
center = [marker_points["lat"].mean(), marker_points["lon"].mean()]

# ------------------------------ гексы ------------------------------------- #
if not hex_mode:
    render_map(center, None, None, "", "", marker_points=marker_points)
    st.stop()

if not city_query:
    st.error("Не определён город — укажите его в поле выше.")
    st.stop()

# данные города: грузим, если ещё не загружены или сменился город
stored = st.session_state.get("data")
if stored is None or stored.get("city") != city_query:
    st.session_state.pop("geocode_trace", None)  # свежая диагностика
    with st.spinner("Загружаю данные OpenStreetMap через Overpass API "
                    "(обычно 1–5 минут; города-миллионники — до 10–15)…"):
        try:
            geo, nodes_df, ways_df = fetch_city_data(city_query)
        except Exception as e:  # noqa: BLE001 — любая сетевая/парсер-ошибка
            st.error(f"Не удалось загрузить данные для «{city_query}»: {e}")
            _trace = st.session_state.get("geocode_trace")
            if _trace:
                with st.expander("Диагностика геокодеров (для поддержки)"):
                    st.code("\n".join(_trace))
            st.stop()
    if geo is None:
        st.error("Город не найден. Уточните название.")
        _trace = st.session_state.get("geocode_trace")
        if _trace:
            with st.expander("Диагностика геокодеров (для поддержки)"):
                st.code("\n".join(_trace))
        st.stop()
    # город хранится ВМЕСТЕ с данными — экран всегда знает, что показывает
    st.session_state["data"] = {"city": city_query, "geo": geo,
                                "nodes": nodes_df, "ways": ways_df,
                                # версия данных — ключ кэша агрегатов
                                "ver": st.session_state.get("_data_ver", 0) + 1}
    st.session_state["_data_ver"] = st.session_state["data"]["ver"]
    stored = st.session_state["data"]

geo, nodes_df, ways_df = stored["geo"], stored["nodes"], stored["ways"]
st.success(f"📍 {geo['display']}")
st.caption(f"Выгрузка OSM: {len(ways_df):,} ways · {len(nodes_df):,} nodes")

# Kontur идёт в своём resolution: сетку принудительно понижаем под источник,
# иначе ячейки разных res не совпадут и фильтр заселённости обнулит сетку
res_eff = res
if map_type.startswith("1.") and kontur_df is not None and not kontur_df.empty:
    src_res = h3.get_resolution(kontur_df["h3"].iloc[0])
    if res > src_res:
        res_eff = src_res
        st.info(f"Kontur Population рассчитан в res{src_res} — сетка понижена "
                f"с res {res} до res{src_res}. Повысить детализацию можно только "
                f"с суррогатным источником.")

grid, in_boundary = make_grid(geo, res_eff)
if not grid:
    st.error("Сетка H3 получилась пустой. Попробуйте другое название города "
             "или перезагрузите приложение (Ctrl+F5).")
    st.stop()
if not in_boundary:
    st.info("Граница города не получена от Nominatim (лимит 0,5 МБ) — сетка "
            "построена по прямоугольной области. Уточните название города "
            "или повторите попытку.")
if len(grid) > MAX_GRID_CELLS:
    st.error(f"Сетка слишком велика ({len(grid)} гексов при res {res}, "
             f"лимит {MAX_GRID_CELLS}). Понизьте resolution.")
    st.stop()

with st.spinner("Считаю агрегаты по гексам…"):
    series, unit, hex_extra, points = _compute_series_cached(
        map_type, sub_option, res_eff, stored.get("ver", 0),
        kontur_df is not None, m2_per_person)

# точки загруженного списка по гексам (тултип «Точек в гексе»).
# Counter (словарь) — без pandas: надёжнее и в .map(), и в render_map
from collections import Counter
pt_cells = Counter(h3.latlng_to_cell(a, b, res_eff) for a, b in
                   zip(marker_points["lat"], marker_points["lon"]))

# суррогатные карты 1–2: не рисуем гексы с 0, кроме кольца вокруг заселённых
if map_type.startswith(("1.", "2.")):
    grid = populated_with_ring(grid, series)
    if not grid:
        _hint = ""
        if map_type.startswith(("1.", "2.")):
            _hint = (" Для карт 1–2 вероятная причина — слабая разметка жилых "
                     "зданий в OSM по этому городу; для точной численности "
                     "загрузите Kontur Population (карта 1).")
        st.warning("Нет данных для выбранного слоя в этом городе — показать "
                   "нечего." + _hint)
        st.stop()

if map_type.startswith("3."):
    st.info("**Из чего складывается индекс спроса.** Каждая строка тултипа — "
            "отдельный показатель, нормированный в индекс 0–100 относительно "
            "максимума по городу. **Жилой фонд** — м² жилых зданий (площадь × "
            "этажность). **Розница/общепит** — непродовольственные магазины, "
            "кафе и рестораны. **Банки/офисы** — банки, обменники, офисы. "
            "**Остановки** — остановки ОТ. **Парковки/АЗС** — парковки и "
            "заправки. Итоговый индекс = 0.40·ЖилойФонд + 0.20·Розница + "
            "0.15·Банки + 0.15·Остановки + 0.10·Авто.")

if map_type.startswith("6."):
    st.info("**Как считается индекс трафика.** Суммарная длина дорог нужного "
            "класса внутри гекса, делённая на площадь гекса (км/км²), затем "
            "нормированная в индекс 0–100 относительно самого загруженного "
            "гекса города. **Пешеходный**: footway, pedestrian, path, steps, "
            "cycleway, living_street. **Автомобильный**: motorway–tertiary и "
            "съезды. Тултип каждого гекса показывает оба индекса сразу. "
            "Это интенсивность сети, а не реальные потоки машин.")

c1, c2, c3 = st.columns(3)
c1.metric("Гексов в сетке", f"{len(grid):,}")
c2.metric("Resolution", f"res {res_eff} (~{RES_SPACING_KM[res_eff]} км между центрами)")
c3.metric("Максимум в ячейке", f"{series.max():,.0f} {unit}" if len(series) else "—")

legend = None
if map_type.startswith("4."):
    _sel = [c for c in POI_CATEGORIES if not sub_option or c in sub_option]
    legend = [(name, color) for name, (color, _) in POI_CATEGORIES.items()
              if name in _sel]
elif map_type.startswith("5."):
    legend = [(name, color) for name, (color, _) in SOCIAL_GROUPS.items()
              if sub_option is None or name in sub_option]
elif map_type.startswith("7."):
    legend = [(name, color) for name, (color, _) in COMPETITOR_TYPES.items()
              if sub_option is None or name in sub_option]

extra_aliases = None
if map_type.startswith("3."):
    extra_aliases = {"c_people": "Жилой фонд: ", "c_retail": "Розница/общепит: ",
                     "c_biz": "Банки/офисы: ", "c_transit": "Остановки: ",
                     "c_auto": "Парковки/АЗС: "}
elif map_type.startswith("4."):
    _sel = [c for c in POI_CATEGORIES if not sub_option or c in sub_option]
    extra_aliases = {name: f"{name}: " for name in POI_CATEGORIES if name in _sel}
elif map_type.startswith("6."):
    extra_aliases = {"ped": "Пешеходный трафик: ",
                     "auto": "Автомобильный трафик: "}
elif map_type.startswith("5."):
    extra_aliases = {name: f"{name}: " for name in SOCIAL_GROUPS
                     if sub_option is None or name in sub_option}
elif map_type.startswith("7."):
    extra_aliases = {name: f"{name}: " for name in COMPETITOR_TYPES
                     if sub_option is None or name in sub_option}

_src = None
if map_type.startswith("1."):
    _src = ("Kontur (2023)" if kontur_df is not None else "OSM: суррогат")

render_map(center, grid, series, unit, map_type, marker_points=marker_points,
           points=points, hex_extra=hex_extra, extra_aliases=extra_aliases,
           legend=legend, source=_src,
           yzoom={7: 13, 8: 15, 9: 16, 10: 17}.get(res_eff, 15),
           npoints=pt_cells)

if map_type.startswith("1.") and kontur_df is None:
    st.caption("⚠️ Оценки по OSM-зданиям — суррогатные: не учитывают реальное "
               "заселение и незавершённое строительство. Для точной численности "
               "загрузите Kontur Population (data.humdata.org, датасет "
               "«Population Density for 400m H3 Hexagons»).")

# таблица по гексам + выгрузка
st.subheader(f"Данные по гексам — {city_query}")
tbl = pd.DataFrame({"cell": grid})
tbl["value"] = tbl["cell"].map(series.reindex(grid)).fillna(0)
tbl["точек_в_гексе"] = tbl["cell"].map(dict(pt_cells)).fillna(0).astype(int)
if hex_extra is not None:
    tbl = tbl.merge(hex_extra, left_on="cell", right_index=True, how="left")
st.dataframe(tbl.sort_values("value", ascending=False),
             use_container_width=True)
csv = tbl.to_csv(index=False).encode("utf-8-sig")
st.download_button("⬇ Скачать таблицу гексов (CSV)", csv, "hexes.csv",
                   "text/csv")
