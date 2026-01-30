#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(same header)
"""

import os
import sys
import logging
import zipfile
import io
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from affine import Affine
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import ee
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds
from rasterio.features import geometry_mask
import folium
from folium.plugins import FloatImage, MeasureControl, MousePosition, Fullscreen
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ═══════════════════════════════════════════════════════════════════
# CONFIGURAÇÕES
# ═══════════════════════════════════════════════════════════════════

PROJECT_ID = "earth-484311"
DAYS_BACK = 60
MAX_CLOUD_COVER = 30
OUTPUT_DIR = Path("out")
TALHOES_DIR = OUTPUT_DIR / "talhoes"
SHAPEFILE_PATH = Path("CEM/CEM.shp")
OUTPUT_GEOTIFF_COMPLETO = OUTPUT_DIR / "ndvi_completo.tif"
OUTPUT_HTML = OUTPUT_DIR / "mapa_ndvi.html"
OUTPUT_CSV = OUTPUT_DIR / "estatisticas.csv"
OUTPUT_LOG = OUTPUT_DIR / "pipeline.log"

SCALE = 10

TALHAO_ID_FIELD = "TALHAO"
TALHAO_NOME_FIELD = "NOME_FAZ"

MAX_WORKERS = 15


# ═══════════════════════════════════════════════════════════════════
# SISTEMA DE LOGS (Thread-safe)
# ═══════════════════════════════════════════════════════════════════

log_lock = Lock()

def setup_logging():
    OUTPUT_DIR.mkdir(exist_ok=True)
    TALHOES_DIR.mkdir(exist_ok=True)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(OUTPUT_LOG, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def log_safe(level, message):
    with log_lock:
        if level == 'info':
            logging.info(message)
        elif level == 'warning':
            logging.warning(message)
        elif level == 'error':
            logging.error(message)


# ═══════════════════════════════════════════════════════════════════
# EARTH ENGINE
# ═══════════════════════════════════════════════════════════════════

def initialize_earth_engine():
    logging.info("🌍 Inicializando Google Earth Engine...")

    try:
        ee.Initialize(project=PROJECT_ID, opt_url='https://earthengine-highvolume.googleapis.com')
        logging.info("✅ Earth Engine inicializado")
        return True
    except Exception:
        logging.warning("⚠️ Tentando autenticação...")
        try:
            ee.Authenticate()
            ee.Initialize(project=PROJECT_ID, opt_url='https://earthengine-highvolume.googleapis.com')
            logging.info("✅ Earth Engine inicializado")
            return True
        except Exception as e:
            logging.error(f"❌ Falha: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════
# PROCESSAMENTO SENTINEL-2
# ═══════════════════════════════════════════════════════════════════

def mask_clouds(image):
    scl = image.select('SCL')
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    return image.updateMask(mask)

def calculate_ndvi(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

def create_ndvi_for_geometry(geometry, start_date, end_date):
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(geometry)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_COVER)))

    collection = collection.map(mask_clouds).map(calculate_ndvi)
    composite = collection.median()
    ndvi = composite.select('NDVI').clip(geometry)
    return ndvi


# ═══════════════════════════════════════════════════════════════════
# PROCESSAMENTO DE TALHÕES
# ═══════════════════════════════════════════════════════════════════

def shapely_to_ee_geometry(shapely_geom):
    if shapely_geom.geom_type == 'Polygon':
        coords = [list(shapely_geom.exterior.coords)]
        return ee.Geometry.Polygon(coords)
    elif shapely_geom.geom_type == 'MultiPolygon':
        coords = [list(poly.exterior.coords) for poly in shapely_geom.geoms]
        return ee.Geometry.MultiPolygon(coords)
    else:
        raise ValueError(f"Tipo não suportado: {shapely_geom.geom_type}")

def download_ndvi_talhao(ndvi_image, geometry, talhao_id, output_path, session=None, timeout=300):
    """
    Faz download do NDVI de um talhão específico.
    - Thread-safe: use 1 Session compartilhada entre threads (recomendado) OU passe None.
    - Suporta resposta ZIP do EE.
    - Download em streaming + escrita atômica.
    - Retry com backoff para 429/5xx/timeouts.

    Returns:
        True se salvou o GeoTIFF com sucesso, False caso contrário.
    """
    import time
    from pathlib import Path
    import zipfile
    import io
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Session (recomendado): melhora throughput e evita "pool is full"
    local_session = session
    created_here = False

    if local_session is None:
        local_session = requests.Session()
        created_here = True

        # Retry automático no nível HTTP (GET)
        retry = Retry(
            total=0,  # vamos controlar retry manualmente abaixo (mais previsível)
            connect=0,
            read=0,
            redirect=0,
            status=0,
            raise_on_status=False,
        )

        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=max(20, MAX_WORKERS * 2),
            pool_maxsize=max(20, MAX_WORKERS * 2),
        )
        local_session.mount("https://", adapter)
        local_session.mount("http://", adapter)

    # retry manual (com backoff)
    max_attempts = 5
    backoff_base = 2.0  # segundos

    # arquivo temporário (escrita atômica)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")

    try:
        # URL do Earth Engine
        url = ndvi_image.getDownloadURL({
            "scale": SCALE,
            "region": geometry,
            "format": "GEO_TIFF",
            "crs": "EPSG:4326",
        })

        for attempt in range(1, max_attempts + 1):
            try:
                # GET streaming
                with local_session.get(url, stream=True, timeout=timeout) as resp:
                    status = resp.status_code

                    # Tratar rate limit e erros temporários
                    if status in (429, 500, 502, 503, 504):
                        raise requests.HTTPError(f"HTTP {status}", response=resp)

                    resp.raise_for_status()

                    content_type = (resp.headers.get("content-type") or "").lower()

                    # Se vier ZIP, precisamos carregar o ZIP (geralmente pequeno, pois é tile/talhão)
                    if "zip" in content_type:
                        # baixa inteiro (zip) e extrai tif
                        data = io.BytesIO(resp.content)
                        with zipfile.ZipFile(data) as zf:
                            tif_files = [f for f in zf.namelist() if f.lower().endswith(".tif") or f.lower().endswith(".tiff")]
                            if not tif_files:
                                raise RuntimeError("ZIP recebido, mas não encontrei .tif dentro.")

                            # extrai o primeiro tif
                            with zf.open(tif_files[0]) as tif_file, open(tmp_path, "wb") as f:
                                while True:
                                    chunk = tif_file.read(1024 * 1024)
                                    if not chunk:
                                        break
                                    f.write(chunk)

                    else:
                        # grava stream direto em disco
                        with open(tmp_path, "wb") as f:
                            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    f.write(chunk)

                # validação rápida: arquivo existe e tem tamanho mínimo
                if not tmp_path.exists() or tmp_path.stat().st_size < 1024:
                    raise RuntimeError("Download terminou mas o arquivo ficou vazio ou pequeno demais.")

                # rename atômico
                tmp_path.replace(output_path)
                return True

            except (requests.Timeout, requests.ConnectionError, requests.HTTPError, RuntimeError) as e:
                # backoff
                if attempt == max_attempts:
                    log_safe("error", f"Erro no download do talhão {talhao_id} após {attempt} tentativas: {e}")
                    # limpa .part
                    try:
                        if tmp_path.exists():
                            tmp_path.unlink()
                    except Exception:
                        pass
                    return False

                sleep_s = backoff_base * (2 ** (attempt - 1))
                log_safe("warning", f"Talhão {talhao_id}: tentativa {attempt}/{max_attempts} falhou ({e}). "
                                    f"Retry em {sleep_s:.1f}s...")
                time.sleep(sleep_s)

        return False

    except Exception as e:
        log_safe("error", f"Erro no download do talhão {talhao_id}: {e}")
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        return False

    finally:
        if created_here:
            try:
                local_session.close()
            except Exception:
                pass


def calculate_talhao_statistics(geotiff_path, shapely_geom):
    try:
        with rasterio.open(geotiff_path) as src:
            data = src.read(1)
            nodata = src.nodata if src.nodata is not None else -9999

            mask = geometry_mask(
                [shapely_geom],
                transform=src.transform,
                invert=True,
                out_shape=data.shape
            )

            valid_data = data[mask & ~np.isnan(data) & (data != nodata)]

            if len(valid_data) > 0:
                pixel_area_m2 = abs(src.transform.a * src.transform.e)
                area_ha = (len(valid_data) * pixel_area_m2) / 10000

                return {
                    'ndvi_medio': float(valid_data.mean()),
                    'ndvi_min': float(valid_data.min()),
                    'ndvi_max': float(valid_data.max()),
                    'ndvi_std': float(valid_data.std()),
                    'area_ha': round(area_ha, 2),
                    'pixels_validos': len(valid_data)
                }
            else:
                return None

    except Exception as e:
        log_safe('error', f"Erro ao calcular estatísticas: {e}")
        return None

def process_single_talhao(talhao_data, start_date, end_date):
    talhao_id = talhao_data['talhao_id']
    talhao_nome = talhao_data['talhao_nome']
    geometry = talhao_data['geometry']

    try:
        ee_geometry = shapely_to_ee_geometry(geometry)
        ndvi_image = create_ndvi_for_geometry(ee_geometry, start_date, end_date)

        output_path = TALHOES_DIR / f"ndvi_{talhao_id}.tif"
        success = download_ndvi_talhao(ndvi_image, ee_geometry, talhao_id, output_path)

        if success:
            stats = calculate_talhao_statistics(output_path, geometry)
            if stats is None:
                stats = {
                    'ndvi_medio': float('nan'),
                    'ndvi_min': float('nan'),
                    'ndvi_max': float('nan'),
                    'ndvi_std': float('nan'),
                    'area_ha': 0.0,
                    'pixels_validos': 0
                }
                log_safe('warning', f"⚠️ {talhao_nome}: sem pixels válidos no período (vai entrar com NDVI=NaN).")

            result = {
                'id': talhao_id,
                'nome': talhao_nome,
                'geometry': geometry,
                'geotiff_path': output_path,
                **stats
            }
            log_safe('info', f"✅ {talhao_nome}: NDVI={stats['ndvi_medio'] if np.isfinite(stats['ndvi_medio']) else 'NaN'}, Área={stats['area_ha']:.2f}ha")
            return result

        else:
            log_safe('error', f"❌ {talhao_nome}: falha no download")
            return None

    except Exception as e:
        log_safe('error', f"❌ {talhao_nome}: erro no processamento - {e}")
        return None

def process_talhoes_parallel(gdf, start_date, end_date, max_workers=MAX_WORKERS):
    logging.info(f"🌾 Processando {len(gdf)} talhões em PARALELO...")
    logging.info(f"⚡ Usando {max_workers} threads simultâneas")

    if TALHAO_ID_FIELD not in gdf.columns:
        logging.warning(f"⚠️ Campo '{TALHAO_ID_FIELD}' não encontrado")
        logging.info(f"   Campos disponíveis: {', '.join(gdf.columns)}")
        logging.info("   Criando IDs automáticos...")
        gdf[TALHAO_ID_FIELD] = [f"talhao_{i+1}" for i in range(len(gdf))]

    talhoes_data = []
    for idx, row in gdf.iterrows():
        talhoes_data.append({
            'idx': idx,
            'talhao_id': str(row[TALHAO_ID_FIELD]),
            'talhao_nome': str(row.get(TALHAO_NOME_FIELD, str(row[TALHAO_ID_FIELD]))),
            'geometry': row.geometry
        })

    talhoes_info = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_talhao = {
            executor.submit(process_single_talhao, talhao_data, start_date, end_date): talhao_data
            for talhao_data in talhoes_data
        }

        with tqdm(total=len(talhoes_data), desc="Talhões", unit="talhão") as pbar:
            for future in as_completed(future_to_talhao):
                talhao_data = future_to_talhao[future]
                try:
                    result = future.result()
                    if result:
                        talhoes_info.append(result)
                except Exception as e:
                    log_safe('error', f"Exceção no talhão {talhao_data['talhao_nome']}: {e}")

                pbar.update(1)
                pbar.set_description(f"Talhões ({len(talhoes_info)} OK)")

    return talhoes_info


# ═══════════════════════════════════════════════════════════════════
# MOSAICO COMPLETO
# ═══════════════════════════════════════════════════════════════════

def create_complete_mosaic(talhoes_info, output_path):
    logging.info("🔗 Criando mosaico completo...")

    if not talhoes_info:
        logging.error("❌ Nenhum talhão processado")
        return False

    try:
        src_files = [rasterio.open(t['geotiff_path']) for t in talhoes_info]
        mosaic, out_trans = merge(src_files)

        out_meta = src_files[0].meta.copy()
        out_meta.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans
        })

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic)

        for src in src_files:
            src.close()

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logging.info(f"✅ Mosaico salvo: {output_path.name} ({file_size_mb:.2f} MB)")
        return True

    except Exception as e:
        logging.error(f"❌ Erro ao criar mosaico: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════
# EXPORTAR ESTATÍSTICAS
# ═══════════════════════════════════════════════════════════════════

def export_statistics(talhoes_info, output_csv):
    logging.info("📊 Exportando estatísticas...")

    data = []
    for t in talhoes_info:
        data.append({
            'ID': t['id'],
            'Nome': t['nome'],
            'NDVI Médio': t['ndvi_medio'],
            'NDVI Mínimo': t['ndvi_min'],
            'NDVI Máximo': t['ndvi_max'],
            'Desvio Padrão': t['ndvi_std'],
            'Área (ha)': t['area_ha'],
            'Pixels Válidos': t['pixels_validos']
        })

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    logging.info(f"✅ CSV salvo: {output_csv.name}")


# ═══════════════════════════════════════════════════════════════════
# HTML INTERATIVO (OPÇÃO 2) — 1 LAYER + DROPDOWN + ZOOM
# ═══════════════════════════════════════════════════════════════════

def get_ndvi_color(ndvi_value):
    if ndvi_value < -0.2:
        return '#0000FF'
    elif ndvi_value < 0.2:
        return '#d73027'
    elif ndvi_value < 0.4:
        return '#fc8d59'
    elif ndvi_value < 0.6:
        return '#fee08b'
    elif ndvi_value < 0.7:
        return '#d9ef8b'
    elif ndvi_value < 0.8:
        return '#91cf60'
    else:
        return '#1a9850'

def escape_js(s: str) -> str:
    """Evita quebrar JS/HTML quando nome tem aspas, etc."""
    return (s or "").replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')

def generate_interactive_html(talhoes_info, mosaic_path, output_html):
    logging.info("🗺️ Gerando mapa HTML interativo (Opção 2: 1 layer + dropdown)...")

    if not talhoes_info:
        logging.error("❌ talhoes_info vazio. Nada para renderizar.")
        return False

    # bounds geral
    all_bounds = [t['geometry'].bounds for t in talhoes_info]
    min_lon = min(b[0] for b in all_bounds)
    min_lat = min(b[1] for b in all_bounds)
    max_lon = max(b[2] for b in all_bounds)
    max_lat = max(b[3] for b in all_bounds)

    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap")

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satélite",
        overlay=False
    ).add_to(m)

    # ✅ 1) montar GDF único, SEM Path (nada de geotiff_path aqui!)
    rows = []
    for t in talhoes_info:
        nome = str(t.get("nome", "Sem nome"))
        talhao_id = str(t.get("id", "sem_id"))

        ndvi_medio = float(t.get("ndvi_medio", np.nan))
        ndvi_min = float(t.get("ndvi_min", np.nan))
        ndvi_max = float(t.get("ndvi_max", np.nan))
        area_ha = float(t.get("area_ha", 0.0))

        rows.append({
            "geometry": t["geometry"],
            "id": talhao_id,
            "nome": nome,
            "ndvi_medio": ndvi_medio,
            "ndvi_min": ndvi_min,
            "ndvi_max": ndvi_max,
            "area_ha": area_ha
        })

    gdf_all = gpd.GeoDataFrame(rows, crs="EPSG:4326")

    # ✅ 2) criar 1 GeoJson com style/popup por feature
    def style_fn(feature):
        nd = feature["properties"].get("ndvi_medio", None)
        try:
            nd = float(nd)
        except Exception:
            nd = np.nan
        color = get_ndvi_color(nd) if np.isfinite(nd) else "#999999"
        return {"fillColor": color, "color": "white", "weight": 1.5, "fillOpacity": 0.55}

    def highlight_fn(feature):
        return {"weight": 3, "fillOpacity": 0.75}

    tooltip = folium.features.GeoJsonTooltip(
        fields=["nome", "id", "ndvi_medio", "area_ha"],
        aliases=["Nome:", "ID:", "NDVI médio:", "Área (ha):"],
        localize=True,
        sticky=False
    )

    popup = folium.features.GeoJsonPopup(
        fields=["nome", "id", "ndvi_medio", "ndvi_min", "ndvi_max", "area_ha"],
        aliases=["Nome:", "ID:", "NDVI médio:", "NDVI mín:", "NDVI máx:", "Área (ha):"],
        localize=True,
        max_width=320
    )

    talhoes_layer = folium.GeoJson(
        gdf_all,
        name="Talhões (NDVI)",
        style_function=style_fn,
        highlight_function=highlight_fn,
        tooltip=tooltip,
        popup=popup
    ).add_to(m)

    # ✅ 3) Dropdown (ID — Nome) + JS pra zoom + abrir popup
    map_name = m.get_name()
    layer_name = talhoes_layer.get_name()

    # lista de opções (7300 options ok; fica MUITO mais leve que 7300 layers)
    options_html = ['<option value="">Selecione um talhão...</option>']
    for r in rows:
        # value = id (mais estável), label = "ID — Nome (NDVI)"
        vid = escape_js(r["id"])
        label = f'{escape_js(r["id"])} — {escape_js(r["nome"])}'
        options_html.append(f'<option value="{vid}">{label}</option>')

    control_html = f"""
    <div id="talhaoControl" style="
        position: fixed;
        top: 12px;
        left: 12px;
        z-index: 9999;
        background: rgba(255,255,255,0.95);
        padding: 10px 12px;
        border: 2px solid #999;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.15);
        font-family: Arial, sans-serif;
        width: 360px;
    ">
      <div style="font-weight:700; margin-bottom:6px;">Filtrar por talhão</div>
      <select id="talhaoSelect" style="width:100%; padding:8px; border-radius:6px; border:1px solid #bbb;">
        {''.join(options_html)}
      </select>
      <div style="margin-top:8px; display:flex; gap:8px;">
        <button id="btnZoomAll" style="flex:1; padding:8px; border-radius:6px; border:1px solid #bbb; background:#f6f6f6; cursor:pointer;">
          Ver tudo
        </button>
        <button id="btnClear" style="flex:1; padding:8px; border-radius:6px; border:1px solid #bbb; background:#f6f6f6; cursor:pointer;">
          Limpar
        </button>
      </div>
      <div style="margin-top:8px; font-size:12px; color:#666;">
        Dica: selecione o talhão para dar zoom e abrir o popup.
      </div>
    </div>

    <script>
      (function() {{
        const map = {map_name};
        const layer = {layer_name};

        // cache: id -> layerLeaflet
        const indexById = {{}};

        function buildIndex() {{
          layer.eachLayer(function(l) {{
            const p = l.feature && l.feature.properties ? l.feature.properties : null;
            if (!p) return;
            const id = String(p.id || "");
            if (id) indexById[id] = l;
          }});
        }}

        function zoomToLayer(l) {{
          try {{
            const b = l.getBounds();
            map.fitBounds(b, {{padding:[20,20]}});
            if (l.openPopup) l.openPopup();
          }} catch(e) {{}}
        }}

        function resetAllStyles() {{
          layer.setStyle(function(feature) {{
            const nd = feature.properties && feature.properties.ndvi_medio;
            let v = parseFloat(nd);
            let color = "#999999";
            if (!isNaN(v)) {{
              // mesma lógica do python
              if (v < -0.2) color = "#0000FF";
              else if (v < 0.2) color = "#d73027";
              else if (v < 0.4) color = "#fc8d59";
              else if (v < 0.6) color = "#fee08b";
              else if (v < 0.7) color = "#d9ef8b";
              else if (v < 0.8) color = "#91cf60";
              else color = "#1a9850";
            }}
            return {{fillColor: color, color: "white", weight: 1.5, fillOpacity: 0.55}};
          }});
        }}

        function highlightOnly(targetId) {{
          layer.setStyle(function(feature) {{
            const id = String(feature.properties && feature.properties.id || "");
            if (id === targetId) {{
              return {{color:"#000", weight:3.5, fillOpacity:0.75}};
            }}
            // "apaga" os demais
            return {{opacity:0.15, fillOpacity:0.08, weight:1}};
          }});
        }}

        buildIndex();

        const sel = document.getElementById("talhaoSelect");
        const btnAll = document.getElementById("btnZoomAll");
        const btnClear = document.getElementById("btnClear");

        sel.addEventListener("change", function() {{
          const id = sel.value;
          resetAllStyles();
          if (!id) return;

          const l = indexById[id];
          if (l) {{
            highlightOnly(id);
            zoomToLayer(l);
          }}
        }});

        btnAll.addEventListener("click", function() {{
          resetAllStyles();
          sel.value = "";
          map.fitBounds([[{min_lat}, {min_lon}], [{max_lat}, {max_lon}]], {{padding:[20,20]}});
        }});

        btnClear.addEventListener("click", function() {{
          resetAllStyles();
          sel.value = "";
        }});
      }})();
    </script>
    """

    m.get_root().html.add_child(folium.Element(control_html))

    # controles
    folium.LayerControl(collapsed=True).add_to(m)
    Fullscreen(position="topleft").add_to(m)
    MeasureControl(position="topleft", primary_length_unit="kilometers", primary_area_unit="hectares").add_to(m)
    MousePosition(position="bottomright", separator=" | ", prefix="Coords:", num_digits=6).add_to(m)

    # legenda
    legend_html = """
    <div style="position: fixed; bottom: 50px; right: 50px; width: 180px;
                background-color: white; border: 2px solid grey; z-index: 9999;
                font-size: 14px; padding: 15px; border-radius: 5px;">
        <p style="margin: 0 0 10px 0; font-weight: bold; font-size: 16px;">NDVI</p>
        <div style="display:flex;align-items:center;margin:5px 0;"><div style="background:#1a9850;width:30px;height:20px;margin-right:10px;"></div><span>0.8-1.0 Ótimo</span></div>
        <div style="display:flex;align-items:center;margin:5px 0;"><div style="background:#91cf60;width:30px;height:20px;margin-right:10px;"></div><span>0.7-0.8 Bom</span></div>
        <div style="display:flex;align-items:center;margin:5px 0;"><div style="background:#d9ef8b;width:30px;height:20px;margin-right:10px;"></div><span>0.6-0.7 Regular</span></div>
        <div style="display:flex;align-items:center;margin:5px 0;"><div style="background:#fee08b;width:30px;height:20px;margin-right:10px;"></div><span>0.4-0.6 Baixo</span></div>
        <div style="display:flex;align-items:center;margin:5px 0;"><div style="background:#fc8d59;width:30px;height:20px;margin-right:10px;"></div><span>0.2-0.4 Muito Baixo</span></div>
        <div style="display:flex;align-items:center;margin:5px 0;"><div style="background:#d73027;width:30px;height:20px;margin-right:10px;"></div><span>-0.2-0.2 Solo</span></div>
        <div style="display:flex;align-items:center;margin:5px 0;"><div style="background:#0000FF;width:30px;height:20px;margin-right:10px;"></div><span>&lt;-0.2 Água</span></div>
        <p style="margin: 10px 0 0 0; font-size: 11px; color: #666;">⚡ 1 camada + filtro</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    output_html = str(output_html)
    m.save(output_html)
    logging.info(f"✅ Mapa HTML salvo: {output_html}")

    return True


# ═══════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("🌱 SMARTGREEN - Pipeline NDVI Paralelo")
    print("⚡ Processamento com múltiplas threads")
    print("=" * 70)
    print()

    setup_logging()
    logging.info("🚀 Iniciando pipeline SMARTGREEN com processamento PARALELO")
    logging.info(f"⚡ Configurado para {MAX_WORKERS} threads simultâneas")

    start_time = datetime.now()

    if not initialize_earth_engine():
        sys.exit(1)

    logging.info(f"📂 Carregando shapefile: {SHAPEFILE_PATH}")

    if not SHAPEFILE_PATH.exists():
        logging.error(f"❌ Shapefile não encontrado: {SHAPEFILE_PATH}")
        sys.exit(1)

    try:
        gdf = gpd.read_file(SHAPEFILE_PATH)
        logging.info(f"✅ Shapefile carregado: {len(gdf)} talhões")
        logging.info(f"   CRS: {gdf.crs}")

        if gdf.crs != "EPSG:4326":
            logging.info("   Reprojetando para WGS84...")
            gdf = gdf.to_crs("EPSG:4326")

    except Exception as e:
        logging.error(f"❌ Erro ao carregar shapefile: {e}")
        sys.exit(1)

    end_date = datetime.now()
    start_date_period = end_date - timedelta(days=DAYS_BACK)
    start_str = start_date_period.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    logging.info(f"📅 Período: {start_str} a {end_str}")
    logging.info(f"📏 Resolução: {SCALE}m (Sentinel-2 máxima)")

    talhoes_info = process_talhoes_parallel(gdf, start_str, end_str, MAX_WORKERS)

    if not talhoes_info:
        logging.error("❌ Nenhum talhão processado com sucesso")
        sys.exit(1)

    logging.info(f"✅ {len(talhoes_info)}/{len(gdf)} talhões processados")

    processing_time = (datetime.now() - start_time).total_seconds()
    avg_time_per_talhao = processing_time / len(talhoes_info) if talhoes_info else 0

    logging.info(f"⏱️  Tempo total: {processing_time:.1f}s")
    logging.info(f"⏱️  Tempo médio por talhão: {avg_time_per_talhao:.1f}s")

    create_complete_mosaic(talhoes_info, OUTPUT_GEOTIFF_COMPLETO)
    export_statistics(talhoes_info, OUTPUT_CSV)

    generate_interactive_html(talhoes_info, OUTPUT_GEOTIFF_COMPLETO, OUTPUT_HTML)

    print()
    print("=" * 70)
    logging.info("✅ Pipeline concluído!")
    logging.info(f"   🌐 Mapa interativo: {OUTPUT_HTML}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("\n⚠️ Interrompido pelo usuário")
        sys.exit(1)
    except Exception as e:
        logging.error(f"\n❌ Erro fatal: {e}", exc_info=True)
        sys.exit(1)
