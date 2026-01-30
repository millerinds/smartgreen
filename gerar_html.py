#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GERAR_HTML.py — gera SOMENTE o mapa HTML a partir de:
- out/ndvi_cem.tif
- CEM/CEM.shp

Uso:
  source venv/bin/activate
  python gerar_html.py
"""

import sys
import logging
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
import rasterio.transform
from affine import Affine

import folium
from folium.plugins import MeasureControl, MousePosition, Fullscreen
from matplotlib.colors import LinearSegmentedColormap


# =========================
# CONFIG
# =========================
SHAPEFILE_PATH = Path("CEM/CEM.shp")
GEOTIFF_PATH   = Path("out/ndvi_cem.tif")
OUTPUT_HTML    = Path("out/mapa_ndvi.html")

# Ajuste se quiser mais leve/rápido no HTML
MAX_HTML_PIXELS = 4000 * 4000  # 16M px (bom)
OVERLAY_OPACITY = 0.7


# =========================
# LOG
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def die(msg: str, code: int = 1):
    logging.error(msg)
    sys.exit(code)


# =========================
# HELPERS
# =========================
def load_shapefile_wgs84(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        die(f"Shapefile não encontrado: {path}")

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        die("Shapefile sem CRS (faltando .prj?)")

    if str(gdf.crs).upper() != "EPSG:4326":
        logging.info("Reprojetando shapefile para EPSG:4326...")
        gdf = gdf.to_crs("EPSG:4326")

    return gdf


def downsample_if_needed(data: np.ndarray, max_pixels: int):
    total = int(data.shape[0]) * int(data.shape[1])
    if total <= max_pixels:
        return data, 1

    factor = int(np.ceil(np.sqrt(total / max_pixels)))
    logging.info(f"Downsample no overlay (fator {factor}x) para melhorar performance do HTML.")
    return data[::factor, ::factor], factor


def rgba_from_ndvi(data: np.ndarray, nodata_value):
    # máscara válida
    valid = ~np.isnan(data)
    if nodata_value is not None:
        valid &= (data != nodata_value)

    # normaliza NDVI (-1..1) -> (0..1)
    norm = np.clip((data + 1.0) / 2.0, 0.0, 1.0)

    colors = ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#91cf60", "#1a9850"]
    cmap = LinearSegmentedColormap.from_list("RdYlGn", colors, N=256)

    rgba = cmap(norm)  # float 0..1
    rgba[..., 3] = valid.astype(float)  # alpha 0/1

    return (rgba * 255).astype(np.uint8)


def calc_bounds_after_downsample(src, data_shape, downsample_factor: int):
    # bounds do raster original
    bounds = src.bounds
    if downsample_factor <= 1:
        return bounds

    # Ajusta transform para refletir amostragem (cada pixel representa "factor" pixels originais)
    new_transform = src.transform * Affine.scale(downsample_factor, downsample_factor)

    # array_bounds retorna (south, west, north, east) -> aqui como (bottom,left,top,right)
    bottom, left, top, right = rasterio.transform.array_bounds(
        data_shape[0], data_shape[1], new_transform
    )

    return rasterio.coords.BoundingBox(left=left, bottom=bottom, right=right, top=top)


# =========================
# MAIN
# =========================
def main():
    logging.info("🗺️ Gerador de HTML (somente) — SmartGreen")
    if not GEOTIFF_PATH.exists():
        die(f"GeoTIFF não encontrado: {GEOTIFF_PATH} (gere o NDVI antes)")

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)

    gdf = load_shapefile_wgs84(SHAPEFILE_PATH)

    with rasterio.open(GEOTIFF_PATH) as src:
        if src.crs is None:
            die("GeoTIFF sem CRS.")
        if str(src.crs).upper() != "EPSG:4326":
            die(f"GeoTIFF CRS não é EPSG:4326 (veio {src.crs}). Reprojete antes ou gere em EPSG:4326.")

        data = src.read(1).astype(np.float32)
        nodata = src.nodata

        # troca nodata por NaN (se existir)
        if nodata is not None:
            data = np.where(data == nodata, np.nan, data)

        valid = data[~np.isnan(data)]
        if valid.size:
            logging.info(f"NDVI stats: min={valid.min():.3f}, max={valid.max():.3f}, média={valid.mean():.3f}")
        else:
            logging.warning("Raster sem dados válidos (tudo NaN).")

        data_ds, factor = downsample_if_needed(data, MAX_HTML_PIXELS)
        rgba = rgba_from_ndvi(data_ds, nodata_value=None)  # já trocamos por NaN

        bounds = calc_bounds_after_downsample(src, data_ds.shape, factor)

    # centro
    center_lat = (bounds.bottom + bounds.top) / 2
    center_lon = (bounds.left + bounds.right) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="OpenStreetMap")

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satélite",
        overlay=False,
    ).add_to(m)

    folium.raster_layers.ImageOverlay(
        name="NDVI",
        image=rgba,
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=OVERLAY_OPACITY,
        interactive=True,
        cross_origin=False,
        zindex=3,
    ).add_to(m)

    # contorno (simplificado pra não pesar)
    geom = gdf.geometry.unary_union
    simplified = geom.simplify(0.001, preserve_topology=True)
    simp_gdf = gpd.GeoDataFrame([{"geometry": simplified}], crs="EPSG:4326")

    folium.GeoJson(
        simp_gdf,
        name="Área",
        style_function=lambda x: {"fillColor": "transparent", "color": "blue", "weight": 2, "fillOpacity": 0},
    ).add_to(m)

    Fullscreen().add_to(m)
    MeasureControl(position="topleft", primary_length_unit="kilometers", primary_area_unit="hectares").add_to(m)
    MousePosition(position="bottomright", separator=" | ", prefix="Coords:", num_digits=6).add_to(m)
    folium.LayerControl().add_to(m)

    legend_html = """
    <div style="position: fixed; bottom: 50px; right: 50px; width: 150px; height: 220px;
                background-color: white; border:2px solid grey; z-index:9999; font-size:14px; padding: 10px">
      <p style="margin:0; font-weight:bold;">NDVI</p>
      <div style="background: linear-gradient(to top, #d73027, #fc8d59, #fee08b, #d9ef8b, #91cf60, #1a9850);
                  height: 120px; width: 30px; margin: 10px 0;"></div>
      <p style="margin:0; font-size:12px;">1.0 (Veg. densa)</p>
      <p style="margin:0; font-size:12px;">0.5 (Veg. moderada)</p>
      <p style="margin:0; font-size:12px;">0.0 (Solo exposto)</p>
      <p style="margin:0; font-size:12px;">-1.0 (Água)</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.fit_bounds([[bounds.bottom, bounds.left], [bounds.top, bounds.right]])
    m.save(str(OUTPUT_HTML))
    logging.info(f"✅ HTML salvo em: {OUTPUT_HTML}")
    logging.info(f"Abrir: xdg-open {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
