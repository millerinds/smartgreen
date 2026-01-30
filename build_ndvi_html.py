#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════
    SMARTGREEN - Pipeline Automatizado de NDVI
    Processamento de imagens Sentinel-2 via Google Earth Engine
    Com divisão automática em tiles para áreas grandes
═══════════════════════════════════════════════════════════════════

🎯 FUNCIONALIDADES:
   1. Lê shapefile da área de interesse (CEM/CEM.shp)
   2. Conecta ao Google Earth Engine (projeto: earth-484311)
   3. Busca imagens Sentinel-2 Surface Reflectance Harmonized
   4. Filtra por período (últimos 60 dias)
   5. Filtra por nuvens (máximo 30%)
   6. Aplica máscara de nuvens usando classificação SCL
   7. Calcula NDVI (Normalized Difference Vegetation Index)
   8. DIVIDE AUTOMATICAMENTE EM TILES se necessário
   9. Mescla tiles em único GeoTIFF (out/ndvi_cem.tif)
   10. Gera HTML interativo com mapa Folium (out/mapa_ndvi.html)

📁 ESTRUTURA ESPERADA:
   SMARTGREEN/
   ├── CEM/
   │   └── CEM.shp (+ .shx .dbf .prj etc)
   ├── out/                    # Criada automaticamente
   └── smartgreen.py           # Este arquivo

⚙️ CONFIGURAÇÕES:
   Edite as constantes abaixo para ajustar o comportamento

🚀 EXECUÇÃO:
   python smartgreen.py

📦 DEPENDÊNCIAS:
   pip install earthengine-api geopandas rasterio numpy folium requests tqdm matplotlib shapely

═══════════════════════════════════════════════════════════════════
"""

import os
import sys
import logging
import zipfile
import io
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from affine import Affine

import ee
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds
import folium
from folium.plugins import FloatImage, MeasureControl, MousePosition, Fullscreen
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ═══════════════════════════════════════════════════════════════════
# CONFIGURAÇÕES - EDITE AQUI
# ═══════════════════════════════════════════════════════════════════

PROJECT_ID = "earth-484311"              # ID do projeto Google Earth Engine
DAYS_BACK = 60                           # Período de busca (últimos N dias)
MAX_CLOUD_COVER = 30                     # Percentual máximo de nuvens (%)
OUTPUT_DIR = Path("out")                 # Diretório de saída
SHAPEFILE_PATH = Path("CEM/CEM.shp")     # Caminho do shapefile
OUTPUT_GEOTIFF = OUTPUT_DIR / "ndvi_cem.tif"     # GeoTIFF de saída
OUTPUT_HTML = OUTPUT_DIR / "mapa_ndvi.html"      # HTML de saída
OUTPUT_LOG = OUTPUT_DIR / "pipeline.log"         # Log de saída

# Escala (metros por pixel)
SCALE = 10

# Limite de download do Earth Engine (bytes) - 50MB
MAX_DOWNLOAD_SIZE = 48 * 1024 * 1024  # 48MB com margem de segurança

# Número de tiles (será calculado automaticamente)
GRID_SIZE = 2  # Começa com 2x2, aumenta se necessário


# ═══════════════════════════════════════════════════════════════════
# SISTEMA DE LOGS
# ═══════════════════════════════════════════════════════════════════

def setup_logging():
    """Configura sistema de logs (console + arquivo)"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Remove handlers anteriores
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configuração
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(OUTPUT_LOG, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )


# ═══════════════════════════════════════════════════════════════════
# INICIALIZAÇÃO DO EARTH ENGINE
# ═══════════════════════════════════════════════════════════════════

def initialize_earth_engine():
    """Inicializa Google Earth Engine"""
    logging.info("🌍 Inicializando Google Earth Engine...")
    
    try:
        ee.Initialize(project=PROJECT_ID, opt_url='https://earthengine-highvolume.googleapis.com')
        logging.info("✅ Earth Engine inicializado com sucesso")
        return True
    except Exception as e:
        logging.warning(f"⚠️ Falha na inicialização direta: {e}")
        logging.info("🔐 Tentando autenticação...")
        
        try:
            ee.Authenticate()
            ee.Initialize(project=PROJECT_ID, opt_url='https://earthengine-highvolume.googleapis.com')
            logging.info("✅ Earth Engine inicializado após autenticação")
            return True
        except Exception as e:
            logging.error(f"❌ Falha ao inicializar Earth Engine: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════
# LEITURA DO SHAPEFILE
# ═══════════════════════════════════════════════════════════════════

def load_shapefile():
    """Carrega shapefile e retorna GeoDataFrame"""
    logging.info(f"📂 Carregando shapefile: {SHAPEFILE_PATH}")
    
    if not SHAPEFILE_PATH.exists():
        logging.error(f"❌ Shapefile não encontrado: {SHAPEFILE_PATH}")
        sys.exit(1)
    
    try:
        gdf = gpd.read_file(SHAPEFILE_PATH)
        logging.info(f"✅ Shapefile carregado: {len(gdf)} feições")
        logging.info(f"   CRS: {gdf.crs}")
        
        if gdf.crs != "EPSG:4326":
            logging.info("   Reprojetando para WGS84...")
            gdf = gdf.to_crs("EPSG:4326")
        
        return gdf
    except Exception as e:
        logging.error(f"❌ Erro ao carregar shapefile: {e}")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
# CONVERSÃO DE GEOMETRIA
# ═══════════════════════════════════════════════════════════════════

def gdf_to_ee_geometry(gdf):
    """Converte GeoDataFrame para ee.Geometry"""
    logging.info("🔄 Convertendo geometria para Earth Engine...")
    
    # Dissolver geometrias
    try:
        from shapely.ops import unary_union
        unified = unary_union(gdf.geometry)
    except:
        unified = gdf.geometry.union_all()
    
    logging.info(f"   Tipo de geometria: {unified.geom_type}")
    
    # Simplificar se necessário
    tolerance = 0.0001
    if unified.geom_type in ['MultiPolygon', 'Polygon']:
        n_coords = sum(len(list(geom.exterior.coords)) for geom in 
                      (unified.geoms if unified.geom_type == 'MultiPolygon' else [unified]))
        
        if n_coords > 10000:
            logging.warning(f"⚠️ Geometria complexa ({n_coords:,} coordenadas)")
            logging.info("   Simplificando...")
            unified = unified.simplify(tolerance, preserve_topology=True)
            n_coords_new = sum(len(list(geom.exterior.coords)) for geom in 
                              (unified.geoms if unified.geom_type == 'MultiPolygon' else [unified]))
            logging.info(f"   Simplificada: {n_coords_new:,} coordenadas")
    
    # Converter para EE
    if unified.geom_type == 'Polygon':
        coords = [list(unified.exterior.coords)]
        ee_geom = ee.Geometry.Polygon(coords)
    elif unified.geom_type == 'MultiPolygon':
        coords = [list(poly.exterior.coords) for poly in unified.geoms]
        ee_geom = ee.Geometry.MultiPolygon(coords)
    else:
        logging.error(f"❌ Tipo não suportado: {unified.geom_type}")
        sys.exit(1)
    
    logging.info("✅ Geometria convertida")
    
    return ee_geom, unified


# ═══════════════════════════════════════════════════════════════════
# PROCESSAMENTO SENTINEL-2
# ═══════════════════════════════════════════════════════════════════

def mask_clouds(image):
    """Aplica máscara de nuvens"""
    scl = image.select('SCL')
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    return image.updateMask(mask)


def calculate_ndvi(image):
    """Calcula NDVI"""
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)


def create_ndvi_composite(aoi, start_date, end_date):
    """Cria composto NDVI para uma área"""
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(aoi)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_COVER)))
    
    collection = collection.map(mask_clouds).map(calculate_ndvi)
    composite = collection.median()
    ndvi = composite.select('NDVI').clip(aoi)
    
    return ndvi


# ═══════════════════════════════════════════════════════════════════
# DIVISÃO EM TILES
# ═══════════════════════════════════════════════════════════════════

def create_grid(bounds, grid_size):
    """
    Cria grid de tiles sobre a área
    
    Args:
        bounds: (minx, miny, maxx, maxy)
        grid_size: número de divisões (ex: 2 = 2x2 = 4 tiles)
    
    Returns:
        list de (tile_bounds, tile_index)
    """
    minx, miny, maxx, maxy = bounds
    
    width = (maxx - minx) / grid_size
    height = (maxy - miny) / grid_size
    
    tiles = []
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            tile_minx = minx + j * width
            tile_miny = miny + i * height
            tile_maxx = tile_minx + width
            tile_maxy = tile_miny + height
            
            tiles.append(((tile_minx, tile_miny, tile_maxx, tile_maxy), idx))
            idx += 1
    
    return tiles


def estimate_tile_size(bounds, scale):
    """
    Estima tamanho de download de um tile em bytes
    Considera: área * (scale^2) * bytes_per_pixel
    """
    minx, miny, maxx, maxy = bounds
    
    # Calcular área aproximada em graus² e converter para m²
    # 1 grau ≈ 111km no equador
    width_m = (maxx - minx) * 111000
    height_m = (maxy - miny) * 111000
    area_m2 = width_m * height_m
    
    # Número de pixels
    n_pixels = area_m2 / (scale ** 2)
    
    # Bytes por pixel (float32 = 4 bytes, com overhead do GeoTIFF ≈ 6 bytes)
    bytes_per_pixel = 6
    
    estimated_size = n_pixels * bytes_per_pixel
    
    return estimated_size


def calculate_optimal_grid_size(bounds, scale, max_size):
    """
    Calcula tamanho de grid necessário para manter tiles abaixo do limite
    """
    total_size = estimate_tile_size(bounds, scale)
    
    logging.info(f"📏 Estimando tamanho total: {total_size / 1024 / 1024:.1f} MB")
    
    if total_size <= max_size:
        logging.info("✅ Área pequena o suficiente - sem necessidade de divisão")
        return 1
    
    # Calcular grid necessário
    ratio = total_size / max_size
    grid_size = int(np.ceil(np.sqrt(ratio)))
    
    # Garantir que seja pelo menos 2
    grid_size = max(2, grid_size)
    
    logging.warning(f"⚠️ Área muito grande! Dividindo em grid {grid_size}x{grid_size} ({grid_size**2} tiles)")
    
    return grid_size


# ═══════════════════════════════════════════════════════════════════
# DOWNLOAD DE TILES
# ═══════════════════════════════════════════════════════════════════

def download_tile(ndvi_image, tile_bounds, tile_idx, scale, temp_dir):
    """
    Faz download de um tile individual
    """
    minx, miny, maxx, maxy = tile_bounds
    
    # Criar geometria do tile
    tile_geom = ee.Geometry.Rectangle([minx, miny, maxx, maxy])
    
    # Recortar NDVI pelo tile
    tile_ndvi = ndvi_image.clip(tile_geom)
    
    # Gerar URL
    try:
        url = tile_ndvi.getDownloadURL({
            'scale': scale,
            'region': tile_geom,
            'format': 'GEO_TIFF',
            'crs': 'EPSG:4326'
        })
        
        # Download
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        # Salvar
        tile_path = temp_dir / f"tile_{tile_idx}.tif"
        
        # Verificar se é ZIP
        content_type = response.headers.get('content-type', '')
        is_zip = 'zip' in content_type
        
        if is_zip:
            # Extrair ZIP
            data = io.BytesIO(response.content)
            with zipfile.ZipFile(data) as zf:
                tif_files = [f for f in zf.namelist() if f.endswith('.tif')]
                if tif_files:
                    with zf.open(tif_files[0]) as tif_file:
                        tile_path.write_bytes(tif_file.read())
        else:
            tile_path.write_bytes(response.content)
        
        return tile_path
        
    except Exception as e:
        logging.error(f"❌ Erro no tile {tile_idx}: {e}")
        return None


def download_tiles_parallel(ndvi_image, tiles, scale, temp_dir):
    """
    Faz download de todos os tiles com barra de progresso
    """
    tile_paths = []
    
    logging.info(f"⬇️ Baixando {len(tiles)} tiles...")
    
    with tqdm(total=len(tiles), desc="Tiles", unit="tile") as pbar:
        for tile_bounds, tile_idx in tiles:
            tile_path = download_tile(ndvi_image, tile_bounds, tile_idx, scale, temp_dir)
            if tile_path:
                tile_paths.append(tile_path)
            pbar.update(1)
    
    return tile_paths


# ═══════════════════════════════════════════════════════════════════
# MESCLAGEM DE TILES
# ═══════════════════════════════════════════════════════════════════

def merge_tiles(tile_paths, output_path):
    """
    Mescla múltiplos tiles em um único GeoTIFF
    """
    logging.info(f"🔗 Mesclando {len(tile_paths)} tiles...")
    
    try:
        # Abrir todos os tiles
        src_files = [rasterio.open(tile) for tile in tile_paths]
        
        # Mesclar
        mosaic, out_trans = merge(src_files)
        
        # Pegar metadados do primeiro tile
        out_meta = src_files[0].meta.copy()
        out_meta.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans
        })
        
        # Salvar
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic)
        
        # Fechar arquivos
        for src in src_files:
            src.close()
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logging.info(f"✅ GeoTIFF mesclado: {output_path.name} ({file_size_mb:.2f} MB)")
        
    except Exception as e:
        logging.error(f"❌ Erro ao mesclar tiles: {e}")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
# PIPELINE DE DOWNLOAD COM TILES
# ═══════════════════════════════════════════════════════════════════

def download_ndvi_with_tiles(ndvi_image, bounds, scale, output_path):
    """
    Pipeline completo de download com divisão automática em tiles
    """
    # Calcular grid necessário
    grid_size = calculate_optimal_grid_size(bounds, scale, MAX_DOWNLOAD_SIZE)
    
    if grid_size == 1:
        # Área pequena - download direto
        logging.info("⬇️ Download direto (sem divisão)...")
        aoi = ee.Geometry.Rectangle(bounds)
        
        try:
            url = ndvi_image.getDownloadURL({
                'scale': scale,
                'region': aoi,
                'format': 'GEO_TIFF',
                'crs': 'EPSG:4326'
            })
            
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            data = io.BytesIO()
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Download") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        data.write(chunk)
                        pbar.update(len(chunk))
            
            data.seek(0)
            
            # Verificar ZIP
            content_type = response.headers.get('content-type', '')
            is_zip = 'zip' in content_type
            
            if is_zip:
                with zipfile.ZipFile(data) as zf:
                    tif_files = [f for f in zf.namelist() if f.endswith('.tif')]
                    if tif_files:
                        with zf.open(tif_files[0]) as tif_file:
                            output_path.write_bytes(tif_file.read())
            else:
                output_path.write_bytes(data.getvalue())
            
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logging.info(f"✅ GeoTIFF salvo: {output_path.name} ({file_size_mb:.2f} MB)")
            
        except Exception as e:
            logging.error(f"❌ Erro no download: {e}")
            sys.exit(1)
    
    else:
        # Área grande - dividir em tiles
        tiles = create_grid(bounds, grid_size)
        
        # Criar diretório temporário
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Download dos tiles
            tile_paths = download_tiles_parallel(ndvi_image, tiles, scale, temp_path)
            
            if not tile_paths:
                logging.error("❌ Nenhum tile foi baixado com sucesso")
                sys.exit(1)
            
            logging.info(f"✅ {len(tile_paths)}/{len(tiles)} tiles baixados")
            
            # Mesclar tiles
            merge_tiles(tile_paths, output_path)


# ═══════════════════════════════════════════════════════════════════
# GERAÇÃO DO HTML
# ═══════════════════════════════════════════════════════════════════

def apply_colormap_with_transparency(data, nodata_value):
    """Aplica colormap RdYlGn ao NDVI"""
    valid_mask = ~np.isnan(data) & (data != nodata_value)
    data_normalized = np.clip((data + 1) / 2, 0, 1)
    
    colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
    cmap = LinearSegmentedColormap.from_list('RdYlGn', colors, N=256)
    
    rgba = cmap(data_normalized)
    rgba[..., 3] = valid_mask.astype(float)
    
    return (rgba * 255).astype(np.uint8)


def downsample_if_needed(data, max_pixels=4000*4000):
    """Downsample para HTML"""
    total_pixels = data.shape[0] * data.shape[1]
    
    if total_pixels <= max_pixels:
        return data, 1
    
    factor = int(np.ceil(np.sqrt(total_pixels / max_pixels)))
    logging.info(f"   Downsample (fator: {factor}x)")
    
    return data[::factor, ::factor], factor


def generate_html_map(geotiff_path, shapefile_gdf, output_html):
    """Gera mapa HTML interativo"""
    logging.info("🗺️ Gerando mapa HTML...")
    
    with rasterio.open(geotiff_path) as src:
        logging.info(f"   Raster: {src.width}x{src.height} pixels")
        
        data = src.read(1)
        nodata = src.nodata if src.nodata is not None else -9999
        
        # Estatísticas
        valid_data = data[~np.isnan(data) & (data != nodata)]
        if len(valid_data) > 0:
            logging.info(f"   NDVI: min={valid_data.min():.3f}, max={valid_data.max():.3f}, média={valid_data.mean():.3f}")
        
        data, downsample_factor = downsample_if_needed(data)
        rgba = apply_colormap_with_transparency(data, nodata)
        
        bounds = src.bounds
        if downsample_factor > 1:
            new_transform = src.transform * src.transform.scale(downsample_factor, downsample_factor)
            bounds = rasterio.coords.BoundingBox(
                *rasterio.transform.array_bounds(data.shape[0], data.shape[1], new_transform)
            )
    
    center_lat = (bounds.bottom + bounds.top) / 2
    center_lon = (bounds.left + bounds.right) / 2
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='OpenStreetMap')
    
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satélite',
        overlay=False
    ).add_to(m)
    
    folium.raster_layers.ImageOverlay(
        name='NDVI',
        image=rgba,
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=0.7,
        interactive=True,
        cross_origin=False,
        zindex=1
    ).add_to(m)
    
    # Simplificar shapefile
    try:
        from shapely.ops import unary_union
        unified = unary_union(shapefile_gdf.geometry)
    except:
        unified = shapefile_gdf.geometry.union_all()
    
    simplified = unified.simplify(0.001, preserve_topology=True)
    simplified_gdf = gpd.GeoDataFrame([{'geometry': simplified}], crs='EPSG:4326')
    
    folium.GeoJson(
        simplified_gdf,
        name='Área',
        style_function=lambda x: {'fillColor': 'transparent', 'color': 'blue', 'weight': 2, 'fillOpacity': 0}
    ).add_to(m)
    
    folium.LayerControl().add_to(m)
    Fullscreen().add_to(m)
    MeasureControl(position='topleft', primary_length_unit='kilometers', primary_area_unit='hectares').add_to(m)
    MousePosition(position='bottomright', separator=' | ', prefix='Coords:', num_digits=6).add_to(m)
    
    legend_html = '''
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
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    m.save(str(output_html))
    logging.info(f"✅ Mapa HTML salvo: {output_html.name}")


# ═══════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════

def main():
    """Pipeline principal"""
    print("=" * 70)
    print("🌱 SMARTGREEN - Pipeline Automatizado de NDVI")
    print("=" * 70)
    print()
    
    setup_logging()
    logging.info("🚀 Iniciando pipeline SMARTGREEN")
    
    # Earth Engine
    if not initialize_earth_engine():
        sys.exit(1)
    
    # Shapefile
    gdf = load_shapefile()
    aoi, shapely_geom = gdf_to_ee_geometry(gdf)
    
    # Datas
    end_date = datetime.now()
    start_date = end_date - timedelta(days=DAYS_BACK)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    logging.info(f"📅 Período: {start_str} a {end_str}")
    
    # Sentinel-2
    logging.info("🛰️ Processando Sentinel-2...")
    
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(aoi)
                  .filterDate(start_str, end_str)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_COVER)))
    
    try:
        count = collection.size().getInfo()
        logging.info(f"   Imagens: {count}")
        
        if count == 0:
            logging.error("❌ Nenhuma imagem encontrada")
            logging.error("   Aumente DAYS_BACK ou MAX_CLOUD_COVER")
            sys.exit(1)
    except:
        logging.info("   Continuando...")
    
    logging.info("   Aplicando máscaras e calculando NDVI...")
    ndvi_image = create_ndvi_composite(aoi, start_str, end_str)
    
    # Obter bounds
    bounds_info = aoi.bounds(maxError=1).getInfo()
    coords = bounds_info['coordinates'][0]
    
    minx = min(c[0] for c in coords)
    maxx = max(c[0] for c in coords)
    miny = min(c[1] for c in coords)
    maxy = max(c[1] for c in coords)
    
    bounds = (minx, miny, maxx, maxy)
    
    # Download com tiles
    download_ndvi_with_tiles(ndvi_image, bounds, SCALE, OUTPUT_GEOTIFF)
    
    # HTML
    generate_html_map(OUTPUT_GEOTIFF, gdf, OUTPUT_HTML)
    
    # Finalizar
    print()
    print("=" * 70)
    logging.info("✅ Pipeline concluído!")
    logging.info(f"   📊 GeoTIFF: {OUTPUT_GEOTIFF}")
    logging.info(f"   🗺️  HTML: {OUTPUT_HTML}")
    logging.info(f"   📝 Log: {OUTPUT_LOG}")
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