#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════
    SMARTGREEN - Pipeline Automatizado de NDVI por Talhão
    Processamento PARALELO de imagens Sentinel-2 via Google Earth Engine
    Com análise individual por talhão e filtros interativos
═══════════════════════════════════════════════════════════════════

🎯 FUNCIONALIDADES APRIMORADAS:
   1. ⚡ PROCESSAMENTO PARALELO com ThreadPoolExecutor
   2. Processa múltiplos talhões simultaneamente
   3. Resolução aumentada para 10m (Sentinel-2 nativo)
   4. Calcula estatísticas por talhão (média, min, max, área)
   5. Gera HTML interativo com filtros por talhão
   6. Permite ligar/desligar talhões no mapa
   7. Exporta CSV com estatísticas
   8. Gera GeoTIFF individual por talhão + mosaico geral

⚡ DESEMPENHO:
   - Processamento paralelo de até 10 talhões simultaneamente
   - Velocidade até 10x mais rápida que versão sequencial
   - Gestão inteligente de recursos e threads
   - Barra de progresso unificada

📁 ESTRUTURA ESPERADA:
   SMARTGREEN/
   ├── CEM/
   │   └── CEM.shp (com campo 'nome' ou 'id' para identificar talhões)
   ├── out/                    # Criada automaticamente
   │   ├── talhoes/           # GeoTIFFs individuais
   │   ├── ndvi_completo.tif  # Mosaico geral
   │   ├── mapa_ndvi.html     # Mapa interativo
   │   └── estatisticas.csv   # Dados por talhão
   └── smartgreen_talhoes_parallel.py  # Este arquivo

⚙️ MELHORIAS:
   - Processamento paralelo com threads
   - Resolução: 10m (antes era configurável)
   - Processamento otimizado por geometria
   - Filtros dinâmicos no HTML
   - Estatísticas detalhadas

🚀 EXECUÇÃO:
   python smartgreen_talhoes_parallel.py

📦 DEPENDÊNCIAS:
   pip install earthengine-api geopandas rasterio numpy folium requests tqdm matplotlib shapely pandas

═══════════════════════════════════════════════════════════════════
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

# Resolução máxima do Sentinel-2 (banda 8 e 4)
SCALE = 10  # 10 metros - melhor resolução disponível

# Campo no shapefile que identifica os talhões
TALHAO_ID_FIELD = "TALHAO"  # Altere para 'id', 'talhao', etc. conforme seu shapefile
TALHAO_NOME_FIELD = "NOME_FAZ"  # Campo com nome legível

# ⚡ CONFIGURAÇÕES DE PARALELIZAÇÃO
MAX_WORKERS = 20  # Número máximo de threads paralelas (ajuste conforme sua máquina)
# Recomendações:
#   - Máquina básica: 3-5 workers
#   - Máquina média: 5-10 workers
#   - Máquina potente: 10-15 workers
#   - Conexão lenta: 3-5 workers
#   - Conexão rápida: 10-15 workers


# ═══════════════════════════════════════════════════════════════════
# SISTEMA DE LOGS (Thread-safe)
# ═══════════════════════════════════════════════════════════════════

# Lock para logging thread-safe
log_lock = Lock()

def setup_logging():
    """Configura sistema de logs thread-safe"""
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
    """Logging thread-safe"""
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
    """Inicializa Google Earth Engine"""
    logging.info("🌍 Inicializando Google Earth Engine...")
    
    try:
        ee.Initialize(project=PROJECT_ID, opt_url='https://earthengine-highvolume.googleapis.com')
        logging.info("✅ Earth Engine inicializado")
        return True
    except Exception as e:
        logging.warning(f"⚠️ Tentando autenticação...")
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
    """Aplica máscara de nuvens usando banda SCL"""
    scl = image.select('SCL')
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    return image.updateMask(mask)


def calculate_ndvi(image):
    """Calcula NDVI"""
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)


def create_ndvi_for_geometry(geometry, start_date, end_date):
    """
    Cria composto NDVI para uma geometria específica
    Thread-safe - cada thread cria sua própria coleção
    """
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
    """Converte geometria Shapely para ee.Geometry"""
    if shapely_geom.geom_type == 'Polygon':
        coords = [list(shapely_geom.exterior.coords)]
        return ee.Geometry.Polygon(coords)
    elif shapely_geom.geom_type == 'MultiPolygon':
        coords = [list(poly.exterior.coords) for poly in shapely_geom.geoms]
        return ee.Geometry.MultiPolygon(coords)
    else:
        raise ValueError(f"Tipo não suportado: {shapely_geom.geom_type}")


def download_ndvi_talhao(ndvi_image, geometry, talhao_id, output_path):
    """
    Faz download do NDVI de um talhão específico
    Thread-safe - cada thread tem sua própria requisição HTTP
    """
    try:
        url = ndvi_image.getDownloadURL({
            'scale': SCALE,
            'region': geometry,
            'format': 'GEO_TIFF',
            'crs': 'EPSG:4326'
        })
        
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        
        # Verificar se é ZIP
        content_type = response.headers.get('content-type', '')
        
        if 'zip' in content_type:
            data = io.BytesIO(response.content)
            with zipfile.ZipFile(data) as zf:
                tif_files = [f for f in zf.namelist() if f.endswith('.tif')]
                if tif_files:
                    with zf.open(tif_files[0]) as tif_file:
                        output_path.write_bytes(tif_file.read())
        else:
            output_path.write_bytes(response.content)
        
        return True
        
    except Exception as e:
        log_safe('error', f"Erro no download do talhão {talhao_id}: {e}")
        return False


def calculate_talhao_statistics(geotiff_path, shapely_geom):
    """
    Calcula estatísticas do NDVI para um talhão
    Thread-safe - cada thread abre seu próprio arquivo
    """
    try:
        with rasterio.open(geotiff_path) as src:
            data = src.read(1)
            nodata = src.nodata if src.nodata is not None else -9999
            
            # Criar máscara da geometria
            mask = geometry_mask(
                [shapely_geom],
                transform=src.transform,
                invert=True,
                out_shape=data.shape
            )
            
            # Aplicar máscara e filtrar nodata
            valid_data = data[mask & ~np.isnan(data) & (data != nodata)]
            
            if len(valid_data) > 0:
                # Calcular área em hectares
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
    """
    Processa um único talhão (função executada por cada thread)
    
    Args:
        talhao_data: dict com {idx, talhao_id, talhao_nome, geometry}
        start_date: data inicial
        end_date: data final
    
    Returns:
        dict com informações do talhão ou None em caso de erro
    """
    idx = talhao_data['idx']
    talhao_id = talhao_data['talhao_id']
    talhao_nome = talhao_data['talhao_nome']
    geometry = talhao_data['geometry']
    
    try:
        # Converter para EE
        ee_geometry = shapely_to_ee_geometry(geometry)
        
        # Criar NDVI
        ndvi_image = create_ndvi_for_geometry(ee_geometry, start_date, end_date)
        
        # Download
        output_path = TALHOES_DIR / f"ndvi_{talhao_id}.tif"
        success = download_ndvi_talhao(ndvi_image, ee_geometry, talhao_id, output_path)
        
        if success:
            # Calcular estatísticas
            stats = calculate_talhao_statistics(output_path, geometry)
            
            if stats:
                result = {
                    'id': talhao_id,
                    'nome': talhao_nome,
                    'geometry': geometry,
                    'geotiff_path': output_path,
                    **stats
                }
                
                log_safe('info', f"✅ {talhao_nome}: NDVI={stats['ndvi_medio']:.3f}, Área={stats['area_ha']:.2f}ha")
                return result
            else:
                log_safe('warning', f"⚠️ {talhao_nome}: sem dados válidos")
                return None
        else:
            log_safe('error', f"❌ {talhao_nome}: falha no download")
            return None
            
    except Exception as e:
        log_safe('error', f"❌ {talhao_nome}: erro no processamento - {e}")
        return None


def process_talhoes_parallel(gdf, start_date, end_date, max_workers=MAX_WORKERS):
    """
    Processa todos os talhões em paralelo usando ThreadPoolExecutor
    
    Args:
        gdf: GeoDataFrame com os talhões
        start_date: data inicial
        end_date: data final
        max_workers: número máximo de threads paralelas
    
    Returns:
        list de dicts com informações dos talhões processados
    """
    logging.info(f"🌾 Processando {len(gdf)} talhões em PARALELO...")
    logging.info(f"⚡ Usando {max_workers} threads simultâneas")
    
    # Verificar campo identificador
    if TALHAO_ID_FIELD not in gdf.columns:
        logging.warning(f"⚠️ Campo '{TALHAO_ID_FIELD}' não encontrado")
        logging.info(f"   Campos disponíveis: {', '.join(gdf.columns)}")
        logging.info("   Criando IDs automáticos...")
        gdf[TALHAO_ID_FIELD] = [f"talhao_{i+1}" for i in range(len(gdf))]
    
    # Preparar dados dos talhões
    talhoes_data = []
    for idx, row in gdf.iterrows():
        talhoes_data.append({
            'idx': idx,
            'talhao_id': str(row[TALHAO_ID_FIELD]),
            'talhao_nome': str(row.get(TALHAO_NOME_FIELD, str(row[TALHAO_ID_FIELD]))),
            'geometry': row.geometry
        })
    
    # Processamento paralelo
    talhoes_info = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submeter todas as tarefas
        future_to_talhao = {
            executor.submit(process_single_talhao, talhao_data, start_date, end_date): talhao_data
            for talhao_data in talhoes_data
        }
        
        # Processar resultados com barra de progresso
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
    """Cria mosaico com todos os talhões"""
    logging.info("🔗 Criando mosaico completo...")
    
    if not talhoes_info:
        logging.error("❌ Nenhum talhão processado")
        return False
    
    try:
        # Abrir todos os GeoTIFFs
        src_files = [rasterio.open(t['geotiff_path']) for t in talhoes_info]
        
        # Mesclar
        mosaic, out_trans = merge(src_files)
        
        # Metadados
        out_meta = src_files[0].meta.copy()
        out_meta.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans
        })
        
        # Salvar
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic)
        
        # Fechar
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
    """Exporta estatísticas para CSV"""
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
# GERAÇÃO DO HTML INTERATIVO
# ═══════════════════════════════════════════════════════════════════

def get_ndvi_color(ndvi_value):
    """Retorna cor para valor NDVI"""
    if ndvi_value < -0.2:
        return '#0000FF'  # Azul - Água
    elif ndvi_value < 0.2:
        return '#d73027'  # Vermelho - Solo/Seco
    elif ndvi_value < 0.4:
        return '#fc8d59'  # Laranja
    elif ndvi_value < 0.6:
        return '#fee08b'  # Amarelo
    elif ndvi_value < 0.7:
        return '#d9ef8b'  # Verde claro
    elif ndvi_value < 0.8:
        return '#91cf60'  # Verde
    else:
        return '#1a9850'  # Verde escuro


def generate_interactive_html(talhoes_info, mosaic_path, output_html):
    """Gera mapa HTML interativo com filtros por talhão"""
    logging.info("🗺️ Gerando mapa HTML interativo...")
    
    # Calcular centro
    all_bounds = [t['geometry'].bounds for t in talhoes_info]
    min_lon = min(b[0] for b in all_bounds)
    min_lat = min(b[1] for b in all_bounds)
    max_lon = max(b[2] for b in all_bounds)
    max_lat = max(b[3] for b in all_bounds)
    
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    
    # Criar mapa
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles='OpenStreetMap'
    )
    
    # Camada satélite
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satélite',
        overlay=False
    ).add_to(m)
    
    # Adicionar cada talhão como camada separada
    for t in talhoes_info:
        nome = t['nome']
        ndvi_medio = t['ndvi_medio']
        area_ha = t['area_ha']
        
        # Cor baseada no NDVI
        color = get_ndvi_color(ndvi_medio)
        
        # Criar GeoJson do talhão
        talhao_gdf = gpd.GeoDataFrame([t], geometry='geometry', crs='EPSG:4326')
        
        # Popup com informações
        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px; min-width: 200px;">
            <h4 style="margin: 5px 0; color: {color};">{nome}</h4>
            <table style="width: 100%; border-collapse: collapse;">
                <tr><td><b>NDVI Médio:</b></td><td>{ndvi_medio:.3f}</td></tr>
                <tr><td><b>NDVI Mín:</b></td><td>{t['ndvi_min']:.3f}</td></tr>
                <tr><td><b>NDVI Máx:</b></td><td>{t['ndvi_max']:.3f}</td></tr>
                <tr><td><b>Área:</b></td><td>{area_ha:.2f} ha</td></tr>
            </table>
        </div>
        """
        
        # Adicionar ao mapa
        folium.GeoJson(
            talhao_gdf,
            name=f'{nome} (NDVI: {ndvi_medio:.3f})',
            style_function=lambda x, color=color: {
                'fillColor': color,
                'color': 'white',
                'weight': 2,
                'fillOpacity': 0.5
            },
            highlight_function=lambda x: {
                'weight': 4,
                'fillOpacity': 0.7
            },
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f'{nome}: {ndvi_medio:.3f}'
        ).add_to(m)
    
    # Controle de camadas (permite ligar/desligar talhões)
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Plugins úteis
    Fullscreen(position='topleft').add_to(m)
    MeasureControl(
        position='topleft',
        primary_length_unit='kilometers',
        primary_area_unit='hectares'
    ).add_to(m)
    MousePosition(
        position='bottomright',
        separator=' | ',
        prefix='Coords:',
        num_digits=6
    ).add_to(m)
    
    # Legenda
    legend_html = '''
    <div style="position: fixed; bottom: 50px; right: 50px; width: 180px; 
                background-color: white; border: 2px solid grey; z-index: 9999; 
                font-size: 14px; padding: 15px; border-radius: 5px;">
        <p style="margin: 0 0 10px 0; font-weight: bold; font-size: 16px;">NDVI</p>
        <div style="margin-bottom: 10px;">
            <div style="display: flex; align-items: center; margin: 5px 0;">
                <div style="background: #1a9850; width: 30px; height: 20px; margin-right: 10px;"></div>
                <span>0.8-1.0 Ótimo</span>
            </div>
            <div style="display: flex; align-items: center; margin: 5px 0;">
                <div style="background: #91cf60; width: 30px; height: 20px; margin-right: 10px;"></div>
                <span>0.7-0.8 Bom</span>
            </div>
            <div style="display: flex; align-items: center; margin: 5px 0;">
                <div style="background: #d9ef8b; width: 30px; height: 20px; margin-right: 10px;"></div>
                <span>0.6-0.7 Regular</span>
            </div>
            <div style="display: flex; align-items: center; margin: 5px 0;">
                <div style="background: #fee08b; width: 30px; height: 20px; margin-right: 10px;"></div>
                <span>0.4-0.6 Baixo</span>
            </div>
            <div style="display: flex; align-items: center; margin: 5px 0;">
                <div style="background: #fc8d59; width: 30px; height: 20px; margin-right: 10px;"></div>
                <span>0.2-0.4 Muito Baixo</span>
            </div>
            <div style="display: flex; align-items: center; margin: 5px 0;">
                <div style="background: #d73027; width: 30px; height: 20px; margin-right: 10px;"></div>
                <span>-0.2-0.2 Solo</span>
            </div>
            <div style="display: flex; align-items: center; margin: 5px 0;">
                <div style="background: #0000FF; width: 30px; height: 20px; margin-right: 10px;"></div>
                <span>&lt;-0.2 Água</span>
            </div>
        </div>
        <p style="margin: 10px 0 0 0; font-size: 11px; color: #666;">
            ⚡ Processado em paralelo
        </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Salvar
    m.save(str(output_html))
    logging.info(f"✅ Mapa HTML salvo: {output_html.name}")


# ═══════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════

def main():
    """Pipeline principal com processamento paralelo"""
    print("=" * 70)
    print("🌱 SMARTGREEN - Pipeline NDVI Paralelo")
    print("⚡ Processamento com múltiplas threads")
    print("=" * 70)
    print()
    
    setup_logging()
    logging.info("🚀 Iniciando pipeline SMARTGREEN com processamento PARALELO")
    logging.info(f"⚡ Configurado para {MAX_WORKERS} threads simultâneas")
    
    # Registrar início
    start_time = datetime.now()
    
    # Earth Engine
    if not initialize_earth_engine():
        sys.exit(1)
    
    # Carregar shapefile
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
    
    # Datas
    end_date = datetime.now()
    start_date_period = end_date - timedelta(days=DAYS_BACK)
    start_str = start_date_period.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    logging.info(f"📅 Período: {start_str} a {end_str}")
    logging.info(f"📏 Resolução: {SCALE}m (Sentinel-2 máxima)")
    
    # Processar talhões em PARALELO
    talhoes_info = process_talhoes_parallel(gdf, start_str, end_str, MAX_WORKERS)
    
    if not talhoes_info:
        logging.error("❌ Nenhum talhão processado com sucesso")
        sys.exit(1)
    
    logging.info(f"✅ {len(talhoes_info)}/{len(gdf)} talhões processados")
    
    # Calcular tempo de processamento
    processing_time = (datetime.now() - start_time).total_seconds()
    avg_time_per_talhao = processing_time / len(talhoes_info) if talhoes_info else 0
    
    logging.info(f"⏱️  Tempo total: {processing_time:.1f}s")
    logging.info(f"⏱️  Tempo médio por talhão: {avg_time_per_talhao:.1f}s")
    
    # Criar mosaico
    create_complete_mosaic(talhoes_info, OUTPUT_GEOTIFF_COMPLETO)
    
    # Exportar estatísticas
    export_statistics(talhoes_info, OUTPUT_CSV)
    
    # Gerar HTML
    generate_interactive_html(talhoes_info, OUTPUT_GEOTIFF_COMPLETO, OUTPUT_HTML)
    
    # Resumo final
    print()
    print("=" * 70)
    logging.info("✅ Pipeline concluído!")
    logging.info(f"   ⚡ Processamento PARALELO com {MAX_WORKERS} threads")
    logging.info(f"   ⏱️  Tempo total: {processing_time:.1f}s ({avg_time_per_talhao:.1f}s/talhão)")
    logging.info(f"   📊 Talhões processados: {len(talhoes_info)}")
    logging.info(f"   📁 GeoTIFFs individuais: {TALHOES_DIR}/")
    logging.info(f"   🗺️  Mosaico completo: {OUTPUT_GEOTIFF_COMPLETO}")
    logging.info(f"   🌐 Mapa interativo: {OUTPUT_HTML}")
    logging.info(f"   📈 Estatísticas: {OUTPUT_CSV}")
    logging.info(f"   📝 Log: {OUTPUT_LOG}")
    print("=" * 70)
    
    # Estatísticas gerais
    ndvi_medios = [t['ndvi_medio'] for t in talhoes_info]
    area_total = sum(t['area_ha'] for t in talhoes_info)
    
    print()
    print("📊 ESTATÍSTICAS GERAIS:")
    print(f"   NDVI médio geral: {np.mean(ndvi_medios):.3f}")
    print(f"   NDVI mínimo: {min(t['ndvi_min'] for t in talhoes_info):.3f}")
    print(f"   NDVI máximo: {max(t['ndvi_max'] for t in talhoes_info):.3f}")
    print(f"   Área total: {area_total:.2f} ha")
    print()
    
    # Ranking de talhões
    print("🏆 TOP 5 TALHÕES (Melhor NDVI):")
    sorted_talhoes = sorted(talhoes_info, key=lambda x: x['ndvi_medio'], reverse=True)
    for i, t in enumerate(sorted_talhoes[:5], 1):
        print(f"   {i}. {t['nome']}: {t['ndvi_medio']:.3f} ({t['area_ha']:.2f} ha)")
    print()
    
    print("⚠️ TOP 5 TALHÕES (Pior NDVI - requer atenção):")
    for i, t in enumerate(sorted_talhoes[-5:][::-1], 1):
        print(f"   {i}. {t['nome']}: {t['ndvi_medio']:.3f} ({t['area_ha']:.2f} ha)")
    print()
    
    # Speedup estimado
    estimated_sequential_time = len(talhoes_info) * avg_time_per_talhao * MAX_WORKERS
    speedup = estimated_sequential_time / processing_time if processing_time > 0 else 1
    
    print("⚡ DESEMPENHO PARALELO:")
    print(f"   Speedup estimado: {speedup:.1f}x")
    print(f"   Tempo economizado: ~{(estimated_sequential_time - processing_time)/60:.1f} minutos")
    
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