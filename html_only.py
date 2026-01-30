#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from pathlib import Path

import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MeasureControl, MousePosition, Fullscreen


# ====== CONFIG ======
SHAPEFILE_PATH = Path("CEM/CEM.shp")
CSV_PATH = Path("out/estatisticas.csv")
OUTPUT_HTML = Path("out/mapa_ndvi.html")

# ajuste se seu shapefile usa outro campo de ID do talhão
TALHAO_ID_FIELD = "TALHAO"


def get_ndvi_color(ndvi_value: float) -> str:
    if ndvi_value < -0.2:
        return "#0000FF"
    elif ndvi_value < 0.2:
        return "#d73027"
    elif ndvi_value < 0.4:
        return "#fc8d59"
    elif ndvi_value < 0.6:
        return "#fee08b"
    elif ndvi_value < 0.7:
        return "#d9ef8b"
    elif ndvi_value < 0.8:
        return "#91cf60"
    else:
        return "#1a9850"


def generate_interactive_html(talhoes_info, output_html: Path):
    logging.info("🗺️ Gerando mapa HTML interativo...")

    # bounds gerais
    all_bounds = [t["geometry"].bounds for t in talhoes_info]
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
        overlay=False,
    ).add_to(m)

    # cada talhão como layer separada
    for t in talhoes_info:
        nome = t["nome"]
        ndvi_medio = float(t["ndvi_medio"])
        color = get_ndvi_color(ndvi_medio)

        # IMPORTANTÍSSIMO: só enviar geometry + propriedades simples (nada de Path/PosixPath)
        feat = {
            "type": "Feature",
            "properties": {
                "id": str(t["id"]),
                "nome": str(nome),
                "ndvi_medio": float(t["ndvi_medio"]),
                "ndvi_min": float(t["ndvi_min"]),
                "ndvi_max": float(t["ndvi_max"]),
                "ndvi_std": float(t["ndvi_std"]),
                "area_ha": float(t["area_ha"]),
                "pixels_validos": int(t["pixels_validos"]),
            },
            "geometry": gpd.GeoSeries([t["geometry"]], crs="EPSG:4326").__geo_interface__["features"][0]["geometry"],
        }

        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px; min-width: 220px;">
            <h4 style="margin: 5px 0; color: {color};">{nome}</h4>
            <table style="width: 100%; border-collapse: collapse;">
                <tr><td><b>NDVI Médio:</b></td><td>{t['ndvi_medio']:.3f}</td></tr>
                <tr><td><b>NDVI Mín:</b></td><td>{t['ndvi_min']:.3f}</td></tr>
                <tr><td><b>NDVI Máx:</b></td><td>{t['ndvi_max']:.3f}</td></tr>
                <tr><td><b>Área:</b></td><td>{t['area_ha']:.2f} ha</td></tr>
                <tr><td><b>Pixels:</b></td><td>{int(t['pixels_validos'])}</td></tr>
            </table>
        </div>
        """

        folium.GeoJson(
            feat,
            name=f"{nome} (NDVI: {ndvi_medio:.3f})",
            style_function=lambda _x, color=color: {
                "fillColor": color,
                "color": "white",
                "weight": 2,
                "fillOpacity": 0.5,
            },
            highlight_function=lambda _x: {"weight": 4, "fillOpacity": 0.7},
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"{nome}: {ndvi_medio:.3f}",
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    Fullscreen(position="topleft").add_to(m)
    MeasureControl(position="topleft", primary_length_unit="kilometers", primary_area_unit="hectares").add_to(m)
    MousePosition(position="bottomright", separator=" | ", prefix="Coords:", num_digits=6).add_to(m)

    legend_html = """
    <div style="position: fixed; bottom: 50px; right: 50px; width: 180px;
                background-color: white; border: 2px solid grey; z-index: 9999;
                font-size: 14px; padding: 15px; border-radius: 5px;">
        <p style="margin: 0 0 10px 0; font-weight: bold; font-size: 16px;">NDVI</p>
        <div style="display:flex; align-items:center; margin:5px 0;"><div style="background:#1a9850;width:30px;height:20px;margin-right:10px;"></div><span>0.8–1.0 Ótimo</span></div>
        <div style="display:flex; align-items:center; margin:5px 0;"><div style="background:#91cf60;width:30px;height:20px;margin-right:10px;"></div><span>0.7–0.8 Bom</span></div>
        <div style="display:flex; align-items:center; margin:5px 0;"><div style="background:#d9ef8b;width:30px;height:20px;margin-right:10px;"></div><span>0.6–0.7 Regular</span></div>
        <div style="display:flex; align-items:center; margin:5px 0;"><div style="background:#fee08b;width:30px;height:20px;margin-right:10px;"></div><span>0.4–0.6 Baixo</span></div>
        <div style="display:flex; align-items:center; margin:5px 0;"><div style="background:#fc8d59;width:30px;height:20px;margin-right:10px;"></div><span>0.2–0.4 Muito Baixo</span></div>
        <div style="display:flex; align-items:center; margin:5px 0;"><div style="background:#d73027;width:30px;height:20px;margin-right:10px;"></div><span>-0.2–0.2 Solo</span></div>
        <div style="display:flex; align-items:center; margin:5px 0;"><div style="background:#0000FF;width:30px;height:20px;margin-right:10px;"></div><span>&lt; -0.2 Água</span></div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    output_html.parent.mkdir(exist_ok=True)
    m.save(str(output_html))
    logging.info(f"✅ HTML salvo: {output_html}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not SHAPEFILE_PATH.exists():
        raise FileNotFoundError(f"Shapefile não encontrado: {SHAPEFILE_PATH}")

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV não encontrado: {CSV_PATH}")

    logging.info("📂 Lendo shapefile...")
    gdf = gpd.read_file(SHAPEFILE_PATH)
    if gdf.crs is None:
        raise ValueError("Shapefile sem CRS.")
    if str(gdf.crs).upper() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    if TALHAO_ID_FIELD not in gdf.columns:
        raise ValueError(f"Campo '{TALHAO_ID_FIELD}' não existe no shapefile. Ajuste TALHAO_ID_FIELD no script.")

    logging.info("📊 Lendo CSV de estatísticas...")
    df = pd.read_csv(CSV_PATH)

    # normaliza nomes de coluna caso variem
    colmap = {
        "ID": "id",
        "Nome": "nome",
        "NDVI Médio": "ndvi_medio",
        "NDVI Mínimo": "ndvi_min",
        "NDVI Máximo": "ndvi_max",
        "Desvio Padrão": "ndvi_std",
        "Área (ha)": "area_ha",
        "Pixels Válidos": "pixels_validos",
    }
    for k in colmap:
        if k not in df.columns:
            raise ValueError(f"CSV sem coluna '{k}'. Colunas disponíveis: {list(df.columns)}")

    df = df.rename(columns=colmap)

    logging.info("🔗 Juntando geometria + estatísticas...")
    # garante que ID é string pra bater com shapefile
    df["id"] = df["id"].astype(str)
    gdf[TALHAO_ID_FIELD] = gdf[TALHAO_ID_FIELD].astype(str)

    talhoes_info = []
    # cria index por id para lookup rápido
    geom_by_id = dict(zip(gdf[TALHAO_ID_FIELD].tolist(), gdf.geometry.tolist()))

    missing = 0
    for _, row in df.iterrows():
        tid = row["id"]
        geom = geom_by_id.get(tid)
        if geom is None:
            missing += 1
            continue

        talhoes_info.append({
            "id": tid,
            "nome": row["nome"],
            "geometry": geom,
            "ndvi_medio": float(row["ndvi_medio"]),
            "ndvi_min": float(row["ndvi_min"]),
            "ndvi_max": float(row["ndvi_max"]),
            "ndvi_std": float(row["ndvi_std"]),
            "area_ha": float(row["area_ha"]),
            "pixels_validos": int(row["pixels_validos"]),
        })

    logging.info(f"✅ Talhões no HTML: {len(talhoes_info)} | sem geometria: {missing}")

    if not talhoes_info:
        raise RuntimeError("Nenhum talhão casou ID do CSV com shapefile. Verifique TALHAO_ID_FIELD e o CSV.")

    generate_interactive_html(talhoes_info, OUTPUT_HTML)


if __name__ == "__main__":
    main()
