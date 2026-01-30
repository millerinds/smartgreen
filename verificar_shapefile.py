#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper script para descobrir o campo identificador do shapefile
"""

import geopandas as gpd
from pathlib import Path

SHAPEFILE = Path("CEM/CEM.shp")

def analisar_shapefile():
    """Analisa o shapefile e sugere o campo identificador"""
    
    print("=" * 70)
    print("🔍 SMARTGREEN - Analisador de Shapefile")
    print("=" * 70)
    print()
    
    if not SHAPEFILE.exists():
        print(f"❌ Shapefile não encontrado: {SHAPEFILE}")
        print("   Verifique se está na pasta correta")
        return
    
    print(f"📂 Carregando: {SHAPEFILE}")
    
    try:
        gdf = gpd.read_file(SHAPEFILE)
        print(f"✅ Carregado: {len(gdf)} feições")
        print(f"   CRS: {gdf.crs}")
        print()
        
        # Listar colunas
        print("=" * 70)
        print("📋 COLUNAS DISPONÍVEIS")
        print("=" * 70)
        print()
        
        for i, col in enumerate(gdf.columns, 1):
            print(f"{i:2d}. {col}")
        
        print()
        print("=" * 70)
        print("🎯 SUGESTÕES PARA TALHAO_ID_FIELD")
        print("=" * 70)
        print()
        
        # Sugestões automáticas
        campos_comuns = ['nome', 'id', 'talhao', 'cod', 'codigo', 'name', 'field']
        sugestoes = []
        
        for col in gdf.columns:
            col_lower = col.lower()
            if any(campo in col_lower for campo in campos_comuns):
                sugestoes.append(col)
        
        if sugestoes:
            print("Campos que parecem identificadores:")
            for sug in sugestoes:
                print(f"   ✅ {sug}")
                
                # Mostrar amostra
                amostra = gdf[sug].head(5).tolist()
                print(f"      Amostra: {amostra}")
                print()
        else:
            print("⚠️ Nenhum campo óbvio encontrado")
            print("   Campos disponíveis:")
            for col in gdf.columns:
                if col != 'geometry':
                    print(f"   - {col}")
        
        print()
        print("=" * 70)
        print("📊 PRÉVIA DOS DADOS")
        print("=" * 70)
        print()
        
        # Mostrar preview (exceto geometria)
        colunas_preview = [c for c in gdf.columns if c != 'geometry']
        print(gdf[colunas_preview].head())
        
        print()
        print("=" * 70)
        print("⚙️ CONFIGURAÇÃO RECOMENDADA")
        print("=" * 70)
        print()
        
        if sugestoes:
            campo_recomendado = sugestoes[0]
            print(f"Edite a linha 58 do smartgreen_talhoes.py:")
            print()
            print(f'   TALHAO_ID_FIELD = "{campo_recomendado}"')
            print()
        else:
            print("Escolha um campo da lista acima e configure:")
            print()
            print('   TALHAO_ID_FIELD = "SEU_CAMPO_AQUI"')
            print()
        
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Erro ao ler shapefile: {e}")
        print()
        print("Possíveis causas:")
        print("   - Shapefile corrompido")
        print("   - Arquivos auxiliares faltando (.shx, .dbf, .prj)")
        print("   - Problemas de codificação")


if __name__ == "__main__":
    try:
        analisar_shapefile()
    except KeyboardInterrupt:
        print("\n⚠️ Interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro: {e}")