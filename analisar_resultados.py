#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════
    SMARTGREEN - Análise de Resultados
    Script para gerar gráficos e relatórios dos dados NDVI
═══════════════════════════════════════════════════════════════════
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Configurações
OUTPUT_DIR = Path("out")
CSV_PATH = OUTPUT_DIR / "estatisticas.csv"
PLOTS_DIR = OUTPUT_DIR / "graficos"

# Criar diretório de gráficos
PLOTS_DIR.mkdir(exist_ok=True)

# Configurar estilo dos gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def load_data():
    """Carrega dados do CSV"""
    if not CSV_PATH.exists():
        print(f"❌ Arquivo não encontrado: {CSV_PATH}")
        print("   Execute primeiro: python smartgreen_talhoes.py")
        return None
    
    df = pd.read_csv(CSV_PATH)
    print(f"✅ Dados carregados: {len(df)} talhões")
    return df


def plot_ndvi_ranking(df):
    """Gráfico de ranking de NDVI por talhão"""
    df_sorted = df.sort_values('NDVI Médio', ascending=True)
    
    # Definir cores baseadas no NDVI
    colors = []
    for ndvi in df_sorted['NDVI Médio']:
        if ndvi >= 0.7:
            colors.append('#1a9850')  # Verde escuro - Ótimo
        elif ndvi >= 0.6:
            colors.append('#91cf60')  # Verde - Bom
        elif ndvi >= 0.4:
            colors.append('#fee08b')  # Amarelo - Regular
        elif ndvi >= 0.2:
            colors.append('#fc8d59')  # Laranja - Baixo
        else:
            colors.append('#d73027')  # Vermelho - Crítico
    
    plt.figure(figsize=(14, max(8, len(df) * 0.4)))
    bars = plt.barh(df_sorted['Nome'], df_sorted['NDVI Médio'], color=colors)
    
    # Adicionar valores nas barras
    for i, (bar, ndvi) in enumerate(zip(bars, df_sorted['NDVI Médio'])):
        plt.text(ndvi + 0.02, i, f'{ndvi:.3f}', 
                va='center', fontsize=9, fontweight='bold')
    
    plt.xlabel('NDVI Médio', fontsize=12, fontweight='bold')
    plt.ylabel('Talhão', fontsize=12, fontweight='bold')
    plt.title('Ranking de NDVI por Talhão', fontsize=14, fontweight='bold', pad=20)
    plt.xlim(-0.1, 1.0)
    
    # Adicionar linhas de referência
    plt.axvline(x=0.7, color='green', linestyle='--', alpha=0.3, label='Ótimo (≥0.7)')
    plt.axvline(x=0.6, color='yellow', linestyle='--', alpha=0.3, label='Bom (≥0.6)')
    plt.axvline(x=0.4, color='orange', linestyle='--', alpha=0.3, label='Regular (≥0.4)')
    
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    output_path = PLOTS_DIR / "01_ranking_ndvi.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico salvo: {output_path.name}")
    plt.close()


def plot_area_vs_ndvi(df):
    """Gráfico de dispersão: Área vs NDVI"""
    plt.figure(figsize=(12, 8))
    
    scatter = plt.scatter(df['Área (ha)'], df['NDVI Médio'], 
                         s=df['Área (ha)'] * 10,  # Tamanho proporcional à área
                         c=df['NDVI Médio'], 
                         cmap='RdYlGn', 
                         alpha=0.6,
                         edgecolors='black',
                         linewidth=1)
    
    # Adicionar nomes dos talhões
    for idx, row in df.iterrows():
        plt.annotate(row['Nome'], 
                    (row['Área (ha)'], row['NDVI Médio']),
                    fontsize=8,
                    alpha=0.7,
                    xytext=(5, 5),
                    textcoords='offset points')
    
    plt.xlabel('Área (hectares)', fontsize=12, fontweight='bold')
    plt.ylabel('NDVI Médio', fontsize=12, fontweight='bold')
    plt.title('Relação entre Área e NDVI dos Talhões', fontsize=14, fontweight='bold', pad=20)
    
    # Colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('NDVI Médio', fontsize=10, fontweight='bold')
    
    # Linhas de referência
    plt.axhline(y=0.7, color='green', linestyle='--', alpha=0.3)
    plt.axhline(y=0.6, color='yellow', linestyle='--', alpha=0.3)
    plt.axhline(y=0.4, color='orange', linestyle='--', alpha=0.3)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = PLOTS_DIR / "02_area_vs_ndvi.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico salvo: {output_path.name}")
    plt.close()


def plot_ndvi_distribution(df):
    """Histograma de distribuição do NDVI"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histograma
    ax1.hist(df['NDVI Médio'], bins=20, color='#91cf60', edgecolor='black', alpha=0.7)
    ax1.axvline(df['NDVI Médio'].mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Média: {df["NDVI Médio"].mean():.3f}')
    ax1.axvline(df['NDVI Médio'].median(), color='blue', linestyle='--', 
               linewidth=2, label=f'Mediana: {df["NDVI Médio"].median():.3f}')
    ax1.set_xlabel('NDVI Médio', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Número de Talhões', fontsize=12, fontweight='bold')
    ax1.set_title('Distribuição do NDVI', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Boxplot
    ax2.boxplot(df['NDVI Médio'], vert=True, patch_artist=True,
                boxprops=dict(facecolor='#91cf60', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_ylabel('NDVI Médio', fontsize=12, fontweight='bold')
    ax2.set_title('Boxplot do NDVI', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Adicionar faixas de classificação no boxplot
    ax2.axhline(y=0.7, color='green', linestyle='--', alpha=0.3, label='Ótimo')
    ax2.axhline(y=0.6, color='yellow', linestyle='--', alpha=0.3, label='Bom')
    ax2.axhline(y=0.4, color='orange', linestyle='--', alpha=0.3, label='Regular')
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    
    output_path = PLOTS_DIR / "03_distribuicao_ndvi.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico salvo: {output_path.name}")
    plt.close()


def plot_classification_pie(df):
    """Gráfico de pizza: Classificação dos talhões"""
    # Classificar talhões
    classifications = []
    for ndvi in df['NDVI Médio']:
        if ndvi >= 0.7:
            classifications.append('Ótimo (≥0.7)')
        elif ndvi >= 0.6:
            classifications.append('Bom (0.6-0.7)')
        elif ndvi >= 0.4:
            classifications.append('Regular (0.4-0.6)')
        elif ndvi >= 0.2:
            classifications.append('Baixo (0.2-0.4)')
        else:
            classifications.append('Crítico (<0.2)')
    
    df['Classificação'] = classifications
    
    # Contar
    counts = df['Classificação'].value_counts()
    
    # Cores
    colors_map = {
        'Ótimo (≥0.7)': '#1a9850',
        'Bom (0.6-0.7)': '#91cf60',
        'Regular (0.4-0.6)': '#fee08b',
        'Baixo (0.2-0.4)': '#fc8d59',
        'Crítico (<0.2)': '#d73027'
    }
    colors = [colors_map.get(c, '#cccccc') for c in counts.index]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pizza
    wedges, texts, autotexts = ax1.pie(counts.values, 
                                        labels=counts.index,
                                        colors=colors,
                                        autopct='%1.1f%%',
                                        startangle=90,
                                        explode=[0.05] * len(counts))
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    ax1.set_title('Distribuição de Talhões por Classificação', 
                 fontsize=12, fontweight='bold')
    
    # Barras
    ax2.bar(range(len(counts)), counts.values, color=colors, edgecolor='black')
    ax2.set_xticks(range(len(counts)))
    ax2.set_xticklabels(counts.index, rotation=45, ha='right')
    ax2.set_ylabel('Número de Talhões', fontsize=11, fontweight='bold')
    ax2.set_title('Quantidade por Classificação', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Adicionar valores nas barras
    for i, v in enumerate(counts.values):
        ax2.text(i, v + 0.1, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    output_path = PLOTS_DIR / "04_classificacao_talhoes.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico salvo: {output_path.name}")
    plt.close()


def plot_variability(df):
    """Gráfico de variabilidade (min-max) do NDVI"""
    df_sorted = df.sort_values('NDVI Médio')
    
    plt.figure(figsize=(14, max(8, len(df) * 0.4)))
    
    # Plotar range (min-max)
    for i, row in enumerate(df_sorted.itertuples()):
        plt.plot([row._4, row._5],  # NDVI Mínimo e Máximo
                [i, i], 
                'o-', 
                linewidth=2, 
                markersize=6,
                color='#666666',
                alpha=0.6)
        
        # Destacar média
        plt.plot(row._3,  # NDVI Médio
                i,
                'D',
                markersize=10,
                color='#1a9850',
                markeredgecolor='black',
                markeredgewidth=1)
    
    plt.yticks(range(len(df_sorted)), df_sorted['Nome'])
    plt.xlabel('NDVI', fontsize=12, fontweight='bold')
    plt.ylabel('Talhão', fontsize=12, fontweight='bold')
    plt.title('Variabilidade do NDVI por Talhão (Min-Média-Max)', 
             fontsize=14, fontweight='bold', pad=20)
    plt.xlim(-0.1, 1.0)
    
    # Legenda customizada
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#1a9850', 
               markersize=10, markeredgecolor='black', label='NDVI Médio'),
        Line2D([0], [0], marker='o', color='#666666', linestyle='-', 
               markersize=6, label='Range (Min-Max)')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    output_path = PLOTS_DIR / "05_variabilidade_ndvi.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico salvo: {output_path.name}")
    plt.close()


def generate_report(df):
    """Gera relatório textual"""
    report_path = PLOTS_DIR / "00_relatorio.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RELATÓRIO DE ANÁLISE NDVI - SMARTGREEN\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Data: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total de Talhões: {len(df)}\n")
        f.write(f"Área Total: {df['Área (ha)'].sum():.2f} ha\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("ESTATÍSTICAS GERAIS\n")
        f.write("-" * 70 + "\n")
        f.write(f"NDVI Médio Geral: {df['NDVI Médio'].mean():.3f}\n")
        f.write(f"Desvio Padrão: {df['NDVI Médio'].std():.3f}\n")
        f.write(f"NDVI Mínimo (global): {df['NDVI Mínimo'].min():.3f}\n")
        f.write(f"NDVI Máximo (global): {df['NDVI Máximo'].max():.3f}\n")
        f.write(f"Mediana: {df['NDVI Médio'].median():.3f}\n\n")
        
        # Classificação
        classifications = []
        for ndvi in df['NDVI Médio']:
            if ndvi >= 0.7:
                classifications.append('Ótimo')
            elif ndvi >= 0.6:
                classifications.append('Bom')
            elif ndvi >= 0.4:
                classifications.append('Regular')
            elif ndvi >= 0.2:
                classifications.append('Baixo')
            else:
                classifications.append('Crítico')
        
        df['Classificação'] = classifications
        counts = df['Classificação'].value_counts()
        
        f.write("-" * 70 + "\n")
        f.write("DISTRIBUIÇÃO POR CLASSIFICAÇÃO\n")
        f.write("-" * 70 + "\n")
        for classe, count in counts.items():
            pct = (count / len(df)) * 100
            f.write(f"{classe:12s}: {count:3d} talhões ({pct:5.1f}%)\n")
        f.write("\n")
        
        f.write("-" * 70 + "\n")
        f.write("TOP 10 MELHORES TALHÕES\n")
        f.write("-" * 70 + "\n")
        top10 = df.nlargest(10, 'NDVI Médio')
        for i, row in enumerate(top10.itertuples(), 1):
            f.write(f"{i:2d}. {row.Nome:20s} | NDVI: {row._3:.3f} | "
                   f"Área: {row._7:.2f} ha | {row.Classificação}\n")
        f.write("\n")
        
        f.write("-" * 70 + "\n")
        f.write("TOP 10 TALHÕES QUE REQUEREM ATENÇÃO (Menor NDVI)\n")
        f.write("-" * 70 + "\n")
        bottom10 = df.nsmallest(10, 'NDVI Médio')
        for i, row in enumerate(bottom10.itertuples(), 1):
            f.write(f"{i:2d}. {row.Nome:20s} | NDVI: {row._3:.3f} | "
                   f"Área: {row._7:.2f} ha | {row.Classificação} ⚠️\n")
        f.write("\n")
        
        f.write("-" * 70 + "\n")
        f.write("TALHÕES COM MAIOR VARIABILIDADE (Maior Desvio Padrão)\n")
        f.write("-" * 70 + "\n")
        varied = df.nlargest(10, 'Desvio Padrão')
        for i, row in enumerate(varied.itertuples(), 1):
            f.write(f"{i:2d}. {row.Nome:20s} | Desvio: {row._6:.3f} | "
                   f"NDVI: {row._3:.3f} | Range: {row._4:.3f} - {row._5:.3f}\n")
        f.write("\n")
        
        f.write("=" * 70 + "\n")
        f.write("RECOMENDAÇÕES\n")
        f.write("=" * 70 + "\n\n")
        
        critical = df[df['NDVI Médio'] < 0.4]
        if len(critical) > 0:
            f.write(f"⚠️  ATENÇÃO URGENTE: {len(critical)} talhões com NDVI < 0.4\n")
            f.write("   Ações: Investigar causa, considerar irrigação, análise de solo\n\n")
        
        low = df[(df['NDVI Médio'] >= 0.4) & (df['NDVI Médio'] < 0.6)]
        if len(low) > 0:
            f.write(f"👁️  MONITORAMENTO: {len(low)} talhões com NDVI entre 0.4 e 0.6\n")
            f.write("   Ações: Acompanhar evolução, verificar necessidades\n\n")
        
        good = df[df['NDVI Médio'] >= 0.6]
        if len(good) > 0:
            f.write(f"✅ BOA CONDIÇÃO: {len(good)} talhões com NDVI ≥ 0.6\n")
            f.write("   Ações: Manter manejo atual, monitoramento de rotina\n\n")
        
        f.write("=" * 70 + "\n")
    
    print(f"📄 Relatório salvo: {report_path.name}")


def main():
    """Função principal"""
    print("=" * 70)
    print("📊 SMARTGREEN - Análise de Resultados")
    print("=" * 70)
    print()
    
    # Carregar dados
    df = load_data()
    if df is None:
        return
    
    print()
    print("🎨 Gerando gráficos...")
    print()
    
    # Gerar gráficos
    plot_ndvi_ranking(df)
    plot_area_vs_ndvi(df)
    plot_ndvi_distribution(df)
    plot_classification_pie(df)
    plot_variability(df)
    
    # Gerar relatório
    print()
    generate_report(df)
    
    print()
    print("=" * 70)
    print("✅ Análise concluída!")
    print(f"   📁 Gráficos salvos em: {PLOTS_DIR}/")
    print(f"   📄 Relatório: {PLOTS_DIR}/00_relatorio.txt")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()