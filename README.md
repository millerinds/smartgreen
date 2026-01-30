# SafraSmart

Pipeline para cálculo de NDVI por talhão usando Google Earth Engine (Sentinel‑2), com processamento paralelo, exportação de estatísticas e geração de mapa interativo.

## O que este projeto faz
- Baixa NDVI por talhão a partir do Sentinel‑2 (Earth Engine).
- Gera GeoTIFFs individuais e um mosaico completo.
- Calcula estatísticas por talhão (média, min, max, desvio, área).
- Exporta um CSV consolidado.
- Gera um mapa HTML interativo com camadas por talhão.

## Pré-requisitos
- Python 3.9+ (recomendado)
- Conta no Google Earth Engine
- Shapefile dos talhões em `CEM/CEM.shp`

## Instalação
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuração rápida
No arquivo `safrasmart.py`, ajuste se necessário:
- `PROJECT_ID`: ID do projeto no Google Cloud/Earth Engine
- `SHAPEFILE_PATH`: caminho do shapefile
- `TALHAO_ID_FIELD` e `TALHAO_NOME_FIELD`: campos do shapefile
- `DAYS_BACK`, `MAX_CLOUD_COVER`, `SCALE`, `MAX_WORKERS`

## Como usar (pipeline principal)
```bash
source venv/bin/activate
python safrasmart.py
```

Na primeira execução, o script pode pedir autenticação do Earth Engine.

## Saídas geradas
Os arquivos são gravados em `out/`:
- `out/talhoes/ndvi_<id>.tif`: GeoTIFF por talhão
- `out/ndvi_completo.tif`: mosaico com todos os talhões
- `out/estatisticas.csv`: estatísticas por talhão
- `out/mapa_ndvi.html`: mapa interativo
- `out/pipeline.log`: log detalhado

## Scripts auxiliares
- `verificar_shapefile.py`: lista campos do shapefile e sugere identificadores
  ```bash
  python verificar_shapefile.py
  ```
- `analisar_resultados.py`: gera gráficos a partir do CSV
  ```bash
  python analisar_resultados.py
  ```
  Saídas em `out/graficos/`

- `gerar_html.py`: gera somente o mapa HTML a partir de um GeoTIFF existente
  ```bash
  python gerar_html.py
  ```
  Observação: ajuste `GEOTIFF_PATH` no arquivo caso o seu mosaico esteja em
  `out/ndvi_completo.tif`.

## Estrutura do projeto
```
SafraSmart/
├── CEM/                    # Shapefile dos talhões
├── out/                    # Saídas do pipeline
├── safrasmart.py   # Pipeline principal (NDVI paralelo)
├── analisar_resultados.py  # Gráficos e relatórios
├── gerar_html.py           # Geração isolada do HTML
├── verificar_shapefile.py  # Inspeção do shapefile
└── requirements.txt        # Dependências
```

## Observações
- O processamento paralelo depende da sua máquina e conexão.
- Se o shapefile não estiver em EPSG:4326, o script reprojeta automaticamente.


