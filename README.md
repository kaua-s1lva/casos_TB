# Identificação de padrões de indivíduos que abandonam o tratamento da tuberculose com técnicas de IA

Código do Trabalho de Conclusão de Curso de **Kauã de Souza da Silva** (Universidade Federal do Espírito Santo), sob orientação do **Prof. Dr. Antonio Almeida de Barros Junior**.

O projeto treina e compara modelos de aprendizado de máquina para prever a **perda de seguimento** (abandono ou abandono primário) do tratamento da tuberculose pulmonar no Brasil. Os dados são as notificações do SINAN de 2018 a 2022, combinadas com indicadores do município de residência: IDHM, índice de Gini, porte populacional, região e cobertura da Atenção Primária à Saúde. A abordagem segue a de [Rodrigues et al. (2024)](https://doi.org/10.1186/s12889-024-18815-0).

| Item | Valor |
|---|---|
| População de estudo | 185.361 notificações de adultos com TB pulmonar sensível, com desfecho cura ou perda de seguimento |
| Divisão | treino: 2018–2021 (148.830 casos; 15,2% de perda) · teste: 2022 (36.531 casos; 18,0% de perda) |
| Modelos | Regressão Logística, Árvore de Decisão, Random Forest, LightGBM e Perceptron Multicamadas (MLP) |
| Métrica principal | AUC no conjunto de teste, com IC 95% e comparação pelo teste de DeLong |

## Como executar

### 1. Preparar o ambiente (apenas na primeira vez)

Requisitos: **Python 3.13** e um servidor **MySQL** em execução.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
```

Copie `.env.example` para `.env` e preencha o acesso ao MySQL. O banco indicado em `DB_NAME` é criado automaticamente se ainda não existir.

### 2. Colocar os dados brutos em `data/raw/`

Os dados **não estão no repositório**, por serem microdados de saúde. São necessários 5 arquivos:

| Arquivo | Conteúdo | Origem |
|---|---|---|
| `fNotificacao.csv` | notificações do SINAN (2018–2022) | exportado do `.pbix` de Santos (2026) |
| `dLocal.csv` | dimensão de municípios | exportado do `.pbix` de Santos (2026) |
| `dTempo.csv` | dimensão de datas (`id_tempo`, `ano_notificacao`) | exportado do `.pbix` de Santos (2026) |
| `censo_total_1991_2010.xlsx` | indicadores municipais | Atlas do Desenvolvimento Humano no Brasil |
| `PacientesTB mais dados muncipais casos maior que 4.xlsx` | cobertura da APS por município | Ministério da Saúde |

O procedimento de obtenção está em [`docs/01_fontes_e_obtencao_dos_dados.md`](docs/01_fontes_e_obtencao_dos_dados.md).

### 3. Executar

```bash
python main.py
```

O `main.py` executa todo o projeto, na ordem da metodologia, e para logo no início se faltar algum arquivo ou a conexão com o banco falhar:

| Etapa | Script | Saída |
|---|---|---|
| 0. Verificações | `main.py` | — |
| 1. Reconstrução da base (ETL) | `src/data/construir_base_dados.py` | tabelas `fnotificacao`, `dlocal` e `dtempo` no MySQL |
| 2a. Fluxograma da população | `src/visualization/fluxograma_populacao_estudo.py` | `reports/fluxograma_populacao_estudo.png` e `.csv` |
| 2b. ABT | `src/features/construir_abt.py` | tabela `tb_abt_tuberculose` e `data/processed/abt_tuberculose.csv` |
| 3. Modelos | `src/models/<modelo>.py` | `models/<modelo>/<cenário>/` |
| 4. Resultados | `src/visualization/gerar_resultados.py` | `reports/resultados/tabelas/` e `reports/resultados/figuras/` |

A execução completa é demorada: o ETL lê uma planilha de cerca de 3 GB, e cada modelo faz validação cruzada em 10 partes. Para repetir só uma etapa, execute o script dela, sempre de dentro da pasta `ia`. Por exemplo:

```bash
python src/models/lightgbm_modelo.py
python src/visualization/gerar_resultados.py
```

## Estrutura do projeto

```
ia/
├── main.py                     executa todo o projeto
├── requirements.txt            dependências com versões fixas
├── .env.example                modelo do arquivo de acesso ao banco
├── data/
│   ├── raw/                    dados brutos (fora do Git)
│   └── processed/              ABT congelada (fora do Git)
├── docs/                       documentação do projeto
├── models/                     resultados de cada modelo e cenário
├── notebooks/                  análises exploratórias
├── references/                 artigos, fontes de dados e ferramentas
├── reports/                    tabelas e figuras usadas no TCC
├── sql/                        consulta SQL da população de estudo (referência)
└── src/
    ├── config.py               caminhos e semente aleatória
    ├── banco_dados.py          conexão com o banco (ConexaoBanco)
    ├── data/                   ETL
    ├── features/               ABT, variáveis e divisão treino/teste
    ├── models/                 um script por modelo, avaliação e registro de resultados
    └── visualization/          fluxograma, tabelas e figuras
```

## Documentação

| Documento | Assunto |
|---|---|
| [`docs/01_fontes_e_obtencao_dos_dados.md`](docs/01_fontes_e_obtencao_dos_dados.md) | fontes e extração dos dados do `.pbix` |
| [`docs/02_reconstrucao_da_base.md`](docs/02_reconstrucao_da_base.md) | ETL: correções e decisões |
| [`docs/03_populacao_de_estudo_e_abt.md`](docs/03_populacao_de_estudo_e_abt.md) | critérios de seleção, desfecho e ABT |
| [`docs/04_dicionario_de_variaveis.md`](docs/04_dicionario_de_variaveis.md) | variáveis, categorias e referências |
| [`docs/05_modelagem_e_avaliacao.md`](docs/05_modelagem_e_avaliacao.md) | protocolo, hiperparâmetros e métricas |
| [`docs/06_saidas.md`](docs/06_saidas.md) | arquivos gerados e como usá-los no TCC |

## Reprodutibilidade e privacidade

- Todos os procedimentos aleatórios usam a semente `SEMENTE = 42` (`src/config.py`).
- As versões das bibliotecas estão fixadas em `requirements.txt`, e cada `models/<modelo>/<cenário>/metricas.json` registra as versões usadas no treino.
- Microdados (`data/`), predições por paciente (`predicoes_teste.csv`), modelos treinados (`modelo.joblib`) e o `.env` **não são versionados**. Do diretório `models/` vão para o Git apenas métricas, importâncias e razões de chances agregadas.
