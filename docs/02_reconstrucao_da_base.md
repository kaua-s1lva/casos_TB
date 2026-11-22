# 02 – Reconstrução da base de dados (ETL)

**Código:** `src/data/construir_base_dados.py` · **Execução:** etapa 1 do `main.py`

O ETL grava três tabelas no MySQL e substitui as existentes a cada execução:

| Tabela | Origem | Linhas |
|---|---|---|
| `dtempo` | `dTempo.csv` | uma por data de notificação (2018–2022) |
| `dlocal` | `dLocal.csv` + Atlas 2010 + cobertura da APS | 3.284 municípios |
| `fnotificacao` | `fNotificacao.csv` | 301.959 notificações |

## Problemas encontrados na auditoria e correções

A análise exploratória das tabelas exportadas do `.pbix` encontrou inconsistências na dimensão geográfica. Cada correção é feita por uma função do script:

| Problema | Correção | Função |
|---|---|---|
| Indicadores socioeconômicos do Censo de 1991, e não do de 2010, e valores com erro de casas decimais | os indicadores são relidos do Atlas, filtrando 2010 | `ler_indicadores_atlas` |
| Municípios instalados depois de 2010 com código IBGE de 6 dígitos e sem dados cadastrais | o código de 7 dígitos e os atributos são completados a partir da linha do mesmo município | `corrigir_codigos_ibge` |
| O mesmo município repetido com `codigo_uf` diferente: o ETL original combinava a UF de notificação com o município de residência | fica uma linha por município (a de UF correta); as chaves da fato são redirecionadas | `remover_municipios_duplicados` |
| `cod_uf` preenchido com "Brasil" | recalculado a partir dos 2 primeiros dígitos do código do município | `ler_dimensao_local_original` |
| Notificações repetidas, idênticas exceto pelo `id_notificacao` | removidas | `carregar_notificacoes` |

## Preenchimentos manuais

São poucos e ficam explícitos no código (constantes no topo das funções):

- **Atlas para municípios criados depois de 2010** (`ATLAS_MANUAL`): Balneário Rincão (SC) e Mojuí dos Campos (PA) recebem os valores de 2010 dos municípios de origem, Içara (SC) e Santarém (PA).
- **Cobertura da APS ausente** (`COBERTURA_APS_MANUAL`): Ibiá (MG) recebe 93,16%, consultado no e-Gestor para a mesma competência (07/2021). O valor só é aplicado quando o dado falta na planilha.

## Leitura da planilha de cobertura

A planilha de cobertura tem cerca de 3 GB descompactada. Em vez de abri-la inteira com o pandas, o ETL lê o XML interno do `.xlsx` em blocos, extraindo apenas as três colunas necessárias (`ler_cobertura_aps`). Essa etapa leva alguns minutos.

## Faixas derivadas

A `dlocal` também recebe faixas de IDHM, pobreza, renda per capita e Gini, mantidas por compatibilidade com o painel original. Os modelos usam os indicadores numéricos, não essas faixas.
