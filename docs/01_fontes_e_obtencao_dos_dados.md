# 01 – Fontes e obtenção dos dados

## Fontes

| Fonte | Conteúdo usado | Referência |
|---|---|---|
| SINAN – tuberculose (via Santos, 2026) | notificações de 2018 a 2022: características individuais, clínicas e situação de encerramento | Santos (2026) |
| Atlas do Desenvolvimento Humano no Brasil | IDHM, índice de Gini, renda per capita, pobreza e esperança de vida, Censo 2010 | PNUD, Ipea e FJP (2013) |
| Cobertura da Atenção Primária à Saúde | percentual de cobertura da APS por município, competência 07/2021 | Ministério da Saúde (e-Gestor) |

## Por que os dados vêm de um arquivo `.pbix`

As notificações vêm do *Data Warehouse* construído por Santos (2026). O ETL original dele (cópia em [`legado/`](legado/)) depende de arquivos intermediários que não estavam mais disponíveis no formato esperado, então não pôde ser reexecutado. As tabelas já tratadas continuavam disponíveis no modelo de dados do painel Power BI (`.pbix`) entregue pelo autor.

## Procedimento de extração (DAX Studio)

1. Abrir o arquivo `.pbix` no **Power BI Desktop** e deixá-lo aberto.
2. Abrir o **DAX Studio** e conectar ao modelo do Power BI aberto (*PBI / SSDT Model*).
3. Em *Advanced → Export Data*, escolher o formato **CSV** e selecionar as tabelas:
   - `fNotificacao` (fato de notificações)
   - `dLocal` (dimensão de municípios)
   - `dTempo` (dimensão de datas de notificação)
4. Salvar os arquivos em `data/raw/` com os nomes `fNotificacao.csv`, `dLocal.csv` e `dTempo.csv`.

O ETL lê `fNotificacao.csv` e `dLocal.csv` com separador `;`. Em `dTempo.csv` o separador é detectado automaticamente, mas o arquivo precisa ter as colunas `id_tempo` e `ano_notificacao`.

## Arquivos de indicadores municipais

| Arquivo em `data/raw/` | Uso |
|---|---|
| `censo_total_1991_2010.xlsx` | aba `MUN 91-00-10` do Atlas; o ETL filtra o ano de 2010 |
| `PacientesTB mais dados muncipais casos maior que 4.xlsx` | planilha do SINAN-TB com indicadores municipais; o ETL lê apenas `codigo_ibge`, `pc_cobertura_aps_2021` e `pc_cobertura_sf_2019` |

## Ética e privacidade

Os dados são públicos e não identificam os pacientes. A pesquisa enquadra-se nas hipóteses de dispensa de apreciação por Comitê de Ética previstas na Resolução CNS nº 510/2016. Mesmo assim, os microdados **não** são versionados no Git (ver `.gitignore`).
