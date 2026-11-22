# Documentação

Documentação técnica do projeto, na ordem do fluxo de dados. A descrição formal e as justificativas estão no Capítulo 3 (Metodologia) do TCC; aqui estão os detalhes práticos de cada etapa e as decisões tomadas no código.

| Documento | Etapa | Código |
|---|---|---|
| [01 – Fontes e obtenção dos dados](01_fontes_e_obtencao_dos_dados.md) | origem dos arquivos de `data/raw/` | — |
| [02 – Reconstrução da base de dados](02_reconstrucao_da_base.md) | ETL para o MySQL | `src/data/construir_base_dados.py` |
| [03 – População de estudo e ABT](03_populacao_de_estudo_e_abt.md) | seleção dos casos e tabela analítica | `src/features/construir_abt.py`, `src/visualization/fluxograma_populacao_estudo.py` |
| [04 – Dicionário de variáveis](04_dicionario_de_variaveis.md) | desfecho e preditoras | `src/features/variaveis.py` |
| [05 – Modelagem e avaliação](05_modelagem_e_avaliacao.md) | treino, validação e métricas | `src/models/` |
| [06 – Saídas](06_saidas.md) | arquivos gerados | `models/`, `reports/` |

A pasta [`legado/`](legado/) guarda o script de carga original do trabalho de Santos (2026), usado apenas como referência na auditoria. Ele **não** faz parte do fluxo atual.
