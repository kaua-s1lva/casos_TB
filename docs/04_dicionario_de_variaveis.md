# 04 – Dicionário de variáveis

**Código:** `src/features/variaveis.py`. É a fonte única das variáveis, usada pela ABT, pelos modelos e pelas tabelas.

Todas as variáveis categóricas têm também a categoria **"Ignorado"**, mantida como categoria própria, e não imputada. A **referência** é a categoria omitida no *one-hot* da Regressão Logística; as razões de chances são interpretadas em relação a ela.

## Desfecho

| Variável | Valores |
|---|---|
| `alvo_preditivo` | 1 = perda de seguimento (abandono + abandono primário); 0 = cura |

## Variáveis individuais (SINAN)

| Variável | Categorias | Referência |
|---|---|---|
| `faixa_etaria` | 19–29, 30–39, 40–49, 50–59 anos; acima de 60 anos | acima de 60 anos |
| `sexo` | Feminino, Masculino | Feminino |
| `teste_hiv` | Positivo, Negativo, Em andamento, Não realizado | Negativo |
| `alcool` | Sim, Não | Não |
| `tabagismo` | Sim, Não | Não |
| `drogas` (outras drogas ilícitas) | Sim, Não | Não |
| `escolaridade` | Analfabeto, Até fundamental, Ensino médio, Ensino superior | Ensino superior |
| `raca` | Branca, Preta, Parda, Amarela/Indígena | Branca |

## Variáveis do município de residência

| Variável | Tipo | Categorias / unidade | Fonte |
|---|---|---|---|
| `faixa_populacao` | categórica | 7 faixas, de "até 5 mil" a "mais de 500 mil" habitantes (referência: mais de 500 mil) | Santos (2026) |
| `regiao` | categórica | Norte, Nordeste, Sudeste, Sul, Centro-Oeste (referência: Sudeste) | IBGE |
| `idhm` | numérica | 0 a 1 | Atlas 2010 |
| `indice_gini` | numérica | 0 a 1 | Atlas 2010 |
| `cobertura_aps_2021` | numérica | % da população coberta pela APS (07/2021) | Ministério da Saúde |

## Cenários de modelagem

| Cenário | Variáveis | Pasta em `models/<modelo>/` |
|---|---|---|
| Só individuais | as 8 individuais | `individuais/` |
| Individuais + municipais | as 8 individuais + as 5 municipais | `individuais_municipais/` |
