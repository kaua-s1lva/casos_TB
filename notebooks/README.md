# Notebooks

Análises exploratórias e de apoio. **Não fazem parte do fluxo principal**: tudo o que entra no TCC é gerado pelos scripts de `src/`, executados pelo `main.py`.

| Notebook | Conteúdo | Pré-requisito |
|---|---|---|
| `01_exploracao_abt.ipynb` | desfecho, perda por ano, proporção de "Ignorado", taxa de perda por categoria e indicadores municipais por desfecho | `data/processed/abt_tuberculose.csv` (etapa 2 do `main.py`) |
| `02_comparacao_resultados.ipynb` | métricas de todos os modelos e cenários, ganho de AUC com as variáveis municipais, hiperparâmetros e importância por permutação | `models/` preenchido (etapa 3 do `main.py`) |

## Como usar

Abra os notebooks com o mesmo ambiente virtual do projeto. Para isso, instale o Jupyter ou use a extensão Jupyter do VS Code:

```bash
pip install notebook
jupyter notebook notebooks/
```

Os notebooks partem do princípio de que a pasta de trabalho é `ia/notebooks` e encontram os dados por `Path.cwd().parent`.

## Boas práticas

- **Limpe as saídas antes do commit** (*Clear All Outputs*). Mesmo agregadas, as saídas derivam de microdados de saúde.
- Numere os novos notebooks na ordem em que devem ser lidos (`03_...`).
- Quando uma análise de notebook for usada no TCC, leve-a para um script em `src/`, para que o `main.py` a reproduza.
