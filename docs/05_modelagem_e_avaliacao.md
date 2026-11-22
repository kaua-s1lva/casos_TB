# 05 – Modelagem e avaliação

**Código:** um script por modelo em `src/models/`, todos com a mesma estrutura: `montar_pipeline` → `treinar` (busca em grade) → `main`, que avalia, imprime e salva os resultados.

| Modelo | Script |
|---|---|
| Regressão Logística | `regressao_logistica.py` |
| Árvore de Decisão | `arvore_decisao.py` |
| Random Forest | `floresta_aleatoria.py` |
| LightGBM | `lightgbm_modelo.py` |
| Perceptron Multicamadas (MLP) | `rede_neural.py` |

Funções compartilhadas: `avaliacao_desempenho.py` (métricas, importância por permutação e teste de DeLong) e `registrar_resultados.py` (gravação em `models/`).

## Pipeline

Cada modelo é um `imblearn.pipeline.Pipeline`. O preparo é ajustado **só no treino**, e dentro de cada parte da validação cruzada, para não vazar informação.

1. **Categóricas:** *one-hot* com "Ignorado" como categoria. Na Regressão Logística, a categoria de referência é removida.
2. **Numéricas:** imputação pela mediana. Na Regressão Logística e na MLP também há padronização (`StandardScaler`); os modelos de árvore não precisam dela.
3. **Balanceamento:** a busca em grade compara duas estratégias.
   - **Pesos por classe:** `class_weight="balanced"`.
   - **Subamostragem aleatória** da classe majoritária (`RandomUnderSampler`).

   A MLP usa só subamostragem, porque o `MLPClassifier` não aceita pesos por classe.

## Busca de hiperparâmetros

`GridSearchCV` com validação cruzada estratificada de 10 partes (`StratifiedKFold`, `shuffle=True`, semente 42), escolhendo pela AUC.

| Modelo | Hiperparâmetros avaliados |
|---|---|
| Regressão Logística | regularização L1 ou L2; C ∈ {0,001; 0,01; 0,1; 1; 10} |
| Árvore de Decisão | critério Gini ou entropia; profundidade ∈ {3, 4, 5, 6, 8, 10}; mínimo por folha ∈ {50, 200, 500, 1.000} |
| Random Forest | 300 árvores, √p variáveis por divisão; profundidade ∈ {10, 20}; mínimo por folha ∈ {50, 200} |
| LightGBM | 300 árvores; taxa de aprendizado ∈ {0,05; 0,1}; máximo de folhas ∈ {15, 31, 63} |
| MLP | camadas ocultas (32) ou (64, 32); taxa inicial ∈ {0,001; 0,01}; α ∈ {0,0001; 0,01}; parada antecipada |

## Avaliação

O melhor modelo da busca é retreinado com todo o treino e avaliado **uma única vez** no teste (2022).

- **Métricas com limiar 0,5:** AUC (métrica principal), sensibilidade, especificidade, precisão, F1-Score e acurácia.
- **Feitas em `gerar_resultados.py`:**
  - intervalo de confiança de 95% da AUC e comparação entre modelos e entre cenários pelo **teste de DeLong** (algoritmo rápido de Sun e Xu);
  - **escore de Brier**;
  - curvas de calibração;
  - limiar ótimo pelo **índice de Youden**, apenas como análise.
- **Interpretação:**
  - **importância por permutação** no teste: queda média da AUC em 5 permutações de cada variável, para todos os modelos;
  - **razões de chances** (`exp(coef)`) na Regressão Logística.
