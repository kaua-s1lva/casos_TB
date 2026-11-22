"""
LightGBM (gradient boosting de árvores, crescimento folha a folha) para prever a perda de
seguimento do tratamento da tuberculose. É o algoritmo de boosting usado por Rodrigues et al. (2024).

Segue o protocolo do Capítulo 3 (Metodologia) do TCC:
  1. treino = notificações de 2018 a 2021 | teste = notificações de 2022;
  2. categóricas: one-hot com "Ignorado" como categoria; numéricas: mediana nos
     faltantes, sem padronização (árvores não são sensíveis à escala) — ajustado só no treino;
  3. busca em grade com validação cruzada estratificada (10 partes), critério AUC:
     taxa de aprendizado, número máximo de folhas por árvore e balanceamento
     (pesos por classe x subamostragem). Número de árvores fixo (300);
  4. avaliação única no teste, com limiar de 0,5;
  5. interpretação: importância por permutação;
  6. comparação: só variáveis individuais x individuais + municipais.

Requer:  pip install imbalanced-learn lightgbm
Uso (dentro da pasta ia):  python src/models/lightgbm_modelo.py
"""
import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

from src.config import SEMENTE
from src.features.dividir_treino_teste import carregar_treino_teste
from src.features.variaveis import INDIVIDUAIS, MUNICIPAIS_CATEGORICAS, MUNICIPAIS_NUMERICAS
from src.models.avaliacao_desempenho import (calcular_importancia_permutacao, calcular_metricas,
                                             imprimir_titulo)
from src.models.registrar_resultados import salvar_resultados

# o LightGBM cresce as árvores folha a folha: a complexidade é controlada pelo número
# máximo de folhas (num_leaves), e não pela profundidade
HIPERPARAMETROS = {
    "modelo__learning_rate": [0.05, 0.1],
    "modelo__num_leaves": [15, 31, 63],
}

# pesos por classe só sem subamostragem; com subamostragem as classes já chegam iguais
GRADE = [
    {**HIPERPARAMETROS, "balanceamento": ["passthrough"], "modelo__class_weight": ["balanced"]},
    {**HIPERPARAMETROS, "balanceamento": [RandomUnderSampler(random_state=SEMENTE)],
     "modelo__class_weight": [None]},
]


def montar_pipeline(categoricas, numericas):
    preparo = ColumnTransformer([
        ("cat", OneHotEncoder(categories=[validas + ["Ignorado"] for validas, _ in categoricas.values()],
                              handle_unknown="ignore"), list(categoricas)),
        ("num", SimpleImputer(strategy="median"), numericas),
    ])
    modelo = LGBMClassifier(n_estimators=300, n_jobs=1, random_state=SEMENTE, verbose=-1)
    return Pipeline([("preparo", preparo), ("balanceamento", "passthrough"), ("modelo", modelo)])


def treinar(X_treino, y_treino, categoricas, numericas):
    busca = GridSearchCV(montar_pipeline(categoricas, numericas), GRADE, scoring="roc_auc",
                         cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=SEMENTE),
                         n_jobs=-1)
    busca.fit(X_treino, y_treino)
    return busca


def main():
    cenarios = {
        "Só individuais": (INDIVIDUAIS, []),
        "Individuais + municipais": ({**INDIVIDUAIS, **MUNICIPAIS_CATEGORICAS}, MUNICIPAIS_NUMERICAS),
    }
    resultados = {}
    for nome, (categoricas, numericas) in cenarios.items():
        X_treino, X_teste, y_treino, y_teste = carregar_treino_teste(list(categoricas) + numericas)
        print(f"Treinando: {nome} (treino: {len(X_treino)} casos | teste: {len(X_teste)} casos)...")
        resultados[nome] = (treinar(X_treino, y_treino, categoricas, numericas), X_teste, y_teste)

    # ---------------------------------------------------------------- resultados
    pd.set_option("display.width", 120)
    imprimir_titulo("DESEMPENHO NO TESTE (limiar = 0,5)")
    tabela = {}
    for nome, (busca, X_teste, y_teste) in resultados.items():
        p = busca.best_params_
        print(f"{nome}: taxa = {p['modelo__learning_rate']} | folhas = {p['modelo__num_leaves']} | "
              f"{'pesos por classe' if p['balanceamento'] == 'passthrough' else 'subamostragem'} | "
              f"AUC na validação cruzada = {busca.best_score_:.4f}")
        tabela[nome] = calcular_metricas(busca.best_estimator_, X_teste, y_teste)
    print()
    print(pd.DataFrame(tabela).round(4))

    modelo, X_teste, y_teste = resultados["Individuais + municipais"]
    imprimir_titulo("IMPORTÂNCIA POR PERMUTAÇÃO NO TESTE (queda média da AUC)")
    importancia = calcular_importancia_permutacao(modelo.best_estimator_, X_teste, y_teste)
    print(importancia.round(4).to_string())

    for nome, (busca, X_teste, y_teste) in resultados.items():
        salvar_resultados("lightgbm_modelo", nome, busca, X_teste, y_teste,
                          importancia=importancia if nome == "Individuais + municipais" else None,
                          tabelas=None)


if __name__ == "__main__":
    main()
