"""
XGBoost (gradient boosting de árvores) para prever a perda de seguimento do tratamento da tuberculose.

Segue o protocolo do Capítulo 3 (Metodologia) do TCC:
  1. treino = notificações de 2018 a 2021 | teste = notificações de 2022;
  2. categóricas: one-hot com "Ignorado" como categoria; numéricas: mediana nos
     faltantes, sem padronização (árvores não são sensíveis à escala) — ajustado só no treino;
  3. busca em grade com validação cruzada estratificada (10 partes), critério AUC:
     taxa de aprendizado, profundidade das árvores e balanceamento
     (pesos por classe x subamostragem). Número de árvores fixo (300) e
     regularização L2 das folhas no padrão do XGBoost (lambda = 1);
  4. avaliação única no teste, com limiar de 0,5;
  5. interpretação: importância por permutação;
  6. comparação: só variáveis individuais x individuais + municipais.

Requer:  pip install imbalanced-learn xgboost
Uso (dentro da pasta ia):  python src/models/xgboost_modelo.py
"""
import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.config import SEMENTE
from src.features.dividir_treino_teste import carregar_treino_teste
from src.features.variaveis import INDIVIDUAIS, MUNICIPAIS_CATEGORICAS, MUNICIPAIS_NUMERICAS
from src.models.avaliacao_desempenho import (calcular_importancia_permutacao, calcular_metricas,
                                             imprimir_titulo)
from src.models.registrar_resultados import salvar_resultados

HIPERPARAMETROS = {
    "modelo__learning_rate": [0.05, 0.1],
    "modelo__max_depth": [4, 6],
}


def grade(y_treino):
    """O XGBoost não tem class_weight: os pesos por classe são dados por scale_pos_weight,
    a razão entre casos de cura e de abandono. Com subamostragem, as classes já chegam
    iguais e o peso volta a 1."""
    razao = (y_treino == 0).sum() / (y_treino == 1).sum()
    return [
        {**HIPERPARAMETROS, "balanceamento": ["passthrough"], "modelo__scale_pos_weight": [razao]},
        {**HIPERPARAMETROS, "balanceamento": [RandomUnderSampler(random_state=SEMENTE)],
         "modelo__scale_pos_weight": [1]},
    ]


def montar_pipeline(categoricas, numericas):
    preparo = ColumnTransformer([
        ("cat", OneHotEncoder(categories=[validas + ["Ignorado"] for validas, _ in categoricas.values()],
                              handle_unknown="ignore"), list(categoricas)),
        ("num", SimpleImputer(strategy="median"), numericas),
    ])
    modelo = XGBClassifier(n_estimators=300, tree_method="hist", eval_metric="logloss",
                           n_jobs=1, random_state=SEMENTE)
    return Pipeline([("preparo", preparo), ("balanceamento", "passthrough"), ("modelo", modelo)])


def treinar(X_treino, y_treino, categoricas, numericas):
    busca = GridSearchCV(montar_pipeline(categoricas, numericas), grade(y_treino), scoring="roc_auc",
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
        print(f"{nome}: taxa = {p['modelo__learning_rate']} | profundidade = {p['modelo__max_depth']} | "
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
        salvar_resultados("xgboost_modelo", nome, busca, X_teste, y_teste,
                          importancia=importancia if nome == "Individuais + municipais" else None,
                          tabelas=None)


if __name__ == "__main__":
    main()
