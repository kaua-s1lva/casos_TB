"""
Regressão logística para prever a perda de seguimento do tratamento da tuberculose.

Segue o protocolo do Capítulo 3 (Metodologia) do TCC:
  1. treino = notificações de 2018 a 2021 | teste = notificações de 2022;
  2. categóricas: one-hot com "Ignorado" como categoria e a categoria de referência removida;
     numéricas: mediana (faltantes) + padronização — tudo ajustado só no treino;
  3. busca em grade com validação cruzada estratificada (10 partes), critério AUC:
     regularização L1/L2, intensidade C e balanceamento (pesos por classe x subamostragem);
  4. avaliação única no teste, com limiar de 0,5;
  5. interpretação: razões de chances e importância por permutação;
  6. comparação: só variáveis individuais x individuais + municipais.

Requer o pacote imbalanced-learn:  pip install imbalanced-learn
Uso (dentro da pasta ia):  python src/models/regressao_logistica.py
"""
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import SEMENTE
from src.features.dividir_treino_teste import carregar_treino_teste
from src.features.variaveis import INDIVIDUAIS, MUNICIPAIS_CATEGORICAS, MUNICIPAIS_NUMERICAS
from src.models.avaliacao_desempenho import (calcular_importancia_permutacao, calcular_metricas,
                                             imprimir_titulo)
from src.models.registrar_resultados import salvar_resultados

# class_weight="balanced" fica sempre ligado: sem subamostragem ele equilibra as classes
# pelos pesos; com subamostragem as classes já chegam iguais e os pesos viram 1.
GRADE = {
    "balanceamento": ["passthrough", RandomUnderSampler(random_state=SEMENTE)],
    "modelo__C": [0.001, 0.01, 0.1, 1, 10],
    "modelo__l1_ratio": [0, 1],   # 0 = L2 (Ridge) | 1 = L1 (Lasso)
}


def montar_pipeline(categoricas, numericas):
    preparo = ColumnTransformer([
        ("cat", OneHotEncoder(categories=[validas + ["Ignorado"] for validas, _ in categoricas.values()],
                              drop=[referencia for _, referencia in categoricas.values()],
                              handle_unknown="ignore"), list(categoricas)),
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numericas),
    ])
    modelo = LogisticRegression(solver="liblinear", class_weight="balanced",
                                max_iter=2000, random_state=SEMENTE)
    return Pipeline([("preparo", preparo), ("balanceamento", "passthrough"), ("modelo", modelo)])


def treinar(X_treino, y_treino, categoricas, numericas):
    busca = GridSearchCV(montar_pipeline(categoricas, numericas), GRADE, scoring="roc_auc",
                         cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=SEMENTE),
                         n_jobs=-1)
    busca.fit(X_treino, y_treino)
    return busca


def razoes_de_chances(modelo):
    nomes = [n.split("__", 1)[1] for n in modelo.named_steps["preparo"].get_feature_names_out()]
    coeficientes = modelo.named_steps["modelo"].coef_[0]
    return pd.Series(np.exp(coeficientes), index=nomes, name="OR").sort_values(ascending=False)


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
        print(f"{nome}: C = {p['modelo__C']} | "
              f"{'L1 (Lasso)' if p['modelo__l1_ratio'] == 1 else 'L2 (Ridge)'} | "
              f"{'pesos por classe' if p['balanceamento'] == 'passthrough' else 'subamostragem'} | "
              f"AUC na validação cruzada = {busca.best_score_:.4f}")
        tabela[nome] = calcular_metricas(busca.best_estimator_, X_teste, y_teste)
    print()
    print(pd.DataFrame(tabela).round(4))

    modelo, X_teste, y_teste = resultados["Individuais + municipais"]
    modelo = modelo.best_estimator_
    imprimir_titulo("RAZÕES DE CHANCES (modelo com variáveis individuais + municipais)")
    print(razoes_de_chances(modelo).round(3).to_string())

    imprimir_titulo("IMPORTÂNCIA POR PERMUTAÇÃO NO TESTE (queda média da AUC)")
    importancia = calcular_importancia_permutacao(modelo, X_teste, y_teste)
    print(importancia.round(4).to_string())

    for nome, (busca, X_teste, y_teste) in resultados.items():
        salvar_resultados("regressao_logistica", nome, busca, X_teste, y_teste,
                          importancia=importancia if nome == "Individuais + municipais" else None,
                          tabelas={"razoes_de_chances": razoes_de_chances(busca.best_estimator_).to_frame()})


if __name__ == "__main__":
    main()
