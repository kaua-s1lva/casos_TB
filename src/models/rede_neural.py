"""
Perceptron Multicamadas (MLP) para prever a perda de seguimento do tratamento da tuberculose.

Segue o protocolo do Capítulo 3 (Metodologia) do TCC:
  1. treino = notificações de 2018 a 2021 | teste = notificações de 2022;
  2. categóricas: one-hot com "Ignorado" como categoria; numéricas: mediana nos
     faltantes + padronização (a rede é sensível à escala) — ajustado só no treino;
  3. balanceamento por subamostragem: o MLPClassifier do scikit-learn não aceita
     pesos por classe (class_weight), então essa é a única opção para a rede;
  4. busca em grade com validação cruzada estratificada (10 partes), critério AUC:
     arquitetura (camadas ocultas), taxa de aprendizado e regularização L2 (alpha);
     ReLU nas camadas ocultas, otimizador Adam e parada antecipada (early stopping);
  5. avaliação única no teste, com limiar de 0,5;
  6. interpretação: importância por permutação;
  7. comparação: só variáveis individuais x individuais + municipais.

Requer o pacote imbalanced-learn:  pip install imbalanced-learn
Uso (dentro da pasta ia):  python src/models/rede_neural.py
"""
import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import SEMENTE
from src.features.dividir_treino_teste import carregar_treino_teste
from src.features.variaveis import INDIVIDUAIS, MUNICIPAIS_CATEGORICAS, MUNICIPAIS_NUMERICAS
from src.models.avaliacao_desempenho import (calcular_importancia_permutacao, calcular_metricas,
                                             imprimir_titulo)
from src.models.registrar_resultados import salvar_resultados

GRADE = {
    "modelo__hidden_layer_sizes": [(32,), (64, 32)],   # 1 camada de 32 | 2 camadas (64 e 32)
    "modelo__learning_rate_init": [0.001, 0.01],
    "modelo__alpha": [0.0001, 0.01],                   # intensidade da regularização L2
}


def montar_pipeline(categoricas, numericas):
    preparo = ColumnTransformer([
        ("cat", OneHotEncoder(categories=[validas + ["Ignorado"] for validas, _ in categoricas.values()],
                              handle_unknown="ignore"), list(categoricas)),
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numericas),
    ])
    modelo = MLPClassifier(activation="relu", solver="adam", max_iter=200,
                           early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
                           random_state=SEMENTE)
    return Pipeline([("preparo", preparo),
                     ("balanceamento", RandomUnderSampler(random_state=SEMENTE)),
                     ("modelo", modelo)])


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
        rede = busca.best_estimator_.named_steps["modelo"]
        print(f"{nome}: camadas = {p['modelo__hidden_layer_sizes']} | "
              f"taxa = {p['modelo__learning_rate_init']} | alpha = {p['modelo__alpha']} | "
              f"parou na época {rede.n_iter_} | AUC na validação cruzada = {busca.best_score_:.4f}")
        tabela[nome] = calcular_metricas(busca.best_estimator_, X_teste, y_teste)
    print()
    print(pd.DataFrame(tabela).round(4))

    modelo, X_teste, y_teste = resultados["Individuais + municipais"]
    imprimir_titulo("IMPORTÂNCIA POR PERMUTAÇÃO NO TESTE (queda média da AUC)")
    importancia = calcular_importancia_permutacao(modelo.best_estimator_, X_teste, y_teste)
    print(importancia.round(4).to_string())

    for nome, (busca, X_teste, y_teste) in resultados.items():
        salvar_resultados("rede_neural", nome, busca, X_teste, y_teste,
                          importancia=importancia if nome == "Individuais + municipais" else None,
                          tabelas=None)


if __name__ == "__main__":
    main()
