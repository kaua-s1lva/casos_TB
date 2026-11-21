"""
Floresta aleatória (Random Forest) para prever a perda de seguimento do tratamento da tuberculose.

Segue o protocolo do Capítulo 3 (Metodologia) do TCC:
  1. treino = notificações de 2018 a 2021 | teste = notificações de 2022;
  2. categóricas: one-hot com "Ignorado" como categoria; numéricas: mediana nos
     faltantes, sem padronização (árvores não são sensíveis à escala) — ajustado só no treino;
  3. busca em grade com validação cruzada estratificada (10 partes), critério AUC:
     profundidade máxima, mínimo de exemplos por folha e balanceamento
     (pesos por classe x subamostragem). Número de árvores e variáveis sorteadas
     por divisão ficam fixos (300 árvores, raiz quadrada do nº de variáveis);
  4. avaliação única no teste, com limiar de 0,5;
  5. interpretação: importância por permutação;
  6. comparação: só variáveis individuais x individuais + municipais.

Requer o pacote imbalanced-learn:  pip install imbalanced-learn
Uso (dentro da pasta ia):  python src/models/floresta_aleatoria.py
"""
import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

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
    "modelo__max_depth": [10, 20],
    "modelo__min_samples_leaf": [50, 200],
}


def montar_pipeline(categoricas, numericas):
    preparo = ColumnTransformer([
        ("cat", OneHotEncoder(categories=[validas + ["Ignorado"] for validas, _ in categoricas.values()],
                              handle_unknown="ignore"), list(categoricas)),
        ("num", SimpleImputer(strategy="median"), numericas),
    ])
    modelo = RandomForestClassifier(n_estimators=300, max_features="sqrt", class_weight="balanced",
                                    random_state=SEMENTE)
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
        print(f"{nome}: profundidade = {p['modelo__max_depth']} | "
              f"mín. por folha = {p['modelo__min_samples_leaf']} | "
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
        salvar_resultados("floresta_aleatoria", nome, busca, X_teste, y_teste,
                          importancia=importancia if nome == "Individuais + municipais" else None,
                          tabelas=None)


if __name__ == "__main__":
    main()
