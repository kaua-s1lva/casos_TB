"""
Treino e avaliação da Árvore de Decisão para prever perda de seguimento
(abandono) no tratamento da tuberculose.

Fluxo:
  1. Lê a ABT (tb_abt_tuberculose) — proporção real de cura x abandono.
  2. Divide por ano: treino = anos anteriores, teste = último ano (ex.: 2018–2021 x 2022).
     O teste não é tocado até a avaliação final.
  3. Pipeline ajustado SÓ no treino: imputação + one-hot encoding + balanceamento + árvore.
  4. Busca de hiperparâmetros com validação cruzada estratificada (10 dobras) no treino,
     otimizando a AUC.
  5. Avaliação no teste (proporção real): AUC, sensibilidade, especificidade, precisão, F1.
  6. Salva modelo, métricas e figuras.

Balanceamento comparado na busca:
  - pesos por classe (class_weight="balanced"), nativo do scikit-learn;
  - subamostragem da classe majoritária (RandomUnderSampler), se o pacote
    imbalanced-learn estiver instalado (pip install imbalanced-learn).

Uso (dentro da pasta ia):  python src/models/treinar_arvore.py   ou   python main.py
"""
import json
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, RocCurveDisplay)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

from src.db_manager import ConexaoBanco
from src.config import MODELS, REPORTS

try:   # subamostragem só se o imbalanced-learn estiver instalado
    from imblearn.pipeline import Pipeline
    from imblearn.under_sampling import RandomUnderSampler
    TEM_IMBLEARN = True
except ImportError:
    from sklearn.pipeline import Pipeline
    TEM_IMBLEARN = False

SEMENTE = 42
ALVO = "alvo_preditivo"
IDENTIFICADORES = ["id_notificacao", "ano_notificacao", "id_municipio_residencia"]  # não são variáveis
CATEGORICAS = ["faixa_etaria", "sexo", "teste_hiv", "alcool", "tabagismo", "drogas",
               "escolaridade", "raca", "faixa_populacao", "regiao"]
NUMERICAS = ["idhm", "indice_gini", "cobertura_aps_2021"]

GRADE = {
    "classificador__max_depth": [3, 4, 5, 6, 8, 10],
    "classificador__min_samples_leaf": [50, 200, 500, 1000],
    "classificador__criterion": ["gini", "entropy"],
}


def carregar_abt():
    db = ConexaoBanco()
    print("[*] Carregando a ABT...")
    df = db.obter_dataframe("SELECT * FROM tb_abt_tuberculose")
    if df is None or df.empty:
        sys.exit("[X] Tabela 'tb_abt_tuberculose' não encontrada ou vazia. Rode process_data.py antes.")
    return df


def dividir_por_ano(df):
    """Teste = último ano disponível; treino = anos anteriores (validação temporal)."""
    ultimo = int(df["ano_notificacao"].max())
    treino, teste = df[df["ano_notificacao"] < ultimo], df[df["ano_notificacao"] == ultimo]
    if treino.empty or teste.empty:
        sys.exit("[X] Não foi possível dividir por ano: confira a coluna ano_notificacao.")
    for nome, parte in [("Treino", treino), ("Teste", teste)]:
        anos = sorted(parte["ano_notificacao"].unique())
        print(f"    {nome}: {len(parte)} casos ({anos[0]}–{anos[-1]}) | "
              f"abandono {parte[ALVO].mean() * 100:.1f}%")
    return treino, teste


def montar_pipeline():
    preparo = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
        # árvore não precisa de normalização; só preenche faltantes com a mediana do treino
        ("num", SimpleImputer(strategy="median"), NUMERICAS),
    ])
    etapas = [("preparo", preparo)]
    if TEM_IMBLEARN:
        etapas.append(("balanceamento", "passthrough"))
    etapas.append(("classificador", DecisionTreeClassifier(random_state=SEMENTE)))
    return Pipeline(etapas)


def grade_balanceamento():
    """Opções de balanceamento comparadas na validação cruzada."""
    opcoes = [{"classificador__class_weight": ["balanced"]}]
    if TEM_IMBLEARN:
        opcoes[0]["balanceamento"] = ["passthrough"]
        opcoes.append({"balanceamento": [RandomUnderSampler(random_state=SEMENTE)],
                       "classificador__class_weight": [None]})
    return [{**GRADE, **o} for o in opcoes]


def nome_balanceamento(params):
    b = params.get("balanceamento", "passthrough")
    return "pesos por classe" if b == "passthrough" else "subamostragem (RandomUnderSampler)"


def avaliar(modelo, X, y):
    prob = modelo.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "auc": roc_auc_score(y, prob),
        "sensibilidade (recall)": recall_score(y, pred),
        "especificidade": tn / (tn + fp),
        "precisao": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred),
        "vp": int(tp), "fp": int(fp), "vn": int(tn), "fn": int(fn),
    }, pred, prob


def salvar_figuras(modelo, X_teste, y_teste, pred):
    REPORTS.mkdir(parents=True, exist_ok=True)

    # matriz de confusão
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_teste, pred),
                           display_labels=["Cura", "Abandono"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Árvore de decisão – matriz de confusão (teste)")
    fig.savefig(REPORTS / "arvore_matriz_confusao.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # curva ROC
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(modelo, X_teste, y_teste, ax=ax, name="Árvore de decisão")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
    ax.set_title("Curva ROC (teste)")
    fig.savefig(REPORTS / "arvore_curva_roc.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # desenho da árvore (só os 3 primeiros níveis, para ficar legível)
    preparo = modelo.named_steps["preparo"]
    nomes = [n.replace("cat__", "").replace("num__", "") for n in preparo.get_feature_names_out()]
    fig = plt.figure(figsize=(36, 12))
    plot_tree(modelo.named_steps["classificador"], feature_names=nomes, class_names=["Cura", "Abandono"],
              filled=True, rounded=True, fontsize=9, max_depth=3, proportion=True)
    plt.title("Árvore de decisão – 3 primeiros níveis")
    fig.savefig(REPORTS / "arvore_decisao_tuberculose.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def importancia_variaveis(modelo, X_teste, y_teste):
    """Importância por permutação no teste: quanto a AUC cai ao embaralhar cada variável
    original (já agrega todas as categorias do one-hot de uma mesma variável)."""
    r = permutation_importance(modelo, X_teste, y_teste, scoring="roc_auc",
                               n_repeats=5, random_state=SEMENTE, n_jobs=-1)
    imp = (pd.DataFrame({"variavel": X_teste.columns, "queda_auc": r.importances_mean,
                         "desvio": r.importances_std})
           .sort_values("queda_auc", ascending=True))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp["variavel"], imp["queda_auc"], xerr=imp["desvio"], color="#2a78d6")
    ax.set_xlabel("Queda média da AUC ao embaralhar a variável")
    ax.set_title("Importância das variáveis (permutação, teste)")
    ax.grid(axis="x", alpha=.3)
    fig.savefig(REPORTS / "arvore_importancia_variaveis.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return imp.sort_values("queda_auc", ascending=False)


def treinar_modelo(df=None):
    df = carregar_abt() if df is None else df
    colunas = CATEGORICAS + NUMERICAS
    treino, teste = dividir_por_ano(df)
    X_treino, y_treino = treino[colunas], treino[ALVO].astype(int)
    X_teste, y_teste = teste[colunas], teste[ALVO].astype(int)

    print(f"[*] Busca de hiperparâmetros (validação cruzada, 10 dobras)"
          f"{'' if TEM_IMBLEARN else ' — imbalanced-learn não instalado: só pesos por classe'}...")
    busca = GridSearchCV(montar_pipeline(), grade_balanceamento(), scoring="roc_auc",
                         cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=SEMENTE),
                         n_jobs=-1, refit=True)
    busca.fit(X_treino, y_treino)
    modelo = busca.best_estimator_
    melhores = {k.replace("classificador__", ""): v for k, v in busca.best_params_.items()
                if k.startswith("classificador__")}
    print(f"    Melhor configuração: {melhores} | balanceamento: {nome_balanceamento(busca.best_params_)}")
    print(f"    AUC média na validação cruzada: {busca.best_score_:.3f}")

    print("[*] Avaliando no conjunto de teste (proporção real)...")
    metricas, pred, _ = avaliar(modelo, X_teste, y_teste)
    for k, v in metricas.items():
        print(f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")
    print("\n" + classification_report(y_teste, pred, target_names=["Cura", "Abandono"], digits=3))

    print("[*] Salvando figuras, importâncias e modelo...")
    salvar_figuras(modelo, X_teste, y_teste, pred)
    imp = importancia_variaveis(modelo, X_teste, y_teste)
    print(imp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, MODELS / "arvore_decisao.joblib")
    imp.to_csv(REPORTS / "arvore_importancia_variaveis.csv", sep=";", decimal=",", index=False)
    with open(REPORTS / "arvore_metricas.json", "w", encoding="utf-8") as f:
        json.dump({"modelo": "Árvore de decisão",
                   "hiperparametros": melhores,
                   "balanceamento": nome_balanceamento(busca.best_params_),
                   "auc_validacao_cruzada": busca.best_score_,
                   "teste": metricas,
                   "n_treino": len(X_treino), "n_teste": len(X_teste)},
                  f, ensure_ascii=False, indent=2, default=float)
    print(f"[+] Modelo em '{MODELS / 'arvore_decisao.joblib'}' e resultados em '{REPORTS}'.")
    return modelo


if __name__ == "__main__":
    treinar_modelo()
