"""
Treino e avaliação da Regressão Logística para prever perda de seguimento
(abandono) no tratamento da tuberculose.

Segue o mesmo protocolo de treinar_arvore.py, para que os modelos sejam comparáveis:
  1. Lê a ABT (tb_abt_tuberculose) — proporção real de cura x abandono.
  2. Divide por ano: treino = anos anteriores, teste = último ano (ex.: 2018–2021 x 2022).
  3. Pipeline ajustado SÓ no treino:
       categóricas -> one-hot com categoria de referência fixa (para ler odds ratios)
       numéricas   -> imputação pela mediana + padronização (z-score)
       balanceamento -> pesos por classe (ou subamostragem, se imbalanced-learn instalado)
  4. Busca de hiperparâmetros (força da regularização C e tipo L1/L2) com validação cruzada
     estratificada de 10 dobras no treino, otimizando a AUC.
  5. Avaliação no teste (proporção real): AUC, sensibilidade, especificidade, precisão, F1.
  6. Salva modelo, métricas, odds ratios e figuras.

Sobre as odds ratios (OR = exp(coeficiente)):
  - categóricas: OR da categoria em relação à categoria de referência (REFERENCIAS);
  - numéricas: OR para o aumento de 1 desvio-padrão do indicador.
  O modelo é regularizado e treinado com pesos por classe, então as OR servem para
  comparar o peso relativo das variáveis neste modelo preditivo; para OR com intervalo
  de confiança e p-valor (análise inferencial), use uma regressão logística sem
  regularização (ex.: statsmodels).

Uso (dentro da pasta ia):  python src/models/treinar_regressao_logistica.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
IGNORADO = "Ignorado"

# Categorias de cada variável (as mesmas validadas em process_data.py) e a categoria
# de REFERÊNCIA, contra a qual as odds ratios das demais são calculadas.
CATEGORIAS = {
    "faixa_etaria": ["19 - 29 anos", "30 - 39 anos", "40 - 49 anos", "50 - 59 anos", "Acima de 60 anos"],
    "sexo": ["Feminino", "Masculino"],
    "teste_hiv": ["Positivo", "Negativo", "Em andamento", "Não realizado"],
    "alcool": ["Sim", "Não"],
    "tabagismo": ["Sim", "Não"],
    "drogas": ["Sim", "Não"],
    "escolaridade": ["Analfabeto", "Até fundamental", "Ensino médio", "Ensino superior"],
    "raca": ["Branca", "Preta", "Parda", "Amarela/Indígena"],
    "faixa_populacao": ["1 - Até 5000", "2 - 5001 até 10000", "3 - 10001 até 20000",
                        "4 - 20001 até 50000", "5 - 50001 até 100000",
                        "6 - 100001 até 500000", "7 - Maior que 500000"],
    "regiao": ["1 - Norte", "2 - Nordeste", "3 - Sudeste", "4 - Sul", "5 - Centro-Oeste"],
}
REFERENCIAS = {
    "faixa_etaria": "Acima de 60 anos",
    "sexo": "Feminino",
    "teste_hiv": "Negativo",
    "alcool": "Não",
    "tabagismo": "Não",
    "drogas": "Não",
    "escolaridade": "Ensino superior",
    "raca": "Branca",
    "faixa_populacao": "7 - Maior que 500000",
    "regiao": "3 - Sudeste",
}
CATEGORICAS = list(CATEGORIAS)
NUMERICAS = ["idhm", "indice_gini", "cobertura_aps_2021"]

GRADE = {
    "classificador__C": [0.001, 0.01, 0.1, 1, 10],
    "classificador__l1_ratio": [0, 1],   # 0 = regularização L2 (Ridge) | 1 = L1 (Lasso)
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
    categorias = [CATEGORIAS[c] + [IGNORADO] for c in CATEGORICAS]
    referencias = [REFERENCIAS[c] for c in CATEGORICAS]
    preparo = ColumnTransformer([
        # categorias fixas + referência removida: evita colinearidade e permite ler as OR
        ("cat", OneHotEncoder(categories=categorias, drop=referencias, handle_unknown="ignore"),
         CATEGORICAS),
        # regressão é sensível à escala: imputa a mediana e padroniza (média 0, desvio 1)
        ("num", Pipeline([("imputacao", SimpleImputer(strategy="median")),
                          ("padronizacao", StandardScaler())]), NUMERICAS),
    ])
    etapas = [("preparo", preparo)]
    if TEM_IMBLEARN:
        etapas.append(("balanceamento", "passthrough"))
    etapas.append(("classificador", LogisticRegression(solver="liblinear", max_iter=2000,
                                                       random_state=SEMENTE)))
    return Pipeline(etapas)


def grade_balanceamento():
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
    }, pred


def odds_ratios(modelo):
    """Tabela de OR por categoria (ou por 1 desvio-padrão, nas numéricas)."""
    nomes = modelo.named_steps["preparo"].get_feature_names_out()
    coef = modelo.named_steps["classificador"].coef_[0]
    linhas = []
    for nome, b in zip(nomes, coef):
        tipo, resto = nome.split("__", 1)
        if tipo == "cat":
            variavel = next(c for c in CATEGORICAS if resto.startswith(c + "_"))
            categoria = resto[len(variavel) + 1:]
            comparacao = f"{categoria} vs {REFERENCIAS[variavel]}"
        else:
            variavel, comparacao = resto, "+1 desvio-padrão"
        linhas.append({"variavel": variavel, "comparacao": comparacao,
                       "coeficiente": b, "odds_ratio": float(np.exp(b))})
    return pd.DataFrame(linhas)


def salvar_figuras(modelo, X_teste, y_teste, pred, ors):
    REPORTS.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_teste, pred),
                           display_labels=["Cura", "Abandono"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Regressão logística – matriz de confusão (teste)")
    fig.savefig(REPORTS / "logistica_matriz_confusao.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(modelo, X_teste, y_teste, ax=ax, name="Regressão logística")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
    ax.set_title("Curva ROC (teste)")
    fig.savefig(REPORTS / "logistica_curva_roc.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # odds ratios (escala log): à direita de 1 aumenta a chance de abandono
    o = ors[ors["variavel"].ne("")].copy()
    o = o[~o["comparacao"].str.startswith(IGNORADO)]          # "Ignorado" fica só na tabela
    o["rotulo"] = o["variavel"] + ": " + o["comparacao"]
    o = o.sort_values("odds_ratio")
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(o) + 1.5))
    cores = ["#eb6834" if v > 1 else "#2a78d6" for v in o["odds_ratio"]]
    ax.barh(o["rotulo"], o["odds_ratio"] - 1, left=1, color=cores)
    ax.axvline(1, color="black", lw=1)
    ax.set_xscale("log")
    marcas = [m for m in [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6]
              if o["odds_ratio"].min() / 1.2 <= m <= o["odds_ratio"].max() * 1.2]
    ax.set_xticks(marcas)
    ax.set_xticklabels([f"{m:g}".replace(".", ",") for m in marcas])
    ax.minorticks_off()
    ax.set_xlabel("Odds ratio (escala log) — à direita de 1: mais chance de abandono")
    ax.set_title("Regressão logística – odds ratios")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", alpha=.3)
    fig.savefig(REPORTS / "logistica_odds_ratios.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def importancia_variaveis(modelo, X_teste, y_teste):
    """Mesma métrica da árvore: queda da AUC ao embaralhar cada variável original."""
    r = permutation_importance(modelo, X_teste, y_teste, scoring="roc_auc",
                               n_repeats=5, random_state=SEMENTE, n_jobs=-1)
    imp = (pd.DataFrame({"variavel": X_teste.columns, "queda_auc": r.importances_mean,
                         "desvio": r.importances_std}).sort_values("queda_auc"))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp["variavel"], imp["queda_auc"], xerr=imp["desvio"], color="#2a78d6")
    ax.set_xlabel("Queda média da AUC ao embaralhar a variável")
    ax.set_title("Importância das variáveis (permutação, teste)")
    ax.grid(axis="x", alpha=.3)
    fig.savefig(REPORTS / "logistica_importancia_variaveis.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return imp.sort_values("queda_auc", ascending=False)


def treinar_regressao_logistica(df=None):
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
    melhores["regularizacao"] = "L1 (Lasso)" if melhores.pop("l1_ratio") == 1 else "L2 (Ridge)"
    print(f"    Melhor configuração: {melhores} | balanceamento: {nome_balanceamento(busca.best_params_)}")
    print(f"    AUC média na validação cruzada: {busca.best_score_:.3f}")

    print("[*] Avaliando no conjunto de teste (proporção real)...")
    metricas, pred = avaliar(modelo, X_teste, y_teste)
    for k, v in metricas.items():
        print(f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")
    print("\n" + classification_report(y_teste, pred, target_names=["Cura", "Abandono"], digits=3))

    print("[*] Odds ratios (maiores efeitos):")
    ors = odds_ratios(modelo)
    destaque = ors.reindex(ors["coeficiente"].abs().sort_values(ascending=False).index).head(12)
    print(destaque[["variavel", "comparacao", "odds_ratio"]].to_string(
        index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n[*] Salvando figuras, importâncias e modelo...")
    salvar_figuras(modelo, X_teste, y_teste, pred, ors)
    imp = importancia_variaveis(modelo, X_teste, y_teste)
    print(imp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, MODELS / "regressao_logistica.joblib")
    ors.to_csv(REPORTS / "logistica_odds_ratios.csv", sep=";", decimal=",", index=False)
    imp.to_csv(REPORTS / "logistica_importancia_variaveis.csv", sep=";", decimal=",", index=False)
    with open(REPORTS / "logistica_metricas.json", "w", encoding="utf-8") as f:
        json.dump({"modelo": "Regressão logística",
                   "hiperparametros": melhores,
                   "balanceamento": nome_balanceamento(busca.best_params_),
                   "auc_validacao_cruzada": busca.best_score_,
                   "teste": metricas,
                   "n_treino": len(X_treino), "n_teste": len(X_teste)},
                  f, ensure_ascii=False, indent=2, default=float)
    print(f"[+] Modelo em '{MODELS / 'regressao_logistica.joblib'}' e resultados em '{REPORTS}'.")
    return modelo


if __name__ == "__main__":
    treinar_regressao_logistica()
