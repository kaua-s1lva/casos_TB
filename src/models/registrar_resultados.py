"""
Registro dos resultados de cada modelo em ia/models/, para o capítulo de Resultados.

Para cada modelo e cenário, grava a pasta models/<modelo>/<cenário>/ com:
  - metricas.json               hiperparâmetros escolhidos, AUC da validação cruzada e métricas do teste
  - validacao_cruzada.csv       AUC de cada combinação da busca em grade, em cada parte
  - predicoes_teste.csv         desfecho real e probabilidade estimada de cada paciente do teste
  - importancia_permutacao.csv  (quando calculada)
  - <outras tabelas>.csv        tabelas específicas do modelo, ex.: razões de chances
  - modelo.joblib               pipeline treinado (pré-processamento + modelo)

As predições permitem calcular depois, sem retreinar, curvas ROC, calibração,
matrizes de confusão, outros limiares e intervalos de confiança da AUC.

Uso (dentro de um script de modelo):
    salvar_resultados("arvore_decisao", "Individuais + municipais", busca, X_teste, y_teste)
"""
import json
import platform
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.config import MODELS
from src.models.avaliacao_desempenho import LIMIAR, calcular_metricas

PASTAS_CENARIOS = {
    "Só individuais": "individuais",
    "Individuais + municipais": "individuais_municipais",
}


def _para_json(valor):
    """Converte os valores dos hiperparâmetros em algo que o JSON aceite."""
    if isinstance(valor, (str, int, float, bool)) or valor is None:
        return valor
    if isinstance(valor, np.generic):
        return valor.item()
    if isinstance(valor, (tuple, list)):
        return [_para_json(v) for v in valor]
    return type(valor).__name__   # ex.: RandomUnderSampler


def _balanceamento(pipeline):
    etapa = pipeline.named_steps.get("balanceamento")
    return "subamostragem" if hasattr(etapa, "fit_resample") else "pesos por classe"


def salvar_resultados(nome_modelo, cenario, busca, X_teste, y_teste, importancia=None, tabelas=None):
    pasta = MODELS / nome_modelo / PASTAS_CENARIOS[cenario]
    pasta.mkdir(parents=True, exist_ok=True)
    pipeline = busca.best_estimator_
    melhor = busca.best_index_

    resumo = {
        "modelo": nome_modelo,
        "cenario": cenario,
        "data_execucao": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "variaveis": list(X_teste.columns),
        "hiperparametros": {k.replace("modelo__", ""): _para_json(v)
                            for k, v in busca.best_params_.items() if k != "balanceamento"},
        "balanceamento": _balanceamento(pipeline),
        "auc_validacao_cruzada": float(busca.best_score_),
        "auc_validacao_cruzada_desvio": float(busca.cv_results_["std_test_score"][melhor]),
        "limiar": LIMIAR,
        "teste": {k: float(v) for k, v in calcular_metricas(pipeline, X_teste, y_teste).items()},
        "n_teste": int(len(y_teste)),
        "versoes": {"python": platform.python_version(), "scikit-learn": sklearn.__version__},
    }
    with open(pasta / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)

    cv = pd.DataFrame(busca.cv_results_)
    cv = cv[[c for c in cv.columns if c.startswith(("param_", "split", "mean_test", "std_test", "rank"))]]
    cv.astype({c: str for c in cv.columns if c.startswith("param_")}).to_csv(
        pasta / "validacao_cruzada.csv", sep=";", index=False)

    pd.DataFrame({"y_real": y_teste.to_numpy(),
                  "probabilidade": pipeline.predict_proba(X_teste)[:, 1]}).to_csv(
        pasta / "predicoes_teste.csv", sep=";", index=False)

    if importancia is not None:
        importancia.rename_axis("variavel").to_csv(pasta / "importancia_permutacao.csv", sep=";")
    for nome, tabela in (tabelas or {}).items():
        tabela.rename_axis("variavel").to_csv(pasta / f"{nome}.csv", sep=";")

    joblib.dump(pipeline, pasta / "modelo.joblib")
    print(f"[+] Resultados salvos em '{pasta}'.")
