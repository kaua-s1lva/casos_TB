"""
Avaliação de desempenho comum a todos os modelos (Seções 2.7 e 2.8 do TCC).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, roc_auc_score

from src.config import SEMENTE

LIMIAR = 0.5


def calcular_metricas(modelo, X, y, limiar=LIMIAR):
    """Métricas no conjunto informado, com o limiar de decisão escolhido."""
    prob = modelo.predict_proba(X)[:, 1]
    vn, fp, fn, vp = confusion_matrix(y, prob >= limiar).ravel()
    sensibilidade, precisao = vp / (vp + fn), vp / (vp + fp)
    return {
        "AUC": roc_auc_score(y, prob),
        "Sensibilidade": sensibilidade,
        "Especificidade": vn / (vn + fp),
        "Precisão": precisao,
        "F1-Score": 2 * precisao * sensibilidade / (precisao + sensibilidade),
        "Acurácia": (vp + vn) / len(y),
    }


def calcular_importancia_permutacao(modelo, X, y):
    """Queda média da AUC ao embaralhar cada variável (5 repetições)."""
    r = permutation_importance(modelo, X, y, scoring="roc_auc", n_repeats=5,
                               random_state=SEMENTE, n_jobs=-1)
    return pd.Series(r.importances_mean, index=X.columns,
                     name="queda da AUC").sort_values(ascending=False)


def delong(y, prob_a, prob_b=None):
    """Teste de DeLong (DeLong et al., 1988; algoritmo rápido de Sun e Xu, 2014).

    Com um modelo: devolve AUC e IC 95%.
    Com dois modelos avaliados nos MESMOS pacientes: devolve também a diferença
    das AUCs (a - b), seu IC 95% e o p-valor do teste de igualdade.
    """
    y = np.asarray(y)
    probs = np.vstack([prob_a] if prob_b is None else [prob_a, prob_b]).astype(float)
    pos, neg = probs[:, y == 1], probs[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    r_pos = np.array([rankdata(x) for x in pos])
    r_neg = np.array([rankdata(x) for x in neg])
    r_tot = np.array([rankdata(x) for x in np.hstack([pos, neg])])
    auc = r_tot[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v10 = (r_tot[:, :m] - r_pos) / n          # componentes dos positivos
    v01 = 1 - (r_tot[:, m:] - r_neg) / m      # componentes dos negativos
    cov = np.atleast_2d(np.cov(v10)) / m + np.atleast_2d(np.cov(v01)) / n
    ep = np.sqrt(np.diag(cov))
    resultado = {"auc": auc[0], "ic_inf": auc[0] - 1.96 * ep[0], "ic_sup": auc[0] + 1.96 * ep[0]}
    if prob_b is not None:
        dif = auc[0] - auc[1]
        ep_dif = np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
        resultado.update({"auc_b": auc[1], "diferenca": dif, "dif_ic_inf": dif - 1.96 * ep_dif,
                          "dif_ic_sup": dif + 1.96 * ep_dif, "p_valor": 2 * norm.sf(abs(dif / ep_dif))})
    return resultado


def imprimir_titulo(texto):
    print("\n" + "=" * 70 + f"\n{texto}\n" + "=" * 70)
