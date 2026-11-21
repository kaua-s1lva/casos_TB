"""
Avaliação de desempenho comum a todos os modelos (Seções 2.7 e 2.8 do TCC).
"""
import pandas as pd
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


def imprimir_titulo(texto):
    print("\n" + "=" * 70 + f"\n{texto}\n" + "=" * 70)
