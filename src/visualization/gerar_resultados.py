"""
Figuras e tabelas do capítulo de Resultados, a partir dos arquivos em ia/models/.

Não treina nada: lê as métricas, predições, importâncias e razões de chances
salvas por cada script de modelo (src/models/registrar_resultados.py).

Saídas em reports/resultados/:
  tabelas/  tabela_desempenho_modelos      AUC (IC 95%), métricas no limiar 0,5 e Brier
            tabela_comparacao_cenarios     AUC sem x com variáveis municipais (teste de DeLong)
            tabela_hiperparametros         configuração escolhida na validação cruzada
  figuras/  fig_curvas_roc                 curvas ROC dos modelos
            fig_comparacao_cenarios        AUC com IC 95% nos dois cenários
            fig_razoes_de_chances          razões de chances da regressão logística
            fig_importancia_permutacao     importância das variáveis em cada modelo
            fig_calibracao                 curvas de calibração
            fig_limiar_melhor_modelo       sensibilidade e especificidade x limiar

Uso (dentro da pasta ia):  python src/visualization/gerar_resultados.py
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from src.config import MODELS
from src.models.avaliacao_desempenho import LIMIAR, delong
from src.visualization.estilo import (CATEGORIAS, CENARIOS, CINZA, CORES, MODELOS, TRACOS, VARIAVEIS,
                                      num, salvar_figura, salvar_tabela)
from src.visualization.tabela_caracteristicas_populacao import gerar_tabela_caracteristicas_populacao

COMPLETO = "individuais_municipais"


# ------------------------------------------------------------------ leitura
def ler_resultados():
    """{(modelo, cenário): {"metricas": dict, "pred": DataFrame, "pasta": Path}}"""
    res = {}
    for modelo in MODELOS:
        for cenario in CENARIOS:
            pasta = MODELS / modelo / cenario
            if not (pasta / "metricas.json").exists():
                print(f"[!] Sem resultados para {modelo}/{cenario} — rode o script do modelo.")
                continue
            res[(modelo, cenario)] = {
                "metricas": json.loads((pasta / "metricas.json").read_text(encoding="utf-8")),
                "pred": pd.read_csv(pasta / "predicoes_teste.csv", sep=";"),
                "pasta": pasta,
            }
    return res


def p_texto(p):
    return "< 0,001" if p < 0.001 else num(p, 3)


def ic_texto(r):
    return f"{num(r['auc'])} ({num(r['ic_inf'])}–{num(r['ic_sup'])})"


def melhor_modelo(res):
    return max((m for m in MODELOS if (m, COMPLETO) in res),
               key=lambda m: res[(m, COMPLETO)]["metricas"]["teste"]["AUC"])


# ------------------------------------------------------------------ tabelas
def tabela_desempenho(res):
    melhor = melhor_modelo(res)
    ref = res[(melhor, COMPLETO)]["pred"]
    linhas = []
    for modelo in MODELOS:
        if (modelo, COMPLETO) not in res:
            continue
        r = res[(modelo, COMPLETO)]
        t, pred = r["metricas"]["teste"], r["pred"]
        brier = np.mean((pred["probabilidade"] - pred["y_real"]) ** 2)
        if modelo == melhor:
            p = "--"
        else:
            p = p_texto(delong(pred["y_real"], ref["probabilidade"], pred["probabilidade"])["p_valor"])
        linhas.append({"Modelo": MODELOS[modelo], "AUC (IC 95%)": ic_texto(delong(pred["y_real"], pred["probabilidade"])),
                       f"p vs. {MODELOS[melhor]}": p, "Sensib.": num(t["Sensibilidade"]),
                       "Especif.": num(t["Especificidade"]), "Precisão": num(t["Precisão"]),
                       "F1": num(t["F1-Score"]), "Brier": num(brier)})
    salvar_tabela(pd.DataFrame(linhas), "tabela_desempenho_modelos",
                  "Desempenho dos modelos no conjunto de teste (2022), com variáveis individuais e municipais.",
                  "tab-res-desempenho", "l c c c c c c c",
                  nota=f"Nota: métricas calculadas com limiar de {num(LIMIAR, 1)}; IC 95% e p-valor pelo teste de DeLong; "
                       "Brier = escore de Brier (quanto menor, melhor).")


def tabela_comparacao_cenarios(res):
    linhas = []
    for modelo in MODELOS:
        if (modelo, "individuais") not in res or (modelo, COMPLETO) not in res:
            continue
        a, b = res[(modelo, COMPLETO)]["pred"], res[(modelo, "individuais")]["pred"]
        assert (a["y_real"].to_numpy() == b["y_real"].to_numpy()).all(), "testes diferentes"
        d = delong(a["y_real"], a["probabilidade"], b["probabilidade"])
        linhas.append({"Modelo": MODELOS[modelo],
                       "Só individuais": ic_texto(delong(b["y_real"], b["probabilidade"])),
                       "Individuais + municipais": ic_texto(delong(a["y_real"], a["probabilidade"])),
                       "Diferença (IC 95%)": f"{num(d['diferenca'])} ({num(d['dif_ic_inf'])}–{num(d['dif_ic_sup'])})",
                       "p-valor": p_texto(d["p_valor"])})
    salvar_tabela(pd.DataFrame(linhas), "tabela_comparacao_cenarios",
                  "AUC no conjunto de teste sem e com as variáveis do município de residência.",
                  "tab-res-cenarios", "l c c c c",
                  nota="Nota: AUC (IC 95%); diferença = com municipais menos só individuais; "
                       "IC e p-valor pelo teste de DeLong para curvas ROC pareadas.")


def _formatar_hiperparametros(h):
    nomes = {"C": "C", "l1_ratio": "regularização", "criterion": "critério", "max_depth": "profundidade",
             "min_samples_leaf": "mín. por folha", "hidden_layer_sizes": "camadas ocultas",
             "learning_rate_init": "taxa de aprendizado", "alpha": "alpha", "learning_rate": "taxa de aprendizado",
             "num_leaves": "máx. de folhas"}
    partes = []
    for k, v in h.items():
        if k == "class_weight":
            continue
        if k == "l1_ratio":
            v = "L1" if v == 1 else "L2"
        elif k == "hidden_layer_sizes":
            v = "(" + ", ".join(str(x) for x in v) + ")"
        elif isinstance(v, float):
            v = num(v, 4).rstrip("0").rstrip(",")
        partes.append(f"{nomes.get(k, k)} = {v}")
    return "; ".join(partes)


def tabela_hiperparametros(res):
    linhas = []
    for (modelo, cenario), r in res.items():
        m = r["metricas"]
        linhas.append({"Modelo": MODELOS[modelo], "Cenário": CENARIOS[cenario],
                       "Hiperparâmetros escolhidos": _formatar_hiperparametros(m["hiperparametros"]),
                       "Balanceamento": m["balanceamento"],
                       "AUC na validação cruzada": f"{num(m['auc_validacao_cruzada'])} ± {num(m['auc_validacao_cruzada_desvio'])}"})
    salvar_tabela(pd.DataFrame(linhas), "tabela_hiperparametros",
                  "Hiperparâmetros selecionados pela validação cruzada em cada modelo e cenário.",
                  "tab-res-hiperparametros", "p{2.4cm} p{2.4cm} p{4.6cm} p{2.0cm} c",
                  nota="Nota: AUC na validação cruzada = média ± desvio-padrão nas 10 partes do conjunto de treinamento.")


# ------------------------------------------------------------------ figuras
def figura_curvas_roc(res):
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], color=CINZA, lw=0.9, ls="--", label="Aleatório (AUC = 0,500)")
    for modelo in MODELOS:
        if (modelo, COMPLETO) not in res:
            continue
        pred = res[(modelo, COMPLETO)]["pred"]
        fpr, tpr, _ = roc_curve(pred["y_real"], pred["probabilidade"])
        auc = res[(modelo, COMPLETO)]["metricas"]["teste"]["AUC"]
        ax.plot(fpr, tpr, color=CORES[modelo], ls=TRACOS[modelo], lw=1.6,
                label=f"{MODELOS[modelo]} (AUC = {num(auc)})")
    ax.set_xlabel("1 $-$ especificidade (taxa de falsos positivos)")
    ax.set_ylabel("Sensibilidade (taxa de verdadeiros positivos)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=7.5)
    salvar_figura(fig, "fig_curvas_roc")


def figura_comparacao_cenarios(res):
    modelos = [m for m in MODELOS if (m, COMPLETO) in res and (m, "individuais") in res]
    fig, ax = plt.subplots(figsize=(5.8, 2.9))
    for i, modelo in enumerate(modelos):
        valores = {}
        for cenario, desloc, marcador in [("individuais", -0.12, "o"), (COMPLETO, 0.12, "s")]:
            pred = res[(modelo, cenario)]["pred"]
            r = delong(pred["y_real"], pred["probabilidade"])
            valores[cenario] = r["auc"]
            ax.errorbar(r["auc"], i + desloc, xerr=[[r["auc"] - r["ic_inf"]], [r["ic_sup"] - r["auc"]]],
                        fmt=marcador, color=CORES[modelo] if cenario == COMPLETO else CINZA,
                        mfc="white" if cenario == "individuais" else CORES[modelo], ms=5, lw=1.1, capsize=2,
                        label=None)
        ax.plot([valores["individuais"], valores[COMPLETO]], [i - 0.12, i + 0.12], color="#bbbbbb", lw=0.8, zorder=0)
    ax.set_yticks(range(len(modelos)), [MODELOS[m] for m in modelos])
    ax.invert_yaxis()
    ax.set_xlabel("AUC no conjunto de teste (IC 95%)")
    ax.grid(axis="y", visible=False)
    ax.plot([], [], "o", color=CINZA, mfc="white", label="Só individuais")
    ax.plot([], [], "s", color=CINZA, label="Individuais + municipais")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.5, frameon=False)
    salvar_figura(fig, "fig_comparacao_cenarios")


def _rotulo_dummy(nome):
    for var in sorted(VARIAVEIS, key=len, reverse=True):
        if nome.startswith(var + "_"):
            cat = nome[len(var) + 1:]
            return VARIAVEIS[var], CATEGORIAS.get(cat, cat)
    return VARIAVEIS.get(nome, nome), None


def figura_razoes_de_chances(res):
    arq = MODELS / "regressao_logistica" / COMPLETO / "razoes_de_chances.csv"
    if not arq.exists():
        return
    oddsr = pd.read_csv(arq, sep=";").set_index("variavel")["OR"]
    linhas = []
    for var in VARIAVEIS:   # ordem da Tabela 3
        for nome, valor in oddsr.items():
            grupo, cat = _rotulo_dummy(nome)
            if grupo == VARIAVEIS[var]:
                linhas.append((grupo, cat, valor))
    fig, ax = plt.subplots(figsize=(5.6, 0.2 * len(linhas) + 0.8))
    rotulos, y, grupo_ant = [], 0, None
    for grupo, cat, valor in linhas:
        if grupo != grupo_ant:
            rotulos.append((y, grupo, True)); y += 1; grupo_ant = grupo
        zerado = np.isclose(valor, 1.0)
        cor = "#bbbbbb" if zerado else ("#e34948" if valor > 1 else "#2a78d6")
        ax.plot(valor, y, "o", color=cor, ms=4)
        rotulos.append((y, cat if cat else "(por desvio-padrão)", False)); y += 1
    ax.axvline(1, color=CINZA, lw=0.9, ls="--")
    ax.set_xscale("log")
    ticks = [0.5, 0.75, 1, 1.5, 2, 3]
    ax.set_xticks(ticks, [num(t, 2).rstrip("0").rstrip(",") for t in ticks])
    ax.minorticks_off()
    ax.set_yticks([r[0] for r in rotulos], [r[1] for r in rotulos], fontsize=7)
    for tick, (_, _, negrito) in zip(ax.get_yticklabels(), rotulos):
        tick.set_fontweight("bold" if negrito else "normal")
    ax.set_ylim(y, -1)
    ax.set_xlabel("Razão de chances (escala logarítmica)")
    ax.grid(axis="y", visible=False)
    salvar_figura(fig, "fig_razoes_de_chances")


def figura_importancia(res):
    tabela = {}
    for modelo in MODELOS:
        arq = MODELS / modelo / COMPLETO / "importancia_permutacao.csv"
        if arq.exists():
            tabela[MODELOS[modelo]] = pd.read_csv(arq, sep=";").set_index("variavel").iloc[:, 0]
    imp = pd.DataFrame(tabela)
    imp = imp.loc[imp.mean(axis=1).sort_values(ascending=False).index]
    imp.index = [VARIAVEIS[v] for v in imp.index]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    im = ax.imshow(imp.clip(lower=0), cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(imp.shape[1]), imp.columns, rotation=20, ha="right")
    ax.set_yticks(range(imp.shape[0]), imp.index)
    ax.grid(False)
    limite = imp.to_numpy().max() * 0.6
    for i in range(imp.shape[0]):
        for j in range(imp.shape[1]):
            v = imp.iat[i, j]
            ax.text(j, i, num(v, 3), ha="center", va="center", fontsize=7,
                    color="white" if v > limite else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Queda média da AUC")
    salvar_figura(fig, "fig_importancia_permutacao")


def figura_calibracao(res):
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], color=CINZA, lw=0.9, ls="--", label="Calibração perfeita")
    for modelo in MODELOS:
        if (modelo, COMPLETO) not in res:
            continue
        pred = res[(modelo, COMPLETO)]["pred"]
        faixas = pd.qcut(pred["probabilidade"], 10, duplicates="drop")
        g = pred.groupby(faixas, observed=True).mean()
        ax.plot(g["probabilidade"], g["y_real"], marker="o", ms=3, color=CORES[modelo],
                ls=TRACOS[modelo], lw=1.4, label=MODELOS[modelo])
    prev = res[next(iter(res))]["pred"]["y_real"].mean()
    ax.axhline(prev, color="#bbbbbb", lw=0.8)
    ax.text(0.02, prev + 0.015, f"prevalência no teste ({num(prev * 100, 1)}%)", fontsize=7, color=CINZA)
    ax.set_xlabel("Probabilidade média estimada (decis)")
    ax.set_ylabel("Proporção observada de perda de seguimento")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=7.5)
    salvar_figura(fig, "fig_calibracao")


def figura_limiar(res):
    melhor = melhor_modelo(res)
    pred = res[(melhor, COMPLETO)]["pred"]
    fpr, tpr, limiares = roc_curve(pred["y_real"], pred["probabilidade"])
    ok = (limiares <= 1) & (limiares >= 0)
    fpr, tpr, limiares = fpr[ok], tpr[ok], limiares[ok]
    k = int(np.argmax(tpr - fpr))
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.plot(limiares, tpr, color="#e34948", lw=1.6, label="Sensibilidade")
    ax.plot(limiares, 1 - fpr, color="#2a78d6", lw=1.6, ls="--", label="Especificidade")
    ax.axvline(LIMIAR, color=CINZA, lw=0.9, ls=":")
    ax.text(LIMIAR + 0.01, 0.04, f"limiar adotado ({num(LIMIAR, 1)})", fontsize=7, color=CINZA)
    ax.axvline(limiares[k], color=CINZA, lw=0.9, ls="--")
    ax.text(limiares[k] + 0.01, 0.92, f"máx. de Youden ({num(limiares[k], 2)})", fontsize=7, color=CINZA)
    ax.set_xlabel(f"Limiar de decisão ({MODELOS[melhor]})")
    ax.set_ylabel("Valor da métrica")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="center right", fontsize=7.5)
    salvar_figura(fig, "fig_limiar_melhor_modelo")
    print(f"    {MODELOS[melhor]}: limiar de Youden no teste = {limiares[k]:.3f} "
          f"(sensibilidade {tpr[k]:.3f}, especificidade {1 - fpr[k]:.3f})")


def gerar_resultados(incluir_tabela_populacao=True):
    res = ler_resultados()
    if incluir_tabela_populacao:
        gerar_tabela_caracteristicas_populacao()
    tabela_desempenho(res)
    tabela_comparacao_cenarios(res)
    tabela_hiperparametros(res)
    figura_curvas_roc(res)
    figura_comparacao_cenarios(res)
    figura_razoes_de_chances(res)
    figura_importancia(res)
    figura_calibracao(res)
    figura_limiar(res)


if __name__ == "__main__":
    gerar_resultados()
