"""
Fluxograma de seleção da população de estudo (no formato da Figura 1 do artigo
de Rodrigues et al., BMC Public Health, 2024).

Os critérios são os mesmos do WHERE de seleção, aplicados em sequência: cada
caso é contado uma única vez, no primeiro critério que o exclui.

Saídas na pasta reports/:
  - fluxograma_selecao.png
  - fluxograma_selecao.csv

Uso (dentro da pasta ia):  python src/visualization/fluxograma_selecao.py
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from src.db_manager import ConexaoBanco
from src.config import REPORTS

PASTA_SAIDA = str(REPORTS)

QUERY = """
    SELECT faixa_etaria, imigrante, sem_teto, profissional_saude, encarcerado,
           gestante, forma_clinica, teste_sensibilidade,
           tempo_entre_encerramento_e_inicio_tratamento,
           situacao_encerramento
    FROM casos_tuberculose.fnotificacao
"""

RESISTENTES = [
    "1",  # Resistente somente à Isoniazida
    "Resistente somente à Rifampicina",
    "Resistente à Isoniazida e Rifampicina",
    "Resistente a outras drogas de 1ª linha",
]
DESFECHOS_VALIDOS = ["Cura", "Abandono", "Abandono Primário"]


# ---------------------------------------------------------------------------
# Funções que dizem quais linhas FICAM em cada condição do WHERE.
# Como no MySQL, uma linha com NULL não passa em !=, = , NOT IN, IN ou LIKE.
# ---------------------------------------------------------------------------
def _txt(s):
    return s.astype("string")


def diferente(col, valor):
    return lambda d: d[col].notna() & (_txt(d[col]) != valor)


def igual(col, valor):
    return lambda d: d[col].notna() & (_txt(d[col]) == valor)


def fora_de(col, valores):
    return lambda d: d[col].notna() & ~_txt(d[col]).isin(valores)


def gestante_nao(d):
    # gestante LIKE '%Não%' OR gestante = 'Ignorado'
    g = _txt(d["gestante"])
    return d["gestante"].notna() & (g.str.contains("Não", regex=False) | (g == "Ignorado"))


def desfecho_diferente(valores):
    """Na última etapa, fica quem NÃO tem um dos desfechos listados."""
    return lambda d: ~_txt(d["situacao_encerramento"]).isin(valores).fillna(False)


# ---------------------------------------------------------------------------
# Etapas do fluxograma: (título da caixa, [(rótulo do excluído, quem fica)])
# A ordem é a do WHERE; mude aqui se quiser outra ordem ou outros rótulos.
# ---------------------------------------------------------------------------
ETAPAS = [
    ("Casos novos de TB pulmonar\nnotificados no SINAN (2018–2022)", [
        ("≤ 18 anos", diferente("faixa_etaria", "<= 18 anos")),
        ("Imigrantes", diferente("imigrante", "Sim")),
        ("Em situação de rua", diferente("sem_teto", "Sim")),
        ("Profissionais de saúde", diferente("profissional_saude", "Sim")),
        ("Privados de liberdade", diferente("encarcerado", "Sim")),
        ("Gestantes", gestante_nao),
    ]),
    ("Adultos fora das\npopulações especiais", [
        ("Pulmonar + extrapulmonar", igual("forma_clinica", "Pulmonar")),
    ]),
    ("Adultos com TB\nexclusivamente pulmonar", [
        ("Resistência a droga de 1ª linha", fora_de("teste_sensibilidade", RESISTENTES)),
        ("Sem tempo de tratamento",
         lambda d: d["tempo_entre_encerramento_e_inicio_tratamento"].notna()),
    ]),
    ("TB pulmonar não resistente,\ncom tempo de tratamento", [
        ("Transferência", desfecho_diferente(["Transferência"])),
        ("Óbito (TB ou outras causas)",
         desfecho_diferente(["Óbito por TB", "Óbito por outras causas"])),
        ("Mudança de esquema", desfecho_diferente(["Mudança de Esquema"])),
        ("TB-DR", desfecho_diferente(["TB-DR"])),
        ("Falência", desfecho_diferente(["Falência"])),
        # garante o mesmo resultado do WHERE: qualquer outro desfecho (ou NULL) sai aqui
        ("Outro desfecho ou sem informação",
         lambda d: d["situacao_encerramento"].notna()
         & _txt(d["situacao_encerramento"]).isin(DESFECHOS_VALIDOS)),
    ]),
]


def _fmt(n):
    return f"{int(n):,}".replace(",", ".")


def _pct(p):
    return f"{p:.1f}".replace(".", ",") + "%"


def calcular_fluxo(df):
    """Aplica os critérios em sequência e devolve as contagens de cada etapa."""
    d = df
    etapas = []
    for titulo, criterios in ETAPAS:
        n_inicio = len(d)
        excluidos = []
        for rotulo, fica in criterios:
            mascara = fica(d).astype(bool)
            n_fora = int((~mascara).sum())
            if n_fora > 0:
                excluidos.append((rotulo, n_fora))
            d = d[mascara]
        etapas.append((titulo, n_inicio, excluidos))
    desfecho = _txt(d["situacao_encerramento"])
    cura = int((desfecho == "Cura").sum())
    return etapas, len(d), cura, len(d) - cura


def desenhar(etapas, total, final, cura, abandono, caminho):
    superficie, tinta, tinta2, borda = "#fcfcfb", "#0b0b0b", "#52514e", "#8f8e89"
    azul, laranja = "#2a78d6", "#eb6834"

    fig, ax = plt.subplots(figsize=(10, 12.5), facecolor=superficie)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.04)
    ax.axis("off")

    def caixa(x, y, w, h, texto, fc="white", ec=borda, negrito=False, ha="center", fs=10):
        ax.add_patch(FancyBboxPatch((x, y), w, h, fc=fc, ec=ec, lw=1.2,
                                    boxstyle="round,pad=0.004,rounding_size=0.008"))
        tx = x + w / 2 if ha == "center" else x + 0.015
        ax.text(tx, y + h / 2, texto, ha=ha, va="center", fontsize=fs, color=tinta,
                fontweight="bold" if negrito else "normal", linespacing=1.35)

    def seta(origem, destino, tracejada=False):
        ax.annotate("", xy=destino, xytext=origem,
                    arrowprops=dict(arrowstyle="-|>", color=tinta2,
                                    lw=1.1 if tracejada else 1.3,
                                    ls=(0, (2, 2)) if tracejada else "-"))

    mx, mw, mh = 0.04, 0.40, 0.075        # caixas principais (esquerda)
    ex, ew = 0.53, 0.44                   # caixas de exclusão (direita)
    cx = mx + mw / 2
    y_final, y_classes, h_classes = 0.145, 0.02, 0.07

    # distribui as caixas principais conforme o tamanho das caixas de exclusão
    alturas_excl = [0.019 * max(len(e), 1) + 0.035 for _, _, e in etapas]
    vaos = [max(a + 0.02, 0.09) for a in alturas_excl]
    espaco = 0.975 - (y_final + mh) - mh * len(etapas)
    escala = espaco / sum(vaos)
    ys, y = [], 0.975
    for vao in vaos:
        y -= mh
        ys.append(y)
        y -= vao * escala

    for i, ((titulo, n, excluidos), y) in enumerate(zip(etapas, ys)):
        caixa(mx, y, mw, mh, f"{titulo}\n{_fmt(n)}", negrito=(i == 0))
        y_prox = ys[i + 1] if i + 1 < len(ys) else y_final
        seta((cx, y), (cx, y_prox + mh))
        if not excluidos:
            continue
        eh = alturas_excl[i]
        y_meio = (y + y_prox + mh) / 2
        tot = sum(c for _, c in excluidos)
        texto = (f"Excluídos: {_fmt(tot)} ({_pct(tot / total * 100)} da base)\n"
                 + "\n".join(f"   {_fmt(c)}  {r}" for r, c in excluidos))
        caixa(ex, y_meio - eh / 2, ew, eh, texto, fc="#f4f3ef", ec="#d9d8d2", ha="left", fs=9.2)
        seta((cx, y_meio), (ex, y_meio), tracejada=True)

    caixa(mx, y_final, mw, mh,
          f"População de estudo\n{_fmt(final)} ({_pct(final / total * 100)} da base)",
          ec=tinta, negrito=True)

    cw = 0.19
    for x, nome, n, cor in [(mx, "Cura", cura, azul), (mx + mw - cw, "Abandono", abandono, laranja)]:
        pct = _pct(n / final * 100) if final else "—"
        caixa(x, y_classes, cw, h_classes, f"{nome}\n{_fmt(n)} ({pct})", ec=cor, negrito=True, fs=9.5)
        seta((cx, y_final), (x + cw / 2, y_classes + h_classes))
    ax.text(ex, y_classes + h_classes / 2,
            "Abandono inclui abandono primário.\nPercentuais das classes sobre a população de estudo.",
            fontsize=8.8, color=tinta2, va="center", linespacing=1.4)

    ax.text(0.04, 1.03, "Fluxograma de seleção da população de estudo",
            fontsize=13, fontweight="bold", color=tinta, va="top")
    fig.savefig(caminho, dpi=200, bbox_inches="tight", facecolor=superficie)
    plt.close(fig)


def gerar_fluxograma(db=None):
    db = db or ConexaoBanco()
    print("[*] Extraindo dados da fNotificacao...")
    df = db.obter_dataframe(QUERY)
    if df is None or df.empty:
        print("[X] Erro: Sem dados.")
        return

    total = len(df)
    etapas, final, cura, abandono = calcular_fluxo(df)

    os.makedirs(PASTA_SAIDA, exist_ok=True)
    caminho = os.path.join(PASTA_SAIDA, "fluxograma_selecao.png")
    desenhar(etapas, total, final, cura, abandono, caminho)

    linhas = []
    for titulo, n, excluidos in etapas:
        linhas.append({"etapa": titulo.replace("\n", " "), "item": "Casos na etapa", "casos": n})
        linhas += [{"etapa": titulo.replace("\n", " "), "item": f"Excluído: {r}", "casos": c}
                   for r, c in excluidos]
    linhas += [{"etapa": "População de estudo", "item": "Total", "casos": final},
               {"etapa": "População de estudo", "item": "Cura", "casos": cura},
               {"etapa": "População de estudo", "item": "Abandono (inclui primário)", "casos": abandono}]
    tab = pd.DataFrame(linhas)
    tab["percentual_da_base"] = tab["casos"] / total * 100
    tab.to_csv(os.path.join(PASTA_SAIDA, "fluxograma_selecao.csv"), sep=";",
               decimal=",", index=False, encoding="utf-8-sig")

    for titulo, n, excluidos in etapas:
        print(f"\n{titulo.replace(chr(10), ' ')}: {_fmt(n)}")
        for r, c in excluidos:
            print(f"   - {_fmt(c)}  {r}")
    print(f"\nPopulação de estudo: {_fmt(final)} ({_pct(final / total * 100)} da base)")
    if final:
        print(f"   Cura: {_fmt(cura)} ({_pct(cura / final * 100)})")
        print(f"   Abandono: {_fmt(abandono)} ({_pct(abandono / final * 100)})")
    print(f"\n[+] Fluxograma salvo em '{caminho}'.")


if __name__ == "__main__":
    gerar_fluxograma()
