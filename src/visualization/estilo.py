"""
Estilo e utilitários comuns às figuras e tabelas do TCC:
nomes legíveis de modelos e variáveis, cores, formatação de números e gravação.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import REPORTS

PASTA_FIGURAS = REPORTS / "resultados" / "figuras"
PASTA_TABELAS = REPORTS / "resultados" / "tabelas"

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e5e5e5", "grid.linewidth": 0.6,
    "legend.frameon": False,
})

# ordem fixa: cada modelo tem sempre a mesma cor e o mesmo traço em todas as figuras
MODELOS = {
    "regressao_logistica": "Regressão Logística",
    "arvore_decisao": "Árvore de Decisão",
    "floresta_aleatoria": "Random Forest",
    "rede_neural": "MLP",
    "lightgbm_modelo": "LightGBM",
}
CORES = {"regressao_logistica": "#2a78d6", "arvore_decisao": "#eb6834", "floresta_aleatoria": "#1baf7a",
         "rede_neural": "#eda100", "lightgbm_modelo": "#e87ba4"}
TRACOS = {"regressao_logistica": "-", "arvore_decisao": "--", "floresta_aleatoria": "-.",
          "rede_neural": ":", "lightgbm_modelo": (0, (5, 1, 1, 1))}
CINZA = "#555555"

CENARIOS = {"individuais": "Só individuais", "individuais_municipais": "Individuais + municipais"}

VARIAVEIS = {
    "faixa_etaria": "Faixa etária", "sexo": "Sexo", "teste_hiv": "Teste de HIV",
    "alcool": "Uso de álcool", "tabagismo": "Tabagismo", "drogas": "Uso de outras drogas",
    "escolaridade": "Escolaridade", "raca": "Raça/cor", "faixa_populacao": "Porte populacional",
    "regiao": "Região", "idhm": "IDHM", "indice_gini": "Índice de Gini",
    "cobertura_aps_2021": "Cobertura da APS",
}
CATEGORIAS = {
    "19 - 29 anos": "19 a 29 anos", "30 - 39 anos": "30 a 39 anos", "40 - 49 anos": "40 a 49 anos",
    "50 - 59 anos": "50 a 59 anos", "Acima de 60 anos": "60 anos ou mais",
    "Até fundamental": "Até o fundamental", "1 - Até 5000": "Até 5 mil",
    "2 - 5001 até 10000": "5 mil a 10 mil", "3 - 10001 até 20000": "10 mil a 20 mil",
    "4 - 20001 até 50000": "20 mil a 50 mil", "5 - 50001 até 100000": "50 mil a 100 mil",
    "6 - 100001 até 500000": "100 mil a 500 mil", "7 - Maior que 500000": "Mais de 500 mil",
    "1 - Norte": "Norte", "2 - Nordeste": "Nordeste", "3 - Sudeste": "Sudeste",
    "4 - Sul": "Sul", "5 - Centro-Oeste": "Centro-Oeste",
}


def num(x, casas=3):
    """Número com vírgula decimal."""
    return f"{x:.{casas}f}".replace(".", ",")


def milhar(n):
    return f"{int(n):,}".replace(",", ".")


def salvar_figura(fig, nome):
    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PASTA_FIGURAS / f"{nome}.pdf", bbox_inches="tight")
    fig.savefig(PASTA_FIGURAS / f"{nome}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[+] Figura: {PASTA_FIGURAS / nome}.pdf")


def _escapar(texto):
    return str(texto).replace("%", "\\%").replace("&", "\\&").replace("_", "\\_")


def salvar_tabela(df, nome, legenda, rotulo, colunas_formato, nota=None, fonte="Fonte: elaborado pelo autor."):
    """Grava a tabela em CSV e em LaTeX (ambiente table pronto para \\input no TCC)."""
    PASTA_TABELAS.mkdir(parents=True, exist_ok=True)
    df.to_csv(PASTA_TABELAS / f"{nome}.csv", sep=";", index=False, encoding="utf-8-sig")
    linhas = [" & ".join(_escapar(v) for v in r) + " \\\\" for r in df.itertuples(index=False)]
    rodape = (f"{_escapar(nota)}\\\\" if nota else "") + fonte
    tex = "\n".join([
        "\\begin{table}[H]", "    \\centering", f"    \\caption{{{legenda}}}", f"    \\label{{{rotulo}}}",
        "    \\footnotesize", f"    \\begin{{tabular}}{{{colunas_formato}}}", "        \\hline",
        "        " + " & ".join(f"\\textbf{{{_escapar(c)}}}" for c in df.columns) + " \\\\",
        "        \\hline", *["        " + l for l in linhas], "        \\hline", "    \\end{tabular}",
        f"    \\legend{{{rodape}}}", "\\end{table}", ""])
    (PASTA_TABELAS / f"{nome}.tex").write_text(tex, encoding="utf-8")
    print(f"[+] Tabela: {PASTA_TABELAS / nome}.tex")
