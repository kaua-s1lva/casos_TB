"""
Tabela 1 — Características da população de estudo, por desfecho do tratamento.
Segue o formato da Tabela 1 de Rodrigues et al. (2024).

Lê a ABT (a mesma usada no treino dos modelos), então a tabela descreve
exatamente os casos e as categorias (inclusive "Ignorado") dos modelos.
  - colunas: Total | Cura | Perda de seguimento | p-valor
  - variáveis categóricas: n (%), com percentual dentro de cada coluna;
    variáveis Sim/Não mostram só a linha "Sim" (e "Ignorado"), como em Rodrigues et al.
  - variáveis numéricas: mediana [intervalo interquartil]
  - p-valor: qui-quadrado de independência (categóricas, incluindo "Ignorado")
    e Mann-Whitney (numéricas), comparando cura x perda de seguimento

Saídas em reports/resultados/tabelas/: tabela_caracteristicas_populacao.csv e .tex
(o .tex é uma longtable em fonte reduzida, com variáveis em negrito e categorias recuadas,
que continua na página seguinte com "(continua)" e "(conclusão)", como pede o guia da UFES)

Uso (dentro da pasta ia):  python src/visualization/tabela_caracteristicas_populacao.py
"""
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

from src.banco_dados import ConexaoBanco
from src.features.construir_abt import TABELA_ABT
from src.features.variaveis import ALVO, CATEGORICAS, NUMERICAS
from src.visualization.estilo import CATEGORIAS, PASTA_TABELAS, VARIAVEIS, _escapar, milhar, num


def formatar_p(p):
    return "< 0,001" if p < 0.001 else num(p, 3)


def p_categorica(abt, var):
    return chi2_contingency(pd.crosstab(abt[var], abt[ALVO]))[1]


def p_numerica(abt, var):
    cura = abt.loc[abt[ALVO] == 0, var].dropna()
    perda = abt.loc[abt[ALVO] == 1, var].dropna()
    return mannwhitneyu(cura, perda).pvalue


def gerar_tabela_caracteristicas_populacao(abt=None):
    abt = abt if abt is not None else ConexaoBanco().consultar(f"SELECT * FROM {TABELA_ABT}")
    grupos = {"Total": abt, "Cura": abt[abt[ALVO] == 0], "Perda de seguimento": abt[abt[ALVO] == 1]}
    cabecalho = {g: f"{g} (n = {milhar(len(d))})" for g, d in grupos.items()}
    vazio = {cabecalho[g]: "" for g in grupos}

    linhas = []
    for var, (categorias, _) in CATEGORICAS.items():
        sim_nao = sorted(categorias) == ["Não", "Sim"]
        mostrar = (["Sim"] if sim_nao else categorias) + ["Ignorado"]
        linhas.append({"Característica": VARIAVEIS[var], **vazio, "p-valor": formatar_p(p_categorica(abt, var))})
        for cat in mostrar:
            if not (abt[var] == cat).any():
                continue
            linha = {"Característica": "   " + CATEGORIAS.get(cat, cat)}
            for g, d in grupos.items():
                k = int((d[var] == cat).sum())
                linha[cabecalho[g]] = f"{milhar(k)} ({num(k / len(d) * 100, 1)}%)"
            linha["p-valor"] = ""
            linhas.append(linha)

    for var in NUMERICAS:
        casas = 1 if var == "cobertura_aps_2021" else 3
        linha = {"Característica": f"{VARIAVEIS[var]}, mediana [IIQ]"}
        for g, d in grupos.items():
            q1, med, q3 = d[var].quantile([.25, .5, .75])
            linha[cabecalho[g]] = f"{num(med, casas)} [{num(q1, casas)}; {num(q3, casas)}]"
        linha["p-valor"] = formatar_p(p_numerica(abt, var))
        linhas.append(linha)

    tabela = pd.DataFrame(linhas)
    salvar(tabela, {g: len(d) for g, d in grupos.items()})
    return tabela


NOME = "tabela_caracteristicas_populacao"
NOTA = ("Nota: variáveis categóricas em n (%), com percentual dentro de cada coluna; "
        "IDHM, índice de Gini e cobertura da APS (%) em mediana [intervalo interquartil]. "
        "Nas variáveis do tipo sim/não, apresenta-se apenas a categoria ``Sim''. "
        "p-valor: teste qui-quadrado de independência (variáveis categóricas, incluindo a categoria "
        "``Ignorado'') e teste de Mann-Whitney (variáveis numéricas), comparando cura e perda de "
        "seguimento. Perda de seguimento = abandono + abandono primário.")


def salvar(tabela, n_por_grupo):
    """CSV com a tabela como está e LaTeX pronto para \\input no TCC."""
    PASTA_TABELAS.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(PASTA_TABELAS / f"{NOME}.csv", sep=";", index=False, encoding="utf-8-sig")

    corpo = []
    for linha in tabela.itertuples(index=False):
        rotulo, *valores, p = linha
        p = p.replace("<", "$<$")
        if rotulo.startswith("   "):                       # categoria: recuada
            rotulo = "\\hspace{3mm}" + _escapar(rotulo.strip())
        else:                                              # variável: em negrito
            rotulo = rotulo.replace(", mediana [IIQ]", "")  # já explicado na nota
            rotulo = f"\\textbf{{{_escapar(rotulo)}}}"
        corpo.append("        " + " & ".join([rotulo, *map(_escapar, valores), p]) + " \\\\")

    # longtable: a tabela ocupa duas páginas. Seguindo o guia da UFES, o título e o
    # cabeçalho se repetem, com "(continua)" na primeira página e "(conclusão)" na
    # última, e a linha de fechamento, a nota e a fonte aparecem só no fim.
    legenda = "Características da população de estudo, segundo o desfecho do tratamento."
    cabecalho = [
        "        \\hline",
        "        \\textbf{Característica} & " + " & ".join(f"\\textbf{{{g}}}" for g in n_por_grupo)
        + " & \\textbf{p-valor} \\\\",
        "         & " + " & ".join(f"(n = {milhar(n)})" for n in n_por_grupo.values()) + " & \\\\",
        "        \\hline",
    ]
    # \Needspace*: se sobrar menos de meia página, a tabela começa na próxima. Assim ela
    # ocupa sempre exatamente duas páginas e "(conclusão)" fica na última.
    tex = "\n".join([
        "\\Needspace*{0.5\\textheight}",
        "\\begingroup",
        "\\captionsetup{font=normalsize}",
        "\\SingleSpacing*\\footnotesize",   # com *: sem a linha em branco que o memoir insere
        "\\setlength{\\tabcolsep}{4pt}\\setlength{\\LTleft}{0pt}\\setlength{\\LTright}{0pt}",
        "\\begin{longtable}{@{}l@{\\extracolsep{\\fill}}rrrc@{}}",
        f"    \\caption{{{legenda}}}\\label{{tab-res-caracteristicas}} \\\\",
        "        \\multicolumn{5}{r@{}}{(continua)} \\\\",
        *cabecalho,
        "    \\endfirsthead",
        f"    \\caption[]{{{legenda}}} \\\\",
        "        \\multicolumn{5}{r@{}}{(conclusão)} \\\\",
        *cabecalho,
        "    \\endhead",
        "    \\endfoot",
        "        \\hline",
        f"        \\multicolumn{{5}}{{@{{}}p{{\\linewidth}}@{{}}}}{{{_escapar(NOTA)}}} \\\\",
        "        \\multicolumn{5}{@{}l}{Fonte: elaborado pelo autor.} \\\\",
        "    \\endlastfoot",
        *corpo,
        "\\end{longtable}",
        "\\endgroup", ""])
    (PASTA_TABELAS / f"{NOME}.tex").write_text(tex, encoding="utf-8")
    print(f"[+] Tabela: {PASTA_TABELAS / NOME}.tex")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(gerar_tabela_caracteristicas_populacao().to_string(index=False))
