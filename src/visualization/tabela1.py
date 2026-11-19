"""
Tabela 1 — características da população de estudo (view vw_populacao_estudo).

Usa a população de estudo da view vw_populacao_estudo (todos os critérios de
seleção já aplicados; desfecho Cura ou Perda de seguimento = Abandono +
Abandono Primário).
Colunas: Total (= Cura + Perda de seguimento), Cura e Perda de seguimento.
  - variáveis categóricas: n (%)  — percentual dentro da coluna
  - variáveis contínuas:   mediana [IQR]

Saídas na pasta reports/:
  - tabela1.csv   (abre no Excel, separador ;)
  - tabela1.html  (abre no navegador; dá para copiar e colar no Word)

Uso (dentro da pasta ia):  python src/visualization/tabela1.py
"""
import os
import pandas as pd
from src.db_manager import ConexaoBanco
from src.config import REPORTS

PASTA_SAIDA = str(REPORTS)

QUERY = """
    SELECT faixa_etaria, sexo, teste_hiv, alcool, tabagismo, drogas,
           escolaridade, raca, situacao_encerramento
    FROM casos_tuberculose.vw_populacao_estudo
"""

CURA = ["Cura"]
PERDA = ["Abandono", "Abandono Primário"]

# (coluna, rótulo na tabela, ordem preferida das categorias)
CATEGORICAS = [
    ("faixa_etaria", "Faixa etária", ["<= 18 anos", "19 - 29 anos", "30 - 39 anos",
                                     "40 - 49 anos", "50 - 59 anos", "Acima de 60 anos"]),
    ("sexo", "Sexo", ["Feminino", "Masculino"]),
    ("teste_hiv", "HIV (resultado do teste)", ["Positivo", "Negativo", "Em andamento", "Não realizado"]),
    ("alcool", "Uso de álcool", ["Sim", "Não"]),
    ("tabagismo", "Tabagismo", ["Sim", "Não"]),
    ("drogas", "Uso de drogas ilícitas", ["Sim", "Não"]),
    ("escolaridade", "Escolaridade", ["Analfabeto", "Até fundamental", "Até fundamentalAnalfabeto",
                                      "Ensino médio", "Ensino superior"]),
    ("raca", "Raça/cor", ["Branca", "Preta", "Parda", "Amarela/Indígena"]),
]

CONTINUAS = []  # nenhuma variável contínua nesta versão

# códigos que ficaram como número no ETL e têm significado conhecido
CORRECOES = {
    "teste_sensibilidade": {"1": "Resistente somente à Isoniazida"},
    "escolaridade": {"ensino superior": "Ensino superior"},
}
IGNORADO = "Ignorado / não informado"


def limpar(df, col, ordem):
    s = df[col].astype("string").str.strip()
    s = s.replace(CORRECOES.get(col, {}))
    # qualquer valor fora do domínio (códigos soltos como 3, 9, 0, 8, nulos) vira "Ignorado"
    return s.where(s.isin(ordem), IGNORADO).fillna(IGNORADO)


def _fmt(n):
    return f"{int(n):,}".replace(",", ".")


def _num(x, casas=1):
    return f"{x:.{casas}f}".replace(".", ",")


def gerar_tabela1(db=None):
    db = db or ConexaoBanco()
    print("[*] Extraindo dados...")
    df = db.obter_dataframe(QUERY)
    if df is None or df.empty:
        print("[X] Erro: Sem dados.")
        return

    desfecho = df["situacao_encerramento"].astype("string")
    grupos = {
        "Total": desfecho.isin(CURA + PERDA).fillna(False),
        "Cura": desfecho.isin(CURA).fillna(False),
        "Perda de seguimento": desfecho.isin(PERDA).fillna(False),
    }
    n = {g: int(m.sum()) for g, m in grupos.items()}
    cab = {g: f"{g} (n={_fmt(n[g])})" for g in grupos}

    linhas = []
    for col, rotulo, ordem in CATEGORICAS:
        s = limpar(df, col, ordem)
        cats = [c for c in ordem if (s == c).any()] + ([IGNORADO] if (s == IGNORADO).any() else [])
        linhas.append({"Característica": rotulo, **{cab[g]: "" for g in grupos}})
        for c in cats:
            linha = {"Característica": f"    {c}"}
            for g, m in grupos.items():
                k = int((s[m] == c).sum())
                linha[cab[g]] = f"{_fmt(k)} ({_num(k / n[g] * 100)}%)"
            linhas.append(linha)

    for col, rotulo in CONTINUAS:
        x = pd.to_numeric(df[col], errors="coerce")
        linha = {"Característica": f"{rotulo}, mediana [IQR]"}
        for g, m in grupos.items():
            v = x[m].dropna()
            linha[cab[g]] = f"{_num(v.median(), 0)} [{_num(v.quantile(.25), 0)}; {_num(v.quantile(.75), 0)}]"
        linhas.append(linha)

    tab = pd.DataFrame(linhas)
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    tab.to_csv(os.path.join(PASTA_SAIDA, "tabela1.csv"), sep=";", index=False, encoding="utf-8-sig")

    nota = ("Valores em n (%), com percentual dentro de cada coluna, salvo indicação. "
            "Perda de seguimento = Abandono + Abandono Primário. Total = Cura + Perda de "
            "seguimento. População de estudo após os critérios de seleção (view "
            "vw_populacao_estudo). "
            "Códigos fora do dicionário do SINAN foram agrupados em \"Ignorado / não informado\".")
    from html import escape
    cab_html = "".join(f"<th>{escape(c)}</th>" for c in tab.columns)
    corpo = []
    for _, r in tab.iterrows():
        nome = r["Característica"]
        sub = nome.startswith("    ")
        estilo_1a = ' style="padding-left:1.8em"' if sub else ' style="font-weight:bold"'
        celulas = [f"<td{estilo_1a}>{escape(nome.strip())}</td>"]
        celulas += [f"<td>{escape(str(r[c]))}</td>" for c in tab.columns[1:]]
        corpo.append("<tr>" + "".join(celulas) + "</tr>")
    html = (f"<table><thead><tr>{cab_html}</tr></thead>"
            f"<tbody>{''.join(corpo)}</tbody></table>")
    estilo = ("<style>body{font-family:Arial,sans-serif;font-size:12px;margin:24px}"
              "table{border-collapse:collapse}th,td{padding:3px 10px;text-align:right}"
              "th:first-child,td:first-child{text-align:left}"
              "thead th{border-top:2px solid #000;border-bottom:1px solid #000}"
              "tbody tr:last-child td{border-bottom:2px solid #000}</style>")
    with open(os.path.join(PASTA_SAIDA, "tabela1.html"), "w", encoding="utf-8") as f:
        f.write(f"<html><head><meta charset='utf-8'>{estilo}</head><body>"
                f"<h3>Tabela 1 — Características da população de estudo: casos novos de TB "
                f"pulmonar notificados no SINAN, 2018–2022</h3>{html}<p style='max-width:900px'>{nota}</p></body></html>")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_colwidth", 60)
    print(tab.to_string(index=False))
    print(f"\n[+] Tabela salva em '{PASTA_SAIDA}/tabela1.csv' e 'tabela1.html'.")


if __name__ == "__main__":
    gerar_tabela1()
