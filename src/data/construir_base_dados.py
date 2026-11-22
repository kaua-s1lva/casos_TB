"""
Reconstrução da base de dados (ETL) — Seção 3.2 do TCC.
Carrega as tabelas no MySQL a partir das fontes originais.

  dtempo        <- data/raw/dTempo.csv (uma linha por data de notificação; precisa das
                   colunas id_tempo e ano_notificacao)
  fnotificacao  <- data/raw/fNotificacao.csv (sem linhas duplicadas e com as chaves de
                   local redirecionadas para a dlocal sem duplicados)
  dlocal        <- data/raw/dLocal.csv (chaves, nomes, códigos,
                   população, faixa populacional, UF e região)
                   + Atlas Brasil 2010 (data/raw/censo_total_1991_2010.xlsx)
                   + cobertura da APS/ESF (data/raw/PacientesTB mais dados muncipais casos maior que 4.xlsx)
                   uma linha por município (duplicados removidos)

Uso (dentro da pasta ia):  python src/data/construir_base_dados.py   ou   python main.py
"""
import html
import re
import zipfile
import numpy as np
import pandas as pd
from src.banco_dados import ConexaoBanco
from src.config import ARQ_ATLAS, ARQ_DLOCAL, ARQ_DTEMPO, ARQ_FATO, ARQ_PACIENTES

ANO_ATLAS = 2010


# ---------------------------------------------------------------------------
# Fato
# ---------------------------------------------------------------------------
def carregar_notificacoes(db, mapa_locais):
    df = pd.read_csv(ARQ_FATO, sep=";", encoding="utf-8")

    # 1) notificações duplicadas: mesma linha em todas as colunas, exceto o id
    antes = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "id_notificacao"])
    print(f"    Notificações duplicadas removidas: {antes - len(df)}")

    # 2) chaves de local apontam para a linha mantida de cada município na dlocal
    for col in ["id_nocal_notificacao", "id_ocal_esidencia"]:
        novos = df[col].map(mapa_locais)
        if novos.isna().any():
            raise ValueError(f"{col}: {novos.isna().sum()} chaves sem correspondência na dlocal")
        print(f"    {col}: {(novos != df[col]).sum()} chaves redirecionadas")
        df[col] = novos.astype("int64")

    db.gravar_tabela(df, "fnotificacao")


# ---------------------------------------------------------------------------
# dTempo
# ---------------------------------------------------------------------------
def carregar_dimensao_tempo(db):
    """Dimensão de datas de notificação (2018 a 2022), usada para obter o ano de cada
    notificação na divisão treino/teste. O separador do CSV é detectado automaticamente."""
    d = pd.read_csv(ARQ_DTEMPO, sep=None, engine="python", encoding="utf-8-sig")
    faltando = {"id_tempo", "ano_notificacao"} - set(d.columns)
    if faltando:
        raise ValueError(f"{ARQ_DTEMPO.name}: colunas ausentes {sorted(faltando)}")
    db.gravar_tabela(d, "dtempo")


# ---------------------------------------------------------------------------
# dLocal
# ---------------------------------------------------------------------------
def ler_dimensao_local_original():
    """Da dLocal original fica só o que não vem do Atlas: chaves, nomes, códigos,
    população, faixa populacional, UF e região."""
    colunas = ["id_local", "codigo_uf", "id_municipio", "municipio_uf", "id_municipio_nome",
               "populacao", "faixa_populacao", "sigla_uf", "cod_uf", "regiao",
               "Estado_Pais", "Estado_Pais_Completo"]
    d = pd.read_csv(ARQ_DLOCAL, sep=";", encoding="utf-8", usecols=colunas)
    d = corrigir_codigos_ibge(d)
    # nome vazio (municípios ausentes do Atlas usado no ETL original): tira do municipio_uf,
    # ex.: "Balneário Rincão (SC)" -> "Balneário Rincão"
    sem_nome = d["id_municipio_nome"].isna()
    d.loc[sem_nome, "id_municipio_nome"] = (d.loc[sem_nome, "municipio_uf"]
                                            .str.replace(r"\s*\([A-Z]{2}\)\s*$", "", regex=True))
    # cod_uf veio preenchido com "Brasil" na original; o código da UF são os 2 primeiros
    # dígitos do código do município
    d["cod_uf"] = (d["id_municipio"] // 10 ** (d["id_municipio"].astype(str).str.len() - 2)).astype("Int64")
    return d


def corrigir_codigos_ibge(d):
    """O ETL original converteu os códigos IBGE de 6 para 7 dígitos usando a lista de
    municípios do Atlas 2010. Municípios instalados depois de 2010 (Balneário Rincão/SC
    e Mojuí dos Campos/PA) não estão no Atlas, então ficaram com o código de 6 dígitos
    (sem o dígito verificador) e sem os dados cadastrais.
    Aqui o código de 6 dígitos é completado a partir da linha do mesmo município que já
    tem o código de 7 dígitos, e os atributos cadastrais são copiados dela.
    O id_local não muda, então a ligação com a fNotificacao continua a mesma."""
    codigo = d["id_municipio"].astype(str)
    completos = d[codigo.str.len() == 7].drop_duplicates("id_municipio")
    por_prefixo = completos.set_index(completos["id_municipio"].astype(str).str[:6])

    cadastrais = ["id_municipio", "id_municipio_nome", "populacao", "faixa_populacao",
                  "sigla_uf", "regiao", "Estado_Pais", "Estado_Pais_Completo"]
    curtos = codigo.str.len() == 6
    for i in d.index[curtos]:
        par = por_prefixo.loc[codigo[i]] if codigo[i] in por_prefixo.index else None
        if par is None:
            print(f"    [!] Código de 6 dígitos sem correspondente: {codigo[i]} ({d.at[i, 'municipio_uf']})")
            continue
        for col in cadastrais:
            d.at[i, col] = par[col]
        print(f"    Código corrigido: {codigo[i]} -> {par['id_municipio']} ({par['municipio_uf']})")
    return d


def remover_municipios_duplicados(d):
    """Mantém uma linha por município (id_municipio).

    O ETL original criou locais de notificação combinando a UF de notificação com o
    município de residência; quando as duas UFs diferem, surgiram linhas repetidas
    do mesmo município com codigo_uf errado. Fora id_local e codigo_uf, as linhas
    repetidas são idênticas.

    Fica a linha cujo codigo_uf confere com o código do município (ou, empatando, o
    menor id_local). Retorna a dlocal sem duplicados e o mapa {id_local antigo:
    id_local mantido}, usado para redirecionar as chaves da fNotificacao."""
    d = d.copy()
    d["_uf_correta"] = d["codigo_uf"] == d["cod_uf"]
    ordem = d.sort_values(["id_municipio", "_uf_correta", "id_local"], ascending=[True, False, True])
    mantido = ordem.groupby("id_municipio")["id_local"].transform("first")
    mapa = dict(zip(ordem["id_local"], mantido))

    unicos = ordem[ordem["id_local"] == mantido].drop(columns="_uf_correta")
    unicos["codigo_uf"] = unicos["cod_uf"]
    unicos = unicos.sort_values("id_local").reset_index(drop=True)
    print(f"    Locais duplicados removidos: {len(d) - len(unicos)} "
          f"({len(d)} -> {len(unicos)} municípios)")
    return unicos, mapa


def ler_indicadores_atlas():
    colunas = {
        "ANO": "ano", "Codmun7": "id_municipio",
        "ESPVIDA": "expectativa_vida",   # esperança de vida ao nascer (anos)
        "GINI": "indice_gini",           # índice de Gini
        "PMPOB": "prop_pobreza",         # % de pobres
        "RDPC": "renda_pc",              # renda per capita (R$)
        "pesotot": "populacao_atlas",    # população total
        "IDHM": "idhm", "IDHM_R": "idhm_r",
    }
    a = pd.read_excel(ARQ_ATLAS, sheet_name="MUN 91-00-10", usecols=list(colunas)).rename(columns=colunas)
    a = a[a["ano"] == ANO_ATLAS].drop(columns="ano")
    for col, casas in [("expectativa_vida", 2), ("indice_gini", 2), ("prop_pobreza", 2),
                       ("renda_pc", 2), ("idhm", 3), ("idhm_r", 3)]:
        a[col] = a[col].round(casas)
    a["populacao_atlas"] = a["populacao_atlas"].astype("Int64")
    return a


def _letra_coluna_excel(i):
    s, n = "", i + 1
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


# Cobertura APS que falta na planilha do SINAN-TB, preenchida a partir da fonte oficial.
# Fonte: e-Gestor Atenção Primária à Saúde - Cobertura APS
#        (https://relatorioaps.saude.gov.br/cobertura/aps), competência 07/2021,
#        a mesma competência indicada no dicionário de dados da planilha.
# Só é aplicada quando o valor está ausente (não sobrescreve dados da planilha).
COBERTURA_APS_MANUAL = {
    3129509: 93.16,   # Ibiá (MG) - competência 07/2021
}


# Indicadores do Atlas para municípios instalados depois do Censo 2010 (sem valor oficial).
# Usam-se os valores de 2010 do município de origem, do qual foram desmembrados,
# já que em 2010 o território deles fazia parte desse município.
# Fonte: Atlas Brasil, censo_total_1991_2010.xlsx, ano 2010.
# populacao_atlas fica vazia (a população do município de origem não representa o novo).
# Só é aplicado quando o valor está ausente.
ATLAS_MANUAL = {
    # Balneário Rincão (SC): desmembrado de Içara (SC, 4207007), instalado em 2013
    4220000: {"expectativa_vida": 76.67, "indice_gini": 0.40, "prop_pobreza": 2.90,
              "renda_pc": 759.54, "idhm": 0.741, "idhm_r": 0.732},
    # Mojuí dos Campos (PA): desmembrado de Santarém (PA, 1506807), instalado em 2013
    1504752: {"expectativa_vida": 73.44, "indice_gini": 0.58, "prop_pobreza": 31.07,
              "renda_pc": 409.07, "idhm": 0.691, "idhm_r": 0.632},
}


def ler_cobertura_aps():
    """Lê só as colunas codigo_ibge, pc_cobertura_aps_2021 e pc_cobertura_sf_2019.
    A planilha tem ~3 GB descompactada, então o XML é lido em blocos em vez de abrir
    a planilha inteira com o pandas (que levaria muito tempo e memória)."""
    alvo = ["codigo_ibge", "pc_cobertura_aps_2021", "pc_cobertura_sf_2019"]
    cab = pd.read_excel(ARQ_PACIENTES, nrows=0).columns.tolist()
    letra_para_nome = {_letra_coluna_excel(cab.index(c)): c for c in alvo}
    padrao = re.compile(
        r'<c r="(' + "|".join(letra_para_nome) + r')(\d+)"[^>]*?'
        r'(?:/>|>(?:<v>([^<]*)</v>|<is><t[^>]*>([^<]*)</t></is>)?</c>)')

    linhas = {}
    with zipfile.ZipFile(ARQ_PACIENTES) as z, z.open("xl/worksheets/sheet1.xml") as f:
        resto = ""
        while True:
            bloco = f.read(64 * 1024 * 1024)
            texto = resto + bloco.decode("utf-8", errors="ignore")
            corte = texto.rfind("</row>") + 6 if bloco else len(texto)
            for col, lin, v, t in padrao.findall(texto[:corte]):
                if lin != "1":
                    linhas.setdefault(lin, {})[letra_para_nome[col]] = html.unescape(v or t)
            resto = texto[corte:]
            if not bloco:
                break

    c = pd.DataFrame.from_dict(linhas, orient="index")
    for col in alvo:
        c[col] = pd.to_numeric(c[col], errors="coerce")
    c = c.dropna(subset=["codigo_ibge"])
    c["id_municipio"] = c["codigo_ibge"].astype("int64")
    # a cobertura é do município (mesmo valor em todas as notificações dele): 1 linha por município
    c = c.groupby("id_municipio", as_index=False)[["pc_cobertura_aps_2021", "pc_cobertura_sf_2019"]].first()
    return c.rename(columns={"pc_cobertura_aps_2021": "cobertura_aps_2021",
                             "pc_cobertura_sf_2019": "cobertura_esf_2019"})


def classificar_em_faixas(serie, cortes, nomes):
    def f(x):
        if pd.isna(x):
            return "Ignorado"
        for c, n in zip(cortes, nomes):
            if x < c:
                return n
        return nomes[-1]
    return serie.map(f)


def montar_dimensao_local():
    print("[*] Lendo dLocal original...")
    d = ler_dimensao_local_original()
    d, mapa_locais = remover_municipios_duplicados(d)
    print(f"[*] Lendo Atlas Brasil ({ANO_ATLAS})...")
    atlas = ler_indicadores_atlas()
    print("[*] Lendo cobertura da APS (leva cerca de 2 minutos)...")
    cob = ler_cobertura_aps()

    d = d.merge(atlas, on="id_municipio", how="left")
    d = d.merge(cob, on="id_municipio", how="left")
    manual = d["id_municipio"].map(COBERTURA_APS_MANUAL)
    preencher = d["cobertura_aps_2021"].isna() & manual.notna()
    d.loc[preencher, "cobertura_aps_2021"] = manual[preencher]
    if preencher.any():
        print(f"    Cobertura APS preenchida manualmente (e-Gestor): "
              f"{', '.join(d.loc[preencher, 'municipio_uf'])}")

    for cod, valores in ATLAS_MANUAL.items():
        linhas = (d["id_municipio"] == cod) & d["idhm"].isna()
        for col, valor in valores.items():
            d.loc[linhas, col] = valor
        if linhas.any():
            print(f"    Atlas preenchido com o município de origem: "
                  f"{', '.join(d.loc[linhas, 'municipio_uf'])}")

    d["Faixa_IDHM"] = classificar_em_faixas(d["idhm"], [.5, .6, .7, .8, np.inf],
                            ["Muito baixo", "Baixo", "Médio", "Alto", "Muito alto"])
    pob = ["Baixa (<10%)", "Média (10–19%)", "Alta (20–39%)", "Muito alta (≥40%)"]
    d["Faixa_Pobreza"] = classificar_em_faixas(d["prop_pobreza"], [10, 20, 40, np.inf], pob)
    d["Faixa_Pobreza_Ordem"] = d["Faixa_Pobreza"].map({n: i + 1 for i, n in enumerate(pob)}).fillna(5).astype(int)
    ren = ["Muito baixa", "Baixa", "Média", "Alta"]
    d["Faixa_Renda_PC"] = classificar_em_faixas(d["renda_pc"], [100, 500, 1000, np.inf], ren)
    d["Faixa_Renda_PC_Ordem"] = d["Faixa_Renda_PC"].map({n: i + 1 for i, n in enumerate(ren)}).fillna(5).astype(int)
    d["Faixa_Gini"] = classificar_em_faixas(d["indice_gini"], [.40, .50, np.inf],
                            ["Baixa desigualdade", "Desigualdade média", "Alta desigualdade"])

    d = d[["id_local", "codigo_uf", "id_municipio", "municipio_uf", "id_municipio_nome",
           "expectativa_vida", "indice_gini", "prop_pobreza", "renda_pc", "populacao_atlas",
           "idhm", "idhm_r", "populacao", "faixa_populacao", "sigla_uf", "cod_uf", "regiao",
           "Estado_Pais", "Estado_Pais_Completo", "Faixa_IDHM", "Faixa_Pobreza",
           "Faixa_Pobreza_Ordem", "Faixa_Renda_PC", "Faixa_Renda_PC_Ordem", "Faixa_Gini",
           "cobertura_aps_2021", "cobertura_esf_2019"]]

    print(f"    {len(d)} locais | sem Atlas: {d['idhm'].isna().sum()} | "
          f"sem cobertura APS: {d['cobertura_aps_2021'].isna().sum()}")
    return d, mapa_locais


def construir_base_dados():
    db = ConexaoBanco()
    print("[*] Carregando a dTempo...")
    carregar_dimensao_tempo(db)
    dlocal, mapa_locais = montar_dimensao_local()
    print("[*] Carregando a fNotificacao...")
    carregar_notificacoes(db, mapa_locais)
    db.gravar_tabela(dlocal, "dlocal")


if __name__ == "__main__":
    construir_base_dados()
