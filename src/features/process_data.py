"""
Construção da ABT (Analytical Base Table) da população de estudo.

A ABT tem uma linha por notificação e guarda os dados LIMPOS, mas NÃO
transformados para o modelo. Aqui só entram transformações determinísticas,
que dependem apenas da própria linha:
  - aplicação dos critérios de seleção da população de estudo;
  - padronização dos rótulos e envio de códigos inválidos para "Ignorado";
  - conversão dos indicadores municipais para número;
  - criação do alvo binário (1 = perda de seguimento, 0 = cura).

Ficam FORA da ABT, dentro do Pipeline de treino (ajustados só com o conjunto
de treino, para não vazar informação do teste):
  one-hot encoding, normalização, imputação de faltantes e balanceamento.

Saídas:
  - tabela tb_abt_tuberculose no MySQL (substituída a cada execução)
  - data/processed/abt_tuberculose.csv (cópia congelada para reprodutibilidade)

Uso (dentro da pasta ia):  python src/features/process_data.py   ou   python main.py
"""
from datetime import datetime

import pandas as pd
from src.db_manager import ConexaoBanco
from src.config import DATA_PROCESSED

TABELA_ABT = "tb_abt_tuberculose"
ARQ_ABT = DATA_PROCESSED / "abt_tuberculose.csv"
IGNORADO = "Ignorado"

# Mesmos critérios de seleção do fluxograma (sql/vw_populacao_estudo.sql).
# Obs.: não usar '%' (LIKE) — o pymysql interpreta '%' como parâmetro.
QUERY = """
    SELECT
        f.id_notificacao,
        t.ano_notificacao,
        l.id_municipio            AS id_municipio_residencia,
        -- características individuais
        f.faixa_etaria, f.sexo, f.teste_hiv, f.alcool, f.tabagismo,
        f.drogas, f.escolaridade, f.raca,
        -- características do município de residência
        l.idhm, l.indice_gini, l.faixa_populacao, l.cobertura_aps_2021, l.regiao,
        -- desfecho
        f.situacao_encerramento
    FROM casos_tuberculose.fnotificacao f
    LEFT JOIN casos_tuberculose.dlocal l ON l.id_local = f.id_ocal_esidencia
    LEFT JOIN casos_tuberculose.dtempo t ON t.id_tempo = f.id_tempo
    WHERE f.faixa_etaria != '<= 18 anos'
      AND f.imigrante != 'Sim'
      AND f.sem_teto != 'Sim'
      AND f.profissional_saude != 'Sim'
      AND f.encarcerado != 'Sim'
      AND (INSTR(f.gestante, 'Não') > 0 OR f.gestante = 'Ignorado')
      AND f.forma_clinica = 'Pulmonar'
      AND f.teste_sensibilidade NOT IN ('1', 'Resistente somente à Rifampicina',
            'Resistente à Isoniazida e Rifampicina', 'Resistente a outras drogas de 1ª linha')
      AND f.tempo_entre_encerramento_e_inicio_tratamento IS NOT NULL
      AND f.situacao_encerramento IN ('Cura', 'Abandono', 'Abandono Primário')
"""

# Categorias válidas de cada variável (dicionário do SINAN). Qualquer outro valor
# (códigos soltos como '3' ou '9', nulos, rótulos corrompidos) vira "Ignorado".
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
# correções de grafia antes da validação
CORRECOES = {
    "escolaridade": {"ensino superior": "Ensino superior"},
}
NUMERICAS = ["idhm", "indice_gini", "cobertura_aps_2021"]
PERDA_SEGUIMENTO = ["Abandono", "Abandono Primário"]


def limpar(df):
    df = df.copy()

    for col, validas in CATEGORIAS.items():
        s = df[col].astype("string").str.strip().replace(CORRECOES.get(col, {}))
        df[col] = s.where(s.isin(validas), IGNORADO).fillna(IGNORADO)

    # indicadores municipais como número (aceita vírgula decimal, caso a dlocal
    # tenha sido carregada como texto); faltantes ficam NaN para o pipeline imputar
    for col in NUMERICAS:
        df[col] = pd.to_numeric(df[col].astype("string").str.replace(",", ".", regex=False),
                                errors="coerce")

    df["alvo_preditivo"] = df["situacao_encerramento"].isin(PERDA_SEGUIMENTO).astype(int)
    df = df.drop(columns="situacao_encerramento")   # o desfecho original não pode virar variável
    return df


def preparar_dados_ml():
    db = ConexaoBanco()

    print("[*] Extraindo a população de estudo do banco...")
    df = db.obter_dataframe(QUERY)
    if df is None or df.empty:
        print("[X] Erro: consulta sem dados.")
        return None

    print("[*] Limpando e padronizando...")
    abt = limpar(df)

    # resumo
    n, n_ab = len(abt), int(abt["alvo_preditivo"].sum())
    print(f"\n    {n} notificações | perda de seguimento: {n_ab} ({n_ab / n * 100:.1f}%)")
    ign = {c: int((abt[c] == IGNORADO).sum()) for c in CATEGORIAS if (abt[c] == IGNORADO).any()}
    print(f"    'Ignorado' por variável: {ign}")
    nul = abt[NUMERICAS].isna().sum()
    print(f"    Indicadores municipais faltantes: {nul[nul > 0].to_dict() or 'nenhum'}")

    # banco + cópia congelada
    db.enviar_dataframe(abt, TABELA_ABT, modo="replace")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    abt.assign(data_extracao=datetime.now().strftime("%Y-%m-%d %H:%M")).to_csv(
        ARQ_ABT, sep=";", index=False, encoding="utf-8")
    print(f"[+] Cópia salva em '{ARQ_ABT}'.")
    return abt


if __name__ == "__main__":
    preparar_dados_ml()
