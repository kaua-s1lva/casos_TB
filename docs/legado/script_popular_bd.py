# etl_sqlalchemy.py
"""
ETL para carregar amostralDados.xlsx para PostgreSQL usando SQLAlchemy.
- Normaliza/valida tipos
- Garante limites SMALLINT
- Trata datas com pd.to_datetime(..., errors='coerce')
- Compatível com schema existente (usa id_tempo_notificao conforme sua tabela)
- Remove duplicidades na fato (dentro do próprio arquivo), mantendo padrão ETL
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, MetaData, Table, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine.url import URL

# -----------------------------
# CONFIGURAÇÃO DO BANCO
# -----------------------------
DB_CONFIG = {
    'drivername': 'mysql+pymysql',
    'username': 'root',
    'password': 'root',
    'host': 'localhost',
    'port': '3306',
    'database': 'casos_tuberculose'
}

# Constrói URL e engine
engine = create_engine(URL.create(**DB_CONFIG), echo=False, future=True)
metadata = MetaData()

# Define tabelas para refletir
dm_local = Table('dm_local', metadata, autoload_with=engine)
dm_tempo = Table('dm_tempo', metadata, autoload_with=engine)
fato_notificacao = Table('fato_notificacao', metadata, autoload_with=engine)

# -----------------------------
# UTILITÁRIOS E VALIDAÇÕES
# -----------------------------
SMALLINT_MIN = -32768
SMALLINT_MAX = 32767

def norm_val(v):
    """Normaliza valores para chaves: converte floats inteiros para int, strip, uppercase; retorna '' para nulos."""
    if pd.isna(v):
        return ''
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    s = str(v).strip()
    return s.upper()

def to_int(x, default=None):
    """
    Converte x para int.
    Se não for conversível, retorna default.
    """
    if pd.isna(x) or x is None:
        return default

    s = str(x).strip()
    if s == '' or s.upper() == '#NULL!':
        return default

    if '.' in s:
        f = float(s)
        i = int(f)
    else:
        i = int(s)

    return i

def to_smallint(x, default=None, allow_negative=False):
    """
    Converte x para int dentro do intervalo SMALLINT.
    Se não for conversível ou exceder limites, lança Exception ou retorna default.
    """
    if pd.isna(x) or x is None:
        return default
    
    s = str(x).strip()
    if s == '' or s.upper() == '#NULL!':
        return default

    if '.' in s:
        f = float(s)
        i = int(f)
    else:
        i = int(s)

    if (not allow_negative and i < 0) or (i < SMALLINT_MIN or i > SMALLINT_MAX):
        raise Exception("Erro no range de valor do SMALLINT")

    return i

def safe_parse_date(value):
    """Parse para Timestamp pandas; retorna pd.Timestamp ou None."""
    if pd.isna(value):
        return None
    try:
        ts = pd.to_datetime(value, errors='coerce')
        if pd.isna(ts):
            return None
        return ts
    except Exception:
        return None

# -----------------------------
# FUNÇÕES DE MAPEAMENTO
# -----------------------------
def tratar_valor_nulo_nan(valor, campo):
    if pd.isna(valor) or valor == '#NULL!' or valor is None or str(valor).strip() == '':
        if campo in ['sexo', 'forma_clinica', 'faixa_etaria', 'raca', 'gestante', 
                    'escolaridade', 'tipo_notificacao', 'situacao_encerramento', 'tratamento']:
            return 9
        elif campo in ['aids', 'alcool', 'diabetes', 'doenca_mental', 'drogas', 'tabagismo',
                      'teste_hiv', 'resultado_teste_rapido', 'raio_x_torax', 'teste_sensibilidade',
                      'historico_patologico', 'beneficio_governo', 'sem_teto', 'encarcerado',
                      'imigrante', 'profissional_saude', 'extrapulmonar_1', 'extrapulmonar_2',
                      'supervisionado_no_tratamento']:
            return 9
        elif campo == 'encerramento_abandono':
            return 0
        else:
            return None
    else:
        return valor

def calcular_faixa_etaria(ano_nasc, ano_notificacao):
    ano_nasc = tratar_valor_nulo_nan(ano_nasc, 'ano_nasc')
    ano_notificacao = tratar_valor_nulo_nan(ano_notificacao, 'ano_notificacao')
    if ano_nasc is None or ano_notificacao is None:
        return 9
    try:
        idade = int(ano_notificacao) - int(ano_nasc)
        if idade <= 18:
            return 1
        elif 19 <= idade <= 29:
            return 2
        elif 30 <= idade <= 39:
            return 3
        elif 40 <= idade <= 49:
            return 4
        elif 50 <= idade <= 59:
            return 5
        else:
            return 6
    except Exception:
        return 9

def calcular_tempo_entre_datas(data1, data2):
    d1 = safe_parse_date(data1)
    d2 = safe_parse_date(data2)
    if d1 is None or d2 is None:
        return None
    try:
        diff = (d2 - d1).days
        return diff
    except Exception:
        return None

def mapear_sexo(cs_sexo):
    """
    Mapeia sexo para códigos:
    1 = FEMININO
    2 = MASCULINO
    3 = IGNORADO / NÃO INFORMADO / QUALQUER OUTRO
    """
    v = tratar_valor_nulo_nan(cs_sexo, 'sexo')

    # 1) Nulo depois do tratamento -> IGNORADO
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return 3

    # 2) Se vier como string
    if isinstance(v, str):
        s = v.strip().upper()

        # casos diretos
        if s == 'M':
            return 2
        if s == 'F':
            return 1

        # alguns valores que podem aparecer em base SUS:
        if s in ('1', 'FEMININO'):
            return 1
        if s in ('2', 'MASCULINO'):
            return 2

        # qualquer coisa diferente (I, IGN, N, NI, etc.) -> categoria “ignorado”
        return 3

    # 3) Se vier numérico (1, 2, 3...)
    try:
        iv = int(v)
    except Exception:
        return 3

    if iv in (1, 2, 3):
        return iv

    # default seguro
    return 3


def mapear_escolaridade(cs_escol_n):
    v = tratar_valor_nulo_nan(cs_escol_n, 'escolaridade')
    if v is None:
        return 10
    try:
        iv = int(v)
    except Exception:
        return 10
    if iv == 0:
        return 0
    if iv in [1, 2, 3, 4]:
        return 1
    if iv in [5, 6]:
        return 2
    if iv in [7, 8]:
        return 3
    if iv == 9:
        return 9
    return 10

def mapear_raca(cs_raca):
    v = tratar_valor_nulo_nan(cs_raca, 'raca')
    if v is None:
        return 10
    try:
        iv = int(v)
    except Exception:
        return 10
    if iv == 1:
        return 1
    if iv == 2:
        return 2
    if iv in [3, 5]:
        return 3
    if iv == 4:
        return 4
    if iv == 9:
        return 9
    return 10

def mapear_gestante(cs_gestant):
    v = tratar_valor_nulo_nan(cs_gestant, 'gestante')
    if v is None:
        return 9
    try:
        iv = int(v)
    except Exception:
        return 9
    if iv in [1, 2, 3, 4]:
        return iv
    if iv == 5:
        return 5
    if iv == 6:
        return 6
    return 9


# -----------------------------
# FUNÇÕES DE CARREGAMENTO TABELA ATLAS
# -----------------------------

ATLAS_CSV_PATH = 'atlas.csv'

def carregar_atlas(path=ATLAS_CSV_PATH):
    """
    Carrega atlas.csv, garante id_municipio com 7 dígitos
    e converte os indicadores.
    """
    atlas = pd.read_csv(path, sep=';', encoding='latin1')

    # >>> IMPORTANTE: NÃO cortar o último dígito aqui <<<
    atlas['id_municipio'] = (
        atlas['id_municipio']
        .astype(str)
        .str.strip()
        .str.zfill(7)    # garante 7 dígitos com zero à esquerda se precisar
    )

    def conv(v, fator):
        if pd.isna(v):
            return None
        try:
            return float(v) / fator
        except Exception:
            return None

    atlas['expectativa_vida'] = atlas['expectativa_vida'].apply(lambda x: conv(x, 100))
    atlas['indice_gini']      = atlas['indice_gini'].apply(lambda x: conv(x, 100))
    atlas['prop_pobreza']     = atlas['prop_pobreza'].apply(lambda x: conv(x, 100))
    atlas['renda_pc']         = atlas['renda_pc'].apply(lambda x: conv(x, 100))
    atlas['idhm']             = atlas['idhm'].apply(lambda x: conv(x, 1000))
    atlas['idhm_r']           = atlas['idhm_r'].apply(lambda x: conv(x, 1000))

    atlas = atlas.rename(columns={
        'id_municipio_nome': 'nome_municipio',
        'populacao': 'populacao_atlas'
    })

    return atlas[
        [
            'id_municipio',
            'nome_municipio',
            'expectativa_vida',
            'indice_gini',
            'prop_pobreza',
            'renda_pc',
            'populacao_atlas',
            'idhm',
            'idhm_r'
        ]
    ]

POP_CSV_PATH = 'Base_MUNIC_2024_20251107.csv'  # ajuste o nome se for outro

def carregar_populacao(path=POP_CSV_PATH):
    """
    Carrega CSV de população / faixa_populacao / regiao por município.
    Espera colunas:
      - id_municipio;sigla_uf;cod_uf;população;faixa_populacao;regiao
    """
    pop = pd.read_csv(path, sep=';', encoding='utf-8')

    # garante string limpinha
    pop['id_municipio'] = (
        pop['id_municipio']
        .astype(str)
        .str.strip()
        .str.zfill(7)   # 7 dígitos, igual ao Atlas
    )

    # deixa só o que interessa (mantém sigla_uf se quiser também)
    # cols = [c for c in ['id_municipio', 'sigla_uf' 'cod_uf', 'populacao', 'faixa_populacao', 'regiao'] if c in pop.columns]
    # pop = pop[cols]

    return pop


# -----------------------------
# ETL: leitura e transformação
# -----------------------------
def processar_dados(df: pd.DataFrame, atlas_df: pd.DataFrame, pop_df: pd.DataFrame):
    # ------------------------------------
    # 0) MAPA 6 → 7 DÍGITOS A PARTIR DO ATLAS
    # ------------------------------------
    # Garante string e 7 dígitos no Atlas
    atlas_df = atlas_df.copy()
    atlas_df['id_municipio'] = (
        atlas_df['id_municipio']
        .astype(str)
        .str.strip()
        .str.zfill(7)
    )

    # Cria um dicionário: prefixo de 6 dígitos -> código completo (7 dígitos)
    map6_to7 = {}
    for v in atlas_df['id_municipio'].dropna():
        v = str(v).strip()
        if not v:
            continue
        k6 = v[:6]          # prefixo (ex.: "110032")
        map6_to7[k6] = v    # mapeia "110032" -> "1100320"

    # ------------------------------------
    # 1) Dimensão Local
    # ------------------------------------
    # NOTIFICAÇÃO
    #locais_notificacao = df[['sg_uf_not', 'codigo_ibge', 'municipio_UF', 'id_regiona']].drop_duplicates()
    locais_notificacao = df[['sg_uf_not', 'codigo_ibge', 'municipio_UF']].drop_duplicates()
    locais_notificacao = locais_notificacao.rename(columns={
        'sg_uf_not': 'codigo_uf',
        'codigo_ibge': 'id_municipio',
        # 'id_regiona': 'id_regional'
    })

    # Normaliza e converte para 7 dígitos usando o mapa 6→7
    def normalizar_id_municipio_not(v):
        if pd.isna(v):
            return None
        s = str(v).strip()
        if not s:
            return None
        k6 = s[:6]                   # pega sempre os 6 primeiros
        return map6_to7.get(k6, s)   # se tiver no Atlas, usa o 7 dígitos, senão mantém

    locais_notificacao['id_municipio'] = locais_notificacao['id_municipio'].apply(normalizar_id_municipio_not)

    # RESIDÊNCIA
    # locais_residencia = df[['sg_uf', 'id_mn_resi', 'municipio_UF', 'id_rg_resi']].drop_duplicates()
    locais_residencia = df[['sg_uf', 'id_mn_resi', 'municipio_UF']].drop_duplicates()
    locais_residencia = locais_residencia.rename(columns={
        'sg_uf': 'codigo_uf',
        'id_mn_resi': 'id_municipio',
        # 'id_rg_resi': 'id_regional'
    })

    def normalizar_id_municipio_res(v):
        if pd.isna(v):
            return None
        s = str(v).strip()
        if not s:
            return None
        s6 = s[:6].zfill(6)          # garante 6 dígitos no prefixo
        return map6_to7.get(s6, s6)  # tenta achar no Atlas, senão fica com os 6 dígitos mesmo

    locais_residencia['id_municipio'] = locais_residencia['id_municipio'].apply(normalizar_id_municipio_res)

    # Junta todos os locais (notificação + residência)
    todos_locais = pd.concat([locais_notificacao, locais_residencia], ignore_index=True).drop_duplicates()

    # garante string
    todos_locais['id_municipio'] = todos_locais['id_municipio'].astype(str).str.strip()

    # ----- JOIN COM O ATLAS (agora 7 dígitos combinam direto) -----
    todos_locais = todos_locais.merge(
        atlas_df,
        on='id_municipio',
        how='left'
    )

    # ===== NOVO: MERGE COM CSV DE POPULAÇÃO / REGIÃO =====
    pop_df = pop_df.copy()
    pop_df['id_municipio'] = (
        pop_df['id_municipio']
        .astype(str)
        .str.strip()
        .str.zfill(7)
    )

    todos_locais = todos_locais.merge(
        pop_df,
        on='id_municipio',
        how='left'   # mantém todos os locais, mesmo se não tiver no CSV
    )

    # ------------------------------------
    # 2) Dimensão Tempo
    # ------------------------------------
    tempos = df[['dt_notific']].drop_duplicates()
    tempos = tempos[tempos['dt_notific'].notna()]
    tempos = tempos[tempos['dt_notific'] != '#NULL!'].copy()

    tempos['ts'] = pd.to_datetime(tempos['dt_notific'], errors='coerce')
    tempos = tempos[tempos['ts'].notna()].copy()
    tempos['data_notificacao'] = tempos['ts'].dt.date
    tempos['dia_notificacao'] = tempos['ts'].dt.day.astype('Int64')
    tempos['mes_notificacao'] = tempos['ts'].dt.month.astype('Int64')
    tempos['ano_notificacao'] = tempos['ts'].dt.year.astype('Int64')
    tempos['trimestre_notificacao'] = tempos['ts'].dt.quarter.astype('Int64')

    # ------------------------------------
    # 3) Fato
    # ------------------------------------
    fato_data = []
    for _, row in df.iterrows():
        faixa = calcular_faixa_etaria(row.get('ano_nasc'), row.get('nu_ano'))

        t_nd = calcular_tempo_entre_datas(row.get('dt_diag'), row.get('dt_notific'))
        t_in = calcular_tempo_entre_datas(row.get('dt_notific'), row.get('dt_inic_tr'))
        t_ei = calcular_tempo_entre_datas(row.get('dt_inic_tr'), row.get('dt_encerra'))

        sexo = mapear_sexo(row.get('cs_sexo'))
        forma_clinica = to_smallint(tratar_valor_nulo_nan(row.get('forma'), 'forma_clinica'), default=9)
        aids = to_smallint(tratar_valor_nulo_nan(row.get('agravaids'), 'aids'), default=9)
        alcool = to_smallint(tratar_valor_nulo_nan(row.get('agravalcoo'), 'alcool'), default=9)
        diabetes = to_smallint(tratar_valor_nulo_nan(row.get('agravdiabe'), 'diabetes'), default=9)
        doenca_mental = to_smallint(tratar_valor_nulo_nan(row.get('agravdoenc'), 'doenca_mental'), default=9)
        drogas = to_smallint(tratar_valor_nulo_nan(row.get('agravdroga'), 'drogas'), default=9)
        tabagismo = to_smallint(tratar_valor_nulo_nan(row.get('agravtabac'), 'tabagismo'), default=9)
        teste_hiv = to_smallint(tratar_valor_nulo_nan(row.get('hiv'), 'teste_hiv'), default=9)
        resultado_teste_rapido = to_smallint(tratar_valor_nulo_nan(row.get('test_molec'), 'resultado_teste_rapido'), default=9)
        raio_x_torax = to_smallint(tratar_valor_nulo_nan(row.get('raiox_tora'), 'raio_x_torax'), default=9)
        teste_sensibilidade = to_smallint(tratar_valor_nulo_nan(row.get('test_sensi'), 'teste_sensibilidade'), default=9)
        historico_patologico = to_smallint(tratar_valor_nulo_nan(row.get('histopatol'), 'historico_patologico'), default=9)
        raca = mapear_raca(row.get('cs_raca'))
        gestante = mapear_gestante(row.get('cs_gestant'))
        escolaridade = mapear_escolaridade(row.get('cs_escol_n'))
        beneficio_governo = to_smallint(tratar_valor_nulo_nan(row.get('benef_gov'), 'beneficio_governo'), default=9)
        sem_teto = to_smallint(tratar_valor_nulo_nan(row.get('pop_rua'), 'sem_teto'), default=9)
        encarcerado = to_smallint(tratar_valor_nulo_nan(row.get('pop_liber'), 'encarcerado'), default=9)
        imigrante = to_smallint(tratar_valor_nulo_nan(row.get('pop_imig'), 'imigrante'), default=9)
        profissional_saude = to_smallint(tratar_valor_nulo_nan(row.get('pop_saude'), 'profissional_saude'), default=9)
        tipo_notificacao = to_smallint(tratar_valor_nulo_nan(row.get('tp_not'), 'tipo_notificacao'), default=9)
        extrapulmonar_1 = to_smallint(tratar_valor_nulo_nan(row.get('extrapu1_n'), 'extrapulmonar_1'), default=9)
        extrapulmonar_2 = to_smallint(tratar_valor_nulo_nan(row.get('extrapu2_n'), 'extrapulmonar_2'), default=9)
        situacao_encerramento = to_smallint(tratar_valor_nulo_nan(row.get('situa_ence'), 'situacao_encerramento'), default=9)
        supervisionado_no_tratamento = to_smallint(tratar_valor_nulo_nan(row.get('trat_super'), 'supervisionado_no_tratamento'), default=9)
        tratamento = to_smallint(tratar_valor_nulo_nan(row.get('tratamento'), 'tratamento'), default=9)
        encerramento_abandono = to_smallint(tratar_valor_nulo_nan(row.get('encerramento_abandono'), 'encerramento_abandono'), default=0)

        id_regional_not = to_int(row.get('id_regiona'))
        id_regional_res = to_int(row.get('id_rg_resi'))

        # --- IDs de município para a FATO usando o mesmo mapa 6→7 ---

        # município de notificação (codigo_ibge)
        codigo_ibge = row.get('codigo_ibge')
        if pd.isna(codigo_ibge):
            id_municip_not = None
        else:
            s = str(codigo_ibge).strip()
            k6 = s[:6]
            id_municip_not = map6_to7.get(k6, k6)

        # município de residência (id_mn_resi)
        id_municip_res_raw = row.get('id_mn_resi')
        if pd.isna(id_municip_res_raw):
            id_municip_res = None
        else:
            s = str(id_municip_res_raw).strip()
            k6 = s[:6].zfill(6)
            id_municip_res = map6_to7.get(k6, k6)

        fato_data.append({
            'sexo': sexo,
            'forma_clinica': forma_clinica,
            'aids': aids,
            'alcool': alcool,
            'diabetes': diabetes,
            'doenca_mental': doenca_mental,
            'drogas': drogas,
            'tabagismo': tabagismo,
            'teste_hiv': teste_hiv,
            'resultado_teste_rapido': resultado_teste_rapido,
            'raio_x_torax': raio_x_torax,
            'teste_sensibilidade': teste_sensibilidade,
            'historico_patologico': historico_patologico,
            'faixa_etaria': to_smallint(faixa, default=9),
            'raca': raca,
            'gestante': gestante,
            'escolaridade': escolaridade,
            'beneficio_governo': beneficio_governo,
            'sem_teto': sem_teto,
            'encarcerado': encarcerado,
            'imigrante': imigrante,
            'profissional_saude': profissional_saude,
            'tipo_notificacao': tipo_notificacao,
            'extrapulmonar_1': extrapulmonar_1,
            'extrapulmonar_2': extrapulmonar_2,
            'situacao_encerramento': situacao_encerramento,
            'supervisionado_no_tratamento': supervisionado_no_tratamento,
            'tratamento': tratamento,
            'encerramento_abandono': encerramento_abandono,
            'tempo_entre_notificacao_e_diagnostico': t_nd,
            'tempo_entre_inicio_tramamento_e_notificacao': t_in,
            'tempo_entre_encerramento_e_inicio_tratamento': t_ei,

            'id_regional_notificacao': id_regional_not,
            'id_regional_residencia': id_regional_res,

            # dados para junção (agora com o mesmo padrão usado em dm_local)
            'sg_uf_not': row.get('sg_uf_not'),
            'id_municip_not': id_municip_not,
            'sg_uf_res': row.get('sg_uf'),
            'id_municip_res': id_municip_res,
            'dt_notific': row.get('dt_notific')
        })

    fato_df = pd.DataFrame(fato_data)

    # ----- REMOÇÃO DE DUPLICIDADES NA FATO -----
    qtd_before = len(fato_df)
    fato_df = fato_df.drop_duplicates()
    qtd_after = len(fato_df)
    print(f"Fato: removidas {qtd_before - qtd_after} linhas duplicadas dentro do arquivo (restaram {qtd_after}).")

    return todos_locais, tempos, fato_df

# -----------------------------
# INSERÇÃO NO BANCO
# -----------------------------
LOG_ARQUIVO = "etl_erros.log"

def registrar_erro(msg):
    """Salva mensagens de erro em etl_erros.log"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    linha = f"[{timestamp}] {msg}\n"
    with open(LOG_ARQUIVO, "a", encoding="utf-8") as f:
        f.write(linha)

def inserir_dados(engine, locais_df, tempos_df, fato_df, batch_size=500):
    conn = engine.connect()
    trans = conn.begin()
    try:
        # 1) dm_local
        locais_dict = {}
        for _, local in locais_df.iterrows():
            codigo = norm_val(local['codigo_uf'])
            id_mun = norm_val(local['id_municipio'])
            municipio_uf = local.get('municipio_UF')
            # id_regional = local.get('id_regional')

            sigla_uf = norm_val(local.get('sigla_uf'))
            cod_uf = norm_val(local.get('cod_uf'))

            chave = f"{codigo}_{id_mun}"
            if chave in locais_dict:
                continue

            sel = select(dm_local.c.id).where(
                (dm_local.c.codigo_uf == (codigo or None)) &
                (dm_local.c.id_municipio == (id_mun or None))
            )
            res = conn.execute(sel).fetchone()
            if res:
                locais_dict[chave] = res[0]
                continue

            ins = dm_local.insert().values(
                codigo_uf       = codigo or None,
                id_municipio   = id_mun or None,
                municipio_uf   = municipio_uf if not pd.isna(municipio_uf) else None,
                # id_regional    = to_int(id_regional if not pd.isna(id_regional) else None),

                # Dados da tabela do Antonio
                populacao =to_int(local.get("populacao")),
                faixa_populacao = local.get('faixa_populacao'),
                sigla_uf = sigla_uf if not pd.isna(sigla_uf) else None,
                cod_uf = cod_uf if not pd.isna(cod_uf) else None,
                regiao = local.get('regiao'),

                # Campos vindos do Atlas (já vêm como float / int do merge)
                id_municipio_nome = local.get('nome_municipio'),
                expectativa_vida  = local.get('expectativa_vida'),
                indice_gini       = local.get('indice_gini'),
                prop_pobreza      = local.get('prop_pobreza'),
                renda_pc          = local.get('renda_pc'),
                populacao_atlas         = to_int(local.get('populacao_atlas')),
                idhm              = local.get('idhm'),
                idhm_r            = local.get('idhm_r')
            )
            r = conn.execute(ins)
            locais_dict[chave] = r.inserted_primary_key[0]

        # 2) dm_tempo
        tempos_dict = {}
        for _, tempo in tempos_df.iterrows():
            data_not = tempo['data_notificacao']
            chave_t = str(data_not)
            if chave_t in tempos_dict:
                continue

            sel = select(dm_tempo.c.id).where(dm_tempo.c.data_notificacao == data_not)
            res = conn.execute(sel).fetchone()
            if res:
                tempos_dict[chave_t] = res[0]
                continue

            ins = dm_tempo.insert().values(
                data_notificacao = data_not,
                dia_notificacao = int(tempo['dia_notificacao']),
                mes_notificacao = int(tempo['mes_notificacao']),
                ano_notificacao = int(tempo['ano_notificacao']),
                trimestre_notificacao = int(tempo['trimestre_notificacao'])
            )
            r = conn.execute(ins)
            tempos_dict[chave_t] = r.inserted_primary_key[0]

        # 3) fato_notificacao
        inserted = 0
        errors = 0

        def fin(x):
            if x is None:
                return None
            if isinstance(x, (int, np.integer)):
                if SMALLINT_MIN <= int(x) <= SMALLINT_MAX:
                    return int(x)
                return None
            return to_smallint(x, default=None)

        for i, row in fato_df.iterrows():
            chave_not = f"{norm_val(row['sg_uf_not'])}_{norm_val(row['id_municip_not'])}"
            chave_res = f"{norm_val(row['sg_uf_res'])}_{norm_val(row['id_municip_res'])}"

            dt = safe_parse_date(row['dt_notific'])
            if dt is None:
                errors += 1
                if errors <= 10:
                    print(f"[WARN] registro {i}: dt_notific inválida -> {row['dt_notific']}")
                continue
            chave_t = str(dt.date())

            id_local_not = locais_dict.get(chave_not)
            id_local_res = locais_dict.get(chave_res)
            id_tempo = tempos_dict.get(chave_t)

            if not (id_local_not and id_local_res and id_tempo):
                errors += 1
                msg = (
                    f"[WARN] registro {i}: chave/local/tempo faltando -> "
                    f"chave_not={chave_not}, chave_res={chave_res}, chave_tempo={chave_t} "
                    f"(id_local_not={id_local_not}, id_local_res={id_local_res}, id_tempo={id_tempo})"
                )
                if errors <= 20:
                    print(msg)
                registrar_erro(msg)
                continue

            values = {
                'id_tempo_notificao': int(id_tempo),
                'id_local_notificacao': int(id_local_not),
                'id_local_residencia': int(id_local_res),

                'id_regional_notificacao': to_int(row['id_regional_notificacao']),
                'id_regional_residencia': to_int(row['id_regional_residencia']),

                'sexo': fin(row['sexo']),
                'forma_clinica': fin(row['forma_clinica']),
                'aids': fin(row['aids']),
                'alcool': fin(row['alcool']),
                'diabetes': fin(row['diabetes']),
                'doenca_mental': fin(row['doenca_mental']),
                'drogas': fin(row['drogas']),
                'tabagismo': fin(row['tabagismo']),
                'teste_hiv': fin(row['teste_hiv']),
                'resultado_teste_rapido': fin(row['resultado_teste_rapido']),
                'raio_x_torax': fin(row['raio_x_torax']),
                'teste_sensibilidade': fin(row['teste_sensibilidade']),
                'historico_patologico': fin(row['historico_patologico']),
                'faixa_etaria': fin(row['faixa_etaria']),
                'raca': fin(row['raca']),
                'gestante': fin(row['gestante']),
                'escolaridade': fin(row['escolaridade']),
                'beneficio_governo': fin(row['beneficio_governo']),
                'sem_teto': fin(row['sem_teto']),
                'encarcerado': fin(row['encarcerado']),
                'imigrante': fin(row['imigrante']),
                'profissional_saude': fin(row['profissional_saude']),
                'tipo_notificacao': fin(row['tipo_notificacao']),
                'extrapulmonar_1': fin(row['extrapulmonar_1']),
                'extrapulmonar_2': fin(row['extrapulmonar_2']),
                'situacao_encerramento': fin(row['situacao_encerramento']),
                'supervisionado_no_tratamento': fin(row['supervisionado_no_tratamento']),
                'tratamento': fin(row['tratamento']),
                'encerramento_abandono': fin(row['encerramento_abandono']),
                'tempo_entre_notificacao_e_diagnostico': to_int(row['tempo_entre_notificacao_e_diagnostico']),
                'tempo_entre_inicio_tramamento_e_notificacao': to_int(row['tempo_entre_inicio_tramamento_e_notificacao']),
                'tempo_entre_encerramento_e_inicio_tratamento': to_int(row['tempo_entre_encerramento_e_inicio_tratamento'])
            }

            try:
                conn.execute(fato_notificacao.insert().values(**values))
                inserted += 1
                if inserted % 100 == 0:
                    print(f"Inseridos {inserted} registros...")
            except SQLAlchemyError as e:
                errors += 1
                if errors <= 20:
                    print(f"[ERRO] ao inserir registro {i}: {e}")
                continue

        trans.commit()
        print(f"Concluído: {inserted} inseridos, {errors} ignorados/erros.")

    except Exception as e_outer:
        trans.rollback()
        print("ERRO TRANSAÇÃO:", e_outer)
        raise
    finally:
        conn.close()

# -----------------------------
# MAIN
# -----------------------------
def main():
    print("Carregando dados do Excel...")
    try:
        df = pd.read_excel('dadosCompleto.xlsx', sheet_name='PacientesTB mais dados muncipai')
        print("Linhas lidas:", len(df))
    except Exception as e:
        print("Erro ao abrir Excel:", e)
        return
    
    print("Carregando Atlas de desenvolvimento humano...")
    try:
        atlas_df = carregar_atlas()
        print("Municípios no Atlas:", len(atlas_df))
    except Exception as e:
        print("Erro ao carregar atlas.csv:", e)
        return
    
    print("Carregando CSV de população / faixa / região...")
    try:
        pop_df = carregar_populacao()   
        print("Municípios no CSV de população:", len(pop_df))
    except Exception as e:
        print("Erro ao carregar municipios_pop.csv:", e)
        return

    print("Processando...")
    locais, tempos, fato = processar_dados(df, atlas_df, pop_df)
    print("Locais únicos:", len(locais))
    print("Datas únicas:", len(tempos))
    print("Registros fato (após dedup):", len(fato))

    print("Conectando e inserindo no banco...")
    inserir_dados(engine, locais, tempos, fato)

if __name__ == "__main__":
    main()
