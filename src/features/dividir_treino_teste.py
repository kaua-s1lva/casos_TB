"""
Divisão temporal da ABT em treino e teste (Seção 3.6 do TCC):
  - treino = notificações dos anos anteriores ao último (2018–2021);
  - teste  = notificações do último ano (2022).

Os dados saem SEM transformação (sem one-hot, padronização ou balanceamento):
essas etapas ficam no pipeline de cada modelo, ajustadas só no treino.

Uso:
    from src.features.dividir_treino_teste import carregar_treino_teste
    X_treino, X_teste, y_treino, y_teste = carregar_treino_teste()
"""
from src.banco_dados import ConexaoBanco
from src.features.construir_abt import TABELA_ABT
from src.features.variaveis import ALVO, PREDITORAS


def carregar_treino_teste(colunas=PREDITORAS):
    """Devolve X_treino, X_teste, y_treino, y_teste com as colunas pedidas."""
    df = ConexaoBanco().consultar(f"SELECT * FROM {TABELA_ABT}")
    ultimo_ano = int(df["ano_notificacao"].max())
    treino = df[df["ano_notificacao"] < ultimo_ano]
    teste = df[df["ano_notificacao"] == ultimo_ano]
    return treino[colunas], teste[colunas], treino[ALVO], teste[ALVO]
