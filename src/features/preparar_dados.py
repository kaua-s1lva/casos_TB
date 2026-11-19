"""
Prepara os conjuntos de treino e teste a partir da ABT, prontos para qualquer modelo.

Passos (os mesmos do notebook regressao_logistica.ipynb):
  1. Carrega a ABT (tabela tb_abt_tuberculose do banco).
  2. One-hot encoding das variáveis categóricas, com categoria de referência escolhida.
  3. Divide em treino e teste (estratificado e com semente fixa: todos os modelos
     usam exatamente o mesmo conjunto de teste).
  4. Preenche faltantes com a mediana do TREINO e normaliza as numéricas (z-score)
     com média e desvio do TREINO.
  5. Balanceia SÓ o treino, sorteando curas até igualar o número de abandonos
     (subamostragem aleatória). O teste mantém a proporção real.

Uso em qualquer script ou notebook:
    from src.features.preparar_dados import carregar_treino_teste
    X_train, X_test, y_train, y_test = carregar_treino_teste()
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.db_manager import ConexaoBanco

ALVO = "alvo_preditivo"
SEMENTE = 42

CATEGORICAS = ["faixa_etaria", "sexo", "teste_hiv", "alcool", "tabagismo",
               "drogas", "escolaridade", "raca", "faixa_populacao", "regiao"]
NUMERICAS = ["idhm", "indice_gini", "cobertura_aps_2021"]

# Categoria de referência de cada variável: a coluna dela é removida no one-hot e as
# demais categorias são comparadas contra ela (é o que define a leitura das odds ratios).
REFERENCIAS = {
    "faixa_etaria": "Acima de 60 anos",
    "sexo": "Feminino",
    "teste_hiv": "Negativo",
    "alcool": "Não",
    "tabagismo": "Não",
    "drogas": "Não",
    "escolaridade": "Ensino superior",
    "raca": "Branca",
    "faixa_populacao": "7 - Maior que 500000",
    "regiao": "3 - Sudeste",
}


def carregar_abt():
    db = ConexaoBanco()
    df = db.obter_dataframe("SELECT * FROM tb_abt_tuberculose")
    if df is None or df.empty:
        raise RuntimeError("Tabela 'tb_abt_tuberculose' não encontrada ou vazia. "
                           "Rode src/features/process_data.py antes.")
    return df


def codificar(df):
    """One-hot das categóricas, removendo a coluna da categoria de referência."""
    X = pd.get_dummies(df[CATEGORICAS + NUMERICAS], columns=CATEGORICAS, dtype=int)
    referencias = [f"{var}_{ref}" for var, ref in REFERENCIAS.items()]
    return X.drop(columns=[c for c in referencias if c in X.columns])


def balancear(X_train, y_train, semente=SEMENTE):
    """Sorteia, entre as curas, a mesma quantidade de casos de abandono (só no treino)."""
    abandono = X_train[y_train == 1]
    cura = X_train[y_train == 0].sample(n=len(abandono), random_state=semente)
    X_bal = pd.concat([abandono, cura]).sample(frac=1, random_state=semente)
    return X_bal, y_train.loc[X_bal.index]


def carregar_treino_teste(tamanho_teste=0.30, balancear_treino=True, normalizar=True,
                          semente=SEMENTE, df=None, mostrar_resumo=True):
    """Devolve X_train, X_test, y_train, y_test prontos para treinar e avaliar.

    tamanho_teste     proporção dos dados reservada para o teste (padrão 30%)
    balancear_treino  True = subamostragem das curas no treino; False = proporção real
                      (nesse caso, use class_weight="balanced" no modelo)
    normalizar        True = z-score nas numéricas (regressão, rede neural);
                      a árvore de decisão não precisa, mas também não é prejudicada
    df                ABT já carregada (opcional; se omitido, lê do banco)
    """
    df = carregar_abt() if df is None else df
    X = codificar(df)
    y = df[ALVO].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=tamanho_teste, stratify=y, random_state=semente)
    X_train, X_test = X_train.copy(), X_test.copy()

    # faltantes e normalização: estatísticas calculadas só com o treino
    medianas = X_train[NUMERICAS].median()
    X_train[NUMERICAS] = X_train[NUMERICAS].fillna(medianas)
    X_test[NUMERICAS] = X_test[NUMERICAS].fillna(medianas)
    if normalizar:
        scaler = StandardScaler()
        X_train[NUMERICAS] = scaler.fit_transform(X_train[NUMERICAS])
        X_test[NUMERICAS] = scaler.transform(X_test[NUMERICAS])

    if balancear_treino:
        X_train, y_train = balancear(X_train, y_train, semente)

    if mostrar_resumo:
        print(f"Treino: {len(X_train)} casos {y_train.value_counts().sort_index().to_dict()}"
              f"{' (balanceado)' if balancear_treino else ''}")
        print(f"Teste:  {len(X_test)} casos {y_test.value_counts().sort_index().to_dict()} (proporção real)")
        print(f"Variáveis: {X_train.shape[1]} colunas")
    return X_train, X_test, y_train, y_test
