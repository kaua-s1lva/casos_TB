"""
Variável-alvo e variáveis preditoras do estudo (Seção 3.4 e Tabela 3 do TCC).
"""

ALVO = "alvo_preditivo"   # 1 = perda de seguimento (abandono + abandono primário) | 0 = cura

# variável: ([categorias válidas], categoria de referência) — Tabela 3 do TCC.
# Todas as categóricas também têm a categoria "Ignorado".
INDIVIDUAIS = {
    "faixa_etaria": (["19 - 29 anos", "30 - 39 anos", "40 - 49 anos", "50 - 59 anos",
                      "Acima de 60 anos"], "Acima de 60 anos"),
    "sexo": (["Feminino", "Masculino"], "Feminino"),
    "teste_hiv": (["Positivo", "Negativo", "Em andamento", "Não realizado"], "Negativo"),
    "alcool": (["Sim", "Não"], "Não"),
    "tabagismo": (["Sim", "Não"], "Não"),
    "drogas": (["Sim", "Não"], "Não"),
    "escolaridade": (["Analfabeto", "Até fundamental", "Ensino médio", "Ensino superior"],
                     "Ensino superior"),
    "raca": (["Branca", "Preta", "Parda", "Amarela/Indígena"], "Branca"),
}
MUNICIPAIS_CATEGORICAS = {
    "faixa_populacao": (["1 - Até 5000", "2 - 5001 até 10000", "3 - 10001 até 20000",
                         "4 - 20001 até 50000", "5 - 50001 até 100000",
                         "6 - 100001 até 500000", "7 - Maior que 500000"], "7 - Maior que 500000"),
    "regiao": (["1 - Norte", "2 - Nordeste", "3 - Sudeste", "4 - Sul", "5 - Centro-Oeste"],
               "3 - Sudeste"),
}
MUNICIPAIS_NUMERICAS = ["idhm", "indice_gini", "cobertura_aps_2021"]

CATEGORICAS = {**INDIVIDUAIS, **MUNICIPAIS_CATEGORICAS}
NUMERICAS = MUNICIPAIS_NUMERICAS
PREDITORAS = list(CATEGORICAS) + NUMERICAS
