"""
Executa todas as etapas do projeto, na ordem da metodologia do TCC (Capítulo 3).
Descomente as etapas que quiser rodar.

Uso (dentro da pasta ia):  python main.py
"""
from src.data.construir_base_dados import construir_base_dados
from src.features.construir_abt import construir_abt
from src.models import (arvore_decisao, floresta_aleatoria, rede_neural, regressao_logistica,
                        xgboost_modelo)
from src.visualization.fluxograma_populacao_estudo import gerar_fluxograma_populacao_estudo
from src.visualization.tabela_caracteristicas_populacao import gerar_tabela_caracteristicas_populacao
from src.models.arvore_decisao import main
from src.models.floresta_aleatoria import main
from src.models.rede_neural import main
from src.models.regressao_logistica import main
from src.models.xgboost_modelo import main

if __name__ == "__main__":
    # 1. reconstrução da base de dados (ETL) — Seção 3.2
    # construir_base_dados()

    # 2. população de estudo e ABT — Seções 3.3 a 3.5
    # construir_abt()
    # gerar_fluxograma_populacao_estudo()
    # gerar_tabela_caracteristicas_populacao()

    # 3. treinamento e avaliação dos modelos — Seções 3.6 e 3.7
    #regressao_logistica.main()
    # arvore_decisao.main()
    # floresta_aleatoria.main()
    # rede_neural.main()
    # xgboost_modelo.main()

    arvore_decisao.main()
    floresta_aleatoria.main()
    rede_neural.main()
    regressao_logistica.main()
    xgboost_modelo.main()