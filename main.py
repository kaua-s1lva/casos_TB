from src.data.carga_mysql import carregar_dados
from src.features.process_data import preparar_dados_ml
from src.models.treinar_arvore import treinar_modelo
from src.models.regressao_logistica import treinar


if __name__ == "__main__":
    # carregar_dados()

    # preparar_dados_ml()

    # treinar_modelo()
    treinar()