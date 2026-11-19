from src.db_manager import ConexaoBanco
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

def treinar():
    db = ConexaoBanco()
    df = db.obter_dataframe("SELECT * FROM tb_abt_tuberculose")

    categoricas = ["faixa_etaria", "sexo", "teste_hiv", "alcool", "tabagismo",
                "drogas", "escolaridade", "raca", "faixa_populacao", "regiao"]
    numericas = ["idhm", "indice_gini", "cobertura_aps_2021"]

    #fazer o one hot encoding
    X = pd.get_dummies(df[categoricas + numericas], columns=categoricas,
                    drop_first=True, dtype=int)

    print(X)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

    # model = LogisticRegression()
    # model.fit(X_train, y_train)

    # y_pred = model.predict(X_test)

    # acc = accuracy_score(y_test, y_pred)

    # print("Acurácia do Classificador = %f" % acc)





# 2. One-hot: cada categoria vira uma coluna 0/1
#    (drop_first=True remove uma categoria de cada variável, que vira a referência)

treinar()