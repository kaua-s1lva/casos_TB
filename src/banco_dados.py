import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
from src.config import ARQ_ENV

load_dotenv(ARQ_ENV)

class ConexaoBanco:
    def __init__(self):
        self.dialect = os.getenv("DB_DIALECT", "mysql+pymysql")
        self.user = os.getenv("DB_USER")
        self.password = os.getenv("DB_PASSWORD")
        self.host = os.getenv("DB_HOST")
        self.port = os.getenv("DB_PORT")
        self.database = os.getenv("DB_NAME")
        self.engine = None

    def conectar(self):
        """Cria e retorna a engine de conexão com qualquer banco suportado pelo SQLAlchemy."""
        if self.engine is None:
            try:
                url_conexao = f"{self.dialect}://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
                
                if self.dialect == "sqlite":
                    url_conexao = f"sqlite:///{self.database}"

                self.engine = create_engine(url_conexao)
                
                with self.engine.connect() as conn:
                    print(f"[*] Conexão estabelecida com o banco '{self.database}' usando {self.dialect}.")
                    
            except SQLAlchemyError as e:
                print(f"[X] Erro ao conectar ao banco de dados: {e}")
                self.engine = None
                
        return self.engine

    def consultar(self, query):
        """Executa uma query SQL e retorna os dados em um DataFrame Pandas."""
        if self.engine is None:
            self.conectar()
            
        if self.engine:
            try:
                df = pd.read_sql(query, con=self.engine)
                return df
            except Exception as e:
                print(f"[X] Erro ao executar a query: {e}")
                return None
        return None

    def gravar_tabela(self, df, nome_tabela, modo='replace'):
        """Envia um DataFrame Pandas para uma tabela no banco de dados."""
        if self.engine is None:
            self.conectar()
            
        if self.engine:
            try:
                df.to_sql(name=nome_tabela, con=self.engine, if_exists=modo, index=False)
                print(f"[+] {len(df)} linhas enviadas para a tabela '{nome_tabela}'.")
            except Exception as e:
                print(f"[X] Erro ao enviar dados: {e}")