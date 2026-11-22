"""
Executa o projeto inteiro, do zero, na ordem da metodologia do TCC (Capítulo 3):

  0. verificações: arquivos de dados brutos, arquivo .env e banco de dados
  1. reconstrução da base de dados (ETL) ............................ Seção 3.2
  2. população de estudo, fluxograma e ABT ..................... Seções 3.3 a 3.5
  3. treinamento e avaliação dos cinco modelos ................. Seções 3.6 e 3.7
  4. tabelas e figuras de resultados ................................ Capítulo 4

Preparação do ambiente (apenas na primeira vez):
  1. Python 3.13 e um ambiente virtual:
         python -m venv .venv
         .venv\\Scripts\\activate            (Windows)  |  source .venv/bin/activate  (Linux/macOS)
         pip install -r requirements.txt
  2. Servidor MySQL em execução. O banco (DB_NAME) é criado automaticamente se não existir.
  3. Copiar .env.example para .env e preencher o acesso ao banco.
  4. Colocar em data/raw/ os arquivos de dados brutos (lista em VERIFICAR_ARQUIVOS abaixo).
     Eles não estão no Git por serem dados de saúde; a obtenção está descrita na Seção 3.2.

Uso (dentro da pasta ia):  python main.py
Saídas: banco de dados (tabelas e ABT), models/<modelo>/ e reports/.
"""
import sys
import time

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from src.banco_dados import ConexaoBanco
from src.config import ARQ_ATLAS, ARQ_DLOCAL, ARQ_DTEMPO, ARQ_ENV, ARQ_FATO, ARQ_PACIENTES
from src.data.construir_base_dados import construir_base_dados
from src.features.construir_abt import construir_abt
from src.models import (arvore_decisao, floresta_aleatoria, lightgbm_modelo, rede_neural,
                        regressao_logistica)
from src.visualization.fluxograma_populacao_estudo import gerar_fluxograma_populacao_estudo
from src.visualization.gerar_resultados import gerar_resultados

VERIFICAR_ARQUIVOS = {
    ARQ_FATO: "notificações do SINAN exportadas do .pbix de Santos (2026)",
    ARQ_DLOCAL: "dimensão de locais exportada do .pbix de Santos (2026)",
    ARQ_DTEMPO: "dimensão de datas de notificação (colunas id_tempo e ano_notificacao)",
    ARQ_ATLAS: "indicadores municipais do Atlas do Desenvolvimento Humano",
    ARQ_PACIENTES: "planilha com a cobertura da Atenção Primária à Saúde",
}

MODELOS = [regressao_logistica, arvore_decisao, floresta_aleatoria, rede_neural, lightgbm_modelo]


# ------------------------------------------------------------------ verificações
def verificar_arquivos():
    faltando = [f"  - {arq.name}: {desc}" for arq, desc in VERIFICAR_ARQUIVOS.items() if not arq.exists()]
    if faltando:
        sys.exit("[X] Arquivos de dados brutos não encontrados em data/raw/:\n" + "\n".join(faltando))
    if not ARQ_ENV.exists():
        sys.exit("[X] Arquivo .env não encontrado: copie .env.example para .env e preencha o acesso ao banco.")


def preparar_banco():
    """Cria o banco no servidor MySQL, se ainda não existir, e testa a conexão."""
    db = ConexaoBanco()
    if db.dialect.startswith("mysql"):
        faltando = [n for n, v in {"DB_USER": db.user, "DB_HOST": db.host, "DB_PORT": db.port,
                                   "DB_NAME": db.database}.items() if not v]
        if faltando:
            sys.exit(f"[X] Variáveis sem valor no .env: {', '.join(faltando)}")
        servidor = create_engine(URL.create(db.dialect, username=db.user, password=db.password,
                                            host=db.host, port=int(db.port)))
        try:
            with servidor.begin() as conn:
                conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db.database}` "
                                  "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
        except Exception as erro:
            sys.exit(f"[X] Não foi possível acessar o servidor MySQL ({erro.__class__.__name__}). "
                     "Verifique se ele está em execução e se os dados do .env estão corretos.")
        finally:
            servidor.dispose()
    if db.conectar() is None:
        sys.exit("[X] Não foi possível conectar ao banco de dados. Verifique o arquivo .env.")


# ------------------------------------------------------------------ execução
def etapa(titulo, funcao):
    print(f"\n{'=' * 80}\n{titulo}\n{'=' * 80}")
    inicio = time.time()
    funcao()
    print(f"[ok] {titulo} — {(time.time() - inicio) / 60:.1f} min")


def main():
    inicio = time.time()
    etapa("0. Verificações", lambda: (verificar_arquivos(), preparar_banco()))
    etapa("1. Reconstrução da base de dados (ETL)", construir_base_dados)
    etapa("2a. Fluxograma da população de estudo", gerar_fluxograma_populacao_estudo)
    etapa("2b. Construção da ABT", construir_abt)
    for modelo in MODELOS:
        etapa(f"3. Modelo: {modelo.__name__.split('.')[-1]}", modelo.main)
    etapa("4. Tabelas e figuras de resultados", gerar_resultados)
    print(f"\nConcluído em {(time.time() - inicio) / 60:.1f} min. "
          "Resultados em models/ e reports/.")


if __name__ == "__main__":
    main()
