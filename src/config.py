from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

# dados
DATA_RAW = RAIZ / "data" / "raw"
DATA_INTERIM = RAIZ / "data" / "interim"
DATA_PROCESSED = RAIZ / "data" / "processed"

ARQ_FATO = DATA_RAW / "fNotificacao.csv"
ARQ_DLOCAL = DATA_RAW / "dLocal.csv"
ARQ_ATLAS = DATA_RAW / "censo_total_1991_2010.xlsx"
ARQ_PACIENTES = DATA_RAW / "PacientesTB mais dados muncipais casos maior que 4.xlsx"

# saídas
REPORTS = RAIZ / "reports"
MODELS = RAIZ / "models"

# configuração do banco
ARQ_ENV = RAIZ / ".env"

# semente de todos os procedimentos aleatórios (reprodutibilidade)
SEMENTE = 42
