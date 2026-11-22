# 03 – População de estudo e ABT

**Código:** `src/features/construir_abt.py` (ABT) e `src/visualization/fluxograma_populacao_estudo.py` (fluxograma). O mesmo filtro está em `sql/vw_populacao_estudo.sql`, só como referência.

## Critérios de seleção

Os critérios seguem os de Rodrigues et al. (2024) e são aplicados à `fnotificacao` nesta ordem:

| # | Exclusão | Regra no código |
|---|---|---|
| 1 | menores de 19 anos | `faixa_etaria != '<= 18 anos'` |
| 2 | imigrantes | `imigrante != 'Sim'` |
| 3 | pessoas em situação de rua | `sem_teto != 'Sim'` |
| 4 | profissionais de saúde | `profissional_saude != 'Sim'` |
| 5 | pessoas privadas de liberdade | `encarcerado != 'Sim'` |
| 6 | gestantes | `gestante` contém "Não" ou é "Ignorado" |
| 7 | TB extrapulmonar ou mista | `forma_clinica = 'Pulmonar'` |
| 8 | resistência a fármaco de 1ª linha | `teste_sensibilidade` fora das categorias de resistência |
| 9 | sem tempo de tratamento | `tempo_entre_encerramento_e_inicio_tratamento` não nulo |
| 10 | desfechos fora do estudo (transferência, óbito, mudança de esquema, TB-DR, falência, sem informação) | `situacao_encerramento` em Cura, Abandono ou Abandono Primário |

O número de casos excluídos em cada etapa fica em `reports/fluxograma_populacao_estudo.csv`. A figura correspondente está em `reports/fluxograma_populacao_estudo.png`.

## Desfecho

`alvo_preditivo = 1` para **perda de seguimento** (Abandono ou Abandono Primário) e `0` para **Cura**. A coluna original `situacao_encerramento` é removida da ABT, para que o desfecho não entre como variável.

## ABT (*Analytical Base Table*)

Uma linha por notificação: **185.361 casos**. A ABT guarda dados **limpos, mas não transformados**. Nela entram só transformações que dependem apenas da própria linha:

- **Padronização dos rótulos:** qualquer valor fora das categorias válidas vira `"Ignorado"`, inclusive nulos, códigos soltos como "9" e rótulos corrompidos. As categorias válidas estão em `CATEGORIAS`.
- **Indicadores municipais como número:** não pode haver valores faltantes (o ETL preenche os que faltavam). Se aparecer algum, a construção da ABT é interrompida com uma mensagem de erro.
- **Criação do alvo binário.**

Tudo o que precisa "aprender" com os dados fica **fora** da ABT, dentro do *pipeline* de cada modelo, e é ajustado só no treino: *one-hot*, padronização e balanceamento.

**Saídas:**
- tabela `tb_abt_tuberculose` no MySQL (substituída a cada execução);
- `data/processed/abt_tuberculose.csv`: cópia congelada, com a data da extração.

## Divisão treino/teste

Feita por `carregar_treino_teste()` (`src/features/dividir_treino_teste.py`). O último ano de notificação (2022) é o **teste**; os anteriores (2018–2021) são o **treino**.

| Conjunto | Casos | Perda de seguimento |
|---|---|---|
| Treino (2018–2021) | 148.830 | 15,2% |
| Teste (2022) | 36.531 | 18,0% |
