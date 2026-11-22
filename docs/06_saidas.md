# 06 – Saídas

## `models/<modelo>/<cenário>/`

Gravado por `salvar_resultados()` (`src/models/registrar_resultados.py`) ao final de cada script de modelo.

| Arquivo | Conteúdo | No Git? |
|---|---|---|
| `metricas.json` | hiperparâmetros escolhidos, balanceamento, AUC na validação cruzada (média e desvio), métricas no teste, n do teste e versões das bibliotecas | sim |
| `importancia_permutacao.csv` | queda média da AUC por variável (só no cenário com municipais) | sim |
| `razoes_de_chances.csv` | razões de chances da Regressão Logística | sim |
| `validacao_cruzada.csv` | resultados de todas as combinações da busca em grade | não |
| `predicoes_teste.csv` | desfecho real e probabilidade prevista para cada caso do teste | **não** (dado por paciente) |
| `modelo.joblib` | pipeline treinado | não |

## `reports/`

| Caminho | Gerado por | Conteúdo |
|---|---|---|
| `fluxograma_populacao_estudo.png` / `.csv` | `fluxograma_populacao_estudo.py` | fluxograma de seleção e contagens por etapa |
| `resultados/tabelas/tabela_caracteristicas_populacao` | `tabela_caracteristicas_populacao.py` | Tabela 1: n (%) por desfecho, com p-valor |
| `resultados/tabelas/tabela_desempenho_modelos` | `gerar_resultados.py` | AUC (IC 95%), p contra o melhor modelo, demais métricas e Brier |
| `resultados/tabelas/tabela_comparacao_cenarios` | `gerar_resultados.py` | AUC sem e com municipais, diferença e p de DeLong |
| `resultados/tabelas/tabela_hiperparametros` | `gerar_resultados.py` | hiperparâmetros escolhidos por modelo e cenário |
| `resultados/figuras/fig_curvas_roc` | `gerar_resultados.py` | curvas ROC no teste |
| `resultados/figuras/fig_comparacao_cenarios` | `gerar_resultados.py` | AUC com IC 95% nos dois cenários |
| `resultados/figuras/fig_razoes_de_chances` | `gerar_resultados.py` | gráfico de floresta das razões de chances |
| `resultados/figuras/fig_importancia_permutacao` | `gerar_resultados.py` | mapa de calor da importância por modelo |
| `resultados/figuras/fig_calibracao` | `gerar_resultados.py` | calibração por decis |
| `resultados/figuras/fig_limiar_melhor_modelo` | `gerar_resultados.py` | sensibilidade e especificidade conforme o limiar |

Cada tabela sai em `.csv` e em `.tex`. O `.tex` traz o ambiente `table` completo, com legenda, rótulo, nota e fonte. Cada figura sai em `.pdf` (para o LaTeX) e em `.png`.

## Usar no TCC

Copie os arquivos para a pasta do LaTeX e inclua. Pelo guia da UFES, o título (`\caption`) fica **acima** da figura e a fonte (`\legend`) **abaixo**:

```latex
\input{tabelas/tabela_desempenho_modelos.tex}

\begin{figure}[H]
    \centering
    \caption{Curvas ROC dos modelos no conjunto de teste.}
    \label{fig-res-roc}
    \includegraphics[width=0.7\textwidth]{figuras/fig_curvas_roc.pdf}
    \legend{Fonte: elaborado pelo autor.}
\end{figure}
```
