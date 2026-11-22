# Referências

Principais referências do projeto, agrupadas pelo papel que têm no código. A lista completa e formatada (ABNT) está em `bibliografia.bib`, no projeto LaTeX do TCC.

Os PDFs dos artigos podem ser guardados nesta pasta para consulta local. Eles **não** vão para o Git (`*.pdf` está no `.gitignore`), por direitos autorais.

## Trabalhos de base

| Referência | Uso no projeto |
|---|---|
| Rodrigues, M. M. S. et al. Machine learning algorithms using national registry data to predict loss to follow-up during tuberculosis treatment. *BMC Public Health*, 24, 1385, 2024. [doi:10.1186/s12889-024-18815-0](https://doi.org/10.1186/s12889-024-18815-0) | estudo de referência: critérios de seleção, desfecho, LightGBM e formato da Tabela 1 |
| Santos, B. V. G. *Desenvolvimento de uma arquitetura de BI para análise de casos de abandono de tratamento da Tuberculose nos municípios brasileiros*. TCC, UFES, 2026. | origem das notificações (`.pbix`) e do ETL original (`docs/legado/`) |
| Chen, J. et al. Predictive machine learning models for anticipating loss to follow-up in tuberculosis patients throughout anti-TB treatment journey. *Scientific Reports*, 14, 2024. [doi:10.1038/s41598-024-74942-z](https://doi.org/10.1038/s41598-024-74942-z) | trabalho relacionado (revisão de literatura) |
| Brasil. Ministério da Saúde. *Manual de Recomendações para o Controle da Tuberculose no Brasil*. 2. ed. Brasília, 2019. | definições de desfecho do tratamento |

## Fontes de dados

| Fonte | Acesso |
|---|---|
| SINAN – Tuberculose (via Santos, 2026) | arquivo `.pbix` fornecido pelo autor |
| PNUD, Ipea e FJP. *Atlas do Desenvolvimento Humano no Brasil*, 2013 (Censo 2010) | <http://www.atlasbrasil.org.br> |
| Ministério da Saúde. e-Gestor: cobertura da Atenção Primária à Saúde (competência 07/2021) | <https://relatorioaps.saude.gov.br/cobertura/aps> |
| Brasil. Conselho Nacional de Saúde. Resolução nº 510, de 7 de abril de 2016 | dispensa de apreciação ética para dados públicos e não identificados |

## Métodos

| Referência | Onde aparece no código |
|---|---|
| Hosmer, D. W.; Lemeshow, S.; Sturdivant, R. X. *Applied Logistic Regression*. 3. ed. Wiley, 2013. | `regressao_logistica.py`: razões de chances |
| Breiman, L. Random forests. *Machine Learning*, 45(1), 5–32, 2001. [doi:10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324) | `floresta_aleatoria.py`; importância por permutação |
| Friedman, J. H. Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232, 2001. [doi:10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451) | `lightgbm_modelo.py`: *gradient boosting* |
| Ke, G. et al. LightGBM: a highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30, 2017. | `lightgbm_modelo.py` |
| Hastie, T.; Tibshirani, R.; Friedman, J. *The Elements of Statistical Learning*. 2. ed. Springer, 2009. | referência geral dos algoritmos |
| Fisher, A.; Rudin, C.; Dominici, F. All models are wrong, but many are useful. *JMLR*, 20(177), 1–81, 2019. | `avaliacao_desempenho.py`: importância por permutação |
| DeLong, E. R.; DeLong, D. M.; Clarke-Pearson, D. L. Comparing the areas under two or more correlated ROC curves: a nonparametric approach. *Biometrics*, 44(3), 837–845, 1988. [doi:10.2307/2531595](https://doi.org/10.2307/2531595) | `avaliacao_desempenho.py`: `delong()` |
| Sun, X.; Xu, W. Fast implementation of DeLong's algorithm for comparing the areas under correlated ROC curves. *IEEE Signal Processing Letters*, 21(11), 1389–1393, 2014. [doi:10.1109/LSP.2014.2337313](https://doi.org/10.1109/LSP.2014.2337313) | `avaliacao_desempenho.py`: algoritmo usado em `delong()` |
| Youden, W. J. Index for rating diagnostic tests. *Cancer*, 3(1), 32–35, 1950. | `gerar_resultados.py`: limiar de Youden |

## Ferramentas

| Ferramenta | Referência |
|---|---|
| pandas | McKinney, W. Data structures for statistical computing in Python. *Proc. 9th Python in Science Conference*, 56–61, 2010. |
| scikit-learn | Pedregosa, F. et al. Scikit-learn: machine learning in Python. *JMLR*, 12, 2825–2830, 2011. |
| imbalanced-learn | Lemaître, G.; Nogueira, F.; Aridas, C. K. Imbalanced-learn. *JMLR*, 18(17), 1–5, 2017. |
| Power BI Desktop | <https://www.microsoft.com/power-platform/products/power-bi/desktop> |
| DAX Studio | <https://daxstudio.org> |
