Projeto Estatístico do Cartório

Este projeto analisa estatisticamente os atos de registros praticados em um cartório entre 2005 e 2025 o 1º semestre de 2025, aplicando medidas de tendência central, dispersão, assimetria e curtose, e gerando gráficos explicativos.
Os dados foram extraídos de relatório oficial disponibilizado pelo CNJ e totalizam 1.702.338 atos no período.

Tecnologias Utilizadas

Python 3.10+ (compatível até 3.13)
NumPy → cálculos estatísticos
Statistics → mediana e moda
Matplotlib → visualização de dados (barras, histogramas e curvas)
Funções personalizadas → cálculo de Assimetria (g₁) e Curtose (excesso)

Funcionalidades do Código

O script main.py realiza:
Cálculo de medidas estatísticas:
Média
Mediana
Moda (ou indicação de série amodal)
Desvio padrão amostral
Coeficiente de variação (CV)
Assimetria (g₁)
Curtose (excesso de curtose)
Geração de gráficos automáticos:
grafico_barras.png → Atos praticados por período
histograma_distribuicao.png → Distribuição da frequência dos atos
curvas_assimetria.png → Exemplos de curvas simétrica e com assimetrias
onda_assimetria_resultado.png → Curva suave ajustada ao valor real da assimetria
onda_curtose_resultado.png → Curva suave ajustada ao valor real da curtose

Exemplo de Saída no Terminal:

Estatísticas do Cartório
Média: 43,649.69 atos
Mediana: 42,255 atos
Moda: amodal (não há valores que se repetem)
Desvio Padrão (amostral): 9,577.21 atos
Coeficiente de Variação: 21.94%
Assimetria (g₁): -0.2462 - Atualizar/Verificar
Curtose (excesso): -0.9817  (Normal=0) - Atualizar/Verificar

Exemplos de Gráficos

🔹 Atos praticados por período

🔹 Distribuição dos Atos

🔹 Onda ajustada da Assimetria

🔹 Onda ajustada da Curtose

Executar no prompt:

Instale as dependências:

**pip install numpy matplotlib**

Execute o script:

**python main.py**

Apresentará os resultados e gráficos.


Estrutura do Projeto

Projeto-Cartorio-Estatistico
│-- main.py
│-- Projeto Cartório Estatístico.pdf
│-- grafico_barras.png
│-- histograma_distribuicao.png
│-- curvas_assimetria.png
│-- onda_assimetria_resultado.png
│-- onda_curtose_resultado.png
│-- README.md - Novo

Autores

José Luiz Oliveira Cavalcante Teles
José Miguel dos Santos Neto
José Matheus de Castro Lima
Samir Freitas dos Santos
