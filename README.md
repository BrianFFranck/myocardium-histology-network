# Pipeline de Análise Histológica de Miocárdio e Redes Espaciais

Este repositório contém o trabalho prático desenvolvido no âmbito do **Projeto I (Variante B)** da unidade curricular de **Programação para as Biociências II** na **Universidade Lusófona**.

O sistema simula um cenário clínico-laboratorial de patologia digital (Serviço de Anatomia Patológica do Hospital Universitário de Santa Helena). A aplicação realiza o processamento e extração de métricas morfológicas locais de biópsias miocárdicas (coloração H&E) e modela a organização celular e os padrões de infiltração através de uma **Rede Espacial (Grafo)** de alta performance.

---

## 📋 Enquadramento do Problema & Requisitos

A classificação manual e visual de biópsias para detetar patologias como miocardite ativa, rejeição de transplantes ou fibrose miocárdica é um processo suscetível a variabilidade interpretativa. Este sistema automatiza e complementa o pipeline através de duas vertentes críticas:

1. **Manipulação de Matrizes de Imagem (Pillow/NumPy)**: Extração de janelas locais de píxeis (64x64 px) centradas nos centróides das células para calcular métricas morfológicas diretamente sobre os valores de intensidade em escala de cinzentos.
2. **Otimização Algorítmica O(n)**: A construção ingénua (*Naïve*) de redes espaciais por distância Euclidiana com datasets de larga escala (ex: `cells_large.csv` com mais de 5.000 células) apresenta complexidade quadrática `O(n²)`. Foi implementada uma otimização baseada em **subdivisão espacial em grelha** (*Grid-Based Neighbor Search*) para reduzir o custo computacional para tempo linear `O(n)`.

---

## 🚀 Funcionalidades Implementadas

* **Leitura Dinâmica com Memória Constante**: Processamento iterativo do ficheiro CSV recorrendo a geradores em Python (`yield`), assegurando suporte para ficheiros de dados de qualquer dimensão sem alocação massiva de memória.
* **Geração Estocástica de Imagens Sintéticas**: Renderização probabilística da lâmina histológica simulada respeitando as restrições morfológicas (raio, alongamento, cor do núcleo e do citoplasma) de cada tipo de célula e aplicando ruído Gaussiano de fundo com sementes determinísticas (`zlib.adler32`).
* **Extração de Métricas Morfológicas Avançadas**:
    * Estatísticas de intensidade em escala de cinzentos (Média, Mediana, Desvio Padrão, Mínimo, Máximo, P25 e P75 recorrendo a ordenação e particionamento linear de arrays).
    * Área aproximada calculada por segmentação e limiarização de intensidades locais.
    * Estimativa de circularidade calculada sobre a máscara binária utilizando um operador adaptado de erosão com vizinhança 4-conectada.
* **Construção de Redes Espaciais**: Abordagem estruturada por um threshold de distância Euclidiana adaptativo (ajusta dinamicamente o threshold de conectividade caso as células adjacentes possuam tipos celulares distintos).
* **Análise Topológica da Rede**:
    * Deteção de componentes conexas (clusters celulares) via algoritmo de pesquisa em largura (BFS).
    * Isolamento estatístico de *hubs* de alta conectividade (onde o grau do nó é maior que a média mais dois desvios padrões).
    * Estatísticas descritivas das arestas, densidade espacial da lâmina e distribuições de grau.
* **Profiling de Performance**: Integração com o módulo nativo `cProfile` para detalhar tempos acumulados e expor quantitativamente o ganho de velocidade (*speedup*) e o impacto do gargalo computacional.

---

## 🛠️ Arquitetura do Software (Princípios POO)

O código cumpre rigorosamente os princípios de Programação Orientada a Objetos e as boas práticas de otimização de baixo nível em Python:

* **Classe `Cell`**: Abstrai as propriedades biológicas e geométricas de cada célula. Implementa a diretiva `__slots__` para desativar a criação do dicionário dinâmico interno (`__dict__`). Esta otimização reduz o consumo de memória RAM em mais de 50%, viabilizando o processamento ágil de datasets massivos no ambiente local.
* **Classe `SpatialNetwork`**: Encapsula a infraestrutura do grafo, gerindo a matriz de adjacência, o mapeamento de vizinhança na grelha bidimensional e a execução dos algoritmos topológicos (BFS, cálculo de caminhos e graus).

Em total conformidade com o enunciado, **não foram utilizadas bibliotecas de alto nível para grafos (como NetworkX) ou inteligência artificial (como Scikit-Learn/PyTorch)**. Todo o motor de análise foi programado recorrendo à bibliotecas padrão, ao `NumPy` e ao `Pillow`.

---

## 📦 Dependências e Instalação

Para correr a aplicação, assegura-te de que tens instalado o Python 3.10 ou superior. Instala as duas bibliotecas científicas requeridas através do terminal:

```bash
pip install numpy pillow
