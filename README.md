# RNAP-T2 - Classificação de Estilos Arquitetônicos

**Felipe Rovigatti Delfino** e **Lucas de Souza Silva**

---

Projeto de classificação de imagens de edifícios em 25 estilos arquitetônicos utilizando uma rede neural convolucional (CNN) implementada com PyTorch.

O experimento compara o desempenho do modelo utilizando diferentes frações do conjunto de treinamento (25% e 100%), repetindo cada configuração com cinco seeds diferentes.


## 1. Preparação do ambiente

### Clone o repositório:

``` bash
git clone https://github.com/felipedelfin00/RNAP-T2.git
cd RNAP-T2
``` 

### Crie um ambiente virtual:

**Windows**:

``` bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux**:

``` bash
python -m venv .venv
source .venv/bin/activate
```

### Instale as dependências:

``` bash
pip install -r requirements.txt
```

**GPU NVIDIA**: caso deseje utilizar CUDA, verifique se a versão instalada do PyTorch é compatível com a GPU e os drivers disponíveis.


## 2. Execução do pipeline

Na raiz do projeto, execute:

``` bash
python main.py
```

O pipeline executa as seguintes etapas:
1. Carregamento ou download do dataset.
2. Divisão estratificada dos dados.
3. Cálculo das estatísticas de normalização.
4. Criação dos conjuntos de treinamento, validação e teste.
5. Execução opcional de um benchmark para estimar o tempo de treinamento.
6. Treinamento das configurações experimentais.
7. Avaliação dos modelos no conjunto de teste.
8. Exportação dos resultados e checkpoints.

São realizados dez execuções no total: duas frações do conjunto de treinamento para cada uma das cinco seeds.

O treinamento utiliza early stopping com base na loss de validação.


## 3. Geração dos gráficos

Após a conclusão dos experimentos, execute:

``` bash
python -m src.visualize
```

O módulo de visualização gera automaticamente dois tipos de gráficos:


### Curvas de treinamento (training_curves.png)

Uma figura consolidada com dois gráficos lado a lado, comparando os experimentos com 25% e 100% dos dados de treinamento. Cada gráfico apresenta:

- Média das curvas de loss de treinamento e validação entre as cinco execuções.
- Faixas de desvio-padrão (± 1 desvio-padrão).
- Identificação da melhor época média de validação.
- F1-Score macro, acurácia e tempo de execução, apresentados como média ± desvio-padrão.
- Quantidade de execuções consideradas nas curvas.


### Matrizes de confusão

Uma imagem para cada execução, gerada a partir dos respectivos arquivos CSV. As matrizes apresentam os valores absolutos de classificação em cada célula, com as cores normalizadas por classe verdadeira para facilitar a comparação entre classes.


## 4. Dataset
O projeto utiliza o dataset [Architectural Styles Dataset](https://www.kaggle.com/datasets/dumitrux/architectural-styles-dataset), disponibilizado no Kaggle por dumitrux.

O download é realizado automaticamente pelo KaggleHub durante a execução do pipeline, caso o dataset ainda não esteja disponível localmente.

Não é necessário baixar ou organizar as imagens manualmente.


## 5. Configurações do experimento

As configurações podem ser encontradas em `src/config.py`.

A configuração experimental utilizada é:

|Parâmetro|Valor|
|-|-:|
|Resolução das imagens|224 x 224|
|Número de classes|25|
|Batch size|32|
|Épocas máximas|35|
|Learning rate|0.001|
|Weight decay|0.0001|
|Patience (early stopping)|7|
|Frações de treinamento|25% e 100%|
|Seeds|6, 12, 18, 24 e 30|

O conjunto de dados é dividido de maneira estratificada em:
- 70% para treinamento;
- 15% para validação;
- 15% para teste.

O treinamento é realizado com uma CNN inicializada do zero, utilizando AdamW e Cross-Entropy Loss.

O dispositivo de execução é selecionado automaticamente: CUDA, quando disponível, ou CPU.


## 6. Arquitetura da CNN

O modelo utilizado é uma rede neural convolucional desenvolvida e treinada do zero, sem utilização de pesos pré-treinados.

A arquitetura é composta por três blocos convolucionais, responsáveis por extrair características progressivamente mais complexas das imagens:

- **Bloco 1:** duas camadas convolucionais com 32 filtros.
- **Bloco 2:** duas camadas convolucionais com 64 filtros.
- **Bloco 3:** duas camadas convolucionais com 128 filtros.

Cada camada convolucional é seguida por Batch Normalization e ativação ReLU. Ao final de cada bloco, uma camada Max Pooling reduz as dimensões espaciais dos mapas de características.

Após os blocos convolucionais, o modelo utiliza:

1. **Adaptive Average Pooling:** agrega os mapas de características em uma representação de tamanho fixo.
2. **Dropout (0.3):** reduz o risco de overfitting durante o treinamento.
3. **Camada totalmente conectada:** produz 25 valores, um para cada classe arquitetônica.