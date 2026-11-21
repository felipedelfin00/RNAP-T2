# RNAP-T2 - Classificação de Estilos Arquitetônicos

**Felipe Rovigatti Delfino** e **Lucas de Souza Silva**

---

Projeto de classificação de imagens de edifícios em 25 estilos arquitetônicos utilizando redes neurais convolucionais (CNNs) implementada com PyTorch.

O experimento compara uma CNN desenvolvida e treinada do zero com uma ResNet-18 pré-treinada no ImageNet, avaliando também o impacto da quantidade de dados de treinamento no desempenho dos modelos.

Cada configuração experimental é executada com duas frações do conjunto de treinamento (25% e 100%) e cinco seeds diferentes, totalizando 20 execuções.

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

**GPU NVIDIA**: o dispositivo de execução é selecionado automaticamente. Para utilizar CUDA, é necessário que a instalação do PyTorch seja compatível com a GPU e os drivers disponíveis.


## 2. Execução do pipeline

Na raiz do projeto, execute:

``` bash
python main.py
```

O pipeline executa as seguintes etapas:

1. Carregamento ou download do dataset.
2. Pré-processamento e armazenamentodas imagens em cache
3. Divisão estratificada dos dados em treinamento, validação e teste.
4. Cálculo das estatísticas de normalização.
5. Criação dos conjuntos de dados e DataLoaders.
6. Execução opcional de um benchmark para estimativa do tempo de treinamento.
7. Treinamento das configurações experimentais.
8. Avaliação dos melhores modelos no conjunto de teste.
9. Exportação dos resultados, históricos e checkpoints.

São realizados 20 execuções, considerando duas arquiteturas, duas frações de treinamento e cinco seeds.

O treinamento utiliza early stopping com base no F1-score macro da validação. Os pessos correspondentes à época com o maior F1-score macro de validação são restaurados e utilizados na avaliação final.


## 3. Geração dos gráficos

Após a conclusão dos experimentos, execute:

``` bash
python -m src.visualize
```

O módulo de visualização gera gráficos para cada arquitetura, permitindo comparar os resultados obtidos com 25% e 100% dos dados de treinamento.


### Curvas de treinamento

As curvas apresentam:

- Média das métricas de treinamento e validação entre as cinco seeds.
- Faixas de desvio-padrão (± 1 desvio-padrão).
- Evolução da loss de treinamento e validação
- Identificação das melhores épocas.
- Métricas finais de F1-score macro e acurácia, apresentadas como média ± desvio-padrão
- Tempo de treinamento, apresentado como média ± desvio-padrão.


### Matrizes de confusão

É gerada uma matriz de confusão para cada execução experimental, utilizando as predições realizadas no conjunto de teste.

As matrizes apresentam os valores abolutos de classificação em cada célula, com as cores normalizadas por classe verdadeira, facilitando a identificação dos estilos arquitetônicos com maior dificuldade de classificação.


## 4. Dataset
O projeto utiliza o [Architectural Styles Dataset](https://www.kaggle.com/datasets/dumitrux/architectural-styles-dataset), disponibilizado no Kaggle por dumitrux.

O conjunto de dados contém 10.113 imagens distribuídas entre 25 estilos arquitetônicos.

O download é realizado automaticamente utilizando KaggleHub durante a execução do pipeline, caso o dataset ainda não esteja disponível localmente.

As imagens são redimensionadas para 128 x 128 pixels e armazenadas em cache para evitar o processamento repetido em execuções posteriores.

Não é necessário baixar ou organizar as imagens manualmente.


## 5. Configurações do experimento

As configurações podem ser encontradas em `src/config.py`.

### Configurações gerais

| Parâmetro | Valor |
| ---| ---: |
| Resolução das imagens | 128 x 128 |
| Número de classes | 25 |
| Frações de treinamento | 25% e 100% |
| Seeds | 6, 12, 18, 24 e 30 |
| Batch size | 64 |
| Épocas máximas | 35 |
| Patience (early stopping) | 5 |
| Otimizador | AdamW |
| Função de perda | Cross-Entropy Loss |
| Métrica principal | F1-score macro |
| Métrica secundária | Acurácia |

O conjunto de dados é dividido de maneira estratificada em:

- 70% para treinamento;
- 15% para validação;
- 15% para teste.

As seeds são utilizadas para controlar a aleatoriedade das execuções e permitir a comparação entre diferentes configurações experimentais.

Ambas as arquiteturas utilizam data augmentation durante o treinamento, incluindo:

- Random Horizontal Flip;
- Random Rotation de até 5%;
- Color Jitter para variações de brilho, contraste, saturação e matiz.

As transformações não são aplicadas aos conjuntos de validação e teste.

A normalização é realizada utilizando as estatísticas calculadas a partir do conjunto de treinamento para a CNN baseline e os valores de normalização do ImageNet para a ResNet-18.

O dispositivo de execução é selecionado automaticamente: CUDA, quando disponível, ou CPU.

### CNN baseline

| Parâmetro | Valor |
| --- | ---: |
| Learning Rate | 0.001 |
| Weight Decay | 0.0001 |
| Inicialização | Pesos aleatórios |

### ResNet-18

| Parâmetro | Valor |
| --- | ---: |
| Learning rate | 0.0001 |
| Weight decay | 0.001 |
| Inicialização | Pesos pré-treinados no ImageNet |


## 6. Arquiteturas utilizadas

### CNN baseline

A CNN baseline é uma rede neural convolucional desenvolvida e treinada do zero, sem utilização de pesos pré-treinados.

A arquitetura é composta por três blocos convolucionais, responsáveis por extrair características progressivamente mais complexas das imagens:

- **Bloco 1:** duas camadas convolucionais com 32 filtros.
- **Bloco 2:** duas camadas convolucionais com 64 filtros.
- **Bloco 3:** duas camadas convolucionais com 128 filtros.

Cada camada convolucional é seguida por Batch Normalization e ativação ReLU. Ao final de cada bloco, uma camada Max Pooling reduz as dimensões espaciais dos mapas de características.

Após os blocos convolucionais, o modelo utiliza:

1. **Adaptive Average Pooling:** agrega os mapas de características em uma representação de tamanho fixo.
2. **Dropout (0.3):** reduz o risco de overfitting durante o treinamento.
3. **Camada totalmente conectada:** produz 25 valores, um para cada classe arquitetônica.

### ResNet-18

A ResNet-18 utiliza uma arquitetura residual originalmente treinada no ImageNet, com pesos pré-treinados transferidos para o problema de classificação de estilos arquitetônicos.

A camada totalmente original foi substituida por uma noca camada de classificação, com dropout de 0.3 e 25 saídas, correspondentes às classes do dataset.

Todos os parâmetros da rede são disponibilizados para treinamento (fine-tuning completo), permitindo adaptar tanto as características extraídas quanto a camada de classificação ao conjunto de dados utilizado.


## 7. Avaliação experimental

O desempenho dos modelos é avaliado utilizando:

- **F1-score macro:** métrica principal ,calculada pela média do F1-score das 25 classes, atribundo o mesmo peso a cada classe.
- **Acurácia:** proporção de imagens classificadas corretamente.
- **Loss:** utilizada para acompanhar o comportamento do treinamento e da validação.

Os modelos são selecionados com base no maior F1-score macro obtido no conjunto de validação. O conjunto de teste é utilizado somentre para a avaliação final dos modelos selecionados.

Os resultados das cinco seeds são consolidados por média e desvio-padrão, permitindo analisar o desempenho e a variabilidade das configurações experimentais.

---

Desenvolvido como parte da disciplina de Redes Neurais e Aprendizado Profundo