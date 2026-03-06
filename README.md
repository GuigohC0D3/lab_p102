# Lab P102 - Transformer Encoder Implementation

Uma implementação do **Transformer Encoder** do zero usando **NumPy e Pandas**, baseado na arquitetura descrita no paper "Attention Is All You Need" (Vaswani et al., 2017).

## Descrição do Projeto

Este projeto implementa um encoder transformer completo com as seguintes componentes:

- **Vocabulary Tokenization**: Mapeia palavras para IDs
- **Embedding Layer**: Converte tokens em vetores densos de embeddings
- **Self-Attention Mechanism**: Calcula pesos de atenção entre tokens
- **Feed-Forward Network**: Redes neurais posicionais dentro de cada camada
- **Layer Normalization**: Normalização de camadas para estabilidade
- **Multi-Layer Stack**: Empilhamento de 6 camadas de transformers

### Entrada Exemplo
```
"O banco bloqueou cartão"
```

### Saída
Um vetor de representação final (shape: 1 × 4 × 64) contendo as embeddings contextualizadas após passarem por todas as 6 camadas do encoder.

## Estrutura do Projeto

```
lab_p102/
├── README.md              # Este arquivo
├── requirements.txt       # Dependências do projeto
└── enconder/
    └── task_encoder.py    # Implementação do Transformer Encoder
```

## Configuração do Ambiente

### Pré-requisitos
- Python 3.8+

### 1. Criar um Virtual Environment

```bash
python -m venv venv
```

### 2. Ativar o Virtual Environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.\venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### Instalar Dependências

```bash
pip install -r requirements.txt
```

## Como Rodar

Após ativar o virtual environment e instalar as dependências, execute:

```bash
python enconder/task_encoder.py
```

## Saída Esperada

O script irá exibir:

1. **Vocabulário:** DataFrame com palavras e seus IDs
2. **Tokens:** IDs da frase tokenizada
3. **Shape dos Embeddings:** Dimensões da tabela de embeddings
4. **Shape de X:** Dimensões da entrada processada
5. **Para cada uma das 6 camadas:**
   - Shape da saída da self-attention
   - Shape dos pesos de atenção
   - Shape após layer normalization 1
   - Shape após feed-forward network
   - Shape após layer normalization 2
6. **Shape Final de X:** Dimensão do vetor de saída final
7. **Vetor Z Final:** Representação contextualizada do texto

## Componentes Principais


### 1. **Self-Attention**
Calcula a relevância de cada token em relação aos outros tokens da sequência:
- Projeta inputs em Query (Q), Key (K) e Value (V)
- Computa scores de compatibilidade entre Q e K
- Aplica softmax para obter pesos de atenção
- Combina values ponderados pelos pesos

### 2. **Feed-Forward Network**
Rede neural posicional com duas camadas:
- Primeira camada: expande dimensionalidade (d_model → d_ff)
- Ativação ReLU
- Segunda camada: reduz para dimensionalidade original (d_ff → d_model)

### 3. **Layer Normalization**
Normaliza as embeddings para manter valores em escala apropriada durante o treinamento.

### 4. **Residual Connections**
Adições de skip connections (X + output) para facilitar o fluxo de gradientes.

## Parâmetros Configuráveis

- **vocab_size**: 4 (número de palavras no vocabulário)
- **d_model**: 64 (dimensionalidade dos embeddings)
- **d_ff**: 256 (dimensionalidade interna do feed-forward)
- **N**: 6 (número de camadas do encoder)

## Referências

- Vaswani et al. (2017): "Attention Is All You Need" - https://arxiv.org/abs/1706.03762
- Implementação baseada em arquitetura padrão de Transformers
