import numpy as np
import pandas as pd

np.random.seed(42)

vocabulario = {
    "O": 0,
    "banco": 1,
    "bloqueou": 2,
    "cartão": 3
}

df_vocab = pd.DataFrame(list(vocabulario.items()), columns=["palavra", "id"])
print("DataFrame do Vocabulário:")
print(df_vocab)
print()

frase = "O banco bloqueou cartão"
tokens = frase.split()

ids_frase = [vocabulario[palavra] for palavra in tokens]
print("IDs da frase:")
print(ids_frase)
print()

vocab_size = len(vocabulario)
d_model = 64

embedding_table = np.random.randn(vocab_size, d_model)

print("Shape da tabela de embeddings:")
print(embedding_table.shape)
print()

X_sem_batch = embedding_table[ids_frase]
X = np.expand_dims(X_sem_batch, axis=0)

print("Shape de X:")
print(X.shape)
print()

def softmax(x):
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def layer_norm(x, epsilon=1e-6):
    media = np.mean(x, axis=-1, keepdims=True)
    variancia = np.var(x, axis=-1, keepdims=True)
    return (x - media) / np.sqrt(variancia + epsilon)

def relu(x):
    return np.maximum(0, x)

def self_attention(X):
    batch_size, sequence_length, d_model = X.shape
    d_k = d_model

    W_Q = np.random.randn(d_model, d_model)
    W_K = np.random.randn(d_model, d_model)
    W_V = np.random.randn(d_model, d_model)

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    K_T = np.transpose(K, (0, 2, 1))
    scores = Q @ K_T
    scores_scaled = scores / np.sqrt(d_k)

    attention_weights = softmax(scores_scaled)
    attention_output = attention_weights @ V

    return attention_output, attention_weights

def feed_forward(X, d_ff=256):
    batch_size, sequence_length, d_model = X.shape

    W1 = np.random.randn(d_model, d_ff)
    b1 = np.random.randn(d_ff)

    W2 = np.random.randn(d_ff, d_model)
    b2 = np.random.randn(d_model)

    hidden = X @ W1 + b1
    hidden_relu = relu(hidden)

    output = hidden_relu @ W2 + b2
    return output

N = 6

for camada in range(N):
    print(f"CAMADA {camada + 1}")

    X_att, attention_weights = self_attention(X)
    print("Shape de X_att:", X_att.shape)
    print("Shape de attention_weights:", attention_weights.shape)

    X_norm1 = layer_norm(X + X_att)
    print("Shape de X_norm1:", X_norm1.shape)

    X_ffn = feed_forward(X_norm1, d_ff=256)
    print("Shape de X_ffn:", X_ffn.shape)

    X_out = layer_norm(X_norm1 + X_ffn)
    print("Shape de X_out:", X_out.shape)
    print()

    X = X_out

print("Shape final de X:", X.shape)
print("Vetor Z final:")
print(X)