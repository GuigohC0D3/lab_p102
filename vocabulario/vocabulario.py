import numpy as np
import pandas as pd

np.random.seed(42)

vocabulario = {
    "O": 0,
    "banco": 1,
    "bloqueou": 2,
    "cartão": 3
}

db_vocab = pd.DataFrame(list(vocabulario.items()), columns=["palavra", "id"])
print("DataFrame do Vocabulário:")
print(db_vocab)
print()

frase = "O banco bloqueou cartão"

token = frase.split()

ids_frase = [vocabulario[palavra] for palavra in token]
print("IDs da frase:")
print(ids_frase)

vocab_size = len(vocabulario)
d_model = 64

embedding_table = np.random.randn(vocab_size, d_model)

print("Shape da tabela de embeddings:")
print(embedding_table.shape)

X_sem_batch = embedding_table[ids_frase]

X = np.expand_dims(X_sem_batch, axis=0)

print("Shape de X:")
print(X.shape)