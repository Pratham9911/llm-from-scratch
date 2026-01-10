From Text to Final Embeddings 

This project shows how raw text is converted into GPT-ready embeddings step by step.

🔹 Step 1: Raw Text

We start with plain text (a book, paragraph, etc.).

"Hello, do you like tea?"

🔹 Step 2: Tokenization

The text is converted into token IDs using a tokenizer.

Text → [10, 45, 78, 91]


Tokens are numbers.
Models cannot understand text directly.

🔹 Step 3: Dataset (Sliding Window)

The token IDs are split into input–target pairs using a sliding window.

Example (max_length = 4):

Input  → [10, 45, 78, 91]
Target → [45, 78, 91, 33]


This trains the model to predict the next token.

🔹 Step 4: DataLoader (Batching)

The DataLoader groups many input–target pairs into batches.

Example (batch_size = 2):

Inputs  → shape (2, 4)
Targets → shape (2, 4)


Batching makes training efficient.

🔹 Step 5: Token Embeddings

Each token ID is converted into a vector using an embedding layer.

Token IDs → Vectors


Shape becomes:

(batch_size, seq_length, embedding_dim)


Example:

(2, 4, 64)

🔹 Step 6: Positional Embeddings

Since models don’t know word order, we add position information.

One vector per position

Same positions shared across the batch

Position 0 → vector
Position 1 → vector
...

🔹 Step 7: Final Embeddings

Token embeddings and positional embeddings are added together.

Final embedding = token embedding + positional embedding


Final shape (GPT input):

(batch_size, seq_length, embedding_dim)


Example:

torch.Size([8, 4, 256])

