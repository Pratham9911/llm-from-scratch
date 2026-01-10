from embeddings import get_embeddings

with open('the-verdict.txt','r',encoding='utf-8') as f:
  raw_text = f.read()

VOCAB_SIZE = 50257
embeddings = get_embeddings(
    raw_text, VOCAB_SIZE, embedding_dim=64, batch_size=2, max_length=16, stride=8, shuffle=True, num_worker=0
)

print(embeddings.shape)  # Should print torch.Size([2, 16, 64])
