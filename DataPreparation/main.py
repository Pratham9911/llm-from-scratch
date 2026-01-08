from processing import gen_vocab , SimpleTokenizerV1

with open('the-verdict.txt','r',encoding='utf-8') as f:
  raw_text = f.read()

vocab = gen_vocab(raw_text)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1 , text2))

print(text)
tokenizer = SimpleTokenizerV1(vocab)
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)

print("(encoded):", encoded)
print("Decoded text :", decoded)

