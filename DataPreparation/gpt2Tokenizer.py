import tiktoken as tk

tokenizer = tk.get_encoding("gpt2")

def encode_text(text):
    return tokenizer.encode(text , allowed_special={"<|endoftext|>"})

def decode_ids(ids):
    return tokenizer.decode(ids)

if __name__ == "__main__":
    sample_text = "Hello, world! This is a UnknownWord."
    encoded = encode_text(sample_text)
    print("Encoded IDs:", encoded)
    decoded = decode_ids(encoded)
    print("Decoded Text:", decoded)
