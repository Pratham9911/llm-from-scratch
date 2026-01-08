import re

def gen_vocab(raw_text):
    result = re.split(r'([,.:;?_!|"()\']|--|\s)', raw_text)

    words = []
    for item in result:
        item = item.strip()
        if item != "":
            words.append(item)

    all_words = sorted(list(set(words)))
    all_words.extend(["<|endoftext|>","<|unk|>"])

    vocab = {}
    for index, word in enumerate(all_words):
        vocab[word] = index

    return vocab

class SimpleTokenizerV1:
  def __init__(self , vocab):
       self.str_to_int = vocab
       self.int_to_str = {}
   
       for word , idx in vocab.items():
         self.int_to_str[idx] = word
 
  def encode(self , text):
      
      preprocessed = re.split(r'([,.;:?_!"|()\']|--|\s)',text)
      cleaned=[]
      for item in preprocessed:
        item = item.strip()
        if item != "":
          cleaned.append(item)
      preprocessed = cleaned
      
      fixed = []
      for word in preprocessed:
       if word in self.str_to_int:
           fixed.append(word)
       else:
           fixed.append("<|unk|>")

      preprocessed = fixed

      ids = []
      for word in preprocessed:
        id = self.str_to_int[word]
        ids.append(id)
      
      return ids
  
  def decode(self , ids):
      words = []
      for token in ids :
        words.append(self.int_to_str[token])
      text = " ".join(words)
      text = re.sub(r'\s+([,.?!"()\'])',r'\1',text)
      return text