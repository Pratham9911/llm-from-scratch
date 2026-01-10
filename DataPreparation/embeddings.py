import torch

from data_loader import create_dataloader_v1

def get_embeddings(raw_text  ,vocab_size, embedding_dim ,   batch_size = 4 , max_length = 4 , stride = 128 , shuffle = True , num_worker = 0):
   
    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
    
    dataloader = create_dataloader_v1(
        raw_text , batch_size , max_length , stride , shuffle , num_worker
     )

    data_iter = iter(dataloader)
    inputs , targets = next(data_iter)

    token_embeddings = embedding_layer(inputs)
 
    pos_embeddings_layer = torch.nn.Embedding(num_embeddings=max_length , embedding_dim=embedding_dim)

    pos_embeddings = pos_embeddings_layer(torch.arange( max_length))

    final_embeddings = token_embeddings + pos_embeddings
    return final_embeddings
    