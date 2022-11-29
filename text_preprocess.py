import torch
import torch.nn as nn


class BERTEmbeddings(nn.Module):
    def __init__(self,config) :
         #super(BERTEmbeddings, self).__init__()
         super().__init__()
         """Construct the embedding module from word, position and token_type embeddings"""
         self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
         self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
         self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
         self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
         self.dropout = nn.Dropout(config.dropout_prob)
    
    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        word_embedding = self.word_embeddings(input_ids)
        position_embedding = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = word_embedding + position_embedding + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BERTLayerNorm(nn.Module):
    def __init__(self) -> None:
         super().__init__()


if __name__ == '__main__':
        pass
    
