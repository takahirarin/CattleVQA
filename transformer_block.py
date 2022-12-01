import torch
import torch.nn as nn
import math
from vilt_config import ViltConfig

class Attention(nn.Module):
    def __init__(self,config):
            super(Attention, self).__init__()
            
    def forward(self, query, key, value, attention_mask,attention_dropout_prob):
        """
            scores: [batch_size, head, emb_size, emb_size]
             (= [batch_size, head, emb_size, hidden_size/head] * [batch_size, head, hidden_size/head, emb_size])
            attention_mask: [batch_size, 1, 1, emb_size]

            masked_fill後
            scores: [batch_size, 1, 1, emb_size]
            
            return: [batch_size, head, emb_size, hidden_size/head]
        """



        scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(query.size(-1)) #(query * key^T) / √d_k

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, 0)
            #0のところを∞に置き換えてマスクする？

        attn = scores.softmax(dim=-1)
        #行方向にsoftmax

        if attention_dropout_prob is not None:
            attn = attention_dropout_prob(scores)
        
        # attn: [batch_size, 1, 1, emb_size], value: [batch_size, head, emb_size, hidden_size/head]
        # attn * value = [batch_size, head, emb_size, hidden_size/head]

        # broadcastされるので、attn:[batch_size, 1, 1, emb_size] -> [batch_size, head, emb_size, emb_size]となり、かけると
        # [batch_size, head, emb_size, hidden_size/head]となる
        # * 実質3,4次元目の部分のみの行列計算
        return torch.matmul(attn, value), attn




class MultiHeadAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.linear_layers = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(config.num_attention_heads)])
        self.output_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention = Attention(config)



    def forward(self,query, key, value, head, hidden_size, attention_mask=None, attention_dropout_prob=None):
        """"
            query, key, value: [batch_size, emb_size, hidden_size]
            attention_mask: [batch_size, emb_size]

            linear_layersでheadの数と合うように変換
            query, key, value: [batch_size, head, emb_size, hidden_size/head ]
            attention_output: [batch_size, head, emb_size, hidden_size/head ]

            attentionのheadの数をかけて元の形に戻す
            attention_output: [batch_size, emb_size, hidden_size ]
        """


        batch_size = query.size(0)
        d_k = hidden_size // head

        query, key, value = [l(x).view(batch_size, -1, head, d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
                            #h個のattention層に分けるために(batch_size, -1, head, d_k)の形にする
        
        attention_output, attn = self.attention(query, key, value, attention_mask, attention_dropout_prob)
        
        attention_output = attention_output.transpose(1,2).contiguous().view(batch_size, -1, head * d_k)
        #41行目でtranspose(1,2)してたので戻すためのtranspose
        #viewを使うときは要素順に並んでいないといけないのでそのためのcontiguous()
        #[batch_size, 1, hidden_size] で出力

        return self.output_linear(attention_output)

class FFN(nn.Module):
    def __init__(self,config):
        super(FFN,self).__init__()
        self.dense  = nn.Linear(config.hidden_size,config.intermediate_size)
        self.activation = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states



class SubLayerConnection(nn.Module):
    def __init__(self,config):
        super(SubLayerConnection,self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self,hidden_states,input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.norm(hidden_states + input_tensor)
        return hidden_states






if __name__ == '__main__':
    config = ViltConfig()
    head = config.num_attention_heads   # h
    hidden_size = config.hidden_size  #  d_model

    batch_size = 2

    input = torch.randn(batch_size, 1, config.hidden_size)
    print(f'input_size: {input.size()}')

    encoder = MultiHeadAttention(config)
    output = encoder(input, input, input, head, hidden_size)

    print(output)





