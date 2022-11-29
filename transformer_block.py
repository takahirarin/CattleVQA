import torch
import torch.nn as nn
import math
from vilt_config import ViltConfig

class Attention(nn.Module):
    def __init__(self,config):
            super(Attention, self).__init__()
            
    def forward(self, query, key, value, attention_mask,attention_dropout_prob):
        scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(query.size(-1)) #(query * key^T) / √d_k
        # scores.size() = [batch_size, head, 1, 1]
        # query*keyで,[batch_size, head, 1, d_k] * [batch_size, head, 1, d_k]^T = [batch_size, head, 1, 1]
        # (実質、[1, d_k]*[d_k,1] = [1,1]となっている)

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
            #0のところを∞に置き換えてマスクする？

        attn = scores.softmax(dim=-1)
        #行方向にsoftmax

        if attention_dropout_prob is not None:
            attn = attention_dropout_prob(scores)
        
        # attn: [batch_size, head, 1, 1], value: [batch_size, head, 1, d_k]
        # attn*value = [batch_size, head, 1, d_k]
        return torch.matmul(attn, value), attn




class MultiHeadAttention(nn.Module):
    def __init__(self,config):
        super().__init__() # super(MultiHeadAttention).__init__()にするとエラーになる
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(head)])
        self.output_linear = nn.Linear(hidden_size, hidden_size)
        self.attention = Attention(config)

        # def __init__(self):
        #     super().__init__()
        # self.conv1 = nn.Conv2d(1, 20, 5)
        # self.conv2 = nn.Conv2d(20, 20, 5)
      
    def forward(self,query, key, value, head, hidden_size, attention_mask=None, attention_dropout_prob=None):
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




