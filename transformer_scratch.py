import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size:int, heads:int):
        '''
        Makes a self attention layer for transformer architecture.
        :param embed_size: The size of the embedding. d_model from paper, Usually the word embedding dimension size.
        :param heads: The number of the heads in the multi-head attention layer.
        '''
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads  # This must not have 나머지.
        assert embed_size % heads == 0, "Embed size must be divisible by heads number."

        # nn.Linear applies linear transformation. -- 그냥 일반 cell하나 아닌감?
        # it takes input dim, output dim. -- 여기서는 head 내부 dimension임.
        self.values = nn.Linear(self.head_dim,self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim,self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim,self.head_dim, bias=False)
        # fully connected feed forward layer -- 이게 두번째 sublayer
        # 이 layer는 모든 head에서 나온 정보를 총합해서 사용한다.
        self.fc_out = nn.Linear(self.head_dim * self.heads, embed_size)  #둘은 같다.

    def forward(self, keys, values, queries, mask):
        # mask는 decoder에서만 사용한다.
        N = queries.shape[0]  # head 내부 dimension -- question들의 갯수
        val_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]
        # feature 갯수, 여기서는 문장속 단어수.