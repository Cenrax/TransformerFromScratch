import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model : int, vocab_size : int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model : int, seq_len : int = 5000, dropout : float=0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.seq_len = seq_len

        # Create a matrix of shape (seq_len, d_model) containing values from 0 to seq_len - 1
        pe = torch.zeros(seq_len, d_model)

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) # (seq_len, d_model / 2)
        pe[:, 1::2] = torch.cos(position * div_term) # (seq_len, d_model / 2)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)
    

class LayerNormalization(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.eps = 1e-6
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model : int, d_ff : int, dropout : float = 0.1) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2
        
    def forward(self, x):
        x = self.dropout(torch.relu(self.linear_1(x)))
        return self.linear_2(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model : int, heads : int, dropout : float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask = None):
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        key = self.w_k(k).view(bs, -1, self.h, self.d_k) # (bs, seq_len, heads, d_k)
        query = self.w_q(q).view(bs, -1, self.h, self.d_k) # (bs, seq_len, heads, d_k)
        value = self.w_v(v).view(bs, -1, self.h, self.d_k) # (bs, seq_len, heads, d_k)
        
        # transpose to get dimensions bs * h * seq_len * d_k
        key = key.transpose(1, 2) # (bs, heads, seq_len, d_k)
        query = query.transpose(1, 2) # (bs, heads, seq_len, d_k)
        value = value.transpose(1, 2) # (bs, heads, seq_len, d_k)
        
        # calculate attention using function we will define next
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout) # (bs, heads, seq_len, d_k)
        
        # concatenate heads and put through final linear layer
        concat = x.transpose(1, 2).contiguous().view(bs, -1, self.d_model) # (bs, seq_len, d_model)
        output = self.out(concat) # (bs, seq_len, d_model)
    
        return output
    
    @staticmethod
    def attention( query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) /  math.sqrt(d_k) # (
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_scores = torch.softmax(scores, dim=-1) # (bs, heads, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores
    

class ResidualConnection(nn.Module):
    def __init__(self, d_model : int, dropout : float = 0.1) -> None:
        super().__init__()
        self.norm = LayerNormalization()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout : float = 0.1) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(self_attention_block.d_model, dropout), ResidualConnection(self_attention_block.d_model, dropout)])
        
        
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
    

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout : float = 0.1) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model : int, vocab_size : int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        return torch.log_softmax(self.linear(x), dim=-1)
    

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        return self.encoder(self.src_pos(self.src_embed(src)), src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)


    def project(self, x):
        return self.projection_layer(x)
    


def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len: int, tgt_seq_len: int, d_model : int = 512, N: int = 6, heads: int =8, dropout = 0.1, d_ff: int = 2048):
    src_embed = InputEmbeddings( d_model, src_vocab_size)
    tgt_embed = InputEmbeddings( d_model, tgt_vocab_size)

    src_pos = PositionalEncoding(d_model, src_seq_len, dropout=dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout=dropout)

    encoder_blocks = []

    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention( d_model,heads=heads, dropout=dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout=dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout=dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []

    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention( d_model, heads=heads, dropout=dropout)
        decoder_cross_attention_block = MultiHeadAttention( d_model, heads=heads, dropout=dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout=dropout)
        decoder_blocks.append(DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout=dropout))

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    #Initialize the parameters

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer




