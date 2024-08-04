import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from einops import rearrange, repeat


class InputEmbedding(nn.Module):
    def __init__(self, nodes_channels=62, input_dim=5, emb_size=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, emb_size)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn(nodes_channels + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _ = x.shape
        x = self.projection(x)
        cls_token = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_token, x], dim=1)
        x += self.positions
        return x


class InputEmbedding_eye(nn.Module):
    def __init__(self, input_dim: int = 31, emb_size: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, emb_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _ = x.shape
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_in_size, emb_out_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_in_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_in_size, emb_in_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_in_size, emb_out_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class EEG_SAM(nn.Sequential):
    def __init__(self, drop_p=0.1, **kwargs):
        super(EEG_SAM, self).__init__()
        self.attention_1 = MultiHeadAttention(128, 64, num_heads=1, dropout=0.1)
        self.normlayer_1 = nn.LayerNorm(64)
        self.attention_2 = MultiHeadAttention(64, 128, num_heads=1, dropout=0.1)
        self.normlayer_2 = nn.LayerNorm(128)
        self.attention_3 = MultiHeadAttention(128, 64, num_heads=1, dropout=0.1)
        self.normlayer_3 = nn.LayerNorm(64)
        self.activate = nn.ReLU()
        self.dropout = nn.Dropout(drop_p)

    def forward(self, input):
        output1 = self.normlayer_1(self.attention_1(input))
        output1_a = self.activate(output1)
        output2 = self.normlayer_2(self.attention_2(output1_a))
        output2_a = self.activate(output2)
        output3 = self.normlayer_3(self.attention_3(output2_a))
        output = self.activate(output1 + output3)
        return output


class EYE_SAM(nn.Sequential):
    def __init__(self, drop_p=0.1, **kwargs):
        super().__init__()
        self.attention_1 = MultiHeadAttention(256, 64, num_heads=1, dropout=0.1)
        self.normlayer_1 = nn.LayerNorm(64)
        self.attention_2 = MultiHeadAttention(64, 128, num_heads=1, dropout=0.1)
        self.normlayer_2 = nn.LayerNorm(128)
        self.attention_3 = MultiHeadAttention(128, 64, num_heads=1, dropout=0.1)
        self.normlayer_3 = nn.LayerNorm(64)
        self.activate = nn.ReLU()
        self.dropout = nn.Dropout(drop_p)

    def forward(self, input):
        output1 = self.normlayer_1(self.attention_1(input))
        output1_a = self.activate(output1)
        output2 = self.normlayer_2(self.attention_2(output1_a))
        output2_a = self.activate(output2)
        output3 = self.normlayer_3(self.attention_3(output2_a))
        output = self.activate(output1 + output3)
        return output


class CrossAttention(nn.Sequential):
    def __init__(self, drop_p=0.2,
                 **kwargs):
        super(CrossAttention, self).__init__()
        self.attention_1 = MultiHeadAttention(128, 32, num_heads=1, dropout=0.1)
        self.normlayer_1 = nn.LayerNorm(32)
        self.attention_2 = MultiHeadAttention(32, 64, num_heads=1, dropout=0.1)
        self.normlayer_2 = nn.LayerNorm(64)
        self.attention_3 = MultiHeadAttention(64, 32, num_heads=1, dropout=0.1)
        self.normlayer_3 = nn.LayerNorm(32)
        self.activate = nn.ReLU()
        self.dropout = nn.Dropout(drop_p)

    def forward(self, input):
        output1 = self.normlayer_1(self.attention_1(input))
        output1_a = self.activate(output1)
        output2 = self.normlayer_2(self.attention_2(output1_a))
        output2_a = self.activate(output2)
        output3 = self.normlayer_3(self.attention_3(output2_a))
        output = self.activate(output1 + output3)
        return output


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias, 0.1)
