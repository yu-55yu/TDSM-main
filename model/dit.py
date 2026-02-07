import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import Mlp
from mamba_ssm import Mamba

# --- 方案二：部件索引定义 (基于 NTU-RGB+D 25 关节点标准) ---
# 这些索引用于在 Loss 计算时将全量关节点特征切分为不同部位 
PART_INDICES = {
    'hands': [4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23, 24], # 手臂与手部节点
    'legs': [12, 13, 14, 15, 16, 17, 18, 19],           # 腿部与脚部节点
    'torso': [0, 1, 2, 3, 20]                           # 躯干、颈部与头部节点
}

''' 鲁棒的调制函数：防止广播机制触发 OOM '''
def modulate(x, shift, scale):
    # 动态增加维度以匹配输入 x，防止 PyTorch 错误的广播导致显存爆炸
    for _ in range(len(x.shape) - len(shift.shape)):
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift

''' Multi-Head Self Attention (MHSA) - 保留备用 '''
class MHSA(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, qk_norm: bool = False,
                 attn_drop: float = 0., proj_drop: float = 0., norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

        self.qkv_c = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm_c = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm_c = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj_c = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        B, N_c, C = cond.shape
        qkv_c = self.qkv_c(cond).reshape(B, N_c, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q_c, k_c, v_c = qkv_c.unbind(0)
        q_c, k_c = self.q_norm_c(q_c), self.k_norm_c(k_c)

        q = torch.cat((q, q_c), dim=-2)
        k = torch.cat((k, k_c), dim=-2)
        v = torch.cat((v, v_c), dim=-2)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v

        out = out.transpose(1, 2).reshape(B, N + N_c, C)
        x, cond = out[:, :N], out[:, N:]

        x = self.proj(x)
        x = self.proj_drop(x)
        cond = self.proj_c(cond)
        cond = self.proj_drop(cond)
        return x, cond

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size, bias=True), nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

''' Mamba Diffusion Block (方案一：引入 Mamba 机制) '''
class MambaDiffusionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

        # 1. 骨骼时序建模：用 Mamba 替代 Self-Attention [cite: 7]
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mamba = Mamba(
            d_model=dim, 
            d_state=d_state,  
            d_conv=d_conv,    
            expand=expand,    
        )

        # 2. 文本交互层：保留 Cross-Attention [cite: 8]
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

        # 3. 前馈网络层
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x, text_emb, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        # --- Mamba 时序建模 (带自适应调制) ---
        res = x
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        x = res + gate_msa.unsqueeze(1) * self.mamba(x_norm)

        # --- Cross-Attention 文本引导 ---
        res = x
        x_attn, _ = self.cross_attn(self.norm2(x), text_emb, text_emb)
        x = res + x_attn

        # --- FFN 前馈层 (带自适应调制) ---
        res = x
        x_norm = modulate(self.norm3(x), shift_mlp, scale_mlp)
        x = res + gate_mlp.unsqueeze(1) * self.ffn(x_norm)
        
        return x, text_emb

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

''' Diffusion Transformer (DiT) '''
class DiT(nn.Module):
    def __init__(self, in_channels=256, cond_size=2048, hidden_size=768, depth=12, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads

        self.x_embedder = nn.Linear(in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.fc_embedder = nn.Sequential(nn.Linear(cond_size, hidden_size, bias=True), nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True))
        self.fl_embedder = nn.Linear(cond_size, hidden_size, bias=True)
        self.fl_pos_embed = nn.Parameter(torch.zeros(1, 35, hidden_size), requires_grad=True)

        # 使用 MambaDiffusionBlock 堆叠
        self.blocks = nn.ModuleList([MambaDiffusionBlock(dim=hidden_size, num_heads=num_heads) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, in_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.fl_pos_embed, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, fc, fl):
        """
        x: [B, 25, 256] 
        t: [B]
        fc: [B, 1024] 
        fl: [B, 35, 1024]
        """
        x = self.x_embedder(x)
        fl = self.fl_embedder(fl) + self.fl_pos_embed
        t = self.t_embedder(t)
        fc = self.fc_embedder(fc)
        
        c = t + fc
        for block in self.blocks:
            x, fl = block(x, fl, c)
            
        x = self.final_layer(x, c)
        return x