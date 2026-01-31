import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import Mlp
from mamba_ssm import Mamba


''' Multi-Head Self Attention (MHSA) '''
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
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # Linear
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        B, N_c, C = cond.shape
        qkv_c = self.qkv_c(cond).reshape(B, N_c, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # Linear
        q_c, k_c, v_c = qkv_c.unbind(0)
        q_c, k_c = self.q_norm_c(q_c), self.k_norm_c(k_c)

        # Token-wise concatenate
        q = torch.cat((q, q_c), dim=-2)
        k = torch.cat((k, k_c), dim=-2)
        v = torch.cat((v, v_c), dim=-2)

        # Multi-head self-attention
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v

        # Token-specific split
        out = out.transpose(1, 2).reshape(B, N + N_c, C)
        x, cond = out[:, :N], out[:, N:]

        x = self.proj(x)
        x = self.proj_drop(x)

        cond = self.proj_c(cond)
        cond = self.proj_drop(cond)
        return x, cond


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)  # Scale-Shift


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


''' CrossDiT Block '''
class CrossDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.norm1_c = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2_c = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = MHSA(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.mlp_c = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.adaLN_modulation_c = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, cond, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        shift_msa_c, scale_msa_c, gate_msa_c, shift_mlp_c, scale_mlp_c, gate_mlp_c = self.adaLN_modulation_c(c).chunk(6, dim=1)

        x_temp = modulate(self.norm1(x), shift_msa, scale_msa)  # Scale-Shift
        cond_temp = modulate(self.norm1_c(cond), shift_msa_c, scale_msa_c)  # Scale-Shift

        x_temp, cond_temp = self.attn(x_temp, cond_temp)  # MHSA

        x = x + gate_msa.unsqueeze(1) * x_temp  # Scale
        cond = cond + gate_msa_c.unsqueeze(1) * cond_temp  # Scale

        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))  # Scale-Shift-FeedForward-Scale
        cond = cond + gate_mlp_c.unsqueeze(1) * self.mlp_c(modulate(self.norm2_c(cond), shift_mlp_c, scale_mlp_c))  # Scale-Shift-FeedForward-Scale
        return x, cond
        




class MambaDiffusionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, d_state=16, d_conv=4, expand=2):
        super().__init__()
        # DiT 的核心：用于注入 timestep 信息 (c) 的调制层
        # 6 * dim 分别对应 mamba 和 ffn 的 shift, scale, gate
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

        # 1. 骨骼时序建模层：用 Mamba 替代原版的 Self-Attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mamba = Mamba(
            d_model=dim, 
            d_state=d_state,  
            d_conv=d_conv,    
            expand=expand,    
        )

        # 2. 文本交互层：保留 Cross-Attention 
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
        """
        x: [B, L, D] 骨骼序列
        text_emb: [B, L_text, D] 文本嵌入
        c: [B, D] 时间步嵌入 (timestep embedding)
        """
        # 计算 adaLN 的调制参数
        # shift, scale 控制归一化分布；gate 控制残差连接的权重
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        # --- 第一步：Mamba 时序建模 (带时间调制) ---
        res = x
        x_norm = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x = res + gate_msa.unsqueeze(1) * self.mamba(x_norm)

        # --- 第二步：Cross-Attention 文本对齐 ---
        res = x
        # 注意：Cross-Attention 通常不需要像 Self-Attention 那样复杂的调制，直接做 norm 即可
        x_attn, _ = self.cross_attn(self.norm2(x), text_emb, text_emb)
        x = res + x_attn

        # --- 第三步：FFN 层 (带时间调制) ---
        res = x
        x_norm = self.norm3(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = res + gate_mlp.unsqueeze(1) * self.ffn(x_norm)
        
        return x,text_emb
        



class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)  # Scale-Shift
        x = self.linear(x)
        return x


''' Diffusion Transformer '''
class DiT(nn.Module):
    def __init__(self, in_channels=256, cond_size=2048, hidden_size=768, depth=12, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads

        self.x_embedder = nn.Linear(in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.fc_embedder = nn.Sequential(nn.Linear(cond_size, hidden_size, bias=True), nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True))
        self.fl_embedder = nn.Linear(cond_size, hidden_size, bias=True)
        self.fl_pos_embed = nn.Parameter(torch.zeros(1, 35, hidden_size), requires_grad=True)  # M_l = 35

        self.blocks = nn.ModuleList([MambaDiffusionBlock(dim=hidden_size) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, in_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # 1. 基础线性层初始化
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # 2. 修正 Positional Embedding
        nn.init.normal_(self.fl_pos_embed, std=0.02)

        # 3. 修正 Timestep Embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # 4. 核心：对每一个 Block 的 adaLN 最后一层清零 (DiT 训练技巧)
        for block in self.blocks:
            if hasattr(block, 'adaLN_modulation'):
                # 最后一层 Linear 置零，保证训练初期残差路径起主导作用
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # 5. Final layer 零初始化
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, fc, fl):
        x = self.x_embedder(x)  # (B, M_x , D)
        fl = self.fl_embedder(fl) + self.fl_pos_embed  # (B, M_l, D)
        t = self.t_embedder(t)  # (B, D)
        fc = self.fc_embedder(fc)  # (B, D)
        fc = t + fc
        for block in self.blocks:
            x, fl = block(x, fl, fc)  # (B, M_x, D)
        x = self.final_layer(x, fc)
        return x