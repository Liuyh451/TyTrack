from models.cvae_gan import Cvae_Gan
from models.modules import *


class ARHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=64):
        super(ARHead, self).__init__()

        # 1. 残差预测网络
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Tanh()  # 锁定在 (-1, 1)
            # nn.Softsign()
        )

        # 2. 定义【范围 2】的物理跨度 (用于尺度缩放)
        # 纬度: 25.271 - 5.7 = 19.571
        # 经度: 129.669 - 102.4 = 27.269
        self.register_buffer('lat_span', torch.tensor(40.228))
        self.register_buffer('lon_span', torch.tensor(78.700))

        # 3. 动态半径物理参数 (单位：度)
        self.lat_k, self.lat_b = 0.015, 0.15  # 南海是 0.008, 0.1
        self.lon_k, self.lon_b = 0.02, 0.25  # 南海是 0.012, 0.15
    def forward(self, features, ensemble_mean, target_t):
        # A. 计算归一化位移 [B, 2]，取值 (-1, 1)
        delta_norm = self.fc(features)

        # B. 计算物理纠偏半径 (物理单位：度)
        r_lat_phys = self.lat_k * target_t + self.lat_b
        r_lon_phys = self.lon_k * target_t + self.lon_b

        # C. 【关键】将物理半径转化为归一化半径
        # 公式: r_norm = r_phys / Span
        r_lat_norm = r_lat_phys / self.lat_span
        r_lon_norm = r_lon_phys / self.lon_span

        # 构造半径张量
        if torch.is_tensor(target_t):
            radius_norm = torch.stack([r_lat_norm, r_lon_norm], dim=-1)
        else:
            radius_norm = torch.tensor([r_lat_norm, r_lon_norm], device=features.device)
        
        # D. 物理映射：最终输出 = 锚点 + (残差 * 归一化半径)
        # delta_norm 控制方向和强度(-1~1)，radius_norm 控制物理边界
        final_coords = ensemble_mean + delta_norm * radius_norm*1.5

        return final_coords


class TrackPredictor(nn.Module):
    def __init__(self, input_dim):
        super(TrackPredictor, self).__init__()
        self.t=12
        # 轨迹编码器
        self.track_encoder = TrackEncoderWithGru(
            input_dim=14,
            hidden_dim=128
        )
        # 环境编码器
        self.env_encoder = TimeAwareEncoderForENVLighting(
            point_num=9,
            image_size=(64, 64),
            hidden_dim=256,
            input_channels=28
        )
        # 多模态融合器
        self.fusion_model = AttentionFusionWithResidual(256, 256)
        # 解码器
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )
        # 输出层
        self.fc_out = nn.Linear(256, 2)
        # Dropout层
        self.dropout = nn.Dropout(0.2)

        self.ar_head = ARHead(input_dim=256)

    def forward(self, x_env, x_path, x_path_mask, X_mean):
        # 路径编码 [1, 256] [1, 1, 256] [1, 8, 8]
        track_encoded, hidden, attention_weights = self.track_encoder(x_path, x_path_mask)  # [B, track_hidden_dim]
        # 使用轨迹编码结果作为环境编码器的node_history_encoded输入
        env_features = self.env_encoder(x_env, track_encoded)  # [B, env_hidden_dim] [1, 256]
        # 融合
        fusion = self.fusion_model(track_encoded, env_features)  # [1, 256]
        # GRU前向传播
        gru_out, _ = self.gru(fusion.unsqueeze(1), hidden)  # [1, 1, 256]
        out = self.ar_head(gru_out.squeeze(1), X_mean, self.t)
        return out


class Cvae_Gan_Seq(nn.Module):
    def __init__(self, model_path):
        super(Cvae_Gan_Seq, self).__init__()
        self.cvae_gan = Cvae_Gan()
        self.cvae_gan.load_state_dict(torch.load(model_path, map_location='cpu'))
        # 冻结 cvae_gan 的所有参数
        for param in self.cvae_gan.parameters():
            param.requires_grad = False
        print("CVAEGan参数加载并冻结")

        self.track_predictor = TrackPredictor(256)
    def forward(self, model_forecasts, combined_mask,X_mean):
        # 生成环境场
        env_generated = self.cvae_gan.inference(model_forecasts, combined_mask)
        out= self.track_predictor(env_generated, model_forecasts, combined_mask,X_mean)
        return out

