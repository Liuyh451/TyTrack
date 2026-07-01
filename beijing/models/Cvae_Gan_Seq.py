import torch
from torch import nn

from models.cvae_gan import Cvae_Gan
from models.modules import TimeAwareEncoderForENVLighting, AttentionFusionWithResidual, TrackEncoderWithGru


class TrackPredictor(nn.Module):
    def __init__(self, input_dim):
        super(TrackPredictor, self).__init__()
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
        #解码器
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

    def forward(self, x_env, x_path, x_path_mask):
        #路径编码 [1, 256] [1, 1, 256] [1, 8, 8]
        track_encoded, hidden,attention_weights = self.track_encoder(x_path, x_path_mask)  # [B, track_hidden_dim]
        # 使用轨迹编码结果作为环境编码器的node_history_encoded输入
        env_features = self.env_encoder(x_env, track_encoded)  # [B, env_hidden_dim] [1, 256]
        #融合
        fusion = self.fusion_model(track_encoded, env_features) #[1, 256]
        # fusion=track_encoded
        # GRU前向传播
        gru_out, _ = self.gru(fusion.unsqueeze(1), hidden)#[1, 1, 256]
        # 测试：断开历史惯性
        # gru_out, _ = self.gru(fusion.unsqueeze(1), torch.zeros_like(hidden))
        out = self.fc_out(gru_out).squeeze(1)#[1, 2]
        return out,attention_weights


class Cvae_Gan_Seq(nn.Module):
    def __init__(self, model_path):
        super(Cvae_Gan_Seq, self).__init__()
        self.cvae_gan = Cvae_Gan()
        self.cvae_gan.load_state_dict(torch.load(model_path))
        # 冻结 cvae_gan 的所有参数
        for param in self.cvae_gan.parameters():
            param.requires_grad = False
        print("CVAEGan参数加载并冻结")
        self.track_predictor = TrackPredictor(256)

    def forward(self, model_forecasts, combined_mask):
        # 生成环境场
        env_generated = self.cvae_gan.inference(model_forecasts, combined_mask)
        out,attention_weights=self.track_predictor(env_generated, model_forecasts, combined_mask)
        return out,attention_weights
