import torch
from torch import nn, optim
from models.modules import track_encoder_for_gan


class Context_Encoder(nn.Module):
    def __init__(self):
        super(Context_Encoder, self).__init__()
        self.track_encoder = track_encoder_for_gan(
            input_dim=14,
            hidden_dim=256
        )

    def forward(self, track_data,mask):
        # 使用轨迹编码器处理轨迹数据
        track_encoded, hidden = self.track_encoder(track_data,mask)  # [B, track_hidden_dim][4, 256]
        return hidden.squeeze(0)

class EnvFieldEncoderWithPath(nn.Module):
    def __init__(self, input_channels=14, time_steps=9, latent_dim=512):
        """
        波场编码器，融合浮标上下文向量，支持时间序列数据
        :param input_channels: 波场输入的通道数（默认为4）
        :param context_dim: 浮标上下文向量的维度（默认为512）
        :param latent_dim: 输出的潜在向量（μ和σ）的维度（默认为512）
        :param time_steps: 时间步数（默认为10）
        """
        super(EnvFieldEncoderWithPath, self).__init__()
        self.time_steps = time_steps

        # 小模块 A：3D卷积 + BatchNorm + ReLU + MaxPooling (处理时空特征)
        self.module_a = nn.Sequential(
            nn.Conv3d(in_channels=input_channels, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # 时间维度保持不变，空间下采样
        )

        self.module_a1 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )

        self.module_a2 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )

        self.module_a3 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )

        # 小模块 B：3D卷积 + BatchNorm + ReLU + 全局平均池化
        self.module_b = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(output_size=(time_steps, 1, 1))  # 保持时间维度，空间全局平均池化
        )

        # 时间维度压缩层
        self.time_compress = nn.Sequential(
            nn.Conv1d(512 * time_steps, 512, kernel_size=1),
            nn.ReLU()
        )

        # 1x1x1 卷积生成 μ 和 σ
        self.conv_mu = nn.Conv3d(in_channels=1024, out_channels=latent_dim, kernel_size=1)
        self.conv_sigma = nn.Conv3d(in_channels=1024, out_channels=latent_dim, kernel_size=1)

    def forward(self, wave_field, context_vector):
        """
        前向传播
        :param wave_field: 波场输入 (batch_size, time_steps, channels, height, width)
        :param context_vector: 浮标上下文向量 (batch_size, context_dim)
        :return: μ 和 σ (batch_size, latent_dim, 1, 1)
        """
        # 调整输入维度顺序: (B, T, C, H, W) -> (B, C, T, H, W)
        wave_field = wave_field.permute(0, 2, 1, 3, 4)

        # 1. 通过小模块 A 提取波场的时空特征
        wave_features = self.module_a(wave_field)  # (batch_size, 64, T, 64, 64)
        wave_features = self.module_a1(wave_features)  # (batch_size, 128, T, 32, 32)
        wave_features = self.module_a2(wave_features)  # (batch_size, 256, T, 16, 16)
        wave_features = self.module_a3(wave_features)  # (batch_size, 512, T, 8, 8)

        # 2. 通过小模块 B 提取最终的特征向量
        wave_features = self.module_b(wave_features)  # (batch_size, 512, T, 1, 1)

        # 3. 压缩时间维度
        batch_size = wave_features.shape[0]
        # 将时间维度和通道维度合并: (B, 512, T, 1, 1) -> (B, 512*T, 1, 1)
        wave_features_flat = wave_features.view(batch_size, -1, 1, 1)
        # 压缩回512维: (B, 512, 1, 1)
        wave_features_compressed = self.time_compress(wave_features_flat.view(batch_size, -1, 1)).view(batch_size, -1, 1, 1)

        # 4. 将浮标上下文向量扩展到与波场特征相同的空间尺寸
        context_vector = context_vector.view(batch_size, -1, 1, 1)  # (batch_size, context_dim, 1, 1)

        # 5. 融合波场特征和上下文向量
        combined_features = torch.cat([wave_features_compressed, context_vector], dim=1)  # (batch_size, 512 + context_dim, 1, 1)
        combined_features = combined_features.unsqueeze(2)  # 增加时间维度 (B, C, 1, H, W)

        # 6. 通过两个1x1x1卷积生成 μ 和 σ
        mu = self.conv_mu(combined_features)  # (batch_size, latent_dim, 1, 1, 1)
        sigma = self.conv_sigma(combined_features)  # (batch_size, latent_dim, 1, 1, 1)

        mu = mu.squeeze(2).squeeze(2).squeeze(2)  # (batch_size, latent_dim)
        sigma = sigma.squeeze(2).squeeze(2).squeeze(2)  # (batch_size, latent_dim)

        return mu, sigma


class EnvFieldDecoder(nn.Module):
    def __init__(self,output_channels=14, time_steps=9, latent_dim=512):
        """
        波场解码器，将潜在向量解码为海浪场，支持时间序列输出
        :param latent_dim: 潜在向量的维度（默认为512）
        :param output_channels: 输出波场的通道数（默认为4）
        :param time_steps: 时间步数（默认为10）
        """
        super(EnvFieldDecoder, self).__init__()
        self.time_steps = time_steps

        # 时间维度扩展层
        self.time_expand = nn.Sequential(
            nn.Conv1d(latent_dim * 2, 512 * time_steps, kernel_size=1),
            nn.ReLU()
        )

        # 小模块 C：3D反卷积 + BatchNorm + ReLU
        self.module_c1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=512, out_channels=1024, kernel_size=(1, 4, 4), stride=1, padding=0),
            nn.BatchNorm3d(1024),
            nn.ReLU()
        )
        self.module_c2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=1024, out_channels=512, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU()
        )
        self.module_c3 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=512, out_channels=256, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )
        self.module_c4 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )

        # 小模块 D：3D反卷积 + Tanh
        self.module_d = nn.Sequential(
            nn.ConvTranspose3d(in_channels=128, out_channels=output_channels, kernel_size=(1, 4, 4), stride=(1, 2, 2),
                               padding=(0, 1, 1)),
            # nn.Tanh()
            nn.Sigmoid()# 必须改用 Sigmoid 匹配 [0, 1] 数据,Tanh会激活到[-1, 1]
        )

    def forward(self, mu, sigma):
        """
        前向传播
        :param mu: 编码器生成的均值向量 (batch_size, latent_dim, 1, 1)
        :param sigma: 编码器生成的方差向量 (batch_size, latent_dim, 1, 1)
        :return: 解码后的波场 (batch_size, time_steps, output_channels, 128, 128)
        """
        batch_size = mu.shape[0]

        # 1. 将 μ 和 σ 融合为一个向量
        combined_vector = torch.cat([mu, sigma], dim=1)  # (batch_size, latent_dim * 2, 1, 1)

        # 2. 扩展时间维度
        combined_vector_flat = combined_vector.view(batch_size, -1, 1)  # (B, latent_dim*2, 1)
        expanded_vector = self.time_expand(combined_vector_flat)  # (B, 512*T, 1)
        expanded_vector = expanded_vector.view(batch_size, 512, self.time_steps, 1, 1)  # (B, 512, T, 1, 1)
        # 3. 通过小模块 C 和小模块 D 逐步解码
        x = self.module_c1(expanded_vector)  # (batch_size, 1024, 1, 4, 4)
        x = self.module_c2(x)  # (batch_size, 512, 1, 8, 8)
        x = self.module_c3(x)  # (batch_size, 256, 1, 16, 16)
        x = self.module_c4(x)  # (batch_size, 128, 1, 32, 32)
        # print(x.shape)
        output = self.module_d(x)  # (batch_size, 4, 1, 128, 128)
        output = output.permute(0, 2, 1, 3, 4).contiguous()

        return output


# 方法一：使用3D卷积（推荐）
class Discriminator(nn.Module):
    def __init__(self, input_channels=14, time_steps=9):
        super(Discriminator, self).__init__()
        self.time_steps = time_steps

        self.A = nn.Sequential(
            nn.Conv3d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        self.A2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, 1, 1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        )
        self.A3 = nn.Sequential(
            nn.Conv3d(64, 128, 3, 1, 1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        )
        self.A4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, 1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        )
        self.fc = nn.Linear(256 * time_steps, 1)

        # 全局平均池化层（保持时间维度）
        self.global_avg_pool = nn.AdaptiveAvgPool3d((time_steps, 1, 1))
        self.sg = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, time_steps, channels, height, width)
        # 调整为3D卷积所需格式: (batch_size, channels, time_steps, height, width)
        x = x.permute(0, 2, 1, 3, 4)

        x = self.A(x)
        x = self.A2(x)
        x = self.A3(x)
        x = self.A4(x)
        x = self.global_avg_pool(x)  # (batch_size, 512, time_steps, 1, 1)
        x = x.view(x.size(0), -1)     # Flatten
        x = self.fc(x)
        # x = self.sg(x)
        return x

# ----------- 判别器主体 -----------

class LambdaScheduler:
    def __init__(self):
        # KL参数: 精度优先。
        # 初始设为极小值，让模型先学会重构。
        # 上限从 0.03 降到 0.005，防止过度压缩 latent space 导致丢失地理细节。
        self.kl_points = [(0, 0.0001), (30, 0.002), (100, 0.005), (200, 0.01)]
        
        # Adv参数: 极低权重。
        # 介入时间提前到第 20 轮（早点让判别器纠正不合理的物理结构）。
        # 最终权重保持在极低水平，仅作为重构损失的辅助。
        self.adv_points = [(0, 0.0), (20, 0.0001), (100, 0.001), (200, 0.005)]

    @staticmethod
    def linear_interp(epoch, points):
        for i in range(len(points) - 1):
            if points[i][0] <= epoch <= points[i+1][0]:
                e0, v0 = points[i]
                e1, v1 = points[i+1]
                t = (epoch - e0) / (e1 - e0)
                return v0 + t * (v1 - v0)
        return points[-1][1]

    def get_lambdas(self, epoch):
        lambda_kl = self.linear_interp(epoch, self.kl_points)
        lambda_adv = self.linear_interp(epoch, self.adv_points)
        # 你也可以在这里返回重构损失的权重，如果想做动态平衡的话
        return lambda_kl, lambda_adv


class Cvae_Gan(nn.Module):
    def __init__(self):
        super(Cvae_Gan, self).__init__()
        self.context_encoder = Context_Encoder()  # 集合预报编码器
        self.encoder = EnvFieldEncoderWithPath()  # 将集合预报结果和上下文向量输入到编码器
        self.decoder = EnvFieldDecoder()
        self.discriminator = Discriminator()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.lambda_adv = 0.001  # 0.1 ~ 1.0
        self.lambda_kl = 0.0005  # 0.001 ~ 0.1
        # 优化器
        self._initialize_optimizers(0.0001)

    def _initialize_optimizers(self, lr):
        """
        初始化生成器和判别器的优化器
        """
        # 生成器参数（包括所有生成模型组件）
        generator_params = list(self.context_encoder.parameters()) + \
                           list(self.encoder.parameters()) + \
                           list(self.decoder.parameters())

        # 判别器参数
        discriminator_params = self.discriminator.parameters()

        # 定义优化器

        self.optimizer_G = optim.AdamW(generator_params, lr=lr,betas=(0.5, 0.999),weight_decay=1e-4)
        self.optimizer_D = optim.AdamW(discriminator_params, lr=5e-5,betas=(0.5, 0.999),weight_decay=1e-4)

        self.lr_schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_D, 200, eta_min=0, last_epoch=-1
        )
        self.lr_schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_G, 200, eta_min=0, last_epoch=-1
        )
        self.l_mse = nn.MSELoss()
        self.l_l1  = nn.L1Loss()
    def kl_loss(self, mu, logvar):
        """
        计算 KL 散度损失
        Args:
            mu: [B, D] 潜变量均值
            logvar: [B, D] 潜变量对数方差
        Returns:
            KL 损失 (标量)
        """
        # -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl.mean()

    def reparameterize(self, mu, logvar):
        """
        mu: (B, latent_dim)
        logvar: (B, latent_dim)  # log(sigma^2)
        return: z (B, latent_dim)
        """
        std = torch.exp(0.5 * logvar)  # sigma = exp(0.5 * logvar)
        eps = torch.randn_like(std)  # epsilon ~ N(0, I)
        z = mu + eps * std
        return z

    def train_step(self, env_data, paths_data, paths_data_mask):
        b = paths_data.shape[0]
        device = paths_data.device
        # 生成上下文向量 c
        context_vector= self.context_encoder(paths_data, paths_data_mask)  # (batch_size, latent_dim) [4, 512]
        # 编码器生成 z 的均值和标准差
        z_mean, z_var = self.encoder(env_data, context_vector)  # (batch_size, latent_dim)
        z = self.reparameterize(z_mean, z_var)  # 重参数化采样,[4, 128]
        # 标签平滑
        real_labels = torch.ones(b, 1)*0.9  # 真实样本标签平滑到 0.9
        fake_labels = torch.zeros(b, 1)+0.1  # 假样本标签平滑到 0.1
        real_labels, fake_labels = real_labels.to(device), fake_labels.to(device)
        # ---------------------
        #  训练判别器
        # ---------------------
        # 清零判别器梯度
        self.optimizer_D.zero_grad()
        # 用真实数据训练判别器
        outputs  = self.discriminator(env_data)
        real_preds=outputs
        d_loss_real = self.bce_loss(outputs , real_labels)
        # 用生成的数据训练判别器
        reconstructed_field = self.decoder(z, context_vector)  # [4, 1, 2]
        # 用假数据训练判别器，detach()阻止梯度流向生成器
        fake_preds = self.discriminator(reconstructed_field.detach())  # 判别器对生成路径的判别
        d_loss_fake = self.bce_loss(fake_preds, fake_labels)
        # 总判别器损失
        l_adv_D= (d_loss_real + d_loss_fake)/2
        # 反向传播并更新判别器参数
        l_adv_D.backward()
        self.optimizer_D.step()
        # 更新学习率调度器
        self.lr_schedulerD.step()
        # ---------------------
        #  训练生成器
        # ---------------------
        # 清零生成器梯度
        self.optimizer_G.zero_grad()
        outputs = self.discriminator(reconstructed_field)
        l_adv_G = self.bce_loss(outputs, real_labels)
        # 重构损失
        alpha = 0.9
        l_rec = alpha * self.l_mse(env_data, reconstructed_field) + (1 - alpha) * self.l_l1(env_data, reconstructed_field)
        # l_rec = self.mse_loss(env_data, reconstructed_field)
        # KL损失
        l_kl = self.kl_loss(z_mean, z_var)
        l_G = l_rec + self.lambda_kl * l_kl + self.lambda_adv * l_adv_G  # 总生成器损失
        # 优化生成器
        l_G.backward()
        self.optimizer_G.step()
        self.lr_schedulerG.step()
        # 记录判别器输出用于监控
        d_real_mean = torch.sigmoid(real_preds).mean().item()
        d_fake_mean = torch.sigmoid(fake_preds).mean().item()
        return l_G, l_adv_D,d_real_mean,d_fake_mean,l_rec

    def inference(self, track_data,mask):
        self.context_encoder.eval()
        device=track_data.device
        self.decoder.eval()
        with torch.no_grad():
            # 获取上下文向量 c
            context = self.context_encoder(track_data,mask)
            # 生成 z 并解码
            batch_size = track_data.size(0)
            paths = []
            num_samples = 1
            for _ in range(num_samples):
                # 从标准正态分布 N(0, I) 中采样 z
                z = torch.randn_like(context, device=device)  # z 形状与 context 相同
                # 解码生成波场
                generated_paths = self.decoder(z, context)  # 形状为 (batch_size,step,f )[4, 1, 2]
                paths.append(generated_paths)

            # 合并生成的波场样本
            paths = torch.cat(paths, dim=1)  # 合并生成的样本
            return paths
