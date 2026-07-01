import torch
from torch import nn
import torch.nn.functional as F

class EncoderWithGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout_rate=0.2):
        super(EncoderWithGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_norm = nn.LayerNorm(input_dim, eps=1e-5)
        # 使用GRU替代LSTM，双向处理
        self.time_gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.init_weights()
        self.dropout = nn.Dropout(dropout_rate)

    def init_weights(self):
        for name, param in self.time_gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, inputs, combined_mask):
        batch_size, max_time_steps, num_models, input_dim = inputs.shape  # [batch_size, time_steps, models, features]

        # 应用mask遮蔽无效数据
        masked_inputs = inputs * combined_mask.unsqueeze(-1)

        # 计算每个样本-模型组合的有效时间步数
        valid_time_steps_per_model = (combined_mask.sum(dim=1)).view(batch_size * num_models)
        valid_time_steps_cpu = valid_time_steps_per_model[valid_time_steps_per_model != 0].cpu()

        # 重新排列维度以便处理时间序列: [batch, models, time, features]
        permuted_inputs = masked_inputs.permute(0, 2, 1, 3).contiguous()  # [batch_size, num_models, max_time_steps, input_dim]
        reshaped_inputs = permuted_inputs.view(batch_size * num_models, max_time_steps, -1)  # [batch_size*num_models, max_time_steps, input_dim]
        valid_inputs = reshaped_inputs[valid_time_steps_per_model != 0]  # 只保留有效的时间序列

        # 层归一化
        normalized_inputs = self.layer_norm(valid_inputs)

        # 打包序列以处理变长序列
        packed_inputs = nn.utils.rnn.pack_padded_sequence(
            normalized_inputs,
            valid_time_steps_cpu,
            batch_first=True,
            enforce_sorted=False
        )

        # GRU前向传播
        gru_outputs, hidden_states = self.time_gru(packed_inputs)
        gru_outputs, _ = nn.utils.rnn.pad_packed_sequence(gru_outputs, batch_first=True)
        gru_outputs = self.dropout(gru_outputs)


        # 获取实际输出长度
        actual_output_length = gru_outputs.shape[1]

        # 重构输出张量以匹配原始形状
        reconstructed_outputs = torch.zeros(
            (reshaped_inputs.shape[0], gru_outputs.shape[1], gru_outputs.shape[2]),
            dtype=gru_outputs.dtype,
            device=inputs.device
        )
        reconstructed_outputs[valid_time_steps_per_model != 0] = gru_outputs

        # 重新组织为 [batch_size, num_models, time_steps, hidden_dim*2] 然后转置为 [batch_size, time_steps, num_models, hidden_dim*2]
        final_outputs = reconstructed_outputs.view(batch_size, num_models, actual_output_length, -1)
        final_outputs = final_outputs.transpose(1, 2)

        # 调整mask以匹配实际输出长度
        adjusted_mask = combined_mask[:, :actual_output_length, :]
        masked_outputs = final_outputs * adjusted_mask.unsqueeze(-1)

        # 初始化隐藏状态张量
        final_hidden_states = torch.zeros(
            self.num_layers * 2, batch_size, self.hidden_dim,
            device=inputs.device
        )

        hidden_index = 0
        for batch_idx in range(batch_size):
            # 当前批次中有效模型的数量
            valid_models_count = (
                        valid_time_steps_per_model[batch_idx * num_models:(batch_idx + 1) * num_models] != 0).sum()
            if valid_models_count > 0:
                # forward 和 backward 两个方向
                final_hidden_states[:, batch_idx, :] = hidden_states[:, hidden_index + valid_models_count - 1, :]
                hidden_index += valid_models_count

        return masked_outputs, final_hidden_states, adjusted_mask, actual_output_length

class Attention(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.2):
        super(Attention, self).__init__()

        self.time_attention = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.model_attention = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        # self.output_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, encoder_outputs, decoder_hidden, combined_mask):
        batch_size, max_length, num_models, hidden_dim_2 = encoder_outputs.shape
        encoder_outputs = self.layer_norm(encoder_outputs)
        # 调整 decoder_hidden 的形状
        # decoder_hidden: [num_layers*2, batch_size, hidden_dim] -> [batch_size, hidden_dim*2]
        # 我们取最后一层的隐藏状态，并将双向的状态连接起来
        decoder_hidden = decoder_hidden.view(2, -1, batch_size, hidden_dim_2 // 2)
        decoder_hidden = torch.cat((decoder_hidden[-2], decoder_hidden[-1]), dim=2)
        decoder_hidden = decoder_hidden.squeeze(0)  # [batch_size, hidden_dim]
        # 时间步注意力
        time_scores = self.time_attention(encoder_outputs)
        time_scores = self.dropout1(time_scores)

        time_scores = torch.einsum('btmh,bh->btm', time_scores, decoder_hidden)

        # 模型注意力
        model_scores = self.model_attention(encoder_outputs)
        model_scores = self.dropout2(model_scores)
        model_scores = torch.einsum('btmh,bh->btm', model_scores, decoder_hidden)

        # 应用combined_mask
        time_scores = time_scores.masked_fill(combined_mask == 0, -1e9)
        model_scores = model_scores.masked_fill(combined_mask == 0, -1e9)

        time_weights = F.softmax(time_scores, dim=1)
        model_weights = F.softmax(model_scores, dim=2)

        # 组合两个注意力权重
        combined_weights = time_weights * model_weights
        combined_weights[~combined_mask.bool()] = 0
        # 对 combined_weights 归一化
        combined_weights_sum = combined_weights.sum(dim=[1, 2], keepdim=True)
        # 防止除以零的情况
        combined_weights_sum = combined_weights_sum.masked_fill(combined_weights_sum == 0, 1)
        combined_weights = combined_weights / combined_weights_sum
        # 计算上下文向量
        context_vector = torch.sum(encoder_outputs * combined_weights.unsqueeze(-1), dim=[1, 2])
        
        # verify_manual_attention(encoder_outputs, context_vector, my_weights, batch_idx=0)
        return context_vector, combined_weights

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout_rate=0.2):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.clip_value = 1.0

        self.layer_norm = nn.LayerNorm(input_dim, eps=1e-5)
        # 只保留时间维度的LSTM
        self.time_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.init_weights()
        self.dropout1 = nn.Dropout(dropout_rate)
        # self.enhancer = PathEncoderEnhance(hidden_dim=self.hidden_dim, num_models=8, dropout_rate=0.2)

    def init_weights(self):
        for name, param in self.time_lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                param.data[self.hidden_dim:2 * self.hidden_dim] = 1.0

    def forward(self, inputs, combined_mask):
        batch_size, max_length, num_models, input_dim = inputs.shape  # [4, 18, 8, 12]

        # 应用mask
        x = inputs * combined_mask.unsqueeze(-1)

        # 处理时间维度
        time_lengths0 = (combined_mask.sum(dim=1)).view(batch_size * num_models)
        time_lengths = time_lengths0[time_lengths0 != 0].cpu()

        x_time = x.permute(0, 2, 1, 3).contiguous()  # [4, 8, 18, 12]
        x_time = x_time.view(batch_size * num_models, max_length, -1)  # [4*8=32, 18, 12]
        # 过滤掉没有有效时间步的 (batch, model) 组合
        x_time0 = x_time[time_lengths0 != 0]  # 只保留那些 mask 中有有效值的时间序列 #[valid,18,12]

        x_time0 = self.layer_norm(x_time0)

        assert not torch.isnan(x_time0).any(), "Input contains NaN values"
        assert not torch.isinf(x_time0).any(), "Input contains Inf values"
        assert (time_lengths > 0).all(), "Sequence lengths must be positive"
        assert time_lengths.max() <= x.size(1), "Max length exceeds sequence length"

        x_time0 = torch.clamp(x_time0, -1e6, 1e6)  # torch.Size([6, 18, 12])，只有6个有效
        # 在每个有效组合中，只处理前 time_lengths[i] 个时间步
        packed_x = nn.utils.rnn.pack_padded_sequence(x_time0, time_lengths, batch_first=True, enforce_sorted=False)
        time_outputs, (hidden, cell) = self.time_lstm(packed_x)  # 再经过mask，lstm输出的只有4个有效步长
        time_outputs, _ = nn.utils.rnn.pad_packed_sequence(time_outputs, batch_first=True)
        time_outputs = self.dropout1(time_outputs)

        if self.training:
            for param in self.time_lstm.parameters():
                nn.utils.clip_grad_norm_(param, self.clip_value)

        actual_length = time_outputs.shape[1]
        new_x_time = torch.zeros((x_time.shape[0], time_outputs.shape[1], time_outputs.shape[2]),
                                 dtype=time_outputs.dtype, device=inputs.device)
        new_x_time[time_lengths0 != 0] = time_outputs

        # 重组回 [batch_size, num_models, max_length, hidden_dim*2]
        outputs = new_x_time.view(batch_size, num_models, actual_length, -1)
        outputs = outputs.transpose(1, 2)
        # 调整mask以匹配实际长度
        combined_mask = combined_mask[:, :actual_length, :]

        # 应用最终的mask
        outputs = outputs * combined_mask.unsqueeze(-1)

        # 调整hidden和cell的形状以匹配原来的接口
        new_hidden = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim,
                                 device=inputs.device)
        new_cell = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim,
                               device=inputs.device)

        # 只保留每个batch中最后一个有效序列的hidden状态
        valid_batch_indices = torch.nonzero(time_lengths0 != 0).squeeze()
        current_batch_idx = 0
        hidden_idx = 0

        for i in range(batch_size):
            num_valid_timesteps = (time_lengths0[i * num_models:(i + 1) * num_models] != 0).sum()
            if num_valid_timesteps > 0:
                new_hidden[:, i, :] = hidden[:, hidden_idx + num_valid_timesteps - 1, :]
                new_cell[:, i, :] = cell[:, hidden_idx + num_valid_timesteps - 1, :]
                hidden_idx += num_valid_timesteps
        #新加
        # enhanced_outputs = self.enhancer(outputs, combined_mask)
        return outputs, (new_hidden, new_cell), combined_mask, actual_length

class track_encoder_for_gan(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, num_layers=1, dropout_rate=0.0)
        self.attention = Attention(hidden_dim)
        self.encoder_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x, mask):
        encoder_outputs, (hidden, cell), combined_mask, _ = self.encoder(x, mask)
        encoder_outputs = self.encoder_norm(encoder_outputs)
        hidden = torch.cat([hidden[0:1], hidden[1:2]], dim=2)
        context_vector, attention_weights = self.attention(encoder_outputs, hidden, combined_mask)
        return context_vector, hidden

class TrackEncoderWithGru(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = EncoderWithGRU(input_dim, hidden_dim, num_layers=1, dropout_rate=0)
        self.encoder_norm = nn.LayerNorm(hidden_dim * 2)
        self.attention = Attention(hidden_dim)
        #使用双注意力机制效果一般24H：70km左右
        # self.attention = EnhancedDualAttention(hidden_dim)
    def forward(self, x, mask):
        encoder_outputs, hidden,combined_mask, _ = self.encoder(x, mask)
        # encoder_outputs = self.encoder_norm(encoder_outputs)
        hidden = torch.cat([hidden[0:1], hidden[1:2]], dim=2)  # [4, 128]-->[4, 256]
        context_vector, attention_weights = self.attention(encoder_outputs, hidden, combined_mask)
        return context_vector, hidden,attention_weights

class TimeAwareEncoderForENVLighting(nn.Module):
    """
    时间感知的2D数据编码器，支持多通道输入
    """

    def __init__(self, point_num, image_size=(30, 60), hidden_dim=256, input_channels=4):
        """
        初始化编码器

        Args:
            point_num: 时间步数量（历史时刻数）
            image_size: 输入图像的尺寸（高，宽），默认(30, 60)
            hidden_dim: 隐藏层维度（输出特征维度）
            input_channels: 输入图像的通道数，默认为2（原始数据 + 中心差值）
        """
        super().__init__()
        self.point_num = point_num
        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.hidden_dim = hidden_dim
        self.input_channels = input_channels  # 添加输入通道数参数

        # 修改卷积编码器以适应多通道输入
        self.conv_layers2 = nn.Sequential(
            # 第一层卷积：input_channels->32通道
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 第一次下采样
            nn.MaxPool2d(kernel_size=2),
            # 空间注意力机制
            SpatialAttention1(in_channels=64),

            # 第二层卷积：64->128通道
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 第二次下采样
            nn.MaxPool2d(kernel_size=2),
            # 空间注意力机制
            SpatialAttention1(in_channels=128),

            # 第三层卷积：128->256通道
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 第三次下采样
            nn.MaxPool2d(kernel_size=2),
            # 最终空间注意力
            SpatialAttention1(in_channels=256)
        )

        # 降维卷积
        self.conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=2, stride=2)
        # 时间编码器
        self.encode_time = SelfAttentionLSTM8(in_channels=1)

    def get_t_center_differ(self, x):
        """
        计算每个时间步相对于中心点的差值特征

        Args:
            x: 输入张量 [B, T, C, H, W]

        Returns:
            x_t_center: 中心点差值 [B, T, C, H, W]
        """
        # 动态计算中心点索引
        center_h = x.size(3) // 2
        center_w = x.size(4) // 2

        # 对每个通道分别计算中心点差值
        x_t_center = x - x[:, :, :, center_h, center_w].unsqueeze(3).unsqueeze(4).expand_as(x)
        return x_t_center

    def get_differ_with_pad(self, x):
        """
        计算相邻时间步之间的差值，并在最后一个时间步补零
        Args:
            x: [B, T, C, H, W]
        Returns:
            x_differ: [B, T, C, H, W]   # 和原始时间步对齐
        """
        B, T, C, H, W = x.size()
        # 相邻差值 [B, T-1, C, H, W]
        differ = x[:, 1:, :, :, :] - x[:, :-1, :, :, :]

        # 最后一个时间步补零
        zero_pad = torch.zeros((B, 1, C, H, W), device=x.device, dtype=x.dtype)

        # 拼接 -> [B, T, C, H, W]
        x_differ = torch.cat([differ, zero_pad], dim=1)
        return x_differ

    def forward(self, x, node_history_encoded):
        """
        前向传播函数

        Args:
            x: 2D环境数据 [B, T, C, H, W] - 批次，时间步，通道，H, W图像
            node_history_encoded: 1D轨迹编码 [B, 256] - 历史轨迹的编码表示

        Returns:
            hidden: 编码后的特征 [B, 256]
        """
        batch_size, time_steps, channels, height, width = x.size()

        # 1. 特征增强：计算时间差值和中心差值特征
        x_differ = self.get_differ_with_pad(x)  # [B, T-1, C, H, W]
        x_t_center = self.get_t_center_differ(x)  # [B, T, C, H, W]

        # 2. 逐时间步处理2D数据
        x_list = []
        for t in range(self.point_num):
            # 方法1：将原始数据和中心差值特征拼接作为多通道输入，多次实验12H平均60-61 补充：96H平均139km2025年10月13日15点01分
            # x_t = torch.cat((x[:, t, :, :, :], x_t_center[:, t, :, :, :]), dim=1)  # [B, 2*C, H, W]
            # 方法2：时间差值,多次实验12H平均60.2，最好58，最差64
            x_t = torch.cat((x[:, t, :, :, :], x_differ[:, t, :, :, :]), dim=1)  # [B, 2*C, H, W]
            # 方法3：原通道处理，多次实验12H平均64，最差70，最好58 补充：96H平均125km2025年10月13日13点01分
            # x_t = x[:, t, :, :, :]  # [B, C, H, W]

            # 3. 空间特征提取
            x_t = self.conv_layers2(x_t)  # [B, 256, H//8, W//8]

            # 4. 特征降维
            # 根据实际输出尺寸重塑
            b, c, h, w = x_t.size()
            x_t = x_t.view(batch_size, 16, h * 4, w * 4)  # [B, 16, H//8, W//8]
            x_t = self.conv(x_t)  # 降维为 [B, 1, 16, 16]
            x_list.append(x_t)

        # 5. 时间维度堆叠
        x = torch.stack(x_list, dim=1)  # [B, T, 1, 16, 16]

        # 6. 1D轨迹信息重塑
        node_history_encoded = node_history_encoded.reshape(batch_size, 16, 16)  # [B, 16, 16]
        node_history_encoded = node_history_encoded.unsqueeze(1)  # [B, 1, 16, 16]

        # 7. 时间融合
        hidden = self.encode_time(x, node_history_encoded)  # [B, 256]

        return hidden




class SpatialAttention1(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        attention_map1 = self.conv1(x)  # x:torch.Size([32, 64, 50, 50]) map:torch.Size([32, 1, 50, 50])
        attention_map1_transposed = torch.transpose(attention_map1, 2, 3)  # 交换通道维度
        attention_map2 = self.conv2(x)

        attention_map = attention_map1_transposed * attention_map2

        attention_weights = torch.sigmoid(attention_map)  # torch.Size([32, 1, 50, 50])
        return x * attention_weights


class SelfAttentionLSTM8(nn.Module):
    """
    自注意力LSTM模块，用于处理时空特征融合
    用于在时间维度上融合2D环境数据特征，并结合1D轨迹信息进行指导。

    主要功能：
    1. 时间序列处理：对8个时间步的2D特征进行时序建模
    2. 自注意力机制：在空间维度上进行特征增强
    3. 记忆机制：保持时间序列中的重要信息
    4. 外部指导：融合1D轨迹编码作为指导信号
    """

    def __init__(self, in_channels=1):
        """
        初始化自注意力LSTM模块

        Args:
            in_channels: 输入特征图的通道数，默认为1
        """
        super(SelfAttentionLSTM8, self).__init__()

        # 自注意力机制中的查询、键、值卷积层
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # 值(Value)变换
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # 键(Key)变换
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # 查询(Query)变换

        # 记忆模块中的卷积层
        self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # 记忆键变换
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # 记忆值变换

        # 特征融合卷积层
        self.conv6 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)  # 注意力特征融合

        # LSTM门控机制卷积层
        self.conv7 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)  # 输出门
        self.conv8 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)  # 候选状态
        self.conv9 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)  # 输入门

        # LSTM状态更新卷积层
        self.conv10 = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)  # 遗忘门1
        self.conv11 = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)  # 遗忘门2
        self.conv12 = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)  # 更新门
        self.conv13 = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)  # 输出门

    def sa_conv_lstm(self, x, en_1d):
        """
        自注意力卷积LSTM核心处理函数

        Args:
            x: 时间序列特征 [T,B, C, H, W] - 时间步，批次，通道，高度，宽度
            en_1d: 1D轨迹编码指导信号 [B, C, H, W]

        Returns:
            H: 最终隐藏状态 [B, C, H, W]
        """
        # 初始化LSTM状态
        memory = torch.zeros_like(x[0])  # 记忆单元 [B, C, H, W]
        H = torch.zeros_like(x[0])  # 隐藏状态 [B, C, H, W]
        C = torch.randn_like(x[0]) * 1e-6  # 细胞状态 [B, C, H, W]

        # 按时间步顺序处理
        for i in range(x.size(0)):  # 遍历8个时间步
            # LSTM门控计算
            # 遗忘门和输入门计算，融合当前输入、隐藏状态和1D指导信号
            gate_input = torch.cat((H, x[i], en_1d), dim=1)  # [B, 3C, H, W]
            a_xh = torch.sigmoid(self.conv10(gate_input))  # 遗忘门
            ca_xh = C * a_xh  # 遗忘旧状态
            ga = torch.sigmoid(self.conv11(gate_input))  # 输入门
            gv = torch.tanh(self.conv12(gate_input))  # 候选状态
            C = ca_xh + ga * gv  # 更新细胞状态
            a_xh1 = torch.sigmoid(self.conv13(gate_input))  # 输出门
            H = a_xh1 * torch.tanh(C)  # 更新隐藏状态

            # 自注意力记忆机制
            memory, H = self.self_attention_memory(memory, H)

        return H

    def self_attention_memory(self, m, h):
        """
        自注意力记忆机制

        该函数实现了一个空间注意力机制，通过查询、键、值的计算来增强特征表示

        Args:
            m: 记忆状态 [B, C, H, W]
            h: 当前隐藏状态 [B, C, H, W]

        Returns:
            mt: 更新后的记忆状态
            ht: 更新后的隐藏状态
        """
        # 当前隐藏状态的自注意力计算
        vh = self.conv1(h)  # 值变换
        kh = self.conv2(h)  # 键变换
        qh = self.conv3(h)  # 查询变换
        qh = torch.transpose(qh, 2, 3)  # 转置查询矩阵
        ah = F.softmax(kh * qh, dim=-1)  # 计算注意力权重
        zh = vh * ah  # 加权值特征

        # 记忆状态的注意力计算
        km = self.conv4(m)  # 记忆键
        vm = self.conv5(m)  # 记忆值
        am = F.softmax(qh * km, dim=-1)  # 记忆注意力权重
        zm = vm * am  # 加权记忆特征

        # 特征融合
        z0 = torch.cat((zh, zm), dim=1)  # 拼接当前特征和记忆特征
        z = self.conv6(z0)  # 融合后的特征
        hz = torch.cat((h, z), dim=1)  # 拼接原始隐藏状态和融合特征

        # LSTM门控更新
        ot = torch.sigmoid(self.conv7(hz))  # 输出门
        gt = torch.tanh(self.conv8(hz))  # 候选状态
        it = torch.sigmoid(self.conv9(hz))  # 输入门

        # 记忆更新
        gi = gt * it  # 新记忆贡献
        mf = (1 - it) * m  # 旧记忆保持
        mt = gi + mf  # 更新记忆状态
        ht = ot * mt  # 更新隐藏状态

        return mt, ht

    def forward(self, x, en_1d):
        """
        前向传播函数

        Args:
            x: 输入特征 [B, T, C, H, W] - 批次，时间步，通道，高度，宽度
            en_1d: 1D轨迹编码 [B, C, H, W] - 作为指导信号

        Returns:
            flattened_tensor: 扁平化输出特征 [B, 256]
        """
        B, _, _, _, _ = x.size()  # 获取批次大小
        # 调整维度顺序：[B, T, C, H, W] -> [T, B, C, H, W]
        x = x.permute(1, 0, 2, 3, 4)

        # 执行自注意力LSTM处理
        H = self.sa_conv_lstm(x, en_1d)

        # 扁平化输出特征
        flattened_tensor = H.view(B, -1)  # [B, 256]
        return flattened_tensor


class AttentionFusionWithResidual(nn.Module):
    """
    Adaptive Residual Attention Fusion (ARAF) Module
    1.原来那个用transformer的方法96H平均135-140
    2.本方法96H平均117.05
    """

    def __init__(self, input_size, fusion_size):
        super().__init__()
        self.attention = nn.Linear(fusion_size * 2, fusion_size)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, gph):
        attn = self.softmax(torch.cat([x, gph], dim=-1))
        fusion = attn[:, :256] * x + attn[:, 256:] * gph
        combined_features = torch.cat([x, fusion], dim=-1)
        attention_weights = self.sigmoid(self.attention(combined_features))
        # fused_features = attention_weights * x + (1 - attention_weights) * fusion
        x = attention_weights * gph + (1 - attention_weights) * x
        residual = x + fusion * 0.001
        return residual



