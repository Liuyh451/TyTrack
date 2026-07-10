import torch
from datetime import datetime


class GlobalScaler:
    """
    存储并在测试时提供全局归一化的极值参数。
    根据传入的 area 参数自动选择极值字典和基准年份。
    """

    def __init__(self, area="scs", device="cpu"):
        self.device = device
        self.area = area

        # 1. 动态确定基准年份与极值字典
        if self.area == "scs":
            self.base_year = 2023
            self.stats = {
                12: {
                    "min": [1546236032.0, 0.0, 5.7, 102.4, 910.0, 6.9679999351501465],
                    "max": [1697846400.0, 24.0, 25.271, 129.669, 1008.0, 65.0],
                },
                24: {
                    "min": [1546236032.0, 0.0, 5.7, 102.4, 910.0, 6.9679999351501465],
                    "max": [1697846400.0, 24.0, 25.271, 129.669, 1008.0, 65.0],
                },
                36: {
                    "min": [1546236032.0, 0.0, 5.7, 102.4, 915.0, 6.681000232696533],
                    "max": [1697824768.0, 36.0, 25.271, 129.669, 1008.0, 62.0],
                },
                48: {
                    "min": [1546236032.0, 0.0, 5.7, 102.4, 915.0, 6.681000232696533],
                    "max": [1697846400.0, 48.0, 25.271, 129.669, 1006.0, 59.0],
                },
                60: {
                    "min": [1561960832.0, 0.0, 5.7, 102.4, 925.0, 6.681000232696533],
                    "max": [1697846400.0, 60.0, 25.271, 129.669, 1006.0, 59.0],
                },
                72: {
                    "min": [1561960832.0, 0.0, 5.7, 102.4, 925.0, 6.681000232696533],
                    "max": [1697846400.0, 72.0, 25.271, 129.669, 1010.0, 62.0],
                },
                96: {
                    "min": [1564369152.0, 0.0, 5.7, 102.4, 925.0, 8.640999794006348],
                    "max": [1697846400.0, 96.0, 25.271, 129.669, 1006.0, 58.0],
                },
                120: {
                    "min": [1564369152.0, 0.0, 5.7, 102.4, 925.0, 8.970999717712402],
                    "max": [1697824768.0, 120.0, 25.271, 129.669, 1004.0, 58.0],
                },
            }
        else:
            # 非 scs 情况（如 EC 等），使用 2021 基准
            self.base_year = 2021
            self.stats = {
                12: {
                    "min": [1546322432.0, 0.0, 4.072, 90.7, 895.0, 9.0930],
                    "max": [1640001536.0, 24.0, 44.3, 169.4, 1008.0, 87.0],
                },
                24: {
                    "min": [1546322432.0, 0.0, 4.072, 90.7, 895.0, 9.0930],
                    "max": [1640001536.0, 24.0, 44.3, 169.4, 1008.0, 87.0],
                },
                36: {
                    "min": [1546322432.0, 0.0, 4.072, 90.7, 895.0, 7.9450],
                    "max": [1639980032.0, 36.0, 44.3, 169.4, 1008.0, 87.0],
                },
                48: {
                    "min": [1546322432.0, 0.0, 4.072, 90.7, 895.0, 7.9450],
                    "max": [1639980032.0, 48.0, 44.3, 169.4, 1008.0, 87.0],
                },
                60: {
                    "min": [1546322432.0, 0.0, 4.072, 90.7, 895.0, 7.9450],
                    "max": [1640001536.0, 60.0, 44.3, 169.4, 1008.0, 87.0],
                },
                72: {
                    "min": [1546322432.0, 0.0, 4.072, 90.7, 895.0, 7.9450],
                    "max": [1640001536.0, 72.0, 44.3, 169.4, 1008.0, 87.0],
                },
                96: {
                    "min": [1546322432.0, 0.0, 4.072, 90.7, 895.0, 7.9450],
                    "max": [1640001536.0, 96.0, 44.3, 169.4, 1008.0, 87.0],
                },
                120: {
                    "min": [1546322432.0, 0.0, 4.072, 90.7, 895.0, 7.9450],
                    "max": [1639980032.0, 120.0, 44.3, 169.4, 1008.0, 85.0],
                },
            }

    def get_tensors(self, t):
        if t not in self.stats:
            raise ValueError(f"时间步 {t} 不在预设的极值字典中！")

        min_vals = torch.tensor(
            self.stats[t]["min"], dtype=torch.float32, device=self.device
        )
        max_vals = torch.tensor(
            self.stats[t]["max"], dtype=torch.float32, device=self.device
        )
        return min_vals, max_vals

    def _get_time_offset(self, current_timestamp):
        """
        动态计算当前时间戳与基准年（2023 或 2021）同月同日之间的秒数差
        """
        dt_current = datetime.fromtimestamp(current_timestamp)

        # 使用动态选定的基准年进行判断
        if dt_current.year <= self.base_year:
            return 0

        target_month = dt_current.month
        target_day = dt_current.day
        if target_month == 2 and target_day == 29:
            target_day = 28

        dt_fake = dt_current.replace(
            year=self.base_year, month=target_month, day=target_day
        )
        offset = current_timestamp - dt_fake.timestamp()
        return offset

    def normalize(self, x_data, t):
        min_vals, max_vals = self.get_tensors(t)

        if torch.isnan(x_data).any():
            print("警告：归一化前，原始数据 x_data 中就已经含有 NaN！")

        x_norm = x_data.clone()

        # --- 逐样本动态时间平移 ---
        sample_ts = x_data[0, 0, 0, 0].item()

        if sample_ts > max_vals[0]:
            offset = self._get_time_offset(sample_ts)
            x_norm[..., 0] -= offset

        # 归一化计算
        denominator = max_vals - min_vals
        denominator[denominator == 0] = 1e-6
        x_norm[..., :6] = (x_norm[..., :6] - min_vals) / denominator

        actual_data_mask = x_data[..., :6] != 0
        x_norm[..., :6][~actual_data_mask] = 0.0

        # 检查归一化结果是否超过 [0, 1] 范围
        out_of_range = (x_norm[..., :6] > 1.0) | (x_norm[..., :6] < 0.0)
        if out_of_range.any():
            for c in range(6):
                channel_out = out_of_range[..., c]
                if channel_out.any():
                    channel_data = x_norm[..., c]
                    max_val = channel_data.max().item()
                    min_val = channel_data.min().item()
                    above_one = channel_data[channel_data > 1.0]
                    below_zero = channel_data[channel_data < 0.0]
                    if above_one.numel() > 0:
                        print(
                            f"  [Area: {self.area}] 通道 {c} 有 {above_one.numel()} 个值 > 1.0, 最大值: {above_one.max().item():.4f}"
                        )
                    if below_zero.numel() > 0:
                        print(
                            f"  [Area: {self.area}] 通道 {c} 有 {below_zero.numel()} 个值 < 0.0, 最小值: {below_zero.min().item():.4f}"
                        )
                    print(f"    通道 {c} 整体范围: [{min_val:.4f}, {max_val:.4f}]")
                    print(
                        f"    对应 min_vals[{c}]={min_vals[c].item():.4f}, max_vals[{c}]={max_vals[c].item():.4f}"
                    )

            # 开启硬裁剪，确保传给模型的数据安全
            x_norm[..., :6] = torch.clamp(x_norm[..., :6], 0.0, 1.0)

        return x_norm


class ARHeadConfig:
    # CONFIGS = {
    #     "scs": {
    #         "lat_span": 19.571,
    #         "lon_span": 27.269,
    #         "lat_k": 0.008,
    #         "lat_b": 0.1,
    #         "lon_k": 0.012,
    #         "lon_b": 0.15,
    #     },

    #     "wpo": {
    #         "lat_span": 40.228,
    #         "lon_span": 78.700,
    #         "lat_k": 0.015,
    #         "lat_b": 0.15,
    #         "lon_k": 0.020,
    #         "lon_b": 0.25,
    #     }
    # }``
    CONFIGS = {
        "scs": {
            "lat_span": 19.571,
            "lon_span": 27.269,
            "lat_k": 0.04,
            "lat_b": 0.3,
            "lon_k": 0.03,
            "lon_b": 0.4,
        },
        "wpo": {
            "lat_span": 40.228,
            "lon_span": 78.700,
            "lat_k": 0.04,
            "lat_b": 0.5,
            "lon_k": 0.03,
            "lon_b": 0.4,
        },
    }

    @classmethod
    def apply(cls, ar_head, region):
        cfg = cls.CONFIGS[region.lower()]

        with torch.no_grad():
            ar_head.lat_span.fill_(cfg["lat_span"])
            ar_head.lon_span.fill_(cfg["lon_span"])

        ar_head.lat_k = cfg["lat_k"]
        ar_head.lat_b = cfg["lat_b"]
        ar_head.lon_k = cfg["lon_k"]
        ar_head.lon_b = cfg["lon_b"]
