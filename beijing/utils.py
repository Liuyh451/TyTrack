import numpy as np
import torch.nn.functional as F
import torch


def denormalize_latlon(data, geo_bounds,isX=True):
    """
    对经纬度坐标进行反归一化处理

    参数:
    - data: 原始坐标数据, 形状为 [..., 2], 其中 index 0 是纬度, index 1 是经度
    - geo_bounds: 包含缩放边界的字典 (lat_trg_min, lat_trg_max, lon_trg_min, lon_trg_max)

    返回:
    - 反归一化后的数据 (与输入 shape 相同)
    """
    # 创建副本避免修改原数据 (如果是 PyTorch 使用 .clone(), NumPy 使用 .copy())
    denorm_data = data.copy()

    # 反归一化经度 (Longitude) - 索引为 1
    lon_min = geo_bounds['lon_min'] if isX else geo_bounds['lon_trg_min']
    lon_max = geo_bounds['lon_max'] if isX else geo_bounds['lon_trg_max']
    lat_max = geo_bounds['lat_max'] if isX else geo_bounds['lat_trg_max']
    lat_min = geo_bounds['lat_min'] if isX else geo_bounds['lat_trg_min']
    denorm_data[..., 1] = denormalize_column(
        data[..., 1],
        lon_min,
        lon_max
    )

    # 反归一化纬度 (Latitude) - 索引为 0
    denorm_data[..., 0] = denormalize_column(
        data[..., 0],
        lat_min,
        lat_max
    )

    return denorm_data


def get_distance(output, target, geo_bounds):
    """
    Calculate the mean distance between predicted and true coordinates.

    :param output: Model's predicted coordinates, shape (batch_size, max_length, 2)
    :param target: True coordinates, shape (batch_size, max_length, 2)
    :return: Mean distance loss
    """
    R = 6372.8  # Earth radius in kilometers

    # Normalize longitudes
    lon1 = normalize_longitude(denormalize_column(target[:, 1], geo_bounds['lon_trg_min'], geo_bounds['lon_trg_max']))
    lat1 = denormalize_column(target[:, 0], geo_bounds['lat_trg_min'], geo_bounds['lat_trg_max'])
    lon2 = normalize_longitude(denormalize_column(output[:, 1], geo_bounds['lon_trg_min'], geo_bounds['lon_trg_max']))
    lat2 = denormalize_column(output[:, 0], geo_bounds['lat_trg_min'], geo_bounds['lat_trg_max'])

    # Convert degrees to radians
    radlat1 = torch.deg2rad(lat1)
    radlat2 = torch.deg2rad(lat2)
    radlon1 = torch.deg2rad(lon1)
    radlon2 = torch.deg2rad(lon2)

    # Compute the distance formula
    A = torch.sin(radlat1) * torch.sin(radlat2) + torch.cos(radlat1) * torch.cos(radlat2) * torch.cos(radlon1 - radlon2)

    # Ensure A is within [-1, 1] to avoid NaNs in arccos
    A = torch.clamp(A, -1, 1)

    # Compute the distance
    Edd = torch.acos(A) * R

    # Compute the mean distance
    mean_distance = torch.nanmean(Edd)

    return mean_distance


def normalize_longitude(lon):
    return (lon + 180) % 360 - 180


def min_max_normalize(matrix):
    matrix_min = np.min(matrix)
    matrix_max = np.max(matrix)
    return (matrix - matrix_min) / (matrix_max - matrix_min)


# 归一化函数
def normalize_column(column, min_val, max_val):
    return (column - min_val) / (max_val - min_val)


def denormalize_column(normalized_column, min_val, max_val):
    return normalized_column * (max_val - min_val) + min_val


def get_ensemble_mean(x_path, mask, target_t,first=True):
    """
    x_path: [B, T, M, F]
    mask:   [B, T, M]
    """

    device = x_path.device
    batch_size, seq_len, num_models, _ = x_path.shape
    # 1. lead time
    if first==True:
        actual_lt = torch.round(x_path[..., 1]) # 直接取值并四舍五入取整
    else:
        norm_lt = x_path[..., 1]
        if target_t <= 24:
            actual_lt = norm_lt * 24.0
        else:
            actual_lt = norm_lt * float(target_t)

        actual_lt = torch.round(actual_lt)
    # 2. coords
    coords = x_path[..., 2:4]  # [B,T,M,2]

    # 3. scoring
    window_mask = (
        (actual_lt >= target_t - 6)
        & (actual_lt <= target_t)
        & (mask > 0)
    )

    scores = torch.full_like(actual_lt, -1.0)
    scores[window_mask] = actual_lt[window_mask]

    exact_match = (
        (actual_lt == target_t)
        & (mask > 0)
    )
    scores[exact_match] = target_t + 1000.0

    max_scores, max_indices = torch.max(scores, dim=1)

    # =========================
    # 4. valid agencies
    # =========================
    valid_agency_mask = (max_scores > -1.0)

    # =========================
    # 5. gather best coords
    # =========================
    best_coords = torch.zeros(
        (batch_size, num_models, 2),
        device=device
    )

    for m in range(num_models):

        idx = (
            max_indices[:, m]
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, 2)
        )

        m_coords = coords[:, :, m, :]

        picked = torch.gather(m_coords, 1, idx).squeeze(1)

        best_coords[:, m, :] = picked

    # =========================
    # 6. coordinate validity filter
    # =========================
    coord_valid = (
        (best_coords[..., 0].abs() > 1e-6)
        & (best_coords[..., 1].abs() > 1e-6)
    )

    valid_agency_mask = valid_agency_mask & coord_valid

    valid_agency_mask = valid_agency_mask.float()

    # =========================
    # 7. ensemble mean
    # =========================
    sum_coords = (
        best_coords * valid_agency_mask.unsqueeze(-1)
    ).sum(dim=1)

    counts = (
        valid_agency_mask.sum(dim=1, keepdim=True)
        .clamp(min=1)
    )

    ensemble_mean = sum_coords / counts

    return ensemble_mean
