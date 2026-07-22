import argparse
import warnings
import numpy as np
import datetime
import torch
import os
import sys
import traceback
from data.dataloader import GeoBounds
from models.new_model import Cvae_Gan_Seq
from utils import get_ensemble_mean, denormalize_latlon
from values import ARHeadConfig, GlobalScaler

warnings.filterwarnings("ignore", category=FutureWarning)

def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble Forecasting Model")
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--test_X', type=str, default='x.npy')
    parser.add_argument('--test_y', type=str, default='y.npy')
    parser.add_argument('--test_mask', type=str, default='x_masks.npy')
    parser.add_argument('--area', type=str, default='wpo')
    parser.add_argument('--trgpath', type=str, default='./checkpoints/')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--ty_name', type=str, default='default')
    parser.add_argument('--ty_number', type=str, default='20260001')
    parser.add_argument('--report_time', type=str, default='2026010100')
    args = parser.parse_args()
    return args

class TyphoonTester:
    def __init__(self, device, base_ckpt_path):
        self.model = None
        self.device = device
        self.base_ckpt_path = base_ckpt_path
        self.current_geo_bounds = None
        
        self.scs_range = {
            'lat': (5.7, 25.271),
            'lon': (102.4, 129.669)
        }
        self.env_models_cache = {}
        self.track_weights_cache = {}

    def get_region_type(self, lat, lon):
        is_in_scs = (self.scs_range['lat'][0] <= lat <= self.scs_range['lat'][1]) and \
                    (self.scs_range['lon'][0] <= lon <= self.scs_range['lon'][1])
        return "scs" if is_in_scs else "wpo"

    def get_env_model(self, region):
        if region not in self.env_models_cache:
            cvae_path = os.path.join(self.base_ckpt_path, 'env_generator', f"best_model_{region}.pt")
            model = Cvae_Gan_Seq(cvae_path).to(self.device)
            self.env_models_cache[region] = model
        return self.env_models_cache[region]

    def get_track_weight(self, region, t, fold_idx):
        cache_key = f"{region}_{t}_{fold_idx}"
        if cache_key not in self.track_weights_cache:
            ckpt_path = os.path.join(self.base_ckpt_path, 'typredictor', region, f"pre_{t}", f"cgan_seq_best_{fold_idx}.pt")
            state = torch.load(ckpt_path, map_location='cpu')
            self.track_weights_cache[cache_key] = state
        return self.track_weights_cache[cache_key]

    def run_inference(self, X, mask, t):
        X = X.to(self.device)
        mask = mask.to(self.device)
        fold_results = []
        X_masked_all = []
        time_scale = 24.0 if t in [12, 24] else float(t)
        with torch.no_grad():
            X_mean = get_ensemble_mean(X, mask, t)
            X_mean_np = X_mean.detach().cpu().numpy()
            mean_lat, mean_lon = X_mean_np[0, 0], X_mean_np[0, 1]
            
            region = self.get_region_type(mean_lat, mean_lon)
            print(f">>> Pos: ({mean_lat:.2f}N, {mean_lon:.2f}E) | Switched to [{region.upper()}] weights")
            
            self.current_geo_bounds = GeoBounds(region)
            dynamic_scaler = GlobalScaler(area=region, device=self.device)
            X = dynamic_scaler.normalize(X, t)
            X_mean = get_ensemble_mean(X, mask, t, False)
            self.model = self.get_env_model(region)

        for fold_idx in range(5):
            state = self.get_track_weight(region, t, fold_idx)
            self.model.track_predictor.load_state_dict(state)
            ARHeadConfig.apply(self.model.track_predictor.ar_head, region)
            self.model.eval()
            with torch.no_grad():
                pred_latlon = self.model(X, mask, X_mean)
                pred_latlon_denorm = denormalize_latlon(pred_latlon.detach().cpu().numpy(), self.current_geo_bounds)
                fold_results.append(pred_latlon_denorm)
                # --- 精确提取机构坐标 ---
                if fold_idx == 0:
                    _, T, M, _ = X.shape
                    X_times = X[0, :, :, 1] * time_scale
                    X_geo_denorm = np.zeros((M, 2))  # [8, 2]
                    for m in range(M):
                        # 针对每一个机构，寻找其时间特征严格等于 t 的索引
                        # 使用 epsilon 容差解决浮点数匹配问题
                        matches = torch.where(torch.abs(X_times[:, m] - t) < 1e-2)[0]
                        if len(matches) > 0:
                            t_idx = matches[-1]  # 取匹配到的该时效对应索引
                            raw_lat = X[0, t_idx, m, 2].item()
                            raw_lon = X[0, t_idx, m, 3].item()
                            # 仅对有效值进行反归一化
                            if raw_lat != 0 and raw_lon != 0:
                                X_geo_denorm[m, 0] = raw_lat * (self.current_geo_bounds['lat_max'] - self.current_geo_bounds['lat_min']) + self.current_geo_bounds[
                                    'lat_min']
                                X_geo_denorm[m, 1] = raw_lon * (self.current_geo_bounds['lon_max'] - self.current_geo_bounds['lon_min']) + self.current_geo_bounds[
                                    'lon_min']
                    X_masked_all.append(X_geo_denorm[np.newaxis, ...])  # [1, M, 2]

        # ====== 结果聚合 ======
        X_masked_final = X_masked_all[0]  # [1, M, 2]
        # 整理形状以供拼接 [1, 1, Channel, 2]
        X_out = X_masked_final.reshape(1, 1, -1, 2)
        if not fold_results:
            return None
        ensemble_pred = np.mean(np.stack(fold_results, axis=0), axis=0)
        pred_out = ensemble_pred.reshape(1, 1, 1, 2)
        # 拼接通道：8机构 + 1预测= 9通道
        all_samples = np.concatenate([X_out, pred_out], axis=2)
        sample_all.append(all_samples)
        return ensemble_pred

def format_report_time(report_time: str) -> str:
    """将 'YYYY-MM-DD HH:00:00' 转换为 'YYYYMMDDHH'"""
    # 示例: '2026-06-04 00:00:00' -> '2026060400'
    return report_time.replace('-', '').replace(' ', '').replace(':', '')[:10]

def load_nanhai_wind(path):
    wind_file = os.path.join(path, "nanhai_wind.txt")
    wind_dict = {}

    if not os.path.exists(wind_file):
        return wind_dict

    with open(wind_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2 or not parts[0].isdigit():
                continue
            try:
                wind_dict[int(parts[0])] = float(parts[1])
            except ValueError:
                wind_dict[int(parts[0])] = 0.0

    return wind_dict

if __name__ == "__main__":
    args = parse_args()
    print(f"启动推理脚本，参数: {vars(args)}")
    
    ty_code = args.ty_number
    ty_dir = f"{ty_code}_{args.ty_name.lower().strip()}"
    path = os.path.join(args.data_path, ty_dir, args.report_time, '')
    # path = args.data_path + ty_code +args.report_time+'/'
    pre_priod = [12, 24, 36, 48, 60, 72, 96, 120]
    y_test_dict = {}
    X_test_dict = {}
    test_mask_dict = {}
    
    # 加载数据
    try:
        y_test_ibt = torch.from_numpy(np.load(path + args.test_y))
        X_test_inst = torch.from_numpy(np.load(path + args.test_X))
        X_test_mask = torch.from_numpy(np.load(path + args.test_mask))
        nanhai_wind = load_nanhai_wind(path)
        print(f"机构数据：{X_test_inst.shape}，真实值：{y_test_ibt.shape}")
    except Exception as e:
        print(f"加载数据失败: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    
    # 创建推理模型实例
    tester = TyphoonTester(device=args.device, base_ckpt_path=args.trgpath)
    
    for isample in range(X_test_inst.shape[0]):
        print(f'----------第{isample + 1}个样本----------')
        y_test0 = y_test_ibt[isample].unsqueeze(0).squeeze(2)  # [1, 41, 4]
        y_test1 = y_test0.clone()
        X_test1 = X_test_inst[isample].unsqueeze(0).float()
        X_test_mask0 = X_test_mask[isample].unsqueeze(0).float()
        sample_results = []
        valid_periods = []
        sample_all = []
        for t in pre_priod:
            print(f'-----------pre_{t}----------')
            if not torch.any(y_test0[:, :, -1] == t):
                continue
            
            current_X_test = X_test1.clone()
            current_mask = X_test_mask0.clone()
            y_index1 = (y_test0[:, :, -1] == t).int()
            y_test_dict[f'pre_{t}'] = y_test1[y_index1 == 1]
            
            s1 = torch.sum(y_index1, dim=1)
            if t == 12:
                a1_index = (current_X_test[:, :, :, 1] <= t + 12).int()
            else:
                a1_index = (current_X_test[:, :, :, 1] <= t).int()

            t_index = (torch.abs(current_X_test[:, :, :, 1] - t) < 1e-2)
            t_latlon_valid = (
                (current_X_test[:, :, :, 2] != 0)
                & (current_X_test[:, :, :, 3] != 0)
            )
            if not torch.any(t_index & t_latlon_valid):
                print(f"时效 {t} 没有有效经纬度，跳过")
                continue
            
            a1_mask = (a1_index == 0)
            a1_mask_expanded = a1_mask.unsqueeze(-1).expand_as(current_X_test)
            current_X_test[:, :, :, :6][a1_mask_expanded[:, :, :, :6]] = 0
            current_mask[a1_index == 0] = 0
            X_test_dict[f'pre_{t}'] = current_X_test
            test_mask_dict[f'pre_{t}'] = current_mask
            
            input_data = X_test_dict[f'pre_{t}']
            mask_test = test_mask_dict[f'pre_{t}']
            
            pred = tester.run_inference(input_data, mask_test, t)
            if pred is not None:
                sample_results.append(pred)
                valid_periods.append(t)
            else:
                print(f"时效 {t} 推理失败，填充零向量")
                sample_results.append(np.zeros((1, 2)))
                valid_periods.append(t)
        
        if not sample_results:
            print(f"样本 {isample+1} 没有生成任何有效预测，跳过保存")
            continue
        if sample_all:
            all_samples_concat = np.concatenate(sample_all, axis=1)   # 假设每个元素是 [1,1,9,2]
            output_npy_dir = os.path.join("./output_npy", str(args.ty_number))
            os.makedirs(output_npy_dir, exist_ok=True)
            save_path = os.path.join(output_npy_dir, f"{args.report_time}.npy")
            np.save(save_path, all_samples_concat)
            print(f"✅ 保存 all_samples 总文件: {save_path}, 形状: {all_samples_concat.shape}")
        final_array = np.concatenate(sample_results, axis=0)
        utc_now_str = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d%H%M%S")
        formatted_time = format_report_time(args.report_time)
        # txt_filename = f"T_SEVP_C_SCSIOEns_{utc_now_str}_P_TYPHOON_TF_{args.ty_number}_{formatted_time}.txt"
        txt_filename = f"{str(args.ty_number)[2:]}_{args.ty_name.lower().strip()}_{formatted_time}.txt"
        output_dir = os.path.join("./output", str(args.ty_number))
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, txt_filename)
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                for i, t in enumerate(valid_periods):
                    lat, lon = final_array[i]
                    if lat == 0 or lon == 0:
                        continue
                    wind = nanhai_wind.get(t, 0.0)
                    line = f"P+{t:02d}HR {lat:.1f}  {lon:.1f}  {wind:.1f}\n"
                    f.write(line)
            print(f"Prediction saved! File: {txt_filename}, Shape: {final_array.shape}")
        except Exception as e:
            print(f"保存文件失败: {e}", file=sys.stderr)
            traceback.print_exc()
