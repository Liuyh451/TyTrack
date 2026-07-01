import argparse
import warnings
import numpy as np
import datetime
from data.dataloader import GeoBounds
from models.new_model import Cvae_Gan_Seq
import torch
from utils import get_ensemble_mean, denormalize_latlon
from values import ARHeadConfig, GlobalScaler
import os
warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)
#############################
ty_name="mekkhala"
# ty_number="2606"
#############################
def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble Forecasting Model")
    # 数据路径
    parser.add_argument('--data_path', type=str, default='./data/case/wpo/')
    parser.add_argument('--test_X', type=str, default='x.pt')
    parser.add_argument('--test_y', type=str, default='y.pt')
    parser.add_argument('--test_mask', type=str, default='x_masks.pt')
    parser.add_argument('--area', type=str, default='wpo')
    parser.add_argument('--trgpath', type=str, default='./checkpoints/')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model', type=str, default='cvae_gan_seq_light')
    parser.add_argument('--ty_name', type=str, default='default')
    parser.add_argument('--ty_number', type=str, default='2601')
    parser.add_argument('--report_time', type=str, default='2026010100')
    args = parser.parse_args()
    return args

class TyphoonTester:
    def __init__(self, device, base_ckpt_path):
        """
        :param model: 初始化的 Cvae_Gan_Seq 模型
        :param device: cuda 或 cpu
        :param base_ckpt_path: 权重根目录 (如 './checkpoints/typredictor/')
        :param geo_bounds_handler: GeoBounds 类，用于获取不同区域的范围
        """
        self.model = None
        self.device = device
        self.base_ckpt_path = base_ckpt_path
        self.current_geo_bounds = None
        
        # 定义 SCS 的严格判定范围
        self.scs_range = {
            'lat': (5.7, 25.271),
            'lon': (102.4, 129.669)
        }
        # ======= 动态缓存字典 =======
        self.env_models_cache = {}    # 缓存环境场模型，格式: {'scs': model, 'wpo': model}
        self.track_weights_cache = {} # 缓存路径预测权重，格式: {'scs_t12_fold0': state_dict, ...}

    def get_region_type(self, lat, lon):
        """根据经纬度判断属于 scs 还是 wpo"""
        is_in_scs = (self.scs_range['lat'][0] <= lat <= self.scs_range['lat'][1]) and \
                    (self.scs_range['lon'][0] <= lon <= self.scs_range['lon'][1])
        return "scs" if is_in_scs else "wpo"

    def get_env_model(self, region):
        """优化：如果缓存有就直接返回，没有才去创建并加载（只加载一次）"""
        if region not in self.env_models_cache:
            cvae_path = os.path.join(self.base_ckpt_path, 'env_generator', f"best_model_{region}.pt")
            # 初始化模型并移至 GPU
            model = Cvae_Gan_Seq(cvae_path).to(self.device)
            # 存入缓存字典
            self.env_models_cache[region] = model
            print(f"[首次加载] 已成功缓存 {region.upper()} 环境场模型至显存")
        return self.env_models_cache[region]

    def get_track_weight(self, region, t, fold_idx):
        """优化：如果缓存有权重字典就直接返回，没有才读硬盘（只读一次）"""
        cache_key = f"{region}_{t}_{fold_idx}"
        if cache_key not in self.track_weights_cache:
            ckpt_path = os.path.join(self.base_ckpt_path, 'typredictor',region, f"pre_{t}", f"cgan_seq_best_{fold_idx}.pt")
            # 加载到 CPU 即可，后面 load_state_dict 会自动处理设备
            state = torch.load(ckpt_path, map_location='cpu')
            self.track_weights_cache[cache_key] = state
            print(f"[首次读取] 已成功缓存权重文件至内存: {cache_key}")
        return self.track_weights_cache[cache_key]

    def run_inference(self, X, mask, t):
        X = X.to(self.device)
        mask = mask.to(self.device)
        fold_results=[]

        with torch.no_grad():
            # 1. 计算集合平均值
            X_mean = get_ensemble_mean(X, mask, t)
            # 2. 判定区域
            X_mean_np = X_mean.detach().cpu().numpy()
            mean_lat, mean_lon = X_mean_np[0, 0], X_mean_np[0, 1] # 直接取值，不需要反归一化
            
            region = self.get_region_type(mean_lat, mean_lon)
            
            print(f">>> Pos: ({mean_lat:.2f}N, {mean_lon:.2f}E) | Switched to [{region.upper()}] weights")
            # 动态更新地理边界与归一化器
            self.current_geo_bounds=GeoBounds(region)
            dynamic_scaler = GlobalScaler(area=region, device=self.device)
            X = dynamic_scaler.normalize(X, t)
            X_mean = get_ensemble_mean(X, mask, t,False)
            #加载环境场生成模型
            self.model = self.get_env_model(region)
        # 3. 循环加载 5 个 Fold
        for fold_idx in range(5):
            state = self.get_track_weight(region, t, fold_idx)
            self.model.track_predictor.load_state_dict(state)
            ARHeadConfig.apply(self.model.track_predictor.ar_head,region)
            self.model.eval()
            with torch.no_grad():
                pred_latlon= self.model(X, mask, X_mean)
                
                # 反归一化 (注意：如果判定区域后 geo_bounds 发生变化，需在此适配)
                pred_latlon_denorm = denormalize_latlon(pred_latlon.detach().cpu().numpy(), self.current_geo_bounds)
                print(pred_latlon_denorm)
                fold_results.append(pred_latlon_denorm)

        if not fold_results:
            return None
        ensemble_pred = np.mean(np.stack(fold_results, axis=0), axis=0) # [1, 2]
        return ensemble_pred


args = parse_args()

path = args.data_path+ty_name+'/'
pre_priod = [12, 24, 36, 48, 60, 72, 96, 120]
y_test_dict = {}
X_test_dict = {}
test_mask_dict = {}
#加载数据
y_test_ibt = torch.load(path + args.test_y)
X_test_inst = torch.load(path + args.test_X)  # ([99, 19, 8, 14])
X_test_mask = torch.load(path + args.test_mask)  # ([99, 19, 8])
print(f"机构数据：{X_test_inst.shape}，真实值：{y_test_ibt.shape}")
#创建推理模型实例
tester = TyphoonTester(device=args.device, base_ckpt_path=args.trgpath)

for isample in range(X_test_inst.shape[0]):
    print(f'----------第{isample + 1}个样本----------')
    y_test0 = y_test_ibt[isample].unsqueeze(0).squeeze(2)  # [1, 41, 1, 4]--->[1, 41, 4]
    y_test1 = y_test0.clone()
    X_test1 = X_test_inst[isample].unsqueeze(0).float()
    X_test_mask0 = X_test_mask[isample].unsqueeze(0).float()
    sample_results = []
    for t in pre_priod:
        print(f'-----------pre_{t}----------')
        # if not torch.any(y_test0[:, :, -1] == t):
        #     continue
        # 创建当前时间步的数据副本，避免影响其他时间步的数据
        current_X_test = X_test1.clone()
        current_mask = X_test_mask0.clone()
        # 根据时间步筛选标签
        y_index1 = (y_test0[:, :, -1] == t).int()  # [1, 41, 4]
        y_test_dict[f'pre_{t}'] = y_test1[y_index1 == 1]

        # 计算有效样本数量
        s1 = torch.sum(y_index1, dim=1)
        # 构造掩码
        if t == 12:
            a1_index = (current_X_test[:, :, :, 1] <= t + 12).int()
        else:
            a1_index = (current_X_test[:, :, :, 1] <= t).int()

        # 屏蔽无效时间步的数据
        a1_mask = (a1_index == 0)
        a1_mask_expanded = a1_mask.unsqueeze(-1).expand_as(current_X_test)
        # 更新掩码
        current_X_test[:, :, :, :6][a1_mask_expanded[:, :, :, :6]] = 0
        current_mask[a1_index == 0] = 0
        # 存储当前时间步的数据
        X_test_dict[f'pre_{t}'] = current_X_test
        test_mask_dict[f'pre_{t}'] = current_mask
        # 准备输入模型的数据
        input_data = X_test_dict[f'pre_{t}']
        mask_test = test_mask_dict[f'pre_{t}']

        # 运行推理
        pred = tester.run_inference(input_data, mask_test, t)
        
        if pred is not None:
            sample_results.append(pred) # 每个 pred 是 [1, 2]
        else:
        # 如果某时效缺失，填充 0 保持形状对齐
            sample_results.append(np.zeros((1, 2)))
        # 收集当前时间步的结果
    # 拼接所有时效 [8, 2]
    final_array = np.concatenate(sample_results, axis=0)

    # 1. 生成当前 UTC 时间字符串 (年月日时分秒)
    utc_now_str = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")

    # 4. 构造文件名
    txt_filename = f"T_SEVP_C_SCSIOEns_{utc_now_str}_P_TYPHOON_TF_{args.ty_number}_{args.report_time}.txt"
    save_path = os.path.join(args.data_path, txt_filename)

    # 5. 写入文件
    with open(save_path, 'w', encoding='utf-8') as f:
        for i, t in enumerate(pre_priod):          # pre_priod = [12, 24, 36, 48, 60, 72, 96, 120]
            lat, lon = final_array[i]               # 每个 pred 是 [纬度, 经度]
            # 格式示例: P+12HR 15.7  114.9
            line = f"P+{t:02d}HR {lat:.1f}  {lon:.1f}\n"
            f.write(line)

    print(f"✅ Prediction saved! File: {txt_filename}, Shape: {final_array.shape}")
    
