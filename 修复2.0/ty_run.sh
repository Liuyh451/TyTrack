#!/bin/bash

# 激活 conda 环境（需要先初始化 conda）
# 方法1：使用 conda 的绝对路径初始化（推荐，适用于脚本）
source /share/anaconda3/etc/profile.d/conda.sh   # 根据实际 conda 安装路径修改
conda activate nanhaisuo

# 或者使用相对路径（如果 conda 已经加入 PATH，且脚本在用户环境下执行）
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate nanhaisuo
cd /share/TY_Forecast_System/ty_deploy/
# 执行 Python 脚本
python ty_main.py

# 可选：退出后打印完成信息
echo "执行完成"
