import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from io import StringIO
from datetime import datetime

# 1. 预报路径数据 (起报时间: 2026-06-24 20:00:00)
forecast_raw = """P+12HR 16.2  137.2
P+24HR 16.8  135.2
P+36HR 18.1  134.0
P+48HR 19.4  133.8
P+60HR 18.7  132.1
P+72HR 19.0  130.4
P+96HR 14.7  127.3"""

# 2. 真实路径数据
actual_raw = """2026-06-24 20:00:00 16.2 139.5 8 18 998 热带风暴(TS)
2026-06-24 23:00:00 16.5 139.0 8 18 998 热带风暴(TS)
2026-06-25 02:00:00 17.2 137.9 8 20 995 热带风暴(TS)
2026-06-25 05:00:00 17.3 137.7 8 20 995 热带风暴(TS)
2026-06-25 08:00:00 17.4 137.0 8 20 995 热带风暴(TS)
2026-06-25 11:00:00 17.9 136.5 8 20 995 热带风暴(TS)
2026-06-25 14:00:00 19.4 135.3 8 20 995 热带风暴(TS)
2026-06-25 17:00:00 19.8 135.0 8 20 995 热带风暴(TS)
2026-06-25 20:00:00 20.6 134.5 8 20 995 热带风暴(TS)
2026-06-25 23:00:00 21.7 134.2 8 20 995 热带风暴(TS)
2026-06-26 02:00:00 22.8 134.0 8 20 995 热带风暴(TS)
2026-06-26 05:00:00 23.1 133.8 9 23 995 热带风暴(TS)
2026-06-26 08:00:00 23.4 133.5 9 23 990 热带风暴(TS)
2026-06-26 11:00:00 24.8 133.7 9 23 990 热带风暴(TS)
2026-06-26 14:00:00 27.6 134.3 9 23 990 热带风暴(TS)
2026-06-26 17:00:00 29.1 134.4 9 23 990 热带风暴(TS)
2026-06-26 20:00:00 30.3 135.2 9 23 990 热带风暴(TS)
2026-06-26 23:00:00 31.1 136.0 9 23 990 热带风暴(TS)
2026-06-27 02:00:00 32.1 137.2 9 23 990 热带风暴(TS)
2026-06-27 05:00:00 33.9 138.9 9 23 990 热带风暴(TS)"""

# 设定本次的起报时间 (Base Time)
base_time = datetime.strptime("2026-06-24 20:00:00", "%Y-%m-%d %H:%M:%S")

# --- 解析预报数据 ---
f_times, f_lats, f_lons = [], [], []
for line in StringIO(forecast_raw):
    parts = line.strip().split()
    if len(parts) == 3:
        # 清洗标签：将 "P+12HR" 转换为 "12h"
        clean_time = parts[0].replace("P+", "").replace("HR", "") + "h"
        f_times.append(clean_time)
        f_lats.append(float(parts[1]))
        f_lons.append(float(parts[2]))

# --- 解析真实路径数据 ---
a_times, a_lats, a_lons = [], [], []
for line in StringIO(actual_raw):
    parts = line.strip().split()
    if len(parts) >= 4:
        # 拼接日期和时间并转换为 datetime 对象
        current_time_str = f"{parts[0]} {parts[1]}"
        current_time = datetime.strptime(current_time_str, "%Y-%m-%d %H:%M:%S")
        
        # 计算相对于 24日20时 的小时差
        lead_hours = int((current_time - base_time).total_seconds() / 3600)
        
        a_times.append(f"{lead_hours}h")
        a_lats.append(float(parts[2]))
        a_lons.append(float(parts[3]))

# --- 开始绘制地图 ---
fig = plt.figure(figsize=(12, 10), dpi=150)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# 添加标准地理要素
ax.coastlines(resolution='50m', linewidth=0.8, color='black')
ax.add_feature(cfeature.LAND, facecolor='#f3f4f6')
ax.add_feature(cfeature.OCEAN, facecolor='#e0f2fe')
ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')

# 动态调整视野范围：
# 经度跨度：预报最西到127.3，真实最东到139.5
# 纬度跨度：预报最南到14.7，真实最北到33.9
ax.set_extent([125, 143, 12, 36], crs=ccrs.PlateCarree())

# 配置经纬度网格线
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# 3. 绘制预报路径 (红色圆形，虚线)
ax.plot(f_lons, f_lats, marker='o', color='red', linewidth=2, markersize=6, 
        linestyle='--', transform=ccrs.PlateCarree(), label='Forecast Track (Base: 06-24 20:00)')

# 4. 绘制真实路径 (蓝色正方形，实线)
ax.plot(a_lons, a_lats, marker='s', color='blue', linewidth=2, markersize=5, 
        linestyle='-', transform=ccrs.PlateCarree(), label='Actual Track')

# 5. 添加预报时效标签 (标注在点右上方)
for i, txt in enumerate(f_times):
    ax.annotate(txt, (f_lons[i], f_lats[i]), 
                xytext=(6, 4), textcoords='offset points', 
                fontsize=8, color='darkred', weight='bold',
                transform=ccrs.PlateCarree())

# 6. 添加真实时效标签 
# 真实数据点比较密集，为了防止图面太乱，我们每隔 6 小时（或者选择整点）标注一次标签
for i, txt in enumerate(a_times):
    hours = int(txt.replace("h", ""))
    # 只标注 0h, 12h, 24h, 36h, 48h 等关键节点
    if hours % 12 == 0 or hours == 0:
        ax.annotate(txt, (a_lons[i], a_lats[i]), 
                    xytext=(-8, -12), textcoords='offset points', 
                    fontsize=8, color='darkblue', weight='bold',
                    horizontalalignment='right',
                    transform=ccrs.PlateCarree())

# 标题与图例
plt.title('Typhoon Track Lead-Time Comparison (Base: 2026-06-24 20:00)', fontsize=14, pad=15)
plt.legend(loc='upper left', fontsize=11)

plt.tight_layout()
plt.show()