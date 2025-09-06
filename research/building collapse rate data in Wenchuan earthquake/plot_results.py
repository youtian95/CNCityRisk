
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cnmaps import get_adm_maps, draw_maps
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('data_20250904_160202_with_coords.csv')

# 获取数据的地理范围
lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
lat_min, lat_max = df['latitude'].min(), df['latitude'].max()

# 扩展边界范围以便更好显示
lon_buffer = (lon_max - lon_min) * 0.1
lat_buffer = (lat_max - lat_min) * 0.1

# 创建倒塌率到数值的映射函数
def get_collapse_rate_value(class_name):
    """根据分类名称获取倒塌率的中位数值"""
    rate_mapping = {
        '90-100%': 95,
        '80-90%': 85,
        '70-80%': 75,
        '60-70%': 65,
        '50-60%': 55,
        '40-50%': 45,
        '30-40%': 35,
        '20-30%': 25,
        '10-20%': 15,
        '0-10%': 5
    }
    return rate_mapping.get(class_name, 50)  # 默认50%

# 创建颜色映射
class_colors = {
    '90-100%': '#800000',  # 深红色
    '80-90%': '#FF4500',   # 橙红色
    '70-80%': '#FFA500',   # 橙色
    '60-70%': '#FFD700',   # 金色
    '50-60%': '#FFFF00',   # 黄色
    '40-50%': '#ADFF2F',   # 黄绿色
    '30-40%': '#00FF00',   # 绿色
    '20-30%': '#00FFFF',   # 青色
    '10-20%': '#0000FF',   # 蓝色
    '0-10%': '#000080'     # 深蓝色
}


# 绘制底图和县区轮廓
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

province = get_adm_maps(province='四川省', record='first', only_polygon=True)
draw_maps(get_adm_maps(province='四川省', level='区县'), color='b', linewidth=0.8)

# 按分类绘制散点
for class_name in df['class_name'].unique():
    class_data = df[df['class_name'] == class_name]
    color = class_colors.get(class_name, '#808080')  # 默认灰色
    
    # 根据倒塌率计算圆圈大小
    collapse_rate = get_collapse_rate_value(class_name)
    # 基础大小为20，最大倒塌率时大小为100
    circle_size = 20 + (collapse_rate / 100) * 80
    
    ax.scatter(class_data['longitude'], class_data['latitude'], 
              c=color, s=circle_size, alpha=0.8, label=f'{class_name} ({len(class_data)}点)',
              edgecolors='black', linewidth=0.3)

# 设置地图显示范围
ax.set_xlim(lon_min - lon_buffer, lon_max + lon_buffer)
ax.set_ylim(lat_min - lat_buffer, lat_max + lat_buffer)

# 设置标题和标签
ax.set_title('汶川地震建筑倒塌率分布图', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('经度 (°)', fontsize=12)
ax.set_ylabel('纬度 (°)', fontsize=12)

# 添加网格
ax.grid(True, alpha=0.3, linestyle='--')

# 添加图例
legend = ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), 
                  frameon=True, fancybox=True, shadow=True)
legend.set_title('倒塌率分类', prop={'weight': 'bold'})

# 设置坐标轴格式
ax.ticklabel_format(style='plain', axis='both')

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('wenchuan_earthquake_collapse_rate_map.png', dpi=300, bbox_inches='tight')
plt.savefig('wenchuan_earthquake_collapse_rate_map.pdf', bbox_inches='tight')

# 显示图片
plt.show()

print(f"\n图片已保存为:")
print("- wenchuan_earthquake_collapse_rate_map.png")
print("- wenchuan_earthquake_collapse_rate_map.pdf")