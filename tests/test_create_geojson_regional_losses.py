
import os
import time
import geopandas as gpd
import json
from CNCityRisk import RegionalLossSim

# 使用sample_city fixture作为城市名称
city_name = "武汉"
district_name = "武昌区"
loss_type = 'DS_Struct'
i_rup = 0
i_sim = 0

start_time = time.time()
print(f"开始为{city_name}{' - ' + district_name if district_name else ''}生成GeoJSON数据...")

# 生成GeoJSON数据
geojson_data = RegionalLossSim.create_geojson_regional_losses(
    CityName=city_name,
    DistrictName=district_name,
    LossType=loss_type,
    i_rup=i_rup,
    iSim=i_sim
)

# 计算生成时间
generation_time = time.time() - start_time
print(f"GeoJSON数据生成完成，耗时: {generation_time:.2f}秒")

# 验证GeoJSON是否有效
# 解析JSON
geojson_obj = json.loads(geojson_data)
feature_count = len(geojson_obj.get('features', []))
print(f"GeoJSON包含{feature_count}个建筑物特征")

# 检查关键属性是否存在
if feature_count > 0:
    sample_feature = geojson_obj['features'][0]
    print("示例特征属性:")
    for key, value in sample_feature['properties'].items():
        print(f"  - {key}: {value}")
    # 保存到文件
output_file = f"regional_loss_{city_name}_{district_name}_{loss_type}_{i_rup}_{i_sim}.geojson"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(geojson_data)

print(f"GeoJSON数据已保存到: {output_file}")

# 使用geopandas加载验证
gdf = gpd.read_file(output_file)
print(f"成功使用GeoPandas加载数据，共{len(gdf)}行，列包括: {', '.join(gdf.columns)}")

# 统计损失值分布
if 'loss_value' in gdf.columns:
    loss_counts = gdf['loss_value'].value_counts().sort_index()
    print("损失值分布:")
    for loss_value, count in loss_counts.items():
        percentage = 100 * count / len(gdf)
        print(f"  损失级别 {loss_value}: {count}个建筑 ({percentage:.2f}%)")

# 计算GeoJSON文件大小
file_size = os.path.getsize(output_file)
file_size_mb = file_size / (1024 * 1024)
print(f"GeoJSON文件大小: {file_size_mb:.2f} MB")
