# 更新日志 (Changelog)

- [ ] 新增增加地震事件次数的流程

## [0.5.0] - 2026-9-11

- [x] 优化IDA的模拟，没有必要考虑建筑的面积，因为不同面积时，刚度、质量都同比增加，所以分析的结果是完全一样的。将结构分析的代表模型大量减少，节省计算时间。

## [0.4.3] - 2026-9-11

- [x] 优化使用cKDTree进行建筑物与POI的空间匹配，减少计算时间

## [0.4.2] - 2026-9-10

- [x] 只匹配 POI 根目录下的城市文件夹，避免把文件夹内同名的 CSV 文件误认为多份城市数据

## [0.4.1] - 2026-9-10

- [x] [PrepareBldData.py](CNCityRisk/PrepareBldData.py) 中修改了设防烈度查询时，行政规划名字不一致的问题

## [0.4.0] - 2026-9-10

### 增加

- [x] 成都市的模拟完成

### 修改

- [x] 优化 `RegionalLossSim._simulate_regional_losses_multiprocess` 中连续损失的模拟
- [x] 增加 `BldPSPA_ReprBld.SimContinuousLossRandomlyArray` 连续损失的模拟函数
- [x] 优化 `RegionalLossSim._simulate_regional_losses_multiprocess`，直接生成按震源分组的损失结果
- [x] 优化 `RegionalLossSim._simulate_regional_losses_multiprocess` 中损伤状态的模拟

## [0.3.15] - 2026-9-2

- 修改 `IDA_ReprBld.py` 并行读取地震动的bug
- 修改 `PrepareBldData.py` 成都部分区县名修改的问题

## [0.3.14] - 2026-8-28

- [x] 增加`cleanup_im_after_maps`流程，清理多余的文件，减小空间占用

## [0.3.13] - 2026-8-22

### 增加

- 跑完南京的标准步骤。

## [0.3.12] - 2026-8-21

### 修复

- 修复损伤概率总和有误差不为1的bug

## [0.3.11] - 2026-8-21

### 修复

- [x] 修复`grp_im_loss.create_dataset`中损伤状态不能保存为压缩数据集的问题，改为保存为标量数据集。

## [0.3.10] - 2026-8-18

### 修复

- 修复`SearchBldPOI_Local`分类1-3的读取

## [0.3.9] - 2026-8-18

### 修复

- 修复`SearchBldPOI_Local`经纬度的读取

## [0.3.8] - 2026-8-18

- 打包标准工作流程`WORKFLOW_STANDARD`

## [0.3.7] - 2026-8-18

- [x]  `Main.py` 中标准步骤
- [x] 重跑3个城市标准步骤（允许缓存的，不是完全重算）
- [x] 修改日志模块，能记录系统错误
- [x] AI修复一些bug

## [0.3.6] - 2026-8-3

### 更改 (Changed)

- [x] 修改bug: event_based_mfd.csv 是发生频率而不是超越频率
- [x] 修改bug: `weights = np.repeat(multiplicity / N_sim, N_sim)` 改成 `np.tile`

## [0.3.5] - 2026-7-31

### 更改 (Changed)

- [x] 增加打包输出函数，将静态结果结果打包成一个压缩文件，方便其他程序对接绘图
- [x] 前面的所有内容，继续

## [0.3.4] - 2025-10-26

### 更改 (Changed)

- [x] IDA分析重新计算
- [x] 损失比lambda计算程序
- [x] 将建筑数据移到`results/`文件夹下

## [0.3.3] - 2025-10-26

### 更改 (Changed)

- [x] 重新生成瓦片地图，提高矢量瓦片地图的响应速度 '-dsco', 'MAX_FEATURES=1500'
- [x] 添加汶川地震建筑群损失验证的分析对比

## [0.3.2] - 2025-09-05

### 更改 (Changed)

- [x] 重新选定重庆市的范围为主城区，并重新计算
- [x] 更新依赖文件`requirements.txt`

## [0.3.1] - 2025-09-05

### 更改 (Changed)

- [x] 更新依赖，添加了cnmaps

## [0.3.0] - 2025-08-20

### 更改 (Changed)

- [x] 新增重庆市的分析
