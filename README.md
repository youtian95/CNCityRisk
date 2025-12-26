# 国内主要城市的抗震韧性概率评估

## 项目简介
本项目评估了国内主要城市的抗震韧性，提供相关数据和分析结果。

## 文件结构

- `config.ini`：【需要手动新建】配置文件，包括APIkey。内容：
  ```
  [API]
  api_key_OpenTopography = <YOUR API KEY>
  ```
- `README.md`：项目简介和文件夹结构说明（即本文件）。
- `Main.py`：主要分析步骤。
- `requirements.txt`：项目所需的Python库列表。
- `CNCityRisk/`： 包含数据处理和分析的脚本文件。
  - `Data/`：存放原始数据和处理后的数据文件。
    - `BldData`：处理后的建筑的基本信息，包括高度、面积、结构类型等，用于生成结构模型。
    - `CNBldFootprintHeight_Che2024`：建筑轮廓和高度的源数据文件。
    - `POI_AMAP_2020`：高德POI数据，用来预测结构类型
    - `GEM-CHN-v2021.1.0`：震源模型，用来生成震源
  - `results/`：存放中间分析结果
    - `AnnualizedRisk`：【不可删除】年均损失计算
    - `EQsources`：【不可删除】震源
    - `IDA_Results`：【不可删除】IDA分析结果，每个文件代表一个Repr代表结构。
    - `IMsim`：（一个城市的全部分析算完之后可以删除）地震动强度随机场模拟结果，建筑IM插值结果
    - `IndBldLoss`：【不可删除】单体建筑的地震损失模拟结果
    - `Maps`：【不可删除】生成用于展示的web地图数据
    - `RegionalLossSim`：（一个城市的全部分析算完之后可以删除）区域损失模拟结果
    - `ReprBld`：【不可删除】代表结构的结构模型参数，用于单体代表结构的IDA分析
    - `ReprBld_PSPA`：【不可删除】代表建筑的信息，用于单体代表建筑地震损失分析
    - `temp`：临时、中间结果。
  - `PrepareBldData.py`：生成建筑基本信息数据。
  - `plot_city.py`：对城市区域进行绘图。
  - `ClassifyBld.py`：对建筑进行分类，生成代表建筑。
  - `IDA_ReprBld.py`：对代表建筑进行IDA分析。
  - ...
  - `Utilities.py`：其他功能函数。


## 开发环境
1. **创建虚拟环境**：
    - 打开Anaconda Prompt（或终端）。
    - 使用以下命令创建一个新的虚拟环境（例如，环境名称为`quake_resilience`）：
      ```bash
      conda create -n quake_resilience python=3.12 numpy==1.26.2
      # 备注：numpy版本其他版本可能会与openquake.engine冲突
      ```
2. **激活虚拟环境**：
      ```bash
      conda activate quake_resilience
      ```
3. **安装依赖**：
      ```bash
      pip install openquake.engine BldStructPred MDOFModel BldCluster h5py folium cartopy dask geojsoncontour addressparser pypinyin py7zr rasterio cnmaps
      # 备注：numpy版本其他版本可能会与openquake.engine冲突
      ```
4. **安装当前项目包**：
      ```bash
      pip install -e .
      ```
5. **运行项目**：
    - 完成以上配置后，就可以运行`Main.py`文件


