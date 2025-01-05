# 国内主要城市的抗震韧性概率评估

## 项目简介
本项目旨在评估国内主要城市的抗震韧性，提供相关数据和分析结果。

## 文件结构

- `config.ini`：【需要手动新建】配置文件，包括APIkey。内容：
  ```
  [API]
  api_key_OpenTopography = <YOUR API KEY>
  ```
- `README.md`：项目简介和文件夹结构说明（即本文件）。
- `Main.py`：主要分析步骤。
- `requirements.txt`：项目所需的Python库列表。
- `src/`： 包含数据处理和分析的脚本文件。
  - `Data/`：存放原始数据和处理后的数据文件。
    - `BldData`：处理后的建筑的基本信息，包括高度、面积、结构类型等，用于生成结构模型。
    - `CNBldFootprintHeight_Che2024`：建筑轮廓和高度的源数据文件。
  - `results/`：存放分析结果和可视化图表。
    - `IDA_Results`：IDA分析结果，每个文件代表一个Repr代表结构。
    - `ReprBld`：代表结构的结构模型参数。
    - `temp`：临时、中间结果。
  - `PrepareBldData.py`：生成建筑基本信息数据。
  - `plot_city.py`：对城市区域进行绘图。
  - `ClassifyBld.py`：对建筑进行分类，生成代表建筑。
  - `IDA_ReprBld.py`：对代表建筑进行IDA分析。
  - `Utilities.py`：其他功能函数。
  - `PythonLib`：其他python包
    - `BldCluster`：对建筑进行分裂，获得代表建筑。
    - `BldStructClassify`：推测建筑的结构类型。
    - `MDOFModel`：结构的多自由度层剪切模型，可以进行单体结构的动力分析和损失模拟等。



## 配置环境步骤
1. **创建虚拟环境**：
    - 打开Anaconda Prompt（或终端）。
    - 使用以下命令创建一个新的虚拟环境（例如，环境名称为`quake_resilience`）：
      ```bash
      conda create --name quake_resilience python=3.11
      ```
1. **激活虚拟环境**：
    - 使用以下命令激活刚刚创建的虚拟环境：
      ```bash
      conda activate quake_resilience
      ```
1. **安装依赖库**：
    - 运行以下命令安装项目所需的Python库：
      ```bash
      pip install -r requirements.txt
      ```
1. **运行项目**：
    - 完成以上配置后，就可以运行`Main.py`文件


