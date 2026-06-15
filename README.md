# DTL-GANet
A Decoupled Time-Lag Infused Graph Attention Network for Regional Wind Power Forecasting

本项目面向区域风电集群短期功率预测任务，主要研究如何利用多风电场历史功率序列中的空间相关性、时间滞后性和多尺度波动特征，提高风电功率预测精度与模型可解释性。

项目整体采用如下技术路线：

```text
原始风电功率序列
→ 数据清洗与标准化处理
→ VMD 多尺度分解
→ 低中频局部趋势片段构建
→ 跨风电场历史相似片段检索
→ 动态时滞图数据集构建
→ 图注意力网络预测低中频分量
→ 高频残差网络预测高频分量
→ 多尺度预测结果融合
→ 输出最终风电功率预测结果
```

## 1. Project Overview

风电功率序列通常具有较强的随机性、波动性和非平稳性。对于区域风电集群而言，不同风电场之间还可能存在空间相关性和时间滞后关系。直接使用单一深度学习模型对原始功率序列建模，往往难以同时刻画长期趋势、中尺度波动和短时随机扰动。

为此，本项目将原始风电功率序列分解为低频、中频和高频分量，并采用差异化建模策略：

* 对低中频分量：构建局部趋势片段，计算跨风电场历史相似片段，并组织为动态时滞图结构，再使用图注意力网络进行预测；
* 对高频分量：使用一维卷积残差网络建模短时扰动和随机尖峰；
* 对最终结果：融合低中频预测结果与高频预测结果，得到最终风电功率预测值。

## 2. Main Features

* **VMD multi-scale decomposition**
  使用变分模态分解将原始风电功率序列分解为低频、中频和高频分量，降低原始非平稳序列的建模难度。

* **Local trend segment construction**
  对低中频分量进行趋势段落划分，提取方向、长度、斜率和持续时间等形态特征，将连续时间序列转换为局部趋势片段。

* **Cross-farm historical pattern matching**
  从多个风电场的历史片段库中检索与当前目标片段最相似的 Top-k 历史片段，显式记录来源风电场、历史位置、相似度和时间滞后量。

* **Dynamic time-lag graph construction**
  将目标趋势片段作为中心节点，将历史相似片段作为邻居节点，构建随样本动态变化的时滞图结构。

* **Graph attention forecasting model**
  使用边特征感知的图注意力网络对历史相似趋势片段进行动态加权聚合，实现低中频趋势分量预测。

* **High-frequency residual modeling**
  针对高频分量的短时扰动和随机尖峰特征，使用残差网络进行高频分量预测。

* **Multi-scale prediction fusion**
  将低中频预测结果与高频预测结果进行融合，输出最终风电功率预测结果。

## 3. Repository Structure

当前项目的核心代码文件如下：

```text
.
├── vmd_preprossed.py
├── 1_XunZhaoZuiYouZhi_piece_matching_full_sweep_E1_E6_save_visdata.py
├── build_lowmid_piece_libraries.py
├── 2b_build_gat_graph_dataset from_piece features.py
├── 3_gat_lowmid_regressor from_piece _graphs.py
├── 3_train_high_resnet_ vmd.py
├── 4 fusion fixed_ weight_gat_ lowmid_high_resnet.py
└── README.md
```

> Note: Some filenames contain spaces. When running from the command line, wrap such filenames in quotation marks. For long-term maintenance, it is recommended to rename files using underscores only.

## 4. File Description

### 4.1 Data Preprocessing

#### `vmd_preprossed.py`

This script performs the data preprocessing and VMD decomposition stage.

Main functions:

* Load the original wind farm power sequence data;
* Clean and standardize multi-farm power data;
* Construct time-series samples using a sliding window;
* Apply Variational Mode Decomposition (VMD);
* Decompose each wind farm power sequence into low-frequency, medium-frequency, and high-frequency components;
* Save processed data for later trend segment construction and model training.

Expected outputs may include:

* Standardized power sequences;
* VMD low-frequency components;
* VMD medium-frequency components;
* VMD high-frequency components;
* Intermediate `.npy`, `.csv`, or output folders used by subsequent scripts.

### 4.2 Local Trend Segment Construction

#### `1_XunZhaoZuiYouZhi_piece_matching_full_sweep_E1_E6_save_visdata.py`

This script is used for local trend segment scale selection and piece matching experiments.

Main functions:

* Construct local trend segments under different edge counts, such as `E=1` to `E=6`;
* Evaluate the matching performance under different segment lengths;
* Compare validation errors under different local trend segment scales;
* Select the optimal or default trend segment length;
* Save visualization data for later figures and analysis.

In the thesis, the final local trend segment scale is selected by comparing matching performance under different values of `E`.

#### `build_lowmid_piece_libraries.py`

This script builds the low-/medium-frequency local trend segment libraries.

Main functions:

* Use low-/medium-frequency components as the trend sequence;
* Segment each input window into local trend edges;
* Combine consecutive trend edges into local trend pieces;
* Extract trend piece features, including:

  * direction sequence;
  * normalized length;
  * normalized slope;
  * total duration;
  * source wind farm;
  * sample index;
* Save local trend piece libraries for graph dataset construction.

This file provides the structural basis for later time-lag graph learning.

### 4.3 Main Forecasting Model

#### `2b_build_gat_graph_dataset from_piece features.py`

This script builds the graph dataset required by the GAT low-/medium-frequency forecasting model.

Main functions:

* Load local trend piece libraries;
* Search Top-k historical similar trend segments for each target segment;
* Construct dynamic graph samples;
* Define target nodes and historical neighbor nodes;
* Construct node features and edge features;
* Save graph data for subsequent GAT training.

Typical edge features include:

* trend similarity;
* time lag;
* ranking information;
* source wind farm;
* source sample index;
* historical subsequent change information.

#### `3_gat_lowmid_regressor from_piece _graphs.py`

This script trains the graph attention regression model for low-/medium-frequency component prediction.

Main functions:

* Load graph datasets built from local trend piece features;
* Train an edge-feature-aware graph attention network;
* Aggregate Top-k historical similar trend nodes;
* Predict the future low-/medium-frequency component;
* Save trained model weights, predictions, metrics, and intermediate results.

This part is the core implementation of the time-lag-aware graph attention forecasting module.

#### `3_train_high_resnet_ vmd.py`

This script trains the high-frequency residual network.

Main functions:

* Load the VMD high-frequency component data;
* Construct high-frequency historical input windows;
* Train a 1D convolutional residual network;
* Model short-term random fluctuations and high-frequency disturbances;
* Save high-frequency predictions and model results.

This branch is used to compensate for the high-frequency part that is difficult to capture through trend matching.

#### `4 fusion fixed_ weight_gat_ lowmid_high_resnet.py`

This script fuses the low-/medium-frequency GAT prediction results and the high-frequency ResNet prediction results.

Main functions:

* Load low-/medium-frequency predictions from the GAT model;
* Load high-frequency predictions from the ResNet model;
* Perform fixed-weight multi-scale fusion;
* Reconstruct the final wind power prediction;
* Compute final evaluation metrics such as MAE, RMSE, and R²;
* Save final prediction results and figures.

## 5. Running Order

The recommended execution order is:

```bash
# 1. Data preprocessing and VMD decomposition
python vmd_preprossed.py

# 2. Local trend segment scale selection
python 1_XunZhaoZuiYouZhi_piece_matching_full_sweep_E1_E6_save_visdata.py

# 3. Build low-/medium-frequency trend piece libraries
python build_lowmid_piece_libraries.py

# 4. Build GAT graph dataset from trend piece features
python "2b_build_gat_graph_dataset from_piece features.py"

# 5. Train low-/medium-frequency GAT regressor
python "3_gat_lowmid_regressor from_piece _graphs.py"

# 6. Train high-frequency ResNet model
python "3_train_high_resnet_ vmd.py"

# 7. Fuse low-/medium-frequency and high-frequency predictions
python "4 fusion fixed_ weight_gat_ lowmid_high_resnet.py"
```

## 6. Data Description

The original experiment uses historical power sequence data from six wind farms. Each column corresponds to the power output of one wind farm, and each row corresponds to one time step.

A typical data format is:

```text
time_index, farm_1, farm_2, farm_3, farm_4, farm_5, farm_6
0, ...
1, ...
2, ...
...
```

The thesis experiment uses a six-wind-farm power dataset. If the original dataset cannot be publicly released due to data authorization or privacy restrictions, users should prepare their own dataset with the same structure.

Recommended data placement:

```text
data/
└── ningxia6.csv
```

If the dataset path differs from the default path in the scripts, please modify the corresponding file path parameters before running.

## 7. Environment

The project is implemented mainly in Python. The recommended environment is:

```text
Python >= 3.8
NumPy
Pandas
Scikit-learn
Matplotlib
PyTorch
```

If the GAT module uses PyTorch Geometric or another graph learning library in your local implementation, install the corresponding package according to your CUDA and PyTorch versions.

Example installation:

```bash
pip install numpy pandas scikit-learn matplotlib torch
```

Optional:

```bash
pip install torch-geometric
```

## 8. Outputs

Depending on the actual path settings in the scripts, the project may generate the following outputs:

```text
outputs/
├── vmd/
│   ├── low_frequency_components
│   ├── mid_frequency_components
│   └── high_frequency_components
├── piece_libraries/
│   ├── lowmid_piece_library_E1
│   ├── lowmid_piece_library_E2
│   └── ...
├── gat_graph_dataset/
│   ├── graph_samples
│   └── edge_features
├── gat_lowmid_regressor/
│   ├── model_weights
│   ├── predictions
│   └── metrics
├── high_resnet_vmd/
│   ├── model_weights
│   ├── predictions
│   └── metrics
└── fusion/
    ├── final_predictions
    ├── evaluation_metrics
    └── figures
```

Actual output folder names may vary depending on the script configuration.

## 9. Evaluation Metrics

The project mainly uses the following metrics to evaluate forecasting performance:

* **MAE**: Mean Absolute Error;
* **RMSE**: Root Mean Square Error;
* **R²**: Coefficient of Determination.

These metrics are used to evaluate the prediction accuracy of the final wind power forecasting results and the performance of different model components.

## 10. Method Summary

The key idea of this project is to avoid directly modeling the raw non-stationary wind power sequence with a single model. Instead, the method first decomposes the sequence into multiple frequency scales and then uses different models for different components.

For low-/medium-frequency components, the method emphasizes trend structure and historical pattern matching. Local trend pieces are used as graph nodes, and historical similar pieces are used as neighbors. The graph structure is dynamically generated for each target sample, so the model can explicitly represent which wind farm and which historical time period are referenced.

For high-frequency components, the method uses a residual network to model short-term disturbances and random fluctuations.

Finally, the low-/medium-frequency prediction and high-frequency prediction are fused to produce the final wind power forecasting result.

## 11. Notes

* The original dataset may not be included in this repository. Please prepare a six-wind-farm power dataset with the same format if you want to reproduce the experiments.
* Some scripts may contain absolute paths from the original local development environment. Please modify them to relative paths before running on another machine.
* Some filenames contain spaces. It is recommended to rename them before long-term maintenance.
* The fixed-weight fusion script can be further extended to gated fusion if dynamic fusion weights are required.
* Random seeds, training epochs, learning rate, hidden dimensions, and Top-k settings should be checked in each script before reproducing the final results.

## 12. License

This project is released for academic learning and research purposes. Please cite or acknowledge this repository if you use the code or ideas in your own work.

## 13. Acknowledgement

This repository is based on the undergraduate thesis project:

**基于时滞感知图注意力网络的风电集群短期功率预测方法**

The project focuses on wind farm cluster power forecasting, VMD-based multi-scale decomposition, local trend segment matching, dynamic time-lag graph construction, graph attention networks, high-frequency residual modeling, and multi-scale prediction fusion.
