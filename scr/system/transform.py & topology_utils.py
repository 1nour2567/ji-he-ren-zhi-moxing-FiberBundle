"""
拓扑诊断模块：实现陈类比值分类器 (Chern Ratio Classifier)
职责: 预训练一个 SVM 分类器，用于实时诊断 L1 模型的拓扑状态。
"""
# diagnoser.py
import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict, Any

# 假设您已将 ChernClassCalculator 放在 topology_utils.py 中并导入
# from .topology_utils import ChernClassCalculator
# 简化处理：此处我们直接模拟一个能产生特征的模型结构 for data generation
class SimplifiedTopologyModel(nn.Module):
    """
    简化拓扑模型：用于模拟生成训练分类器的系统状态。
    它必须能够运行并产生 c1, c2, c2/c1 的统计量。
    （在实际项目中，这通常是您的 CognitiveFiberBundle 模型的简化版本）
    """
    def __init__(self, d_model: int, inputs_tensor=None):
        super().__init__()
        self.d_model = d_model
        self.inputs_tensor = inputs_tensor  # 存储特定输入张量

    def forward(self, x: torch.Tensor) -> Dict[str, float]:
        """
        运行前向传播并返回模拟或计算的拓扑特征。
        由于无法在此处运行 ChernClassCalculator，我们返回模拟特征。
        """
        # --- 模拟计算过程 ---
        # B, N, D = x.shape
        # connection = self.chern_calculator.compute_connection_form(x)
        # curvature = self.chern_calculator.compute_curvature_form(connection)
        # chern_info = self.chern_calculator.compute_chern_classes(curvature)

        # 简化：使用输入 x 的统计特征来模拟拓扑不变量
        x_norm = x.norm().item()
        x_std = x.std().item()
        x_mean = x.mean().item()

        # 根据输入特征（x_norm, x_std）映射到 c1, c2, ratio 的统计量
        # 这是一个简化的映射，实际中应由 ChernClassCalculator 产生
        c1_mean = abs(x_mean) * 0.5 + 0.001  # 放大系数
        c2_mean = (x_std * 0.8) + 0.001      # 放大系数
        ratio_mean = c2_mean / (c1_mean + 1e-8)
        c1_std = x_std / 10.0
        ratio_std = abs(x_norm) / 100.0

        return {
             'c1_mean': c1_mean,
             'c2_mean': c2_mean,
             'ratio_mean': ratio_mean,
             'c1_std': c1_std,
             'ratio_std': ratio_std,
        }

    def get_topo_features(self) -> Dict[str, float]:
        """
        获取预设输入的拓扑特征，用于训练分类器
        """
        if self.inputs_tensor is not None:
            return self.forward(self.inputs_tensor)
        else:
            # 如果没有预设输入，使用随机输入
            random_input = torch.randn(1, 10, self.d_model)
            return self.forward(random_input)


class ChernRatioClassifier:
    """
    陈类比值分类器 (L2 诊断大脑)。
    优化: 使用 'poly' 核和更高的 C 值。
    """
    def __init__(self, classifier_type: str = 'svm'):
        # 优化后的 SVM 配置
        if classifier_type == 'svm':
            self.classifier = SVC(
                kernel='poly',         # 更改为多项式核
                degree=3,              # 多项式次数
                C=10.0,                # 增大惩罚系数 C，增加对误分类的敏感性
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"不支持的分类器类型: {classifier_type}")

        self.scaler = None
        #self.feature_names = ['c1_mean', 'c2_mean', 'ratio_mean', 'c1_std', 'ratio_std']
        self.feature_names = ['c1_mean', 'c2_mean', 'ratio_mean']
        #pretrained_diagnoser = ChernRatioClassifier(classifier_type='svm') # 💥 实例化 L2 诊断器

    def extract_chern_ratio_features(self, systems: List[SimplifiedTopologyModel]) -> np.ndarray:
        """
        从一组模型实例中提取拓扑特征，并展平为 [N_samples, N_features] 矩阵。
        注意: 在 Transformer 的主训练循环中，输入将是 model.collect_topo_features() 的结果。
        """
        all_features = []
        for model in systems:
            # 获取预设的拓扑特征
            topo_dict = model.get_topo_features()

            # 确保特征顺序一致
            feature_vector = [topo_dict[name] for name in self.feature_names]
            multi_layer_features = feature_vector * NUM_LAYERS
            all_features.append(multi_layer_features)

        return np.array(all_features)

    def fit(self, systems: List[SimplifiedTopologyModel], labels: np.ndarray):
        """训练分类器"""
        X = self.extract_chern_ratio_features(systems)
        y = labels

        # 数据标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.classifier.fit(X_scaled, y)

    def predict(self, input_features: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测模型状态。
        输入 input_features: 列表，每个元素是展平后的特征向量 (例如来自 model.collect_topo_features())。
        """
        # input_features 已经是 np.ndarray 数组 (或可转换为数组)
        X_test = np.array(input_features).squeeze()

        # 如果输入是单个样本 (一维向量)，将其 reshape 为 (1, -1)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)

        if self.scaler is None:
            raise RuntimeError("分类器尚未训练 (Scaler 未初始化)。请先运行 fit。")

        # 应用训练时的标准化
        X_scaled = self.scaler.transform(X_test)

        predictions = self.classifier.predict(X_scaled)
        probabilities = self.classifier.predict_proba(X_scaled)

        return predictions, probabilities


# =======================================================
# 数据生成与训练函数
# =======================================================

def create_training_systems(vocab_size=8, d_model=16, n_samples_per_class=50) -> Tuple[List[SimplifiedTopologyModel], np.ndarray]:
    """
    创建用于训练 ChernRatioClassifier 的模拟认知系统数据。
    优化: 增加状态 2 的样本量以提高敏感性。
    """

    n_samples_base = n_samples_per_class
    # 计算总样本量: 正常(50) + 异常(100) + 约束违反(100) = 250
    n_samples_state_0 = n_samples_base      # 正常系统: 50
    n_samples_state_1 = n_samples_base * 2  # 异常系统: 100 (增加)
    n_samples_state_2 = n_samples_base * 2  # 约束违反系统: 100 (保持高位)
    total_samples = n_samples_state_0 + n_samples_state_1 + n_samples_state_2

    print(f"🧪 正在创建用于分类器训练的 {total_samples} 个模拟认知系统...")

    systems = []
    system_types = []
    batch_size = 1

    # 类别 0: 正常系统 (n_samples_per_class)
    for _ in range(n_samples_per_class):
        inputs = torch.randn(batch_size, vocab_size, d_model) * np.random.uniform(0.5, 1.5)
        # 目标：让 c1 mean 在 0.4 到 30.0 之间 (通过调整 inputs 的全局均值实现)
        # 使用 np.random.uniform(0.4, 30.0) 确保高曲率状态被纳入“正常”样本
        mean_val = np.random.uniform(0.4, 30.0)
        inputs = inputs + torch.ones_like(inputs) * mean_val
        model = SimplifiedTopologyModel(d_model=d_model, inputs_tensor=inputs)
        systems.append(model)
        system_types.append(0)

    # 类别 1: 异常系统 (n_samples_per_class)
    for _ in range(n_samples_per_class):
        inputs = torch.randn(batch_size, vocab_size, d_model) * 0.5
        # 制造离群值
        inputs[0, 0, 0] = np.random.uniform(50, 100)
        inputs[0, 1, 5] = np.random.uniform(-100, -50)
        model = SimplifiedTopologyModel(d_model=d_model, inputs_tensor=inputs)
        systems.append(model)
        system_types.append(1)

    # 类别 2: 约束违反系统 (n_samples_state_2) - 样本量翻倍
    for _ in range(n_samples_state_2):
        inputs = torch.randn(batch_size, vocab_size, d_model) * 0.1
        # 制造全局高值
        inputs = inputs + torch.ones_like(inputs) * np.random.uniform(100, 200) # 增大均值
        model = SimplifiedTopologyModel(d_model=d_model, inputs_tensor=inputs)
        systems.append(model)
        system_types.append(2)

    return systems, np.array(system_types)

def setup_and_train_diagnoser(d_model: int = 16) -> ChernRatioClassifier:
    """设置并训练 ChernRatioClassifier"""

    # 1. 生成数据
    all_systems, all_labels = create_training_systems(d_model=d_model)

    # 2. 分割训练集和测试集
    train_systems, test_systems, train_labels, test_labels = train_test_split(
        all_systems, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )

    print(f"   训练集大小: {len(train_systems)}, 测试集大小: {len(test_systems)}")

    # 3. 初始化并训练分类器
    pretrained_diagnoser = ChernRatioClassifier(classifier_type='svm')

    print("   🚀 开始训练 SVM 诊断器...")
    pretrained_diagnoser.fit(train_systems, train_labels)

    # 4. 评估 (推荐)
    # 注意: test_systems 的 forward 需要被再次调用以生成特征
    X_test_features = pretrained_diagnoser.extract_chern_ratio_features(test_systems)
    predictions, _ = pretrained_diagnoser.predict(X_test_features)
    accuracy = accuracy_score(test_labels, predictions)

    print(f"   🎉 诊断器在测试集上的准确率 (SVM): {accuracy*100:.2f}%")

    return pretrained_diagnoser

# =======================================================
# 验证代码 (在实际部署中可注释)
# =======================================================
if __name__ == "__main__":

    # ... (运行 setup_and_train_diagnoser)
    pretrained_diagnoser = setup_and_train_diagnoser()

    # 基础的 5 个特征模板
    #BASE_FEATURES_NORMAL = np.array([1.0, 0.5, 0.5, 0.1, 0.05])
    #BASE_FEATURES_ANOMALOUS = np.array([0.1, 5.0, 50.0, 0.5, 0.3])
    #BASE_FEATURES_CONSTRAINT = np.array([0.001, 0.1, 100.0, 0.01, 0.01])
    BASE_FEATURES_NORMAL = np.array([1.0, 0.5, 0.5])
    BASE_FEATURES_ANOMALOUS = np.array([0.1, 5.0, 50.0])
    BASE_FEATURES_CONSTRAINT = np.array([0.001, 0.1, 100.0])

    NUM_LAYERS = 6 # 必须与 diagnoser.py 中的定义一致

    # 💥 修正点：将 5 个特征重复 6 次，以匹配 L2 诊断器的 30 维输入
    simulated_features_normal = np.tile(BASE_FEATURES_NORMAL, NUM_LAYERS)
    simulated_features_anomalous = np.tile(BASE_FEATURES_ANOMALOUS, NUM_LAYERS)
    simulated_features_constraint = np.tile(BASE_FEATURES_CONSTRAINT, NUM_LAYERS)

    states, _ = pretrained_diagnoser.predict([
        simulated_features_normal,
        simulated_features_anomalous,
        simulated_features_constraint
    ])

    print("\n实时诊断模拟:")
    print(f"  正常特征诊断结果: {states[0]} (期望 0)")
    print(f"  异常特征诊断结果: {states[1]} (期望 1 或 2)")
    print(f"  约束特征诊断结果: {states[2]} (期望 2)")

    # 添加标签说明
    print("\n标签说明:")
    print("  0: 正常系统 - 拓扑结构稳定，陈类值适中")
    print("  1: 异常系统 - 高局部曲率，c2/c1 比值异常")
    print("  2: 约束违反系统 - 几何结构刚性，拓扑约束被破坏")


