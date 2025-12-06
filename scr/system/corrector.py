# =======================================================
# corrector.py
# 元学习修正模块：根据拓扑诊断结果执行结构修正
# 职责: 包含调整正则化权重、重置联络参数等关键操作。
# =======================================================

import torch
import torch.nn as nn
from typing import TYPE_CHECKING
import torch.optim as optim

# 为了避免循环依赖和简化类型提示，使用 TYPE_CHECKING
if TYPE_CHECKING:
    from .transformer import TopologyAwareTransformer # 假定 Transformer 类定义在 transformer.py 中

class TopologicalCorrector:
    """
    元学习修正器：根据诊断状态 (0, 1, 2) 执行修正动作。

    状态定义:
    - 0: 正常系统
    - 1: 异常系统 (高局部曲率/离群值 -> c2/c1 放大)
    - 2: 约束违反系统 (几何结构刚性/锁定 -> c1/c2 值异常)
    """

    def __init__(self, model: 'TopologyAwareTransformer', optimizer: optim.Optimizer, initial_lambda: float = 0.01):
        """
        初始化修正器。
        :param model: L1 学习模型 (TopologyAwareTransformer 实例)
        :param optimizer: 模型的优化器
        :param initial_lambda: 拓扑正则化项的初始权重 (λ)
        """
        self.model = model
        self.optimizer = optimizer
        self.initial_lambda = initial_lambda
        self.current_lambda = initial_lambda
        # 冻结状态标志
        self.frozen_learning_rate = None

    def adjust_lambda(self, factor: float):
        """
        调整拓扑正则化权重 (self.current_lambda)。
        :param factor: 乘法因子 (例如 2.0 增大，0.5 减小)
        """
        # 确保 λ 不会低于一个微小值
        self.current_lambda = max(self.initial_lambda * 0.01, self.current_lambda * factor)
        print(f"   [Corrector] 调整拓扑正则化权重 λ: new_lambda = {self.current_lambda:.6f}")

    def reset_connection_forms(self, layer_index: int = 0):
        """
        打破约束：重置关键层的联络形式 A 参数。
        这是一种激进的修正，用于打破拓扑冻结或约束违反状态。
        :param layer_index: 要重置的层索引。
        """
        if layer_index < len(self.model.layers):
            # 获取 ChernClassCalculator 模块
            calc = self.model.layers[layer_index].chern_calculator

            # 重置联络形式参数 A (connection_form) 为随机值
            # 保持原始标准差，以保持初始曲率量级
            nn.init.normal_(calc.connection_form.data, mean=0., std=0.1)

            # 可选：重置曲率权重
            nn.init.normal_(calc.curvature_weight.data, mean=0., std=0.01)

            print(f"   [Corrector] 🚨 激进修正: 重置 Layer {layer_index} 的联络形式 (A)。")

    def temporary_freeze_lr(self, duration: float = 100):
        """
        暂时冻结 L1 任务的学习率，强制模型通过拓扑正则化来修正结构。
        :param duration: 冻结的步数或时间。
        """
        if self.frozen_learning_rate is None:
            # 保存当前学习率并设置一个非常小的值
            current_lr = self.optimizer.param_groups[0]['lr']
            self.frozen_learning_rate = current_lr
            self.optimizer.param_groups[0]['lr'] = current_lr * 0.01 # 降为 1%
            print(f"   [Corrector] 冻结 LR: L1 任务学习率降至 {self.optimizer.param_groups[0]['lr']:.8f}")

    def unfreeze_lr(self):
        """
        解除 L1 任务学习率的冻结。
        """
        if self.frozen_learning_rate is not None:
            self.optimizer.param_groups[0]['lr'] = self.frozen_learning_rate
            self.frozen_learning_rate = None
            print(f"   [Corrector] 解除冻结: L1 任务学习率恢复至 {self.optimizer.param_groups[0]['lr']:.8f}")

    def execute_correction(self, predicted_state: int, topo_info: dict):
        """
        执行基于诊断状态的元学习修正。
        :param predicted_state: L2 诊断器预测的系统状态 (0, 1, or 2)。
        """
        self.unfreeze_lr() # 每次诊断前尝试解除冻结

        #avg_c1 = topo_info.get('first_chern_class_mean', 0.0)

        #if avg_c1 > 20.0:
          # 我们知道 25.93 是一个锁定的状态
          #predicted_state = 2
          #print(f"   [Corrector] 🚨 硬触发：平均 c1 ({avg_c1:.2f}) 超过安全阈值 20.0，强制诊断为状态 2。")

        HIGH_LAMBDA_THRESHOLD = 5.0 # 定义智能触发的高阈值 (可根据需要调整)

        #if predicted_state == 1: # 如果诊断器预测为状态 1 (异常/高曲率)

          # 检查当前 λ 是否已超过高阈值
          #if self.current_lambda > HIGH_LAMBDA_THRESHOLD:
            # λ 已经很高，表明温和修正无效，强制转为状态 2 (激进修正)
            #predicted_state = 2
            #print(f"   [Corrector] 💡 智能升级: 诊断为状态 1，但 λ ({self.current_lambda:.4f}) > {HIGH_LAMBDA_THRESHOLD:.1f}，强制转为状态 2。")


        if predicted_state == 1:

            # 状态 1: 异常系统 (高局部曲率)
            # 状态 1 修正：在 λ <= 5.0 时执行温和修正
            print("   [Corrector] 诊断结果: 状态 1 (异常/高曲率)。")
            # 措施: 1. 增加正则化力度以平滑流形； 2. 短暂冻结 LR，强调拓扑修正。
            self.adjust_lambda(factor=1.2) # 激进增加 λ
            #self.temporary_freeze_lr()

        elif predicted_state == 2:

            # 状态 2: 约束违反系统 (结构锁定/几何刚性)
            print("   [Corrector] 诊断结果: 状态 2 (约束违反/锁定)。")
            # 措施: 1. 重置联络打破锁定； 2. 降低 λ，给模型重新学习几何结构的空间。
            self.reset_connection_forms(layer_index=0) # 修正第一层
            self.adjust_lambda(factor=0.5) # 大幅降低 λ

        elif predicted_state == 0:
            # 状态 0: 正常系统 (健康)
            print("   [Corrector] 诊断结果: 状态 0 (正常/健康)。")
            # 措施: 缓慢恢复到初始正则化权重
            #if self.current_lambda > self.initial_lambda * 1.1:
                 #self.adjust_lambda(factor=0.8) # 缓慢衰减 λ
            #else:
                 #self.current_lambda = self.initial_lambda # 稳定在初始值
            
            if self.current_lambda > self.initial_lambda:# 措施: 迅速恢复到初始正则化权重 (0.005)，L1 学习率保持解锁
              self.current_lambda = self.initial_lambda
              print("   [Corrector] 措施: λ 恢复到初始值。")
            else:
              pass


        else:
            print(f"   [Corrector] 警告: 无法识别的诊断状态 {predicted_state}。未执行修正。")

        print(f"   [Corrector] 调整拓扑正则化权重 λ: new_lambda = {self.current_lambda:.6f}") # 添加新 λ 的输出


# =======================================================
# 验证代码 (在实际部署中需 Transformer 和 Optimizer 实例)
# =======================================================
if __name__ == "__main__":


    # 占位符类：模拟 Transformer 和 Optimizer
    class MockLayer(nn.Module):
        def __init__(self, d):
            super().__init__()
            # 模拟 ChernClassCalculator 的 connection_form 存在
            self.chern_calculator = type('MockCalc', (object,), {
                'connection_form': nn.Parameter(torch.randn(d, d) * 0.1),
                'curvature_weight': nn.Parameter(torch.randn(d, d) * 0.01)
            })()

    class MockTransformer:
        def __init__(self, d, L):
            self.layers = [MockLayer(d) for _ in range(L)]

    # 1. 初始化
    D_MODEL = 16
    mock_model = MockTransformer(D_MODEL, L=4)
    mock_optimizer = optim.Adam([p for l in mock_model.layers for p in [l.chern_calculator.connection_form, l.chern_calculator.curvature_weight]], lr=1e-4)

    corrector = TopologicalCorrector(mock_model, mock_optimizer, initial_lambda=0.01)

    print(f"初始 λ: {corrector.current_lambda}")
    print(f"初始 LR: {corrector.optimizer.param_groups[0]['lr']}")

    # 模拟一个拓扑信息字典
    # 状态 1 模拟（c1 均值正常）
    simulated_topo_info_normal = {'first_chern_class_mean': 0.4}
    # 状态 2/锁定模拟（c1 均值极高，用于触发硬修正）
    simulated_topo_info_locked = {'first_chern_class_mean': 25.0}


    # 2. 模拟 状态 1 (异常)
    print("\n--- 模拟诊断状态 1 (异常) ---")
    corrector.execute_correction(1, simulated_topo_info_normal)
    print(f"修正后 λ: {corrector.current_lambda:.6f}")
    print(f"修正后 LR: {corrector.optimizer.param_groups[0]['lr']:.8f}") # 应该被冻结

    # 3. 模拟 状态 2 (约束违反)
    print("\n--- 模拟诊断状态 2 (约束违反) ---")
    # 获取重置前的 connection_form (Layer 0)
    old_conn = mock_model.layers[0].chern_calculator.connection_form.data.clone().mean().item()
    print(f"Layer 0 A 均值 (重置前): {old_conn:.4f}")

    corrector.execute_correction(2, simulated_topo_info_locked)
    new_conn = mock_model.layers[0].chern_calculator.connection_form.data.mean().item()

    print(f"Layer 0 A 均值 (重置后): {new_conn:.4f} (应变化)")
    print(f"修正后 λ: {corrector.current_lambda:.6f}") # 应该大幅降低
    print(f"修正后 LR: {corrector.optimizer.param_groups[0]['lr']:.8f}") # 应该解除冻结

    # 4. 模拟 状态 0 (正常)
    print("\n--- 模拟诊断状态 0 (正常) ---")
    corrector.execute_correction(0, simulated_topo_info_normal)
    print(f"修正后 λ: {corrector.current_lambda:.6f}") # 应该恢复到初始 λ 或略高于初始 λ
