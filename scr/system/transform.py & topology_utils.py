# =======================================================
# transformer.py
# 主体模型模块：拓扑感知 Transformer (L1 学习系统)
# 职责: 执行核心任务 (如序列处理)，并生成拓扑特征供 L2 诊断。
# =======================================================

# =======================================================
# topology_utils.py
# 拓扑几何工具模块：实现陈类和曲率形式的计算
# =======================================================

import torch
import torch.nn as nn
import math
from typing import Dict, Optional

class ChernClassCalculator(nn.Module):
    """
    陈类计算器 - 计算网络激活流形上的拓扑不变量 (增强版: 包含 c2 和 c2/c1)
    
    Chenn-Weil 近似:
    1. 联络形式 A
    2. 曲率形式 F = A² - AᵀA + ...
    3. 陈类 c1 ≈ tr(F) / (2π)
    4. 陈类 c2 ≈ (tr(F²) - tr(F)²) / (8π²)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # 1. 联络形式参数 A - 学习联络（Connection Form）
        # 这是一个可训练参数，代表流形上的基础联络结构
        self.connection_form = nn.Parameter(torch.randn(d_model, d_model) * 0.1)
        
        # 2. 曲率权重 (用于输入非线性项 A_pert)
        # 用于根据输入 x 局部扰动联络 A
        self.curvature_weight = nn.Parameter(torch.randn(d_model, d_model) * 0.01)

    def compute_connection_form(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算联络形式 A.
        输入 x: [B, N, D]
        输出 connection: [B * N, D, D]
        """
        batch_size, n, d = x.shape
        x_flat = x.reshape(-1, d)  # [B*N, D]
        num_patches = batch_size * n
        
        # 1. 基础联络形式 A_base - [B*N, D, D]
        A_base = self.connection_form.unsqueeze(0).expand(num_patches, -1, -1)
        
        # 2. 局部扰动项 A_pert
        # input_effect: [B*N, D]
        input_effect = torch.einsum('bi,ij->bj', x_flat, self.curvature_weight)
        input_diag = torch.diag_embed(input_effect)
        
        # 3. 最终联络形式 A = A_base + A_pert * 0.1
        connection = A_base + input_diag * 0.1
        
        return connection

    def compute_curvature_form(self, connection: torch.Tensor) -> torch.Tensor:
        """
        计算曲率形式 F = dA + A∧A 的近似。
        输入 connection: [B * N, D, D]
        输出 curvature: [B * N, D, D]
        """
        A = connection
        
        # 近似公式：F = A² - AᵀA + 0.01 * A³ (加入高阶项增强区分度)
        A_squared = torch.bmm(A, A)
        A_transposed = A.transpose(-2, -1)
        A_transposed_squared = torch.bmm(A_transposed, A)
        A_cubed = torch.bmm(A_squared, A)
        
        curvature = A_squared - A_transposed_squared + 0.01 * A_cubed
        return curvature

    def compute_chern_classes(self, curvature: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算陈类 (包含 c1, c2 和 c2/c1)
        输入 curvature: [B * N, D, D]
        输出 Dict[str, torch.Tensor]: 包含 c1, c2, 比例等 [B * N] 形状的张量
        """
        
        # 1. 第一陈类 c1
        trace_F = torch.diagonal(curvature, dim1=-2, dim2=-1).sum(dim=-1)  # tr(F): [B * N]
        # c1 = tr(F) / (2πi) -> 在实际计算中取实部近似
        first_chern_class = trace_F / (2 * math.pi)
        
        # 2. 第二陈类 c2
        F_squared = torch.bmm(curvature, curvature)
        trace_F_squared = torch.diagonal(F_squared, dim1=-2, dim2=-1).sum(dim=-1)  # tr(F²): [B * N]
        
        # c2 = (tr(F²) - tr(F)²) / (8π²)
        second_chern_class = (trace_F_squared - trace_F**2) / (8 * math.pi**2)
        
        # 3. 陈类比值 c2/c1 (放大效应)
        # 为避免除零和确保可导性，使用 c1 绝对值加上一个小的 epsilon
        chern_ratio = second_chern_class / (first_chern_class.abs() + 1e-8)
        
        return {
            'c1': first_chern_class,
            'c2': second_chern_class,
            'c2_div_c1': chern_ratio,
            'trace_F': trace_F,
        }

# =======================================================
# 验证示例 (在实际部署中可注释或删除)
# =======================================================
if __name__ == "__main__":
    D_MODEL = 16
    BATCH_SIZE = 4
    SEQ_LEN = 5
    
    # 实例化计算器
    calc = ChernClassCalculator(d_model=D_MODEL)
    
    # 模拟输入数据
    # 启用梯度，模拟在训练循环中
    test_input = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, requires_grad=True)
    
    # 1. 计算联络
    connection = calc.compute_connection_form(test_input)
    print(f"联络形式 A 形状: {connection.shape}") # 应为 [B*N, D, D] = [20, 16, 16]
    
    # 2. 计算曲率
    curvature = calc.compute_curvature_form(connection)
    print(f"曲率形式 F 形状: {curvature.shape}") # 应为 [20, 16, 16]
    
    # 3. 计算陈类
    chern_data = calc.compute_chern_classes(curvature)
    
    c1 = chern_data['c1']
    c2 = chern_data['c2']
    ratio = chern_data['c2_div_c1']
    
    print(f"\n计算结果:")
    print(f"  第一陈类 c1 (前5项): {c1[:5]}")
    print(f"  第二陈类 c2 (前5项): {c2[:5]}")
    print(f"  陈类比值 c2/c1 (前5项): {ratio[:5]}")
    print(f"\n梯度检查 (确保可导性):")
    
    # 4. 检查可导性 (必须可导才能用于训练)
    loss = c1.sum() + c2.sum()
    loss.backward()
    
    # 检查联络参数的梯度
    if calc.connection_form.grad is not None:
        print(f"  联络参数梯度范数: {calc.connection_form.grad.norm().item():.6f}")
    else:
        print("  警告: 联络参数梯度为 None")
    
    print("✅ topology_utils.py 模块构建完成，计算逻辑通过初步验证。")
#except ImportError:
 #   print("Warning: ChernClassCalculator not found. Using placeholder.")
  #  class ChernClassCalculator(nn.Module):
   #     def __init__(self, d_model: int):
    ##        super().__init__()
      #      self.connection_form = nn.Parameter(torch.zeros(d_model, d_model))
       #     self.curvature_weight = nn.Parameter(torch.zeros(d_model, d_model))
        #def compute_connection_form(self, x): return torch.zeros(x.shape[0]*x.shape[1], x.shape[2], x.shape[2])
        #def compute_curvature_form(self, conn): return conn 
        #def compute_chern_classes(self, curv): 
         #   B_N = curv.shape[0]
            # 模拟 c1=1.0, c2=0.5, ratio=0.5，用于占位和测试
          #  return {'c1': torch.ones(B_N), 'c2': torch.ones(B_N)*0.5, 'c2_div_c1': torch.ones(B_N)*0.5}


# -------------------------------------------------------
# 拓扑感知层 (TopologicalLayer)
# -------------------------------------------------------

class TopologicalLayer(nn.Module):
    """拓扑感知层 - 基于增强版陈类 (c1, c2, c2/c1) 动态调整"""
    
    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        
        # 标准 Transformer 组件
        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn_projection = nn.Linear(d_model, d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4), 
            nn.GELU(), 
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 拓扑几何工具
        self.chern_calculator = ChernClassCalculator(d_model)
        
        # 拓扑自适应参数 (可学习)
        self.topological_adaptation = nn.Parameter(torch.tensor(0.1))
        self.chern_threshold = nn.Parameter(torch.tensor(0.8)) # c1 触发阈值
        
        # 存储诊断特征和拓扑信息
        self.topology_info: Dict[str, float] = {}
        self.chern_ratio_features: Dict[str, float] = {}

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, N, D]
        
        # --- 1. 拓扑计算 (获取 c1, c2, c2/c1) ---
        connection = self.chern_calculator.compute_connection_form(x)
        curvature = self.chern_calculator.compute_curvature_form(connection)
        chern_data = self.chern_calculator.compute_chern_classes(curvature)
        
        c1 = chern_data['c1']
        c2 = chern_data['c2']
        ratio = chern_data['c2_div_c1']

        # --- 2. MHA 子层 ---
        residual = x
        attn_output, _ = self.multihead_attn(x, x, x, attn_mask=mask)
        attn_output = self.attn_projection(attn_output)
        x = self.norm1(residual + attn_output)
        
        # --- 3. FFN 子层 (带 c1 驱动的拓扑自适应) ---
        residual = x
        ff_output = self.feed_forward(x)
        
        # c1 驱动的残差调整逻辑 (L1 局部修正)
        c1 = chern_data['c1'] # 形状为 [B * N] 的张量

        num_elements = c1.numel()
        if num_elements > 0:
            # 高曲率时，削弱残差连接的权重
            first_chern_mean = torch.sum(torch.abs(c1)) / num_elements
        else:
            first_chern_mean = torch.tensor(0.0)
        
        
        
        residual_weight = 1.0

        if first_chern_mean > self.chern_threshold:
          esidual_weight = torch.sigmoid(1.0 - self.topological_adaptation * first_chern_mean)

        x = self.norm2(residual * residual_weight + ff_output)# 注意: 乘法应作用于 residual 项，但这里我们保持原样 * ff_output  
        x = self.norm2(residual + ff_output * residual_weight)
        # --- 4. 存储拓扑信息和诊断特征 (供 L2 诊断器提取) ---
        
        # 存储 L1 拓扑信息 (用于正则化和局部修正)
        self.topology_info = {
            'first_chern_class_mean': first_chern_mean.item(),
            'second_chern_class_mean': torch.mean(torch.abs(c2)).item(),
            'chern_ratio_mean': torch.mean(torch.abs(ratio)).item(),
            'chern_threshold': self.chern_threshold.item()
        }
        
        # 存储 L2 诊断特征 (c2/c1 分类器所需)
        # 必须保证特征名称和顺序与 diagnoser.py 中的约定一致！
        self.chern_ratio_features = {
             'c1_mean': first_chern_mean.item(),
             'c2_mean': self.topology_info['second_chern_class_mean'],
             'ratio_mean': self.topology_info['chern_ratio_mean'],
             'c1_std': torch.std(c1.abs()).item(),
             'ratio_std': torch.std(ratio.abs()).item(),
             # **注意:** 如果 diagnoser 提取了更多特征，请在此处添加
        }
        
        return x

# -------------------------------------------------------
# 拓扑感知 Transformer (TopologyAwareTransformer)
# -------------------------------------------------------

class TopologyAwareTransformer(nn.Module):
    """整个 L1 学习系统 - 由多个拓扑感知层构成"""
    def __init__(self, num_layers: int, vocab_size: int, d_model: int, n_heads: int = 8):
        super().__init__()
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 使用自定义的 TopologicalLayer
        self.layers = nn.ModuleList([
            TopologicalLayer(d_model, n_heads) for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # input_ids: [B, N] (Token IDs)
        x = self.embedding(input_ids) # x: [B, N, D]
        
        for layer in self.layers:
            x = layer(x, mask=mask)
        
        x = self.output_norm(x)
        logits = self.output_proj(x) # [B, N, Vocab_size]
        return logits
    
    def collect_topo_features(self) -> np.ndarray:
        """
        L2 诊断器的接口：收集所有层级的 c2/c1 诊断特征。
        必须保证输出的特征向量顺序和结构与 diagnoser.py 中期望的输入完全一致。
        """
        all_features = []
        feature_order = ['c1_mean', 'c2_mean', 'ratio_mean', 'c1_std', 'ratio_std']
        
        for layer in self.layers:
            if hasattr(layer, 'chern_ratio_features'):
                # 按照固定的 feature_order 提取并展平
                features_for_layer = [layer.chern_ratio_features[name] for name in feature_order]
                all_features.extend(features_for_layer)
            else:
                 # 如果某些层尚未运行 (仅在调试时出现)，则填充零
                all_features.extend([0.0] * len(feature_order)) 
        
        return np.array(all_features).reshape(1, -1) # 确保输出形状为 (1, N_features)

    def get_current_topo_info(self):
      """返回第一层的拓扑信息字典，供 Corrector 使用。"""
      # 假设拓扑信息存储在第一个 ToplogicalLayer 的 topology_info 属性中
      if hasattr(self.layers[0], 'topology_info'):
        return self.layers[0].topology_info
      return {'first_chern_class_mean': 0.0} # 返回一个默认的安全字典

# =======================================================
# 验证代码
# =======================================================
if __name__ == "__main__":
    VOCAB_SIZE = 100
    D_MODEL = 64
    NUM_LAYERS = 4
    
    model = TopologyAwareTransformer(NUM_LAYERS, VOCAB_SIZE, D_MODEL)
    
    # 模拟输入序列
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 20)) # Batch 2, Seq Len 20
    
    # 前向传播
    output = model(input_ids)
    print(f"模型输出 Logits 形状: {output.shape}") # 应为 [2, 20, 100]
    
    # 收集诊断特征
    topo_features = model.collect_topo_features()
    
    expected_feature_len = NUM_LAYERS * 5 # 4层 * 5个特征
    
    print(f"\n收集到的拓扑诊断特征 (用于 L2 诊断器):")
    print(f"  形状: {topo_features.shape}") # 应为 (1, 4 * 5 = 20)
    print(f"  前5个特征值: {topo_features[0, :5]}")
    
    # 检查拓扑信息是否已存储在第一层
    print(f"\nLayer 0 拓扑信息:")
    print(f"  c1 Mean: {model.layers[0].topology_info['first_chern_class_mean']:.4f}")
    print(f"  c2/c1 Mean: {model.layers[0].topology_info['chern_ratio_mean']:.4f}")
    
    print("\n✅ transformer.py 模块构建完成。")
