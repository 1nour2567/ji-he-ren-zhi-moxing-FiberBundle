# =======================================================
# main_trainer.py
# 核心集成与训练模块：实现元学习反馈回路
# 职责: 预训练诊断器，初始化 L1/L2 模型，并执行带有修正逻辑的主训练。
# =======================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

# --- 导入所有模块 (假设它们位于同一目录下或已正确配置Python路径) ---
# 注意: 在实际运行前，请确保这些导入路径是正确的
 # (Module I)


# -------------------------------------------------------
# 辅助函数: 模拟数据加载器
# -------------------------------------------------------
def get_mock_data_loader(vocab_size: int, seq_len: int, batch_size: int, num_batches: int):
    """生成一个模拟序列数据的迭代器"""
    for _ in range(num_batches):
        # 模拟输入序列 (B, N)
        inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
        # 模拟目标序列 (下一个词预测)
        targets = torch.roll(inputs, shifts=-1, dims=1)
        yield inputs, targets

# -------------------------------------------------------
# 主训练函数: 带有元学习指导的训练
# -------------------------------------------------------

def train_with_meta_guidance(
    model: TopologyAwareTransformer,
    diagnoser: ChernRatioClassifier,
    corrector: TopologicalCorrector,
    # 💥 修改: 不再传入 data_loader，而是传入创建 data_loader 所需的参数
    get_data_loader_func, # 新参数: 传入创建 data_loader 的函数 (即 get_mock_data_loader)
    data_loader_params: dict, # 新参数: 传入创建 data_loader 所需的参数
    epochs: int,
    meta_check_freq: int = 50
):
    """
    带有陈类诊断的元学习训练循环。

    :param model: L1 学习模型
    :param diagnoser: L2 诊断器
    :param corrector: L2 修正器
    :param data_loader: 训练数据加载器
    :param epochs: 训练轮数
    :param meta_check_freq: 运行 L2 诊断的频率 (每隔多少个 batch)
    """
    optimizer = corrector.optimizer
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    C_TARGET = 25.039  # 我们希望 E[c1] 稳定在这个值附近

    for epoch in range(epochs):
      # 重新创建数据加载器
      data_loader = get_data_loader_func(**data_loader_params)
      num_batches = data_loader_params.get('num_batches')
      if num_batches is None:
        raise ValueError("data_loader_params 必须包含 'num_batches' 键!")



    print("\n===========================================================")
    print(f"🚀 开始元学习训练 (Device: {device})")
    print(f"   L2 诊断频率: 每 {meta_check_freq} 步检查一次")
    print("===========================================================")

    for epoch in range(1, epochs + 1):
        total_loss_epoch = 0
        total_task_loss = 0

        # 💥 修正点：在每个 Epoch 开始时重新创建数据加载器
        data_loader = get_data_loader_func(**data_loader_params)

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # --- L1: 主学习任务与拓扑正则化 ---
            optimizer.zero_grad()
            outputs = model(inputs) # 前向传播触发拓扑特征计算

            # 1. 任务损失 (如语言建模)
            loss_task = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

            # 2. 拓扑正则化损失 (基于 c1 均值)
            C_TARGET = 25.039
            # 收集所有 Layer 的 c1 均值
            c1_means = [layer.topology_info['first_chern_class_mean'] for layer in model.layers]
            c1_tensor = torch.tensor(c1_means, device=device) # 假设 device 变量可用

            # 目标导向损失: L_topo = sum((E[c1] - C_TARGET)^2)
            deviation_loss = torch.sum((c1_tensor - C_TARGET)**2)

            #{topo_loss_c1 = 0.0
            #for layer in model.layers:
                # 拓扑损失: 最小化 c1 的绝对值均值
                #topo_loss_c1 += torch.tensor(layer.topology_info['first_chern_class_mean']).abs()}

            # 3. 总损失 = 任务损失 + λ * 拓扑损失
            total_loss = loss_task + corrector.current_lambda * deviation_loss

            total_loss.backward()
            optimizer.step()

            total_loss_epoch += total_loss.item()
            total_task_loss += loss_task.item()

            # --- L2: 元学习诊断与修正 ---
            if (batch_idx + 1) % meta_check_freq == 0:

                print(f"\n[Meta-Check] Epoch {epoch}/{epochs}, Step {batch_idx + 1}")
                print(f"  当前 L1 任务损失: {loss_task.item():.4f}")
                print(f"  当前 λ: {corrector.current_lambda:.6f}")

                # 1. 提取 L1 特征
                # 特征是 NumPy 数组，形状为 (1, N_features)
                topo_features = model.collect_topo_features()
                SCALE_CORRECTION_FACTOR = 1.0 / 21.5
                for i in range(topo_features.shape[1]):
                  # 只修正均值和标准差特征 (索引 0, 1, 3, 4)
                  if i % 5 != 2:
                     topo_features[0, i] *= SCALE_CORRECTION_FACTOR# 忽略 c2/c1 比值 (索引 2)



                # 2. L2 诊断
                predicted_states, _ = diagnoser.predict(topo_features)
                predicted_state = predicted_states[0]

                print(f"  🧠 诊断器预测状态: {predicted_state} (0:正常, 1:异常, 2:约束)")

                # 3. 提取 L1 拓扑信息 (包含 c1 mean, c2/c1 mean 等)
                # 假设您的 transformer 模块有一个 get_topo_info 方法来返回 self.topology_info
                current_topo_info = model.get_current_topo_info()
                corrector.execute_correction(predicted_state, current_topo_info)



            # 简单的进度打印
            if (batch_idx + 1) % 10 == 0:
                print(f"  [Epoch {epoch}] Step {batch_idx + 1} | Loss: {total_loss.item():.4f}", end='\r')

        avg_loss = total_loss_epoch / (batch_idx + 1)
        avg_task_loss = total_task_loss / (batch_idx + 1)

        print(f"\n\n--- Epoch {epoch} 总结 ---")
        print(f"  平均总损失: {avg_loss:.4f}")
        print(f"  平均任务损失: {avg_task_loss:.4f}")
        print(f"  平均第一陈类 (最后): {model.layers[0].topology_info.get('first_chern_class_mean', 0):.6f}")


# -------------------------------------------------------
# 主运行逻辑
# -------------------------------------------------------
if __name__ == "__main__":
    # --- 配置参数 ---
    VOCAB_SIZE = 5000
    D_MODEL = 128
    NUM_LAYERS = 6
    BATCH_SIZE = 32
    SEQ_LEN = 50
    NUM_BATCHES = 200 # 模拟总训练步数
    EPOCHS = 3
    INITIAL_LAMBDA = 0.005
    META_CHECK_FREQ = 20 # 每 20 步进行一次元学习诊断

    # ----------------------------------------
    # I. 预训练 L2 诊断器 (ChernRatioClassifier)
    # ----------------------------------------
    start_time = time.time()
    # 诊断器只需要一个较小的 d_model 即可训练其分类逻辑
    pretrained_diagnoser = setup_and_train_diagnoser(d_model=32)
    print(f"\n>>> 诊断器预训练耗时: {time.time() - start_time:.2f} 秒 <<<")

    # ----------------------------------------
    # II. 初始化 L1 模型和修正器
    # ----------------------------------------
    model = TopologyAwareTransformer(NUM_LAYERS, VOCAB_SIZE, D_MODEL)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n初始化 L1 模型: TopologyAwareTransformer")
    print(f"  模型参数总数: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    corrector = TopologicalCorrector(model, optimizer, initial_lambda=INITIAL_LAMBDA)

    # ----------------------------------------
    # III. 启动元学习训练
    # ----------------------------------------

    # 模拟数据加载器
    #data_loader = get_mock_data_loader(VOCAB_SIZE, SEQ_LEN, BATCH_SIZE, NUM_BATCHES)
    loader_params = {
        'vocab_size': VOCAB_SIZE,
        'seq_len': SEQ_LEN,
        'batch_size': BATCH_SIZE,
        'num_batches': NUM_BATCHES
    }

    # 开始训练
    train_with_meta_guidance(
        model,
        pretrained_diagnoser,
        corrector,
        get_mock_data_loader,
        loader_params,
        epochs=EPOCHS,
        meta_check_freq=META_CHECK_FREQ
    )

    print("\n===========================================================")
    print("✨ 拓扑元学习系统训练完成。")
    print("===========================================================")

    #import torch # 确保 torch 已经被导入

    print("\n## 🚀 最终拓扑特征分析 (所有 Layer)")

    # 确保模型处于 CPU/评估模式 (尽管这里只读取信息)
    model.eval()

    for i, layer in enumerate(model.layers):
        # 假设 model.layers[i] 是您的 TopologyAwareTransformerLayer 实例
        # 并且 info 字典被正确存储在 layer.topology_info 中
        info = layer.topology_info

        # 使用 .get() 确保键不存在时程序不会崩溃
        c1_mean = info.get('first_chern_class_mean', float('nan'))
        c2_c1_mean = info.get('chern_ratio_mean', float('nan'))
        c1_std = info.get('first_chern_class_std', float('nan'))

        # 如果 mean/std 是 torch.Tensor，需要转换为 float
        if isinstance(c1_mean, torch.Tensor):
            c1_mean = c1_mean.item()
        if isinstance(c2_c1_mean, torch.Tensor):
            c2_c1_mean = c2_c1_mean.item()
        if isinstance(c1_std, torch.Tensor):
            c1_std = c1_std.item()

        print(f"Layer {i}:")
        print(f"  c1 Mean: {c1_mean:.4f}")
        print(f"  c2/c1 Mean: {c2_c1_mean:.4f}")
        print(f"  c1 Std: {c1_std:.4f}")
    # =======================================================
