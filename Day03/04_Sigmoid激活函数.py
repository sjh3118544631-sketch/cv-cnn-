import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# 设置中文字体和随机种子
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
torch.manual_seed(42)


class SigmoidAnalysis:
    """Sigmoid函数分析类"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

    def visualize_sigmoid(self):
        """可视化Sigmoid函数及其导数"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 生成数据
        x = torch.linspace(-10, 10, 1000)
        sigmoid = torch.sigmoid(x)
        sigmoid_manual = 1 / (1 + torch.exp(-x))

        # 1. Sigmoid函数图像
        axes[0, 0].plot(x.numpy(), sigmoid.numpy(), 'b-', linewidth=2)
        axes[0, 0].set_title('Sigmoid函数图像')
        axes[0, 0].set_xlabel('输入值')
        axes[0, 0].set_ylabel('输出值')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5)

        # 2. 手动实现与PyTorch实现对比
        axes[0, 1].plot(x.numpy(), sigmoid.numpy(), 'b-', label='torch.sigmoid', linewidth=2)
        axes[0, 1].plot(x.numpy(), sigmoid_manual.numpy(), 'r--', label='手动实现', linewidth=2, alpha=0.7)
        axes[0, 1].set_title('PyTorch实现 vs 手动实现')
        axes[0, 1].set_xlabel('输入值')
        axes[0, 1].set_ylabel('输出值')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Sigmoid导数
        # 导数公式: σ'(x) = σ(x) * (1 - σ(x))
        sigmoid_derivative = sigmoid * (1 - sigmoid)
        axes[1, 0].plot(x.numpy(), sigmoid_derivative.numpy(), 'g-', linewidth=2)
        axes[1, 0].set_title('Sigmoid导数图像')
        axes[1, 0].set_xlabel('输入值')
        axes[1, 0].set_ylabel('导数值')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 不同输入范围的输出分布
        small_x = torch.linspace(-2, 2, 100)
        large_x = torch.linspace(-10, 10, 100)

        axes[1, 1].plot(small_x.numpy(), torch.sigmoid(small_x).numpy(),
                        'b-', label='小范围输入(-2,2)', linewidth=2)
        axes[1, 1].plot(large_x.numpy(), torch.sigmoid(large_x).numpy(),
                        'r--', label='大范围输入(-10,10)', linewidth=2, alpha=0.7)
        axes[1, 1].set_title('不同输入范围的输出对比')
        axes[1, 1].set_xlabel('输入值')
        axes[1, 1].set_ylabel('输出值')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def gradient_vanish_demo(self):
        """梯度消失演示"""
        print("\n" + "=" * 50)
        print("梯度消失问题演示")
        print("=" * 50)

        # 创建一个简单的多层网络
        class DeepNet(nn.Module):
            def __init__(self, num_layers=10):
                super().__init__()
                self.layers = nn.ModuleList()
                for i in range(num_layers):
                    self.layers.append(nn.Linear(10, 10))
                    self.layers.append(nn.Sigmoid())
                self.output = nn.Linear(10, 1)

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return self.output(x)

        # 创建网络实例
        model = DeepNet(num_layers=10).to(self.device)

        # 模拟输入
        x = torch.randn(1, 10, requires_grad=True).to(self.device)
        x.retain_grad()
        target = torch.tensor([[1.0]]).to(self.device)

        # 前向传播
        output = model(x)
        loss = F.mse_loss(output, target)

        # 反向传播
        loss.backward()

        # 检查梯度
        print(f"\n输入 x 的梯度范数: {x.grad.norm().item():.6f}")

        # 检查各层梯度
        gradients = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradients.append(grad_norm)
                print(f"{name} 梯度范数: {grad_norm:.6f}")

        # 可视化梯度变化
        plt.figure(figsize=(10, 4))
        plt.plot(range(len(gradients)), gradients, 'bo-', linewidth=2)
        plt.title('各层梯度范数变化（梯度消失现象）')
        plt.xlabel('网络层深度')
        plt.ylabel('梯度范数')
        plt.grid(True, alpha=0.3)
        plt.show()

    def binary_classification_example(self):
        """二分类问题示例"""
        print("\n" + "=" * 50)
        print("二分类问题实战")
        print("=" * 50)

        # 创建二分类数据集
        np.random.seed(42)
        n_samples = 500

        # 生成两个类别的数据
        class0_x = np.random.randn(n_samples // 2, 2) * 0.5 + np.array([-2, -2])
        class0_y = np.zeros(n_samples // 2)

        class1_x = np.random.randn(n_samples // 2, 2) * 0.5 + np.array([2, 2])
        class1_y = np.ones(n_samples // 2)

        X = np.vstack([class0_x, class1_x])
        y = np.hstack([class0_y, class1_y])

        # 转换为Tensor
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1)

        # 创建数据集和数据加载器
        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # 定义使用Sigmoid的模型
        class SigmoidClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(2, 16)
                self.fc2 = nn.Linear(16, 8)
                self.fc3 = nn.Linear(8, 1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.fc3(x)  # 最后一层不激活
                return x

            def predict_proba(self, x):
                """预测概率"""
                with torch.no_grad():
                    logits = self.forward(x)
                    return self.sigmoid(logits)

            def predict(self, x, threshold=0.5):
                """预测类别"""
                proba = self.predict_proba(x)
                return (proba > threshold).float()

        # 训练模型
        model = SigmoidClassifier()
        criterion = nn.BCEWithLogitsLoss()  # 结合Sigmoid和二分类交叉熵
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # 训练循环
        n_epochs = 100
        losses = []

        for epoch in range(n_epochs):
            epoch_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            losses.append(epoch_loss / len(train_loader))
            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')

        # 可视化训练过程
        plt.figure(figsize=(12, 4))

        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(losses, 'b-', linewidth=2)
        plt.title('训练损失曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)

        # 决策边界
        plt.subplot(1, 2, 2)
        xx, yy = np.meshgrid(np.linspace(-5, 5, 100),
                             np.linspace(-5, 5, 100))
        grid_points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

        with torch.no_grad():
            probs = torch.sigmoid(model(grid_points)).numpy()

        probs = probs.reshape(xx.shape)
        plt.contourf(xx, yy, probs, levels=20, cmap='RdBu_r', alpha=0.8)
        plt.colorbar(label='预测概率')

        # 绘制数据点
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y,
                              cmap='bwr', edgecolors='k', s=30)
        plt.title('Sigmoid分类器决策边界')
        plt.xlabel('特征1')
        plt.ylabel('特征2')
        plt.legend(*scatter.legend_elements(), title='类别')

        plt.tight_layout()
        plt.show()

        # 测试模型
        test_points = torch.FloatTensor([[-2, -2], [2, 2], [0, 0]])
        with torch.no_grad():
            probabilities = model.predict_proba(test_points)
            predictions = model.predict(test_points)

        for i, point in enumerate(test_points):
            print(f"点 {point.numpy()} -> "
                  f"概率: {probabilities[i].item():.4f}, "
                  f"预测: {predictions[i].item():.0f}")

    def compare_with_relu(self):
        """与ReLU对比"""
        print("\n" + "=" * 50)
        print("Sigmoid vs ReLU 对比")
        print("=" * 50)

        # 对比函数图像
        x = torch.linspace(-5, 5, 1000)
        sigmoid = torch.sigmoid(x)
        relu = F.relu(x)
        leaky_relu = F.leaky_relu(x, negative_slope=0.1)

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(x.numpy(), sigmoid.numpy(), 'b-', linewidth=2)
        plt.title('Sigmoid激活函数')
        plt.xlabel('输入')
        plt.ylabel('输出')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 2)
        plt.plot(x.numpy(), relu.numpy(), 'r-', linewidth=2)
        plt.title('ReLU激活函数')
        plt.xlabel('输入')
        plt.ylabel('输出')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 3)
        plt.plot(x.numpy(), leaky_relu.numpy(), 'g-', linewidth=2)
        plt.title('Leaky ReLU激活函数')
        plt.xlabel('输入')
        plt.ylabel('输出')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 优缺点对比
        print("\nSigmoid优缺点总结:")
        print("✓ 优点:")
        print("  - 输出范围(0,1)，适合表示概率")
        print("  - 平滑可微，有利于梯度计算")
        print("  - 单调函数，保证误差曲面有单峰性")

        print("\n✗ 缺点:")
        print("  - 梯度消失问题（输入绝对值大时梯度接近0）")
        print("  - 输出不是零中心的（zero-centered）")
        print("  - 计算涉及指数运算，较慢")
        print("  - 容易使神经元饱和（输出接近0或1）")

        print("\n适用场景:")
        print("  - 二分类问题的输出层")
        print("  - 需要输出概率的场景")
        print("  - 早期的神经网络（现在隐藏层多用ReLU）")


def main():
    """主函数"""
    analyzer = SigmoidAnalysis()

    # 1. 可视化Sigmoid函数
    print("1. Sigmoid函数可视化")
    analyzer.visualize_sigmoid()

    # 2. 梯度消失演示
    analyzer.gradient_vanish_demo()

    # 3. 二分类实战
    analyzer.binary_classification_example()

    # 4. 与ReLU对比
    analyzer.compare_with_relu()


if __name__ == "__main__":
    main()