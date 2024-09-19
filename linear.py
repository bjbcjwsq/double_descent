import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 设置随机种子以保证结果的可复现性
np.random.seed(0)
torch.manual_seed(0)

# 定义数据集大小和参数
num_samples = 1000
input_dim = 1  # 输入维度
output_dim = 1  # 输出维度
learning_rate = 0.01
epochs = 100000

# 创建一些线性数据加上一点噪声
X = np.random.rand(num_samples, input_dim) * 10
# Y = 3 * X + 2 + np.random.randn(num_samples, output_dim) * 2
Y = 3 * X + 2 

# 转换为Tensor
X_tensor = torch.from_numpy(X).float()
Y_tensor = torch.from_numpy(Y).float()

# 初始化线性模型
model = nn.Linear(input_dim, output_dim)

# 定义损失函数
criterion = nn.MSELoss()

# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(epochs):
    # 前向传播
    outputs = model(X_tensor)
    
    # 计算损失
    loss = criterion(outputs, Y_tensor)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    
    # 更新权重
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 打印最终学习到的权重和偏置
print('Learned weights:', list(model.parameters())[0].item())
print('Learned bias:', list(model.parameters())[1].item())