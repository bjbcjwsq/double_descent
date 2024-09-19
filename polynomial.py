import numpy as np
import matplotlib.pyplot as plt

# 定义生成数据的函数
def generate_data(n_points, noise_level=0.1):
    x = np.linspace(1, 11, n_points)
    y = 3*x**3 - 5*x**2 + 2*x + 10
    return x, y

# 多项式函数
def poly(x, coeffs):
    return sum(c * x**i for i, c in enumerate(coeffs))

def grad(x, coeffs):
    return np.array([i * c * x**(i-1) for i, c in enumerate(coeffs)])

# 计算损失（均方误差）
def compute_loss(coeffs, x, y):
    predictions = poly(x, coeffs)
    return np.mean((predictions - y)**2)

# 梯度下降函数
def gradient_descent(x, y, degree, learning_rate, epochs):
    # 初始化系数
    coeffs = np.ones(degree + 1)
    print("initial loss:{}".format(compute_loss(coeffs, x, y)))
    
    for epoch in range(epochs):
        # 计算预测值
        predictions = poly(x, coeffs)
        
        # 计算目标函数对每个分量的梯度,用grad()函数计算
        gradients = np.mean(2 * (predictions - y) * grad(x, coeffs), axis=1)
        
        # 更新系数
        coeffs -= learning_rate * 2 * gradients
        
        # 打印损失
        if epoch % 1 == 0:
            loss = compute_loss(coeffs, x, y)
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss:.4f}')
    
    return coeffs

# 主程序
if __name__ == "__main__":
    # 用户指定多项式的次数
    degree = 3
    
    # 学习率和迭代次数
    learning_rate = 0.001
    epochs = 10
    
    # 生成数据点
    x_data, y_data = generate_data(5)
    
    # 进行梯度下降拟合
    fitted_coeffs = gradient_descent(x_data, y_data, degree, learning_rate, epochs)
    
    # # 使用拟合的系数绘制多项式曲线
    # x_fit = np.linspace(min(x_data), max(x_data), 1000)
    # y_fit = poly(x_fit, fitted_coeffs)
    
    # # 绘制原始数据点
    # plt.scatter(x_data, y_data, label='Data Points', color='blue')
    
    # # 绘制拟合曲线
    # plt.plot(x_fit, y_fit, label=f'Polynomial Fit (Degree {degree})', color='red')
    
    # # 添加标题和图例
    # plt.title(f'Polynomial Fitting of Degree {degree} Using Gradient Descent')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    # # 输出拟合得到的系数
    # print('Fitted coefficients:', fitted_coeffs)