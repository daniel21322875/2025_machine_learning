import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 定義 Runge 函數
def runge_function(x):
    return 1 / (1 + 25 * x**2)

# 生成數據
np.random.seed(42)  # 固定隨機種子
num_points = 1000
x_train = np.random.uniform(-1, 1, num_points)
y_train = runge_function(x_train)

# 分割訓練/驗證集 (80/20)
split_idx = int(0.8 * num_points)
x_train_split, x_val_split = x_train[:split_idx], x_train[split_idx:]
y_train_split, y_val_split = y_train[:split_idx], y_train[split_idx:]

# 轉換為 PyTorch 張量
x_train_tensor = torch.tensor(x_train_split.reshape(-1, 1), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_split.reshape(-1, 1), dtype=torch.float32)
x_val_tensor = torch.tensor(x_val_split.reshape(-1, 1), dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_split.reshape(-1, 1), dtype=torch.float32)

# 生成測試數據（密集點用於繪圖和評估）
x_test = np.linspace(-1, 1, 1000)
y_test_true = runge_function(x_test)
x_test_tensor = torch.tensor(x_test.reshape(-1, 1), dtype=torch.float32)

# 定義神經網絡 (MLP: 1 輸入, 2 隱藏層 (各 20 神經元), 1 輸出)
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(1, 20)  # 輸入到隱藏層1
        self.fc2 = nn.Linear(20, 20)  # 隱藏層1到隱藏層2
        self.fc3 = nn.Linear(20, 1)   # 隱藏層2到輸出

    def forward(self, x):
        x = torch.tanh(self.fc1(x))  # tanh 激活
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)  # 輸出層線性
        return x

# 初始化模型、損失、優化器
model = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 訓練
num_epochs = 1000
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    # 驗證
    model.eval()
    with torch.no_grad():
        val_outputs = model(x_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
    
    train_losses.append(loss.item())
    val_losses.append(val_loss.item())
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}')

# 評估
model.eval()
with torch.no_grad():
    y_pred_tensor = model(x_test_tensor)
    y_pred = y_pred_tensor.numpy().flatten()

# 計算錯誤
mse = np.mean((y_test_true - y_pred)**2)
max_error = np.max(np.abs(y_test_true - y_pred))
print(f'MSE: {mse:.6f}')
print(f'Max Error: {max_error:.6f}')

# 繪圖 1: 真實函數 vs 預測
plt.figure(figsize=(10, 5))
plt.plot(x_test, y_test_true, label='True Runge Function', color='blue')
plt.plot(x_test, y_pred, label='Neural Network Prediction', color='red', linestyle='--')
plt.title('Runge Function Approximation')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.savefig('runge_approx.png')  # 保存圖片用於報告
plt.show()

# 繪圖 2: 訓練/驗證損失曲線
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_curves.png')  # 保存圖片用於報告
plt.show()