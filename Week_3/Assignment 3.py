import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 定義 Runge 函數及其導數
def runge_function(x):
    return 1 / (1 + 25 * x**2)

def runge_derivative(x):
    return -50 * x / (1 + 25 * x**2)**2

# 生成數據
def generate_data(num_points=1000, seed=42):
    np.random.seed(seed)
    x_train = np.random.uniform(-1, 1, num_points)
    y_train_func = runge_function(x_train)
    y_train_deriv = runge_derivative(x_train)
    
    # 分割訓練/驗證集 (80/20)
    split_idx = int(0.8 * num_points)
    x_train_split, x_val_split = x_train[:split_idx], x_train[split_idx:]
    y_train_func_split, y_val_func_split = y_train_func[:split_idx], y_train_func[split_idx:]
    y_train_deriv_split, y_val_deriv_split = y_train_deriv[:split_idx], y_train_deriv[split_idx:]
    
    # 轉換為 PyTorch 張量
    x_train_tensor = torch.tensor(x_train_split.reshape(-1, 1), dtype=torch.float32)
    y_train_tensor = torch.tensor(np.stack([y_train_func_split, y_train_deriv_split], axis=1), dtype=torch.float32)
    x_val_tensor = torch.tensor(x_val_split.reshape(-1, 1), dtype=torch.float32)
    y_val_tensor = torch.tensor(np.stack([y_val_func_split, y_val_deriv_split], axis=1), dtype=torch.float32)
    
    # 生成測試數據
    x_test = np.linspace(-1, 1, 1000)
    y_test_func_true = runge_function(x_test)
    y_test_deriv_true = runge_derivative(x_test)
    x_test_tensor = torch.tensor(x_test.reshape(-1, 1), dtype=torch.float32)
    
    return (x_train_tensor, y_train_tensor, x_val_tensor, y_val_tensor,
            x_test, y_test_func_true, y_test_deriv_true, x_test_tensor)

# 定義神經網絡 (輸出 2 維：f(x) 和 f'(x))
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(1, 20)  # 輸入到隱藏層1
        self.fc2 = nn.Linear(20, 20) # 隱藏層1到隱藏層2
        self.fc3 = nn.Linear(20, 2)  # 隱藏層2到輸出 (2 維：f(x) 和 f'(x))

    def forward(self, x):
        x = torch.tanh(self.fc1(x))  # tanh 激活
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)  # 輸出層直接預測 f(x) 和 f'(x)
        return x

# 訓練函數
def train_model(model, x_train, y_train, x_val, y_val, num_epochs=1000, lr=0.01, w1=0.5, w2=0.5):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss_func = criterion(outputs[:, 0].unsqueeze(1), y_train[:, 0].unsqueeze(1))  # 函數損失
        loss_deriv = criterion(outputs[:, 1].unsqueeze(1), y_train[:, 1].unsqueeze(1))  # 導數損失
        total_loss = w1 * loss_func + w2 * loss_deriv  # 總損失
        total_loss.backward()
        optimizer.step()
        
        # 驗證
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val)
            val_loss_func = criterion(val_outputs[:, 0].unsqueeze(1), y_val[:, 0].unsqueeze(1))
            val_loss_deriv = criterion(val_outputs[:, 1].unsqueeze(1), y_val[:, 1].unsqueeze(1))
            val_total_loss = w1 * val_loss_func + w2 * val_loss_deriv
        
        train_losses.append(total_loss.item())
        val_losses.append(val_total_loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {total_loss.item():.6f}, '
                  f'Val Loss: {val_total_loss.item():.6f}, Func Loss: {loss_func.item():.6f}, '
                  f'Deriv Loss: {loss_deriv.item():.6f}')
    
    return train_losses, val_losses

# 主程式
if __name__ == "__main__":
    # 生成數據
    (x_train, y_train, x_val, y_val,
     x_test, y_test_func_true, y_test_deriv_true, x_test_tensor) = generate_data()
    
    # 初始化並訓練模型
    model = NeuralNet()
    train_losses, val_losses = train_model(model, x_train, y_train, x_val, y_val, w1=0.5, w2=0.5)
    
    # 評估
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(x_test_tensor)
        y_pred_func = y_pred_tensor[:, 0].numpy()  # 函數預測
        y_pred_deriv = y_pred_tensor[:, 1].numpy()  # 導數預測
    
    # 計算錯誤
    mse_func = np.mean((y_test_func_true - y_pred_func)**2)
    max_error_func = np.max(np.abs(y_test_func_true - y_pred_func))
    mse_deriv = np.mean((y_test_deriv_true - y_pred_deriv)**2)
    max_error_deriv = np.max(np.abs(y_test_deriv_true - y_pred_deriv))
    
    print(f'Function MSE: {mse_func:.6f}')
    print(f'Function Max Error: {max_error_func:.6f}')
    print(f'Derivative MSE: {mse_deriv:.6f}')
    print(f'Derivative Max Error: {max_error_deriv:.6f}')
    
    # 繪圖 1: 真實函數 vs 預測
    plt.figure(figsize=(10, 5))
    plt.plot(x_test, y_test_func_true, label='True Runge Function', color='blue')
    plt.plot(x_test, y_pred_func, label='Neural Network Prediction', color='red', linestyle='--')
    plt.title('Runge Function Approximation')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.savefig('runge_approx.png')
    plt.show()
    
    # 繪圖 2: 真實導數 vs 預測導數
    plt.figure(figsize=(10, 5))
    plt.plot(x_test, y_test_deriv_true, label='True Derivative', color='blue')
    plt.plot(x_test, y_pred_deriv, label='Neural Network Derivative', color='red', linestyle='--')
    plt.title('Runge Derivative Approximation')
    plt.xlabel('x')
    plt.ylabel("f'(x)")
    plt.legend()
    plt.grid(True)
    plt.savefig('runge_deriv_approx.png')
    plt.show()
    
    # 繪圖 3: 訓練/驗證損失曲線
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curves.png')
    plt.show()