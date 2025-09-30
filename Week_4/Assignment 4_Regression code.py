import numpy as np
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 解析 XML 檔案，僅取有溫度的點，回傳 X (經緯度) 及 y (溫度)
def parse_xml_temp(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        ns = {'ns': 'urn:cwa:gov:tw:cwacommon:0.1'}
        lon_start = float(root.find('.//ns:BottomLeftLongitude', ns).text)
        lat_start = float(root.find('.//ns:BottomLeftLatitude', ns).text)
        lon_step = 0.03
        lat_step = 0.03
        content = root.find('.//ns:Content', ns)
        if content is not None:
            temp_data = content.text.strip().split('\n')
            temp_grid = [row.strip().split(',') for row in temp_data if row.strip()]
        else:
            raise ValueError("無法找到 <content> 標籤")
        X = []
        y = []
        for i, row in enumerate(temp_grid):
            lat = lat_start + i * lat_step
            for j, temp_str in enumerate(row):
                lon = lon_start + j * lon_step
                temp = float(temp_str)
                if temp != -999.0:
                    X.append([lon, lat])
                    y.append(temp)
        return np.array(X), np.array(y)
    except Exception as e:
        print(f"解析錯誤: {e}")
        raise

if __name__ == "__main__":
    # 讀取 XML 並僅取有溫度的點
    file_path = 'O-A0038-003.xml'
    X, y = parse_xml_temp(file_path)

    # 分割資料集為訓練集、驗證集與測試集（6:2:2）
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # 建立並訓練隨機森林回歸器
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, y_train)

    # 驗證集預測與 MSE
    y_val_pred = reg.predict(X_val)
    mse_val = mean_squared_error(y_val, y_val_pred)
    print(f"驗證集 MSE: {mse_val:.4f}")

    # 測試集預測與 MSE
    y_test_pred = reg.predict(X_test)
    mse_test = mean_squared_error(y_test, y_test_pred)
    print(f"測試集 MSE: {mse_test:.4f}")

    # 視覺化預測結果
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.title('Validation: True vs Pred')
    plt.scatter(y_val, y_val_pred, s=10, alpha=0.7)
    plt.xlabel('True Temp')
    plt.ylabel('Predicted Temp')
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    plt.subplot(1,2,2)
    plt.title('Test: True vs Pred')
    plt.scatter(y_test, y_test_pred, s=10, alpha=0.7)
    plt.xlabel('True Temp')
    plt.ylabel('Predicted Temp')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.tight_layout()
    plt.show()

    # 使用者輸入經緯度，輸出預測與實際溫度
    while True:
        try:
            user_input = input('請輸入經度,緯度（例如 120.5,23.5，或直接按 Enter 結束）：')
            if not user_input.strip():
                break
            lon_str, lat_str = user_input.split(',')
            lon = float(lon_str.strip())
            lat = float(lat_str.strip())
            pred_temp = reg.predict(np.array([[lon, lat]]))[0]
            # 查找最接近的實際點
            dists = np.linalg.norm(X - np.array([lon, lat]), axis=1)
            idx = np.argmin(dists)
            real_temp = y[idx]
            print(f'預測溫度: {pred_temp:.2f}，最接近資料點實際溫度: {real_temp:.2f} (距離: {dists[idx]:.4f})')
        except Exception as e:
            print('格式錯誤或發生例外，請重新輸入。')