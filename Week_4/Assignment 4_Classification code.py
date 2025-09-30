# 匯入必要的套件
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt  # 匯入繪圖套件

# 解析 XML 檔案，將氣象網格資料轉為經緯度與標籤陣列
# 回傳 X (經緯度座標) 及 y (是否有溫度資料的標籤)
def parse_xml(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        # 取得經緯度範圍與步進
        ns = {'ns': 'urn:cwa:gov:tw:cwacommon:0.1'}
        lon_start = float(root.find('.//ns:BottomLeftLongitude', ns).text)
        lat_start = float(root.find('.//ns:BottomLeftLatitude', ns).text)
        lon_end = float(root.find('.//ns:TopRightLongitude', ns).text)
        lat_end = float(root.find('.//ns:TopRightLatitude', ns).text)
        lon_step = 0.03
        lat_step = 0.03
        lon_points = int((lon_end - lon_start) / lon_step) + 1
        lat_points = int((lat_end - lat_start) / lat_step) + 1

        # 解析溫度網格資料
        content = root.find('.//ns:Content', ns)
        if content is not None:
            temp_data = content.text.strip().split('\n')
            temp_grid = [row.strip().split(',') for row in temp_data if row.strip()]
        else:
            raise ValueError("無法找到 <content> 標籤")

        X = []  # 經緯度座標
        y = []  # 標籤(1:有溫度, 0:缺值)
        for i, row in enumerate(temp_grid):
            lat = lat_start + i * lat_step
            for j, temp_str in enumerate(row):
                lon = lon_start + j * lon_step
                temp = float(temp_str)
                label = 1 if temp != -999.0 else 0
                X.append([lon, lat])
                y.append(label)
        return np.array(X), np.array(y)

    except AttributeError as e:
        print("錯誤：XML 標籤路徑不正確，請檢查 <BottomLeftLongitude>, <BottomLeftLatitude>, <TopRightLongitude>, <TopRightLatitude> 等標籤是否存在。")
        raise
    except ValueError as e:
        print(f"錯誤：{e}")
        raise
    except Exception as e:
        print(f"發生未知錯誤：{e}")
        raise

if __name__ == "__main__":
    # 讀取 XML 並轉為資料集
    file_path = 'O-A0038-003.xml'
    X, y = parse_xml(file_path)

    # 分割資料集為訓練集、驗證集與測試集（6:2:2）
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # 建立並訓練隨機森林分類器
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # 在驗證集上預測並計算準確率
    y_val_pred = clf.predict(X_val)
    acc_val = accuracy_score(y_val, y_val_pred)
    print(f"驗證集準確率: {acc_val:.4f}")

    # 在測試集上預測並計算準確率
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"測試集準確率: {acc:.4f}")

    # 產生隨機經緯度測試資料，並用模型預測
    import random
    num_test = 100
    lon_min, lon_max = X[:,0].min(), X[:,0].max()
    lat_min, lat_max = X[:,1].min(), X[:,1].max()
    X_random = np.array([[random.uniform(lon_min, lon_max), random.uniform(lat_min, lat_max)] for _ in range(num_test)])
    y_random_pred = clf.predict(X_random)
    # ...資料儲存已註解...

    # 以缺值定義邊界點，分析模型在邊界與非邊界的表現
    temp_grid = None
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            if '<Content' in line:
                content_lines = []
                for l in f:
                    if '</Content>' in l:
                        break
                    content_lines.append(l.strip())
                temp_grid = [row.split(',') for row in content_lines if row.strip()]
                break
    if temp_grid is None:
        raise ValueError('無法重建溫度網格')
    temp_grid = np.array(temp_grid, dtype=float)
    nrow, ncol = temp_grid.shape
    boundary_flags = np.zeros(len(X), dtype=bool)
    for idx, (lon, lat) in enumerate(X):
        i = round((lat - lat_min) / 0.03)
        j = round((lon - lon_min) / 0.03)
        if 0 <= i < nrow and 0 <= j < ncol:
            if temp_grid[i, j] != -999.0:
                neighbors = []
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < nrow and 0 <= nj < ncol:
                        neighbors.append(temp_grid[ni, nj])
                    else:
                        neighbors.append(-999.0)
                if any(n == -999.0 for n in neighbors):
                    boundary_flags[idx] = True
    # 將測試集依據邊界標記分類，分別計算準確率
    X_tuple = [tuple(x) for x in X]
    X_test_idx = [X_tuple.index(tuple(x)) for x in X_test]
    boundary_mask = boundary_flags[X_test_idx]
    y_pred = clf.predict(X_test)
    acc_boundary = accuracy_score(y_test[boundary_mask], y_pred[boundary_mask]) if np.any(boundary_mask) else None
    acc_non_boundary = accuracy_score(y_test[~boundary_mask], y_pred[~boundary_mask]) if np.any(~boundary_mask) else None
    print("--- 以缺值定義的邊界錯誤分析 ---")
    if acc_boundary is not None:
        print(f"邊界區域準確率: {acc_boundary:.4f} (樣本數: {boundary_mask.sum()})")
    else:
        print("邊界區域無測試樣本")
    if acc_non_boundary is not None:
        print(f"非邊界區域準確率: {acc_non_boundary:.4f} (樣本數: {(~boundary_mask).sum()})")
    else:
        print("非邊界區域無測試樣本")

    # 驗證集預測結果視覺化
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.title('Validation: True Label')
    plt.scatter(X_val[:,0], X_val[:,1], c=y_val, cmap='coolwarm', s=10, label='True')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.subplot(1,2,2)
    plt.title('Validation: Predicted Label')
    plt.scatter(X_val[:,0], X_val[:,1], c=y_val_pred, cmap='coolwarm', s=10, label='Pred')
    plt.xlabel('Longitude')
    plt.tight_layout()
    plt.show()

    # 測試集預測結果視覺化
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.title('Test: True Label')
    plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap='coolwarm', s=10, label='True')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.subplot(1,2,2)
    plt.title('Test: Predicted Label')
    plt.scatter(X_test[:,0], X_test[:,1], c=y_pred, cmap='coolwarm', s=10, label='Pred')
    plt.xlabel('Longitude')
    plt.tight_layout()
    plt.show()

    # 使用者輸入經緯度，輸出預測標籤與實際標籤
    while True:
        try:
            user_input = input('請輸入經度,緯度（例如 120.5,23.5，或直接按 Enter 結束）：')
            if not user_input.strip():
                break
            lon_str, lat_str = user_input.split(',')
            lon = float(lon_str.strip())
            lat = float(lat_str.strip())
            pred_label = clf.predict(np.array([[lon, lat]]))[0]
            # 查找最接近的實際點
            dists = np.linalg.norm(X - np.array([lon, lat]), axis=1)
            idx = np.argmin(dists)
            real_label = y[idx]
            print(f'預測標籤: {pred_label}，最接近資料點實際標籤: {real_label} (距離: {dists[idx]:.4f})')
        except Exception as e:
            print('格式錯誤或發生例外，請重新輸入。')