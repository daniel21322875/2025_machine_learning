# 匯入必要的套件
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt  # 匯入繪圖套件
import os
import sys


# ---- 簡易取代 sklearn 的工具函式（方便在沒有 sklearn 的環境執行） ----
def train_test_split(X, y, test_size=0.25, random_state=None):
    """簡化版的 train_test_split：回傳 X_train, X_test, y_train, y_test。

    - test_size: 若為 float，視為測試集比例 (0,1)；若為 int，視為測試樣本數。
    - random_state: int 隨機種子以確保可重現性。
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError('X and y must have same number of samples')
    rng = np.random.RandomState(random_state)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    if isinstance(test_size, float):
        if not 0.0 < test_size < 1.0:
            raise ValueError('test_size float must be between 0 and 1')
        n_test = int(np.round(X.shape[0] * test_size))
    else:
        n_test = int(test_size)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError('y_true and y_pred must have same shape')
    return float((y_true == y_pred).mean())

# -------------------------------------------------------------------------------

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
        temps = []  # 溫度值，缺值使用 -999.0
        for i, row in enumerate(temp_grid):
            lat = lat_start + i * lat_step
            for j, temp_str in enumerate(row):
                lon = lon_start + j * lon_step
                temp = float(temp_str)
                label = 1 if temp != -999.0 else 0
                X.append([lon, lat])
                y.append(label)
                temps.append(temp)
        return np.array(X), np.array(y), np.array(temps)

    except AttributeError as e:
        print("錯誤：XML 標籤路徑不正確，請檢查 <BottomLeftLongitude>, <BottomLeftLatitude>, <TopRightLongitude>, <TopRightLatitude> 等標籤是否存在。")
        raise
    except ValueError as e:
        print(f"錯誤：{e}")
        raise
    except Exception as e:
        print(f"發生未知錯誤：{e}")
        raise

# Quadratic Discriminant Analysis (QDA) implementation
class QDA:
    """Quadratic Discriminant Analysis: each class has its own covariance Sigma_k."""
    def __init__(self, reg=1e-6):
        self.reg = reg

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        m, n = X.shape
        # priors
        self.phi = np.clip(y.mean(), 1e-12, 1 - 1e-12)
        # class means
        if np.any(y == 0):
            self.mu0 = X[y == 0].mean(axis=0)
        else:
            self.mu0 = np.zeros(n)
        if np.any(y == 1):
            self.mu1 = X[y == 1].mean(axis=0)
        else:
            self.mu1 = np.zeros(n)

        # class covariances
        X0 = X[y == 0]
        X1 = X[y == 1]
        # If a class has no samples, set covariance to identity
        if X0.shape[0] > 0:
            S0 = np.cov(X0, rowvar=False, bias=True)  # divide by N
        else:
            S0 = np.eye(n)
        if X1.shape[0] > 0:
            S1 = np.cov(X1, rowvar=False, bias=True)
        else:
            S1 = np.eye(n)

        # regularize
        S0 += self.reg * np.eye(n)
        S1 += self.reg * np.eye(n)
        self.Sigma0 = S0
        self.Sigma1 = S1
        # inverses and log-dets
        self.Sigma0_inv = np.linalg.inv(S0)
        self.Sigma1_inv = np.linalg.inv(S1)
        sign0, logdet0 = np.linalg.slogdet(S0)
        sign1, logdet1 = np.linalg.slogdet(S1)
        self.logdet0 = logdet0
        self.logdet1 = logdet1

    def _log_gaussian(self, X, mu, Sigma_inv, logdet):
        """Compute log N(x|mu,Sigma) up to constant (without -0.5*n*log(2pi)).

        Returns array of shape (m,)
        """
        X = np.atleast_2d(X)
        diffs = X - mu
        # mahalanobis: (diffs @ Sigma_inv) * diffs, sum over features
        mahal = np.sum((diffs.dot(Sigma_inv)) * diffs, axis=1)
        return -0.5 * mahal - 0.5 * logdet

    def decision(self, X):
        X = np.atleast_2d(X)
        # log p(x|y=1) + log phi  - (log p(x|y=0) + log(1-phi))
        lg1 = self._log_gaussian(X, self.mu1, self.Sigma1_inv, self.logdet1)
        lg0 = self._log_gaussian(X, self.mu0, self.Sigma0_inv, self.logdet0)
        return (lg1 + np.log(self.phi)) - (lg0 + np.log(1 - self.phi))

    def predict_proba(self, X):
        z = self.decision(X)
        p1 = 1.0 / (1.0 + np.exp(-z))
        p0 = 1 - p1
        return np.vstack([p0, p1]).T

    def predict(self, X):
        return (self.decision(X) > 0).astype(int).ravel()

if __name__ == "__main__":
    # 讀取 XML 並轉為資料集
    # 支援： 1) 命令列參數指定檔名；2) 若未指定則嘗試預設檔名或搜尋當前資料夾第一個 .xml
    file_path = None
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        default = 'O-A0038-003.xml'
        if os.path.isfile(default):
            file_path = default
        else:
            xmls = [f for f in os.listdir('.') if f.lower().endswith('.xml')]
            if len(xmls) > 0:
                print(f"警告: 未指定檔名，使用當前資料夾中的 '{xmls[0]}'")
                file_path = xmls[0]

    if file_path is None:
        print(
            "找不到 XML 檔案。請將 XML 放在程式相同資料夾，或以命令列參數提供檔案名稱：\n"
            "python Assignment 6_Classification code.py yourfile.xml"
        )
        sys.exit(1)

    X, y, temps = parse_xml(file_path)

    # 分割資料集為訓練集、驗證集與測試集（6:2:2）
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # 建立並訓練二次判別分析 (QDA) 分類器
    clf = QDA(reg=1e-6)
    clf.fit(X_train, y_train)

    # 印出診斷資訊（供報告或檢查）
    try:
        print('phi =', clf.phi)
        print('mu0 =', clf.mu0)
        print('mu1 =', clf.mu1)
        print('logdet Sigma0 =', getattr(clf, 'logdet0', None))
        print('logdet Sigma1 =', getattr(clf, 'logdet1', None))
    except Exception:
        pass

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

    # ------------------ Regression model (R) ------------------
    # Build a simple polynomial regression (degree 2) from available temperature data
    class Poly2Regression:
        """Quadratic polynomial regression: features [1, x, y, x^2, x*y, y^2]"""
        def __init__(self, reg=1e-8):
            self.reg = reg
            self.coef_ = None

        def _design(self, X):
            # X: (m,2)
            x = X[:,0]
            y = X[:,1]
            return np.vstack([np.ones_like(x), x, y, x*x, x*y, y*y]).T

        def fit(self, X, t):
            Phi = self._design(X)
            # ridge regularized least squares
            A = Phi.T.dot(Phi) + self.reg * np.eye(Phi.shape[1])
            b = Phi.T.dot(t)
            self.coef_ = np.linalg.solve(A, b)

        def predict(self, X):
            Phi = self._design(np.atleast_2d(X))
            return Phi.dot(self.coef_)

    # 訓練回歸模型只使用有溫度的點 (temps != -999.0)
    avail_mask = temps != -999.0
    X_avail = X[avail_mask]
    temps_avail = temps[avail_mask]
    if X_avail.shape[0] < 6:
        print("警告: 用於回歸的有效樣本太少，回歸模型可能不穩定。")

    R = Poly2Regression(reg=1e-6)
    if X_avail.shape[0] > 0:
        R.fit(X_avail, temps_avail)
        # 計算在可用點上的 RMSE 作為診斷
        preds_train = R.predict(X_avail)
        rmse = np.sqrt(np.mean((preds_train - temps_avail)**2))
        print(f"Regression trained on {X_avail.shape[0]} points, RMSE = {rmse:.4f}")
    else:
        print("沒有可用的溫度資料來訓練回歸模型。")

    # 組合模型 h(x): 若 C(x)=1 回傳 R(x)，否則回傳 -999
    def h_predict(Xq):
        Xq = np.atleast_2d(Xq)
        c = clf.predict(Xq)
        r = R.predict(Xq) if X_avail.shape[0] > 0 else np.full(Xq.shape[0], np.nan)
        out = np.full(Xq.shape[0], -999.0)
        out[c == 1] = r[c == 1]
        return out

    # --- 移除以缺值定義的邊界分析，改為繪製「真實標籤 vs 預測標籤」的左右比較圖 ---
    # 在 Validation 與 Test 各繪製一張 figure，左右兩個子圖：
    # 左：實際標籤 + 決策邊界；右：預測標籤 + 決策邊界（錯誤點以黃圈標示），右圖標題顯示準確率
    y_val_pred = clf.predict(X_val)
    acc_val = accuracy_score(y_val, y_val_pred)
    y_test_pred = clf.predict(X_test)
    acc_test = accuracy_score(y_test, y_test_pred)

    def plot_decision_boundary(clf, X_all, X_pts, y_true, y_pred=None, ax=None, grid_res=300, cmap='coolwarm'):
        """在 XY 平面上畫出決策區域與決策邊界，並疊上資料點。

        - clf: 已訓練的分類器，需有 decision(X) 與 predict(X)
        - X_all: 用於決定繪圖範圍的所有點 (m,2)
        - X_pts: 要疊上的點 (k,2)
        - y_true: 真實標籤 (k,)
        - y_pred: 若提供，會以不同 marker 顯示預測標籤
        """
        ax = ax or plt.gca()
        xmin, xmax = X_all[:,0].min() - 0.01, X_all[:,0].max() + 0.01
        ymin, ymax = X_all[:,1].min() - 0.01, X_all[:,1].max() + 0.01
        xx = np.linspace(xmin, xmax, grid_res)
        yy = np.linspace(ymin, ymax, grid_res)
        XX, YY = np.meshgrid(xx, yy)
        pts = np.c_[XX.ravel(), YY.ravel()]
        Z = clf.decision(pts).reshape(XX.shape)

        # 決策區域
        ax.contourf(XX, YY, Z, levels=[Z.min(), 0, Z.max()], alpha=0.22, cmap=cmap)
        # 決策邊界（加粗）
        ax.contour(XX, YY, Z, levels=[0], colors='k', linewidths=2.0)
        # 真實標籤（點顏色較淺）
        # 使用 alpha 使紅色點看起來較淺，並減小 marker edge width
        sc = ax.scatter(X_pts[:,0], X_pts[:,1], c=y_true, cmap=cmap, s=24, edgecolor='k', linewidths=0.4, alpha=0.6)

        # 若有預測，標註預測錯誤點
        if y_pred is not None:
            wrong = y_pred != y_true
            if np.any(wrong):
                ax.scatter(X_pts[wrong,0], X_pts[wrong,1], facecolors='none', edgecolors='yellow', s=60, linewidths=1.6, label='wrong')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        return sc

    def plot_true_vs_pred(clf, X_all, X_pts, y_true, y_pred, title_prefix=''):
        """繪製左右比較：左為真實標籤；右為預測標籤（錯誤點以黃圈標示）。

        - clf: 已訓練模型
        - X_all: 用於決定繪圖範圍的所有點 (m,2)
        - X_pts: 要疊上的資料點 (k,2)
        - y_true: 真實標籤 (k,)
        - y_pred: 預測標籤 (k,)
        """
        fig, axes = plt.subplots(1,2, figsize=(12,5))
        fig.suptitle(title_prefix)
        # 左：真實標籤
        ax = axes[0]
        plot_decision_boundary(clf, X_all, X_pts, y_true, y_pred=None, ax=ax)
        ax.set_title('True labels with decision boundary')

        # 右：預測標籤，並標示錯誤點
        ax2 = axes[1]
        plot_decision_boundary(clf, X_all, X_pts, y_pred, y_pred=y_pred, ax=ax2)
        wrong = (y_pred != y_true)
        if np.any(wrong):
            ax2.scatter(X_pts[wrong,0], X_pts[wrong,1], facecolors='none', edgecolors='yellow', s=80, linewidths=1.5, label='wrong')
            ax2.legend(loc='upper right')
        acc = accuracy_score(y_true, y_pred)
        ax2.set_title(f'Predicted labels (acc={acc:.4f})')
        plt.tight_layout(rect=[0,0.03,1,0.95])
        plt.show()

    # --- 單張圖：以 XML 所有資料點 (X, y) 作為底圖，疊上決策區域與邊界 ---
    plt.figure(figsize=(8,6))
    plt.title('All data: true labels with QDA decision boundary')
    # 使用整體資料 X 與真實標籤 y
    plot_decision_boundary(clf, X, X, y, y_pred=None, ax=plt.gca(), grid_res=300)
    plt.tight_layout()
    plt.show()