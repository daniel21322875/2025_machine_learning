
"""
h_x_pipeline.py

組合分類與回歸模型為 h(x)：先用 C(x) 判斷是否有溫度，若 C(x)=1 則回傳 R(x)，否則回傳 -999。
此檔案提供訓練、評估、儲存模型與單點查詢功能，並產生簡單圖形來說明行為。
"""
import argparse
import numpy as np
import xml.etree.ElementTree as ET
import sys
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import os


def parse_xml_all(file_path):
    """Parse XML and return X (lon,lat), y_presence (0/1), y_temp (float or np.nan)
    """
    ns = {'ns': 'urn:cwa:gov:tw:cwacommon:0.1'}
    tree = ET.parse(file_path)
    root = tree.getroot()

    lon_start = float(root.find('.//ns:BottomLeftLongitude', ns).text)
    lat_start = float(root.find('.//ns:BottomLeftLatitude', ns).text)
    lon_step = 0.03
    lat_step = 0.03

    content = root.find('.//ns:Content', ns)
    if content is None:
        raise ValueError('No <Content> found in xml')

    temp_data = content.text.strip().split('\n')
    temp_grid = [row.strip().split(',') for row in temp_data if row.strip()]

    X = []
    y_presence = []
    y_temp = []
    for i, row in enumerate(temp_grid):
        lat = lat_start + i * lat_step
        for j, temp_str in enumerate(row):
            lon = lon_start + j * lon_step
            temp = float(temp_str)
            X.append([lon, lat])
            if temp != -999.0:
                y_presence.append(1)
                y_temp.append(temp)
            else:
                y_presence.append(0)
                y_temp.append(np.nan)

    return np.array(X), np.array(y_presence), np.array(y_temp)


def train_models(file_path, out_prefix='h_x', save_models=True):
    # New function to plot error map from models
    X, y_presence, y_temp = parse_xml_all(file_path)

    # compute bounds of training grid
    lon_min, lon_max = X[:,0].min(), X[:,0].max()
    lat_min, lat_max = X[:,1].min(), X[:,1].max()

    # Split full dataset for classifier (presence/absence)
    X_train, X_temp, y_train_pres, y_temp_pres = train_test_split(
        X, y_presence, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val_pres, y_test_pres = train_test_split(
        X_temp, y_temp_pres, test_size=0.5, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train_pres)

    print('Classifier trained')
    y_val_pred = clf.predict(X_val)
    print(f'Classifier val acc: {accuracy_score(y_val_pres, y_val_pred):.4f}')
    y_test_pred = clf.predict(X_test)
    print(f'Classifier test acc: {accuracy_score(y_test_pres, y_test_pred):.4f}')

    # Prepare regression data: start from true-present samples
    present_mask = ~np.isnan(y_temp)
    X_present = X[present_mask]
    y_present = y_temp[present_mask]

    # Filter regression training set by classifier predictions (simulating deploy filtering):
    # keep only those true-present points that classifier also predicts as present
    clf_preds_on_present = clf.predict(X_present)
    selected_mask = clf_preds_on_present == 1
    if selected_mask.sum() == 0:
        # fallback: use all true-present samples if classifier selected none
        print('Warning: classifier predicted no present samples among true-present points; using all true-present samples for regressor training.')
        Xr_all = X_present
        yr_all = y_present
    else:
        Xr_all = X_present[selected_mask]
        yr_all = y_present[selected_mask]

    # Split regression data
    Xr_train, Xr_temp, yr_train, yr_temp = train_test_split(
        Xr_all, yr_all, test_size=0.4, random_state=42
    )
    Xr_val, Xr_test, yr_val, yr_test = train_test_split(Xr_temp, yr_temp, test_size=0.5, random_state=42)

    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(Xr_train, yr_train)

    print('Regressor trained')
    yr_val_pred = reg.predict(Xr_val)
    print(f'Regressor val MSE: {mean_squared_error(yr_val, yr_val_pred):.4f}')
    yr_test_pred = reg.predict(Xr_test)
    print(f'Regressor test MSE: {mean_squared_error(yr_test, yr_test_pred):.4f}')

    # Save models unless disabled
    clf_path = f'{out_prefix}_clf.joblib'
    reg_path = f'{out_prefix}_reg.joblib'
    if save_models:
        joblib.dump(clf, clf_path)
        joblib.dump(reg, reg_path)
    else:
        # intentionally do not print when skipping saving models
        pass

    bounds = (lon_min, lon_max, lat_min, lat_max)
    return clf, reg, X, y_presence, y_temp, bounds


def plot_error_map_from_models(clf=None, reg=None, clf_path=None, reg_path=None, xml_file='O-A0038-003.xml', show=False, out_prefix='error'):
    """Load models if needed, parse XML and produce error map + CSV.

    If clf/reg models are provided they are used; otherwise attempt to find candidate model files in the working directory.
    """
    MODEL_CLF_CAND = ['h_x_clfsel_clf.joblib', 'h_x_clf_clf.joblib', 'h_x_clf.joblib']
    MODEL_REG_CAND = ['h_x_clfsel_reg.joblib', 'h_x_reg.joblib', 'h_x_clfreg.joblib']

    def find_model(cands):
        for c in cands:
            if os.path.exists(c):
                return c
        return None

    # load models if not provided
    if clf is None:
        if clf_path is None:
            clf_path = find_model(MODEL_CLF_CAND)
        if clf_path is None:
            raise FileNotFoundError('Classifier model not found. Train or provide clf_path.')
        clf = joblib.load(clf_path)
    if reg is None:
        if reg_path is None:
            reg_path = find_model(MODEL_REG_CAND)
        if reg_path is None:
            raise FileNotFoundError('Regressor model not found. Train or provide reg_path.')
        reg = joblib.load(reg_path)

    X, y_presence, y_temp = parse_xml_all(xml_file)
    # reuse save_error_map logic (which also saves CSV)
    save_error_map(clf, reg, X, y_presence, y_temp, out_prefix=out_prefix, show=show)
def h_x(clf, reg, lon, lat, bounds=None, strict_boundary=False):
    """Combined model h(x): returns R(x) if C(x)=1, else -999.

    If strict_boundary=True and bounds provided (lon_min, lon_max, lat_min, lat_max),
    then points outside bounds return -999 without calling classifier/regressor.
    """
    if strict_boundary and bounds is not None:
        lon_min, lon_max, lat_min, lat_max = bounds
        if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
            return -999.0

    x = np.array([[lon, lat]])
    c = int(clf.predict(x)[0])
    if c == 1:
        r = float(reg.predict(x)[0])
        return r
    else:
        return -999.0


def evaluate_and_plot(clf, reg, X, y_presence, y_temp):
    # Evaluate classifier on all X
    y_pred = clf.predict(X)
    acc = accuracy_score(y_presence, y_pred)
    print(f'Overall classifier accuracy: {acc:.4f}')

    # Evaluate h(x) on whole grid: produce h_pred and compare to ground truth y_temp
    h_pred = np.array([h_x(clf, reg, lon, lat, bounds=None, strict_boundary=False) for lon, lat in X])

    # For plotting, prepare ground truth where present use temp, else -999
    y_true = np.where(~np.isnan(y_temp), y_temp, -999.0)

    # Scatter plot: true vs h_pred for present points and non-present points colored separately
    present_mask = y_true != -999.0

    if not _HAS_MPL:
        print('matplotlib not available: skipping plots (install matplotlib to enable plotting)')
        return

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.title('True temperature (present points)')
    plt.scatter(X[present_mask,0], X[present_mask,1], c=y_true[present_mask], cmap='viridis', s=8)
    plt.colorbar(label='Temp')
    plt.xlabel('Lon')
    plt.ylabel('Lat')

    plt.subplot(1,2,2)
    plt.title('h(x) predictions (only where classifier->present)')
    # show predictions only where classifier predicted present
    clf_pred_present = clf.predict(X) == 1
    # use NaN for plotting where not predicted present
    h_plot_vals = np.where(clf_pred_present, h_pred, np.nan)
    sc = plt.scatter(X[:,0], X[:,1], c=h_plot_vals, cmap='viridis', s=8)
    plt.colorbar(sc, label='h(x) Temp or NaN')
    plt.xlabel('Lon')
    plt.ylabel('Lat')

    plt.tight_layout()
    plt.show()


def save_error_map(clf, reg, X, y_presence, y_temp, out_prefix='error', show=False):
    """Compute errors on true-present points and save or show error_map and save error_values.csv

    If show=True and matplotlib available, the image will be displayed with plt.show() instead of saved.
    """
    import csv
    present_mask = ~np.isnan(y_temp)
    X_present = X[present_mask]
    y_present = y_temp[present_mask]

    clf_pred = clf.predict(X_present)
    pred_temps = np.full(len(X_present), np.nan)
    mask_predpos = clf_pred == 1
    if mask_predpos.sum() > 0:
        pred_temps[mask_predpos] = reg.predict(X_present[mask_predpos])

    abs_err = np.abs(pred_temps - y_present)

    if _HAS_MPL:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.title('Absolute error (where classifier predicted present)')
        sc = plt.scatter(X_present[:,0], X_present[:,1], c=np.where(np.isnan(abs_err), np.nan, abs_err), cmap='hot', s=8, vmin=0, vmax=np.nanpercentile(abs_err[~np.isnan(abs_err)], 95))
        plt.colorbar(sc, label='abs error')
        fn_mask = ~mask_predpos
        if fn_mask.sum() > 0:
            plt.scatter(X_present[fn_mask,0], X_present[fn_mask,1], facecolors='none', edgecolors='cyan', s=12, label='FN (missed)')
            plt.legend()
        plt.xlabel('Lon')
        plt.ylabel('Lat')

        plt.subplot(1,2,2)
        plt.title('Residuals: predicted - true')
        resids = pred_temps - y_present
        plt.hist(resids[~np.isnan(resids)], bins=50)
        plt.xlabel('Pred - True')
        plt.ylabel('Count')

        plt.tight_layout()
        map_path = f'{out_prefix}_map.png'
        if show:
            plt.show()
        else:
            plt.savefig(map_path, dpi=150)
            plt.close()
            print(f'Saved {map_path}')
    else:
        print('matplotlib not available: skipping map image (CSV will still be saved)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='O-A0038-003.xml')
    parser.add_argument('--out', type=str, default='h_x')
    parser.add_argument('--predict', nargs=2, type=float, help='lon lat to run h(x)')
    parser.add_argument('--strict-boundary', action='store_true', help='Return -999 for points outside training grid bounds')
    parser.add_argument('--save-error-map', action='store_true', help='(deprecated) Save an error map PNG and CSV after training')
    parser.add_argument('--show-error-map', action='store_true', help='Display an error map and save CSV after training')
    parser.add_argument('--no-save-models', action='store_true', help='Do not save trained models to disk')
    parser.add_argument('--plot-error-map', action='store_true', help='Plot error map using existing models (if any)')
    args = parser.parse_args()

    # If script is invoked with no CLI arguments, run full pipeline automatically (train, eval, show error map)
    if len(sys.argv) == 1:
        # set defaults for direct operation
        args.no_save_models = True
        args.show_error_map = True
        args.save_error_map = False
        args.plot_error_map = False


    clf, reg, X, y_presence, y_temp, bounds = train_models(args.file, out_prefix=args.out, save_models=not args.no_save_models)

    evaluate_and_plot(clf, reg, X, y_presence, y_temp)

    # Backwards compatibility: --save-error-map is deprecated — display the map instead of saving PNG
    if args.save_error_map:
        print('Note: --save-error-map is deprecated; displaying error map instead of saving PNG.')
        save_error_map(clf, reg, X, y_presence, y_temp, out_prefix='error', show=True)
    elif args.show_error_map:
        save_error_map(clf, reg, X, y_presence, y_temp, out_prefix='error', show=True)

    # If user explicitly requests plotting from existing models (separate command)
    if args.plot_error_map:
        # prefer interactive show if either show flag set
        show_flag = args.show_error_map or args.save_error_map
        plot_error_map_from_models(clf=None, reg=None, xml_file=args.file, show=show_flag, out_prefix='error')

    if args.predict:
        lon, lat = args.predict
        val = h_x(clf, reg, lon, lat, bounds=bounds, strict_boundary=args.strict_boundary)
        print(f'h(({lon},{lat})) = {val}')


if __name__ == '__main__':
    main()
