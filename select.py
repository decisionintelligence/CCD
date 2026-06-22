import os
import argparse
import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")


def parse_window_arg(x):
    """
    支持：
    - "7D", "1D", "24H"
    - "512", "128"
    """
    if x is None:
        return None

    x = str(x)

    if x.isdigit():
        return int(x)

    return x


def safe_minmax(x, eps=1e-8):
    x = np.asarray(x, dtype=float)

    if np.all(~np.isfinite(x)):
        return np.zeros_like(x)

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    x_min = np.min(x)
    x_max = np.max(x)

    if abs(x_max - x_min) < eps:
        return np.zeros_like(x)

    return (x - x_min) / (x_max - x_min + eps)


def split_continuous_segments(df, time_col="time", gap_factor=3.0):
    """
    自动检测时间断点。
    如果相邻时间差 > 中位数时间间隔 * gap_factor，则切成新片段。
    """
    df = df.sort_values(time_col).reset_index(drop=True)

    delta = df[time_col].diff()
    median_delta = delta.dropna().median()

    if pd.isna(median_delta) or median_delta == pd.Timedelta(0):
        return [df]

    gap_threshold = median_delta * gap_factor
    segment_id = (delta > gap_threshold).cumsum()

    segments = []

    for _, part in df.groupby(segment_id):
        part = part.reset_index(drop=True)
        if len(part) > 0:
            segments.append(part)

    return segments


def iter_windows(
    df,
    time_col="time",
    window_size="7D",
    stride="1D",
    min_points=128,
):
    """
    自动分窗。

    支持两类：

    1. 时间分窗：
       window_size="7D", stride="1D"
       window_size="24H", stride="6H"

    2. 行数分窗：
       window_size=512, stride=128
    """
    if isinstance(window_size, int):
        if stride is None:
            stride = max(1, window_size // 4)

        if not isinstance(stride, int):
            raise ValueError("使用行数分窗时，stride 也必须是整数，例如 --stride 128")

        n = len(df)

        for start in range(0, n - window_size + 1, stride):
            end = start + window_size
            win = df.iloc[start:end].copy()

            if len(win) >= min_points:
                yield win

        return

    window_td = pd.Timedelta(window_size)

    if stride is None:
        stride_td = window_td / 4
    else:
        stride_td = pd.Timedelta(stride)

    start_time = df[time_col].iloc[0]
    end_time = df[time_col].iloc[-1]

    cur = start_time

    while cur + window_td <= end_time:
        left = cur
        right = cur + window_td

        win = df[(df[time_col] >= left) & (df[time_col] < right)].copy()

        if len(win) >= min_points:
            yield win

        cur = cur + stride_td


def count_windows(
    df,
    time_col="time",
    window_size="7D",
    stride="1D",
    min_points=128,
):
    cnt = 0

    for _ in iter_windows(
        df,
        time_col=time_col,
        window_size=window_size,
        stride=stride,
        min_points=min_points,
    ):
        cnt += 1

    return cnt


def pearson_group_score(
    X,
    y,
    n_groups=78,
    group_size=25,
    topk=5,
    eps=1e-8,
):
    """
    X: [T, 1950]
    y: [T]

    返回：
    group_score: [78]
    feature_corr: [78, 25]
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    T = X.shape[0]

    X_mean = np.nanmean(X, axis=0, keepdims=True)
    X_std = np.nanstd(X, axis=0, keepdims=True) + eps

    y_mean = np.nanmean(y)
    y_std = np.nanstd(y) + eps

    X_norm = (X - X_mean) / X_std
    y_norm = (y - y_mean) / y_std

    X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)
    y_norm = np.nan_to_num(y_norm, nan=0.0, posinf=0.0, neginf=0.0)

    corr = np.abs(X_norm.T @ y_norm / max(T - 1, 1))

    feature_corr = corr.reshape(n_groups, group_size)

    sorted_corr = np.sort(feature_corr, axis=1)[:, ::-1]
    group_score = sorted_corr[:, :topk].mean(axis=1)

    return group_score, feature_corr


def elasticnet_group_score_fast(
    X,
    y,
    selected_groups,
    n_groups=78,
    group_size=25,
    alpha=0.01,
    l1_ratio=0.7,
    random_state=42,
):
    """
    快速版本：固定 alpha 和 l1_ratio，不做交叉验证。
    适合先跑通、快速筛变量。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(y) < 50 or len(selected_groups) == 0:
        return np.zeros(n_groups)

    selected_feature_indices = []

    for g in selected_groups:
        start = g * group_size
        end = (g + 1) * group_size
        selected_feature_indices.extend(range(start, end))

    X_sel = X[:, selected_feature_indices]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("enet", ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=5000,
            random_state=random_state,
            selection="random",
        )),
    ])

    model.fit(X_sel, y)

    coef = np.abs(model.named_steps["enet"].coef_)
    coef = coef.reshape(len(selected_groups), group_size)

    group_score = np.zeros(n_groups)

    for local_idx, g in enumerate(selected_groups):
        group_score[g] = coef[local_idx].sum()

    return group_score


def elasticnet_group_score_cv(
    X,
    y,
    selected_groups,
    n_groups=78,
    group_size=25,
    n_splits=3,
    random_state=42,
):
    """
    慢速但更稳版本：ElasticNetCV。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    T = len(y)

    if T < 80 or len(selected_groups) == 0:
        return np.zeros(n_groups)

    selected_feature_indices = []

    for g in selected_groups:
        start = g * group_size
        end = (g + 1) * group_size
        selected_feature_indices.extend(range(start, end))

    X_sel = X[:, selected_feature_indices]

    usable_splits = min(n_splits, max(2, T // 80))

    if usable_splits < 2:
        return np.zeros(n_groups)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("enet", ElasticNetCV(
            l1_ratio=[0.3, 0.5, 0.7, 0.9],
            cv=TimeSeriesSplit(n_splits=usable_splits),
            max_iter=10000,
            n_jobs=-1,
            random_state=random_state,
        )),
    ])

    model.fit(X_sel, y)

    coef = np.abs(model.named_steps["enet"].coef_)
    coef = coef.reshape(len(selected_groups), group_size)

    group_score = np.zeros(n_groups)

    for local_idx, g in enumerate(selected_groups):
        group_score[g] = coef[local_idx].sum()

    return group_score


def run_selection(
    csv_path,
    output_dir,
    time_col="time",
    target_col="power",
    ws_col="ws",
    start_date="2026-02-01",
    n_groups=78,
    group_size=25,
    window_size="7D",
    stride="1D",
    horizon=1,
    pearson_top_groups=30,
    final_top_groups=20,
    min_points=128,
    gap_factor=3.0,
    use_cv=False,
    elastic_alpha=0.01,
    elastic_l1_ratio=0.7,
):
    os.makedirs(output_dir, exist_ok=True)

    print(f"[1] 读取 CSV: {csv_path}", flush=True)
    df = pd.read_csv(csv_path)

    print(f"[2] 检查列名", flush=True)

    if time_col not in df.columns:
        raise ValueError(f"找不到时间列: {time_col}")

    if target_col not in df.columns:
        raise ValueError(f"找不到目标列: {target_col}")

    if ws_col not in df.columns:
        print(f"    警告：没有找到 ws 列: {ws_col}，将只排除 time 和 power", flush=True)

    print(f"[3] 解析时间列并排序: {time_col}", flush=True)
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    print(f"[4] 只保留 {start_date} 及之后的数据", flush=True)
    start_ts = pd.Timestamp(start_date)

    df = df[df[time_col] >= start_ts].copy().reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError(f"{start_date} 之后没有数据。")

    print(f"    过滤后数据量: {len(df)}", flush=True)
    print(f"    时间范围: {df[time_col].min()} ~ {df[time_col].max()}", flush=True)

    exclude_cols = {time_col, target_col, ws_col}

    covariate_cols = [
        c for c in df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    expected_dim = n_groups * group_size

    print(f"[5] 自动识别气象协变量列", flush=True)
    print(f"    排除列: {sorted(list(exclude_cols))}", flush=True)
    print(f"    识别到协变量列数: {len(covariate_cols)}", flush=True)
    print(f"    期望协变量列数: {expected_dim}", flush=True)

    if len(covariate_cols) != expected_dim:
        raise ValueError(
            f"协变量列数量不对。\n"
            f"当前识别到 {len(covariate_cols)} 列，但期望是 {expected_dim} 列。\n"
            f"请检查 CSV 里是否有额外数值列没有排除，或者气象协变量列是否缺失。\n"
            f"当前识别到的前 20 个协变量列：{covariate_cols[:20]}"
        )

    group_mapping_records = []

    for g in range(n_groups):
        start = g * group_size
        end = (g + 1) * group_size
        cols = covariate_cols[start:end]

        for local_idx, col in enumerate(cols):
            group_mapping_records.append({
                "group_id": g,
                "feature_id_in_group": local_idx,
                "column_name": col,
            })

    group_mapping = pd.DataFrame(group_mapping_records)
    group_mapping_path = os.path.join(output_dir, "group_column_mapping.csv")
    group_mapping.to_csv(group_mapping_path, index=False)

    print(f"[6] 自动检测时间断点并切段", flush=True)
    segments = split_continuous_segments(
        df,
        time_col=time_col,
        gap_factor=gap_factor,
    )

    print(f"    连续片段数量: {len(segments)}", flush=True)

    print(f"[7] 统计窗口数量", flush=True)

    total_windows = 0
    segment_window_counts = []

    for seg in segments:
        cnt = count_windows(
            seg,
            time_col=time_col,
            window_size=window_size,
            stride=stride,
            min_points=min_points,
        )
        segment_window_counts.append(cnt)
        total_windows += cnt

    print(f"    总窗口数: {total_windows}", flush=True)

    if total_windows == 0:
        raise RuntimeError(
            "没有生成任何窗口。请调小 --window_size 或 --min_points。"
        )

    all_pearson_scores = []
    all_enet_scores = []

    selected_count = np.zeros(n_groups)
    valid_window_count = 0

    window_records = []

    print(f"[8] 开始分窗筛选", flush=True)
    print(f"    window_size        = {window_size}", flush=True)
    print(f"    stride             = {stride}", flush=True)
    print(f"    horizon            = {horizon}", flush=True)
    print(f"    pearson_top_groups = {pearson_top_groups}", flush=True)
    print(f"    final_top_groups   = {final_top_groups}", flush=True)
    print(f"    use_cv             = {use_cv}", flush=True)

    pbar = tqdm(
        total=total_windows,
        desc="Selecting",
        unit="win",
        dynamic_ncols=True,
    )

    for seg_id, seg in enumerate(segments):
        seg_start = seg[time_col].min()
        seg_end = seg[time_col].max()

        pbar.write(
            f"Segment {seg_id}: {seg_start} ~ {seg_end}, "
            f"n={len(seg)}, windows={segment_window_counts[seg_id]}"
        )

        for win_id, win in enumerate(iter_windows(
            seg,
            time_col=time_col,
            window_size=window_size,
            stride=stride,
            min_points=min_points,
        )):
            pbar.set_postfix({
                "stage": "load",
                "valid": valid_window_count,
                "seg": seg_id,
                "win": win_id,
            })

            X = win[covariate_cols].to_numpy(dtype=float)
            y = win[target_col].to_numpy(dtype=float)

            # 用 X_t 对齐 y_{t+horizon}
            if horizon > 0:
                if len(win) <= horizon + min_points:
                    pbar.update(1)
                    continue

                X_aligned = X[:-horizon]
                y_aligned = y[horizon:]
            else:
                X_aligned = X
                y_aligned = y

            valid_mask = np.isfinite(y_aligned)

            X_aligned = X_aligned[valid_mask]
            y_aligned = y_aligned[valid_mask]

            if len(y_aligned) < min_points:
                pbar.update(1)
                continue

            pbar.set_postfix({
                "stage": "pearson",
                "valid": valid_window_count,
                "n": len(y_aligned),
            })

            pearson_score, _ = pearson_group_score(
                X_aligned,
                y_aligned,
                n_groups=n_groups,
                group_size=group_size,
                topk=5,
            )

            pearson_selected = np.argsort(pearson_score)[::-1][:pearson_top_groups]

            pbar.set_postfix({
                "stage": "elasticnet",
                "valid": valid_window_count,
                "n": len(y_aligned),
                "groups": len(pearson_selected),
            })

            if use_cv:
                enet_score = elasticnet_group_score_cv(
                    X_aligned,
                    y_aligned,
                    selected_groups=pearson_selected,
                    n_groups=n_groups,
                    group_size=group_size,
                )
            else:
                enet_score = elasticnet_group_score_fast(
                    X_aligned,
                    y_aligned,
                    selected_groups=pearson_selected,
                    n_groups=n_groups,
                    group_size=group_size,
                    alpha=elastic_alpha,
                    l1_ratio=elastic_l1_ratio,
                )

            enet_selected = np.where(enet_score > 1e-8)[0]

            for g in enet_selected:
                selected_count[g] += 1

            all_pearson_scores.append(pearson_score)
            all_enet_scores.append(enet_score)

            valid_window_count += 1

            window_records.append({
                "segment_id": seg_id,
                "window_id": win_id,
                "start_time": win[time_col].iloc[0],
                "end_time": win[time_col].iloc[-1],
                "n_points": len(win),
                "n_points_aligned": len(y_aligned),
                "pearson_selected_groups": ",".join(map(str, pearson_selected.tolist())),
                "elasticnet_selected_groups": ",".join(map(str, enet_selected.tolist())),
            })

            pbar.set_postfix({
                "stage": "done",
                "valid": valid_window_count,
                "selected": len(enet_selected),
            })

            pbar.update(1)

    pbar.close()

    if valid_window_count == 0:
        raise RuntimeError(
            "没有生成任何有效窗口。请调小 --window_size、--min_points，或者检查 power 是否全是缺失。"
        )

    print(f"[9] 汇总所有窗口结果", flush=True)

    pearson_mean = np.mean(all_pearson_scores, axis=0)
    enet_mean = np.mean(all_enet_scores, axis=0)
    select_freq = selected_count / valid_window_count

    final_score = (
        0.4 * safe_minmax(select_freq)
        + 0.2 * safe_minmax(pearson_mean)
        + 0.4 * safe_minmax(enet_mean)
    )

    summary = pd.DataFrame({
        "group_id": np.arange(n_groups),
        "final_score": final_score,
        "select_freq": select_freq,
        "pearson_mean": pearson_mean,
        "elasticnet_mean": enet_mean,
    })

    summary = summary.sort_values("final_score", ascending=False).reset_index(drop=True)

    selected_summary = summary.head(final_top_groups).copy()

    group_to_cols = {
        g: covariate_cols[g * group_size:(g + 1) * group_size]
        for g in range(n_groups)
    }

    selected_summary["columns"] = selected_summary["group_id"].apply(
        lambda g: ",".join(group_to_cols[int(g)])
    )

    window_info = pd.DataFrame(window_records)

    summary_path = os.path.join(output_dir, "weather_group_selection_summary.csv")
    selected_path = os.path.join(output_dir, "selected_weather_groups.csv")
    window_path = os.path.join(output_dir, "window_selection_info.csv")
    selected_txt_path = os.path.join(output_dir, "selected_group_ids.txt")

    summary.to_csv(summary_path, index=False)
    selected_summary.to_csv(selected_path, index=False)
    window_info.to_csv(window_path, index=False)

    with open(selected_txt_path, "w", encoding="utf-8") as f:
        f.write(",".join(map(str, selected_summary["group_id"].tolist())))

    print(f"[10] 完成", flush=True)
    print(f"    有效窗口数: {valid_window_count}", flush=True)
    print(f"    最终选择 group 数: {final_top_groups}", flush=True)
    print("", flush=True)
    print("    输出文件:", flush=True)
    print(f"    - 总排名: {summary_path}", flush=True)
    print(f"    - 最终选择: {selected_path}", flush=True)
    print(f"    - 分窗详情: {window_path}", flush=True)
    print(f"    - group 到列名映射: {group_mapping_path}", flush=True)
    print(f"    - 选择的 group id: {selected_txt_path}", flush=True)
    print("", flush=True)
    print("    Top selected groups:", flush=True)
    print(
        selected_summary[[
            "group_id",
            "final_score",
            "select_freq",
            "pearson_mean",
            "elasticnet_mean",
        ]].to_string(index=False),
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./weather_covariate_selection_output")

    parser.add_argument("--time_col", type=str, default="time")
    parser.add_argument("--target_col", type=str, default="power")
    parser.add_argument("--ws_col", type=str, default="ws")

    parser.add_argument("--start_date", type=str, default="2026-02-01")

    parser.add_argument("--n_groups", type=int, default=78)
    parser.add_argument("--group_size", type=int, default=25)

    parser.add_argument("--window_size", type=str, default="7D")
    parser.add_argument("--stride", type=str, default="1D")

    parser.add_argument("--horizon", type=int, default=1)

    parser.add_argument("--pearson_top_groups", type=int, default=30)
    parser.add_argument("--final_top_groups", type=int, default=20)

    parser.add_argument("--min_points", type=int, default=128)
    parser.add_argument("--gap_factor", type=float, default=3.0)

    parser.add_argument("--use_cv", action="store_true")

    parser.add_argument("--elastic_alpha", type=float, default=0.01)
    parser.add_argument("--elastic_l1_ratio", type=float, default=0.7)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    window_size = parse_window_arg(args.window_size)
    stride = parse_window_arg(args.stride)

    run_selection(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        time_col=args.time_col,
        target_col=args.target_col,
        ws_col=args.ws_col,
        start_date=args.start_date,
        n_groups=args.n_groups,
        group_size=args.group_size,
        window_size=window_size,
        stride=stride,
        horizon=args.horizon,
        pearson_top_groups=args.pearson_top_groups,
        final_top_groups=args.final_top_groups,
        min_points=args.min_points,
        gap_factor=args.gap_factor,
        use_cv=args.use_cv,
        elastic_alpha=args.elastic_alpha,
        elastic_l1_ratio=args.elastic_l1_ratio,
    )