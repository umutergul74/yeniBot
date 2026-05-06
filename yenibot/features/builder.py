from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Iterable

import numpy as np
import pandas as pd

from yenibot.features.wavelet import causal_wavelet_denoise

RAW_COLUMNS = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "num_trades",
    "taker_buy_base_vol",
    "taker_buy_quote_vol",
    "ignore",
}

METADATA_COLUMNS = {
    "4h_source_timestamp",
    "4h_available_timestamp",
}

LABEL_COLUMNS = {
    "label",
    "fwd_return_10h",
    "tb_return",
    "hit_type",
    "exit_timestamp",
    "exit_bar",
}


@dataclass(frozen=True)
class FeatureResult:
    frame: pd.DataFrame
    feature_columns: list[str]


def _safe_divide(numerator: pd.Series, denominator: pd.Series, default: float = 0.0) -> pd.Series:
    result = numerator.astype(float) / denominator.replace(0, np.nan).astype(float)
    return result.replace([np.inf, -np.inf], np.nan).fillna(default)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    return ((series - mean) / std.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _rolling_rank_score(series: pd.Series, window: int) -> pd.Series:
    def rank_last(values: np.ndarray) -> float:
        if np.isnan(values[-1]):
            return np.nan
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            return np.nan
        current = valid[-1]
        rank = ((valid < current).sum() + 0.5 * (valid == current).sum()) / len(valid)
        return float(2.0 * rank - 1.0)

    return series.rolling(window, min_periods=window).apply(rank_last, raw=True)


def _log_return(series: pd.Series) -> pd.Series:
    positive = series.where(series > 0)
    return np.log(positive / positive.shift(1)).replace([np.inf, -np.inf], np.nan)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x = x - x.mean()
    denom = float(np.dot(x, x))

    def slope(values: np.ndarray) -> float:
        y = values.astype(float)
        y = y - y.mean()
        return float(np.dot(x, y) / denom)

    return series.rolling(window, min_periods=window).apply(slope, raw=True)


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period, min_periods=period).mean()


def _adx(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    alpha = 1.0 / period
    atr = true_range.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()


def _config_get(config: object, path: Iterable[str], default: object) -> object:
    current = config
    for key in path:
        if isinstance(current, dict):
            if key not in current:
                return default
            current = current[key]
        else:
            if not hasattr(current, key):
                return default
            current = getattr(current, key)
    return current


def compute_bar_features(frame: pd.DataFrame, config: object) -> FeatureResult:
    """Compute causal microstructure features for a single timeframe."""

    df = frame.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    wavelet_enabled = bool(_config_get(config, ["features", "wavelet", "enabled"], True))
    if wavelet_enabled:
        wavelet_cfg = _config_get(config, ["features", "wavelet"], {})
        df["close_denoised"] = causal_wavelet_denoise(
            df["close"],
            window=int(_config_get(wavelet_cfg, ["window"], 256)),
            wavelet=str(_config_get(wavelet_cfg, ["wavelet"], "db4")),
            level=int(_config_get(wavelet_cfg, ["level"], 2)),
            threshold_scale=float(_config_get(wavelet_cfg, ["threshold_scale"], 0.5)),
        )
        df["volume_denoised"] = causal_wavelet_denoise(
            df["volume"],
            window=int(_config_get(wavelet_cfg, ["window"], 256)),
            wavelet=str(_config_get(wavelet_cfg, ["wavelet"], "db4")),
            level=int(_config_get(wavelet_cfg, ["level"], 2)),
            threshold_scale=float(_config_get(wavelet_cfg, ["threshold_scale"], 0.5)),
        )

    cvd_window = int(_config_get(config, ["features", "order_flow", "cvd_zscore_window"], 100))
    cvd_rate_window = int(_config_get(config, ["features", "order_flow", "cvd_rate_window"], 100))
    imbalance_span = int(_config_get(config, ["features", "order_flow", "imbalance_ema_span"], 14))
    vpt_window = int(_config_get(config, ["features", "whale", "vpt_zscore_window"], 100))
    whale_threshold = float(_config_get(config, ["features", "whale", "whale_zscore_threshold"], 2.0))
    whale_buy_ratio = float(_config_get(config, ["features", "whale", "whale_buy_ratio"], 0.55))
    whale_sell_ratio = float(_config_get(config, ["features", "whale", "whale_sell_ratio"], 0.45))
    large_trade_window = int(_config_get(config, ["features", "whale", "large_trade_window"], 100))
    realized_vol_window = int(_config_get(config, ["features", "structure", "realized_vol_window"], 14))
    gk_window = int(_config_get(config, ["features", "structure", "gk_vol_window"], 14))
    atr_period = int(_config_get(config, ["features", "structure", "atr_period"], 14))
    adx_period = int(_config_get(config, ["features", "structure", "adx_period"], 14))
    vwap_window = int(_config_get(config, ["features", "structure", "vwap_window"], 24))
    stationarity_cfg = _config_get(config, ["features", "stationarity"], {})
    stationarity_enabled = bool(_config_get(stationarity_cfg, ["enabled"], True))
    stationarity_window = int(_config_get(stationarity_cfg, ["normalization_window"], cvd_window))

    df["taker_buy_ratio"] = _safe_divide(df["taker_buy_base_vol"], df["volume"], default=0.5).clip(0.0, 1.0)
    df["taker_sell_ratio"] = 1.0 - df["taker_buy_ratio"]
    taker_sell_base = df["volume"] - df["taker_buy_base_vol"]
    df["true_cvd_delta"] = df["taker_buy_base_vol"] - taker_sell_base
    df["true_cvd_zscore"] = _rolling_zscore(df["true_cvd_delta"], cvd_window)
    cumulative_cvd = df["true_cvd_delta"].cumsum()
    df["cvd_cumulative_rate"] = _rolling_slope(cumulative_cvd, cvd_rate_window)
    df["buy_sell_imbalance_ema"] = (df["taker_buy_ratio"] - 0.5).ewm(
        span=imbalance_span,
        adjust=False,
        min_periods=imbalance_span,
    ).mean()

    df["vol_per_trade"] = _safe_divide(df["volume"], df["num_trades"])
    df["vpt_zscore"] = _rolling_zscore(df["vol_per_trade"], vpt_window)
    df["whale_buy_flag"] = ((df["vpt_zscore"] > whale_threshold) & (df["taker_buy_ratio"] > whale_buy_ratio)).astype(float)
    df["whale_sell_flag"] = ((df["vpt_zscore"] > whale_threshold) & (df["taker_buy_ratio"] < whale_sell_ratio)).astype(float)
    large_trade_volume = df["volume"].where(df["vpt_zscore"] > whale_threshold, 0.0)
    df["large_trade_ratio"] = _safe_divide(
        large_trade_volume.rolling(large_trade_window, min_periods=large_trade_window).sum(),
        df["volume"].rolling(large_trade_window, min_periods=large_trade_window).sum(),
    )

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["realized_vol_14"] = df["log_return"].rolling(realized_vol_window, min_periods=realized_vol_window).std(ddof=0)
    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])
    gk_var = 0.5 * log_hl.pow(2) - (2.0 * np.log(2.0) - 1.0) * log_co.pow(2)
    df["gk_vol_14"] = np.sqrt(gk_var.clip(lower=0).rolling(gk_window, min_periods=gk_window).mean())
    df["atr_14"] = _atr(df, atr_period)
    df["adx_14"] = _adx(df, adx_period)
    rolling_vwap = _safe_divide(
        df["quote_volume"].rolling(vwap_window, min_periods=vwap_window).sum(),
        df["volume"].rolling(vwap_window, min_periods=vwap_window).sum(),
        default=np.nan,
    )
    df["vwap_dist_atr"] = (df["close"] - rolling_vwap) / df["atr_14"].replace(0, np.nan)

    if stationarity_enabled:
        df = _add_stationary_features(df, config, stationarity_window)

    feature_columns = select_feature_columns(df)
    return FeatureResult(df, feature_columns)


def _add_stationary_features(df: pd.DataFrame, config: object, window: int) -> pd.DataFrame:
    if window <= 1:
        raise ValueError("features.stationarity.normalization_window must be greater than 1")

    rolling_volume = df["volume"].rolling(window, min_periods=window).mean()
    df["volume_log_zscore"] = _rolling_zscore(np.log1p(df["volume"].clip(lower=0)), window)
    df["true_cvd_delta_norm"] = _safe_divide(df["true_cvd_delta"], rolling_volume, default=np.nan)
    df["cvd_cumulative_rate_norm"] = _safe_divide(df["cvd_cumulative_rate"], rolling_volume, default=np.nan)
    df["vol_per_trade_log_zscore"] = _rolling_zscore(np.log1p(df["vol_per_trade"].clip(lower=0)), window)
    df["atr_14_pct"] = _safe_divide(df["atr_14"], df["close"], default=np.nan)

    if "close_denoised" in df.columns:
        df["close_denoised_log_return"] = _log_return(df["close_denoised"])
    if "volume_denoised" in df.columns:
        volume_denoised = df["volume_denoised"].clip(lower=0)
        df["volume_denoised_log_zscore"] = _rolling_zscore(np.log1p(volume_denoised), window)

    df = _add_structure_stability_features(df, config, window)
    return _add_order_flow_v2_features(df, config, window)


def _add_structure_stability_features(df: pd.DataFrame, config: object, default_window: int) -> pd.DataFrame:
    cfg = _config_get(config, ["features", "structure_stability"], {})
    if not bool(_config_get(cfg, ["enabled"], False)):
        return df

    stable_window = int(_config_get(cfg, ["stable_window"], default_window))
    stable_clip_abs = float(_config_get(cfg, ["stable_clip_abs"], 5.0))
    stable_transforms = set(_config_get(cfg, ["stable_transforms"], ["zscore", "rank"]) or [])
    source_columns = list(
        _config_get(
            cfg,
            ["source_columns"],
            [
                "log_return",
                "close_denoised_log_return",
                "realized_vol_14",
                "gk_vol_14",
                "atr_14_pct",
                "adx_14",
                "vwap_dist_atr",
                "volume_log_zscore",
                "volume_denoised_log_zscore",
            ],
        )
        or []
    )
    stable_scores = _stable_rolling_score_frame(
        df,
        [str(column) for column in source_columns],
        stable_window,
        stable_clip_abs,
        stable_transforms,
        error_context="features.structure_stability.stable_window",
        missing_ok=True,
    )
    if stable_scores.empty:
        return df
    return pd.concat([df, stable_scores], axis=1)


def _add_order_flow_v2_features(df: pd.DataFrame, config: object, default_window: int) -> pd.DataFrame:
    cfg = _config_get(config, ["features", "order_flow_v2"], {})
    if not bool(_config_get(cfg, ["enabled"], False)):
        return df

    ratio_window = int(_config_get(cfg, ["ratio_zscore_window"], default_window))
    ratio_delta_periods = int(_config_get(cfg, ["ratio_delta_periods"], 1))
    slope_window = int(_config_get(cfg, ["imbalance_slope_window"], 24))
    pressure_windows = list(_config_get(cfg, ["pressure_windows"], [3, 6, 12, 24]) or [])
    efficiency_epsilon = float(_config_get(cfg, ["efficiency_epsilon"], 1e-3))
    stable_window = int(_config_get(cfg, ["stable_window"], default_window))
    stable_clip_abs = float(_config_get(cfg, ["stable_clip_abs"], 5.0))
    stable_transforms = set(_config_get(cfg, ["stable_transforms"], ["zscore", "rank"]) or [])

    taker_imbalance = (df["taker_buy_ratio"] - df["taker_sell_ratio"]).clip(-1.0, 1.0)
    large_trade_intensity = (df["vpt_zscore"] - df["vpt_zscore"].rolling(default_window, min_periods=default_window).median()).clip(lower=0.0)
    signed_large_trade_pressure = large_trade_intensity * taker_imbalance
    normalized_return = _safe_divide(
        df["log_return"],
        df["realized_vol_14"].replace(0, np.nan),
        default=0.0,
    ).clip(-10.0, 10.0)
    abs_cvd_pressure = df["true_cvd_delta_norm"].abs()

    df["taker_imbalance"] = taker_imbalance
    df["taker_buy_ratio_zscore"] = _rolling_zscore(df["taker_buy_ratio"], ratio_window)
    df["taker_buy_ratio_delta"] = df["taker_buy_ratio"].diff(ratio_delta_periods)
    df["taker_imbalance_slope"] = _rolling_slope(taker_imbalance, slope_window)
    df["signed_large_trade_pressure"] = signed_large_trade_pressure
    df["orderflow_efficiency"] = _safe_divide(
        normalized_return,
        abs_cvd_pressure + efficiency_epsilon,
        default=0.0,
    ).clip(-10.0, 10.0)
    df["absorption_pressure"] = (-normalized_return * taker_imbalance).clip(-10.0, 10.0)
    df["cvd_price_divergence"] = (df["true_cvd_delta_norm"] - normalized_return).clip(-10.0, 10.0)
    stable_frames = [
        _stable_rolling_score_frame(
            df,
            [
                "signed_large_trade_pressure",
                "orderflow_efficiency",
                "absorption_pressure",
                "cvd_price_divergence",
            ],
            stable_window,
            stable_clip_abs,
            stable_transforms,
            error_context="features.order_flow_v2.stable_window",
        )
    ]

    for item in pressure_windows:
        window = int(item)
        if window <= 1:
            raise ValueError("features.order_flow_v2.pressure_windows values must be greater than 1")
        df[f"taker_imbalance_mean_{window}"] = taker_imbalance.rolling(window, min_periods=window).mean()
        df[f"cvd_pressure_{window}"] = df["true_cvd_delta_norm"].rolling(window, min_periods=window).sum()
        df[f"large_trade_pressure_{window}"] = signed_large_trade_pressure.rolling(window, min_periods=window).mean()
        df[f"absorption_pressure_{window}"] = df["absorption_pressure"].rolling(window, min_periods=window).mean()
        df[f"cvd_price_divergence_{window}"] = df["cvd_price_divergence"].rolling(window, min_periods=window).mean()
        stable_frames.append(
            _stable_rolling_score_frame(
                df,
                [
                    f"cvd_pressure_{window}",
                    f"large_trade_pressure_{window}",
                    f"absorption_pressure_{window}",
                    f"cvd_price_divergence_{window}",
                ],
                stable_window,
                stable_clip_abs,
                stable_transforms,
                error_context="features.order_flow_v2.stable_window",
            )
        )
    stable_frames = [frame for frame in stable_frames if not frame.empty]
    if stable_frames:
        return pd.concat([df, *stable_frames], axis=1)
    return df


def _stable_rolling_score_frame(
    df: pd.DataFrame,
    columns: list[str],
    window: int,
    clip_abs: float,
    transforms: set[str],
    *,
    error_context: str,
    missing_ok: bool = False,
) -> None:
    if window <= 1:
        raise ValueError(f"{error_context} must be greater than 1")
    additions: dict[str, pd.Series] = {}
    for column in columns:
        if column not in df.columns:
            if missing_ok:
                continue
            raise KeyError(f"Stable source column is missing: {column}")
        series = df[column].replace([np.inf, -np.inf], np.nan)
        if "zscore" in transforms:
            additions[f"{column}_stable_zscore"] = _rolling_zscore(series, window).clip(-clip_abs, clip_abs)
        if "rank" in transforms:
            additions[f"{column}_stable_rank"] = _rolling_rank_score(series, window)
    return pd.DataFrame(additions, index=df.index)


def select_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = RAW_COLUMNS | METADATA_COLUMNS | LABEL_COLUMNS
    excluded_prefixes = ("pred_", "regime_", "fold")
    columns: list[str] = []
    for column in frame.columns:
        if column in excluded:
            continue
        if any(column.startswith(prefix) for prefix in excluded_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    return sorted(columns)


def filter_feature_columns(feature_columns: list[str], config: object) -> list[str]:
    exclude_columns = set(_config_get(config, ["features", "exclude_columns"], []) or [])
    exclude_patterns = list(_config_get(config, ["features", "exclude_patterns"], []) or [])
    profile = resolve_feature_profile(config)
    include_patterns = list(profile.get("include_patterns", []) or [])
    profile_exclude_patterns = list(profile.get("exclude_patterns", []) or [])
    stationarity_cfg = _config_get(config, ["features", "stationarity"], {})
    if bool(_config_get(stationarity_cfg, ["exclude_nonstationary"], False)):
        exclude_patterns.extend(list(_config_get(stationarity_cfg, ["exclude_patterns"], []) or []))
    exclude_patterns.extend(profile_exclude_patterns)
    filtered = []
    raw_order_flow_v2_columns = raw_order_flow_v2_model_exclusions(config)
    for column in feature_columns:
        if column in exclude_columns:
            continue
        if column in raw_order_flow_v2_columns:
            continue
        if include_patterns and not any(fnmatch(column, pattern) for pattern in include_patterns):
            continue
        if any(fnmatch(column, pattern) for pattern in exclude_patterns):
            continue
        filtered.append(column)
    if not filtered:
        raise ValueError("Feature filtering removed every feature column")
    return filtered


def feature_availability_columns(feature_columns: list[str], config: object) -> list[str]:
    """Columns that must be finite before saving the feature matrix.

    The model profile controls trainable inputs, while HMM and labeling need a
    few non-model columns later. Inactive experimental features must not change
    the row universe simply because their warmup window is longer.
    """

    available = set(feature_columns)
    required = list(filter_feature_columns(feature_columns, config))
    extra_columns = list(_config_get(config, ["hmm", "features"], []) or [])
    atr_column = str(_config_get(config, ["labeling", "atr_column"], "atr_14"))
    extra_columns.append(atr_column)

    missing = [column for column in extra_columns if column not in available]
    if missing:
        raise ValueError(f"Missing required non-model feature columns: {missing}")
    for column in extra_columns:
        if column not in required:
            required.append(column)
    return required


def resolve_feature_profile(config: object) -> dict[str, object]:
    active = _config_get(config, ["features", "active_profile"], None)
    if not active:
        return {"name": None, "include_patterns": [], "exclude_patterns": []}

    profiles = _config_get(config, ["features", "profiles"], {}) or {}
    if not isinstance(profiles, dict) or active not in profiles:
        raise ValueError(f"Unknown features.active_profile: {active}")

    def load_profile(name: str, seen: set[str] | None = None) -> dict[str, object]:
        seen = set() if seen is None else seen
        if name in seen:
            raise ValueError(f"Cyclic feature profile inheritance detected at {name}")
        seen.add(name)
        current = profiles[name]
        if not isinstance(current, dict):
            raise ValueError(f"Feature profile must be a mapping: {name}")
        parent_name = current.get("inherit")
        parent = load_profile(str(parent_name), seen) if parent_name else {"include_patterns": [], "exclude_patterns": []}
        include_patterns = _dedupe_patterns(
            list(parent.get("include_patterns", []) or []) + list(current.get("include_patterns", []) or [])
        )
        if "exclude_patterns" in current:
            exclude_patterns = list(current.get("exclude_patterns", []) or [])
        else:
            exclude_patterns = list(parent.get("exclude_patterns", []) or [])
        exclude_patterns.extend(list(current.get("append_exclude_patterns", []) or []))
        return {
            "name": name,
            "description": current.get("description", ""),
            "include_patterns": include_patterns,
            "exclude_patterns": _dedupe_patterns(exclude_patterns),
        }

    return load_profile(str(active))


def _dedupe_patterns(patterns: list[object]) -> list[str]:
    deduped: list[str] = []
    for pattern in patterns:
        text = str(pattern)
        if text not in deduped:
            deduped.append(text)
    return deduped


def raw_order_flow_v2_model_exclusions(config: object) -> set[str]:
    cfg = _config_get(config, ["features", "order_flow_v2"], {})
    if not bool(_config_get(cfg, ["enabled"], False)):
        return set()
    if not bool(_config_get(cfg, ["stable_only"], False)):
        return set()
    pressure_windows = [int(item) for item in (list(_config_get(cfg, ["pressure_windows"], [3, 6, 12, 24]) or []))]
    base = {
        "signed_large_trade_pressure",
        "orderflow_efficiency",
        "absorption_pressure",
        "cvd_price_divergence",
    }
    for window in pressure_windows:
        base.update(
            {
                f"cvd_pressure_{window}",
                f"large_trade_pressure_{window}",
                f"absorption_pressure_{window}",
                f"cvd_price_divergence_{window}",
            }
        )
    return base | {f"4h_{column}" for column in base}


def build_feature_matrix(primary_frame: pd.DataFrame, htf_frame: pd.DataFrame, config: object) -> FeatureResult:
    """Build 1H features plus correctly delayed 4H features."""

    primary = compute_bar_features(primary_frame, config).frame
    htf_result = compute_bar_features(htf_frame, config)
    htf = htf_result.frame

    shift_hours = int(_config_get(config, ["features", "mtf", "shift_hours"], 4))
    htf_features = htf[["timestamp", *htf_result.feature_columns]].copy()
    htf_features["4h_source_timestamp"] = htf_features["timestamp"]
    htf_features["timestamp"] = htf_features["timestamp"] + pd.Timedelta(hours=shift_hours)
    htf_features["4h_available_timestamp"] = htf_features["timestamp"]
    htf_features = htf_features.rename(
        columns={column: f"4h_{column}" for column in htf_result.feature_columns}
    )

    merged = pd.merge_asof(
        primary.sort_values("timestamp"),
        htf_features.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )

    warmup_rows = int(_config_get(config, ["features", "warmup_rows"], 300))
    if warmup_rows > 0:
        merged = merged.iloc[warmup_rows:].copy()
    merged = merged.ffill()
    feature_columns = select_feature_columns(merged)
    model_feature_columns = filter_feature_columns(feature_columns, config)
    availability_columns = feature_availability_columns(feature_columns, config)
    merged = merged.dropna(subset=availability_columns).reset_index(drop=True)
    if merged[availability_columns].isna().any().any():
        bad = merged[availability_columns].columns[merged[availability_columns].isna().any()].tolist()
        raise ValueError(f"Feature matrix contains NaNs after warmup/fill: {bad}")
    return FeatureResult(merged, model_feature_columns)
