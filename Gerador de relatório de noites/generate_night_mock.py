#!/usr/bin/env python3
"""
generate_night_mock.py

Gera um dataset sintético representando uma noite de sono, no formato que
seria produzido por agregação real de sensores do Apple Watch (HealthKit
para frequência cardíaca, CoreMotion para acelerômetro), por época de 30s.

As métricas e distribuições foram calibradas estatisticamente com base no
dataset real de treinamento (sleep_stage_train.csv) para garantir que o modelo
CoreML (SleepClassifier4Classes) atinja a acurácia esperada (~55% - 60%).

Uso:
    python3 generate_night_mock.py --duration-hours 8 --seed 42 \
        --output night_mock.csv
"""

import argparse
import datetime as dt
import numpy as np
import pandas as pd

STAGES = ["Wake", "Light", "N3", "REM"]

# Parâmetros fisiológicos calibrados estatisticamente com base no dataset real de treino
HR_PARAMS = {
    "Wake":  dict(mean=69.85, std=1.87),
    "Light": dict(mean=64.49, std=1.41),
    "N3":    dict(mean=65.99, std=1.11),
    "REM":   dict(mean=65.44, std=1.81),
}

MOTION_PARAMS = {
    "Wake":  dict(active_ratio=0.0508, std_x=0.0609, std_y=0.0783, std_z=0.0746),
    "Light": dict(active_ratio=0.0061, std_x=0.0117, std_y=0.0178, std_z=0.0170),
    "N3":    dict(active_ratio=0.0040, std_x=0.0084, std_y=0.0124, std_z=0.0119),
    "REM":   dict(active_ratio=0.0083, std_x=0.0159, std_y=0.0241, std_z=0.0237),
}

HR_SAMPLES_PER_EPOCH = 5       # ~1 leitura de HR a cada 6s
MOTION_HZ = 50                 # taxa do acelerômetro
ACTIVE_THRESHOLD = 0.05        # enorm acima disso conta como "amostra ativa"
STAGE_PERSISTENCE = 0.90       # prob. de permanecer no mesmo estágio a cada época


def stage_weights(phase: float) -> dict:
    """Distribuição de probabilidade dos estágios conforme o progresso da
    noite (0.0 = deitou, 1.0 = acordou), aproximando o ciclo ultradiano:
    mais N3 no início, mais REM/despertares perto do fim."""
    if phase < 0.08:
        return {"Wake": 0.60, "Light": 0.35, "N3": 0.05, "REM": 0.00}
    if phase < 0.50:
        return {"N3": 0.40, "Light": 0.45, "REM": 0.10, "Wake": 0.05}
    if phase < 0.85:
        return {"N3": 0.10, "Light": 0.55, "REM": 0.25, "Wake": 0.10}
    return {"N3": 0.02, "Light": 0.43, "REM": 0.20, "Wake": 0.35}


def simulate_stage_timeline(n_epochs: int, rng: np.random.Generator) -> list:
    """Gera a sequência de estágios com persistência (evita alternância
    época-a-época irreal) mas ainda variando conforme a fase da noite."""
    timeline = []
    current = "Wake"
    for i in range(n_epochs):
        phase = i / max(n_epochs - 1, 1)
        weights = stage_weights(phase)
        if timeline and rng.random() < STAGE_PERSISTENCE:
            stage = current
        else:
            stages, probs = zip(*weights.items())
            stage = rng.choice(stages, p=probs)
        timeline.append(stage)
        current = stage
    return timeline


def simulate_epoch_motion(stage: str, epoch_seconds: int, rng: np.random.Generator) -> dict:
    params = MOTION_PARAMS[stage]
    n = MOTION_HZ * epoch_seconds

    x = rng.normal(0.0, params["std_x"], n)
    y = rng.normal(0.0, params["std_y"], n)
    z = 1.0 + rng.normal(0.0, params["std_z"], n)

    if params["active_ratio"] > 0:
        active_mask = rng.random(n) < params["active_ratio"]
        burst_mult = 3.0
        x[active_mask] += rng.normal(0.0, params["std_x"] * burst_mult, active_mask.sum())
        y[active_mask] += rng.normal(0.0, params["std_y"] * burst_mult, active_mask.sum())
        z[active_mask] += rng.normal(0.0, params["std_z"] * burst_mult, active_mask.sum())

    vm = np.sqrt(x**2 + y**2 + z**2)
    enorm = np.abs(vm - 1.0)
    active_count = int((enorm > ACTIVE_THRESHOLD).sum())

    return dict(
        motion_vm_mean=float(vm.mean()),
        motion_vm_std=float(vm.std(ddof=1)),
        motion_vm_max=float(vm.max()),
        motion_vm_min=float(vm.min()),
        motion_enorm_mean=float(enorm.mean()),
        motion_enorm_std=float(enorm.std(ddof=1)),
        motion_enorm_max=float(enorm.max()),
        motion_x_std=float(x.std(ddof=1)),
        motion_y_std=float(y.std(ddof=1)),
        motion_z_std=float(z.std(ddof=1)),
        motion_active_count=float(active_count),
        motion_sample_count=float(n),
        motion_active_ratio=active_count / n,
    )


def build_night_dataset(
    duration_hours: float,
    epoch_seconds: int,
    start_time: dt.datetime,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_epochs = int(duration_hours * 3600 / epoch_seconds)

    stages = simulate_stage_timeline(n_epochs, rng)

    rows = []
    prev_hr_epoch_mean = 65.0

    for i, stage in enumerate(stages):
        params_hr = HR_PARAMS[stage]

        # Dinâmica autonômica entre épocas (estabilidade no N3, variabilidade no REM/Wake)
        if stage == "N3":
            target_hr = params_hr["mean"] + rng.normal(0.0, 0.3)
            epoch_hr_mean = 0.90 * prev_hr_epoch_mean + 0.10 * target_hr
            epoch_hr_std = max(0.4, rng.normal(params_hr["std"], 0.15))
        elif stage == "REM":
            target_hr = params_hr["mean"] + rng.normal(0.0, 4.0)
            epoch_hr_mean = target_hr
            epoch_hr_std = max(0.9, rng.normal(params_hr["std"], 0.4))
        elif stage == "Wake":
            target_hr = params_hr["mean"] + rng.normal(0.0, 3.0)
            epoch_hr_mean = target_hr
            epoch_hr_std = max(0.9, rng.normal(params_hr["std"], 0.5))
        else:  # Light
            target_hr = params_hr["mean"] + rng.normal(0.0, 1.5)
            epoch_hr_mean = 0.70 * prev_hr_epoch_mean + 0.30 * target_hr
            epoch_hr_std = max(0.5, rng.normal(params_hr["std"], 0.25))

        prev_hr_epoch_mean = epoch_hr_mean

        hr_samples = rng.normal(epoch_hr_mean, epoch_hr_std, HR_SAMPLES_PER_EPOCH)
        hr_samples = np.clip(hr_samples, 40.0, 120.0)

        row = {
            "epoch": i,
            "true_stage": stage,
            "hr_mean": float(hr_samples.mean()),
            "hr_std": float(hr_samples.std(ddof=1)),
            "hr_min": float(hr_samples.min()),
            "hr_max": float(hr_samples.max()),
            "hr_count": float(len(hr_samples)),
        }
        row.update(simulate_epoch_motion(stage, epoch_seconds, rng))
        rows.append(row)

    df = pd.DataFrame(rows)

    # ---- Features derivadas em relação à noite inteira ----
    night_mean_hr = df["hr_mean"].mean()
    night_min_hr = df["hr_mean"].min()
    df["hr_diff_night_mean"] = df["hr_mean"] - night_mean_hr
    df["hr_ratio_night_mean"] = df["hr_mean"] / (night_mean_hr if night_mean_hr > 0 else 1.0)
    df["hr_diff_night_min"] = df["hr_mean"] - night_min_hr

    # ---- Janelas móveis ----
    df["hr_mean_roll3"] = df["hr_mean"].rolling(window=3, center=True, min_periods=1).mean()
    df["hr_mean_roll5"] = df["hr_mean"].rolling(window=5, center=True, min_periods=1).mean()
    df["hr_std_roll5"] = df["hr_mean"].rolling(window=5, center=True, min_periods=1).std().fillna(0.0)
    df["motion_enorm_mean_roll3"] = df["motion_enorm_mean"].rolling(window=3, center=True, min_periods=1).mean()
    df["motion_enorm_mean_roll5"] = df["motion_enorm_mean"].rolling(window=5, center=True, min_periods=1).mean()
    df["motion_active_ratio_roll5"] = df["motion_active_ratio"].rolling(window=5, center=True, min_periods=1).mean()

    # ---- Lag (época anterior/seguinte) ----
    df["hr_mean_prev1"] = df["hr_mean"].shift(1).bfill()
    df["hr_mean_next1"] = df["hr_mean"].shift(-1).ffill()
    df["motion_vm_std_prev1"] = df["motion_vm_std"].shift(1).fillna(0.0)
    df["motion_vm_std_next1"] = df["motion_vm_std"].shift(-1).fillna(0.0)

    # ---- Tempo ----
    df["relative_time_in_night"] = df["epoch"] / float(n_epochs)
    df["time_from_start_minutes"] = df["epoch"] * (epoch_seconds / 60.0)
    df["timestamp"] = [start_time + dt.timedelta(seconds=i * epoch_seconds) for i in df["epoch"]]

    feature_columns = [
        "hr_mean", "hr_std", "hr_min", "hr_max", "hr_count",
        "motion_vm_mean", "motion_vm_std", "motion_vm_max", "motion_vm_min",
        "motion_enorm_mean", "motion_enorm_std", "motion_enorm_max",
        "motion_x_std", "motion_y_std", "motion_z_std",
        "motion_active_count", "motion_sample_count", "motion_active_ratio",
        "hr_diff_night_mean", "hr_ratio_night_mean", "hr_diff_night_min",
        "hr_mean_roll3", "hr_mean_roll5", "hr_std_roll5",
        "motion_enorm_mean_roll3", "motion_enorm_mean_roll5", "motion_active_ratio_roll5",
        "hr_mean_prev1", "hr_mean_next1", "motion_vm_std_prev1", "motion_vm_std_next1",
        "relative_time_in_night", "time_from_start_minutes",
    ]
    return df[["timestamp", "epoch", "true_stage"] + feature_columns]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-hours", type=float, default=8.0)
    parser.add_argument("--epoch-seconds", type=int, default=30)
    parser.add_argument("--start-time", type=str, default="2026-08-10T23:00:00",
                        help="ISO 8601, ex: 2026-08-10T23:00:00")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="night_mock.csv")
    args = parser.parse_args()

    start_time = dt.datetime.fromisoformat(args.start_time)
    df = build_night_dataset(args.duration_hours, args.epoch_seconds, start_time, args.seed)
    df.to_csv(args.output, index=False)

    print(f"{len(df)} épocas geradas ({args.duration_hours}h, {args.epoch_seconds}s/época)")
    print(f"Distribuição de estágios: {df['true_stage'].value_counts().to_dict()}")
    print(f"Salvo em: {args.output}")


if __name__ == "__main__":
    main()
