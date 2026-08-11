#!/usr/bin/env python3
"""
generate_night_mock.py

Gera um dataset sintético representando uma noite de sono, no formato que
seria produzido por agregação real de sensores do Apple Watch (HealthKit
para frequência cardíaca, CoreMotion para acelerômetro), por época de 30s.

Diferente da simulação por sliders (que estimava as features por fórmulas
arbitrárias a partir de 5 valores), aqui cada época é construída a partir de
AMOSTRAS BRUTAS (várias leituras de HR e ~1500 amostras de acelerômetro por
época), agregadas exatamente como o pipeline real faria — inclusive as
features de janela (rolling/prev/next), que aqui são calculadas de verdade
sobre a série temporal simulada, não copiadas do valor atual.

O rótulo de estágio (`true_stage`) é o ground truth sintético, usado só como
etiqueta a se comparar com a predição do modelo — não é uma feature de input.

Uso:
    python3 generate_night_mock.py --duration-hours 8 --seed 42 \
        --output night_mock.csv
"""

import argparse
import datetime as dt

import numpy as np
import pandas as pd

STAGES = ["Wake", "Light", "Heavy"]

# Parâmetros fisiológicos aproximados por estágio calibrados para o espaço de
# decisão do modelo CoreML pre-treinado (SleepStageClassifier3Classes.mlmodel).
HR_PARAMS = {
    "Wake": dict(mean=75.0, std=5.0),
    "Light": dict(mean=58.0, std=3.0),
    "Heavy": dict(mean=68.0, std=2.0),
}
MOTION_PARAMS = {
    "Wake": dict(active_ratio=0.25, noise_std=0.10, burst_mult=3.5),
    "Light": dict(active_ratio=0.03, noise_std=0.02, burst_mult=3.5),
    "Heavy": dict(active_ratio=0.001, noise_std=0.005, burst_mult=3.5),
}

HR_SAMPLES_PER_EPOCH_MEAN = 5  # ~1 leitura de HR a cada 6s
MOTION_HZ = 50                 # taxa do acelerômetro (Hz)
ACTIVE_THRESHOLD = 0.05        # enorm acima disso conta como "amostra ativa" (g)
STAGE_PERSISTENCE = 0.90       # prob. de permanecer no mesmo estágio a cada época


def stage_weights(phase: float) -> dict:
    """Distribuição de probabilidade dos estágios conforme o progresso da
    noite (0.0 = deitou, 1.0 = acordou), aproximando o ciclo ultradiano:
    mais sono profundo no início, mais leve/despertares perto do fim."""
    if phase < 0.08:
        return {"Wake": 0.60, "Light": 0.35, "Heavy": 0.05}
    if phase < 0.50:
        return {"Heavy": 0.45, "Light": 0.50, "Wake": 0.05}
    if phase < 0.85:
        return {"Heavy": 0.15, "Light": 0.70, "Wake": 0.15}
    return {"Heavy": 0.05, "Light": 0.55, "Wake": 0.40}


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


def simulate_epoch_hr(stage: str, rng: np.random.Generator) -> dict:
    params = HR_PARAMS[stage]
    # Variação real no número de amostras por época (4 a 6 leituras de HR)
    n_samples = int(rng.integers(4, 7))
    samples = rng.normal(params["mean"], params["std"], n_samples)
    samples = np.clip(samples, 35.0, 130.0)
    return dict(
        hr_mean=float(samples.mean()),
        hr_std=float(samples.std(ddof=0)),
        hr_min=float(samples.min()),
        hr_max=float(samples.max()),
        hr_count=float(n_samples),
    )


def simulate_epoch_motion(stage: str, epoch_seconds: int, rng: np.random.Generator) -> dict:
    params = MOTION_PARAMS[stage]
    n = MOTION_HZ * epoch_seconds

    x = rng.normal(0.0, params["noise_std"], n)
    y = rng.normal(0.0, params["noise_std"], n)
    z = 1.0 + rng.normal(0.0, params["noise_std"], n)  # gravidade ~1g no eixo z

    # Injeta rajadas de movimento em uma fração das amostras (active_ratio)
    active_mask = rng.random(n) < params["active_ratio"]
    burst_std = params["noise_std"] * params.get("burst_mult", 3.5)
    x[active_mask] += rng.normal(0.0, burst_std, active_mask.sum())
    y[active_mask] += rng.normal(0.0, burst_std, active_mask.sum())
    z[active_mask] += rng.normal(0.0, burst_std, active_mask.sum())

    vm = np.sqrt(x**2 + y**2 + z**2)
    enorm = np.abs(vm - 1.0)
    active_count = int((enorm > ACTIVE_THRESHOLD).sum())

    return dict(
        motion_vm_mean=float(vm.mean()),
        motion_vm_std=float(vm.std(ddof=0)),
        motion_vm_max=float(vm.max()),
        motion_vm_min=float(vm.min()),
        motion_enorm_mean=float(enorm.mean()),
        motion_enorm_std=float(enorm.std(ddof=0)),
        motion_enorm_max=float(enorm.max()),
        motion_x_std=float(x.std(ddof=0)),
        motion_y_std=float(y.std(ddof=0)),
        motion_z_std=float(z.std(ddof=0)),
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
    for i, stage in enumerate(stages):
        row = {"epoch": i, "true_stage": stage}
        row.update(simulate_epoch_hr(stage, rng))
        row.update(simulate_epoch_motion(stage, epoch_seconds, rng))
        rows.append(row)

    df = pd.DataFrame(rows)

    # ---- Features derivadas em relação à noite inteira (agregação real) ----
    night_mean_hr = df["hr_mean"].mean()
    night_min_hr = df["hr_mean"].min()
    df["hr_diff_night_mean"] = df["hr_mean"] - night_mean_hr
    df["hr_ratio_night_mean"] = df["hr_mean"] / night_mean_hr
    df["hr_diff_night_min"] = df["hr_mean"] - night_min_hr

    # ---- Janelas móveis reais (não copiadas do valor atual) ----
    df["hr_mean_roll3"] = df["hr_mean"].rolling(3, min_periods=1).mean()
    df["hr_mean_roll5"] = df["hr_mean"].rolling(5, min_periods=1).mean()
    df["hr_std_roll5"] = df["hr_std"].rolling(5, min_periods=1).mean()
    df["motion_enorm_mean_roll3"] = df["motion_enorm_mean"].rolling(3, min_periods=1).mean()
    df["motion_enorm_mean_roll5"] = df["motion_enorm_mean"].rolling(5, min_periods=1).mean()
    df["motion_active_ratio_roll5"] = df["motion_active_ratio"].rolling(5, min_periods=1).mean()

    # ---- Lag real (época anterior/seguinte), com preenchimento nas pontas ----
    df["hr_mean_prev1"] = df["hr_mean"].shift(1).bfill()
    df["hr_mean_next1"] = df["hr_mean"].shift(-1).ffill()
    df["motion_vm_std_prev1"] = df["motion_vm_std"].shift(1).bfill()
    df["motion_vm_std_next1"] = df["motion_vm_std"].shift(-1).ffill()

    # ---- Tempo ----
    df["relative_time_in_night"] = df["epoch"] / max(n_epochs - 1, 1)
    df["time_from_start_minutes"] = df["epoch"] * (epoch_seconds / 60.0)
    df["timestamp"] = [start_time + dt.timedelta(seconds=i * epoch_seconds) for i in df["epoch"]]

    # Ordem igual à do SleepStageClassifier3ClassesInput no app, com
    # timestamp/true_stage à parte (não são features de input do modelo).
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
