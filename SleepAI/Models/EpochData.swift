//
//  EpochData.swift
//  SleepAI
//
//  Created by Isaque da Silva Targino on 10/08/26.
//
//  Representa uma única época (30 s) do dataset sintético de uma noite de sono,
//  com todas as 33 features que o modelo CoreML SleepStageClassifier3Classes espera.

import Foundation
import CoreML

// MARK: - Model

struct EpochData: Identifiable {
    let id: Int                   // epoch index
    let timestamp: Date
    let trueStage: SleepStage?    // ground truth do dataset (apenas para comparação)

    // ── Heart-rate features ──────────────────────────────────────────────────
    let hrMean: Double
    let hrStd: Double
    let hrMin: Double
    let hrMax: Double
    let hrCount: Double

    // ── Motion features ──────────────────────────────────────────────────────
    let motionVmMean: Double
    let motionVmStd: Double
    let motionVmMax: Double
    let motionVmMin: Double
    let motionEnormMean: Double
    let motionEnormStd: Double
    let motionEnormMax: Double
    let motionXStd: Double
    let motionYStd: Double
    let motionZStd: Double
    let motionActiveCount: Double
    let motionSampleCount: Double
    let motionActiveRatio: Double

    // ── Derived night features ───────────────────────────────────────────────
    let hrDiffNightMean: Double
    let hrRatioNightMean: Double
    let hrDiffNightMin: Double

    // ── Rolling window features ──────────────────────────────────────────────
    let hrMeanRoll3: Double
    let hrMeanRoll5: Double
    let hrStdRoll5: Double
    let motionEnormMeanRoll3: Double
    let motionEnormMeanRoll5: Double
    let motionActiveRatioRoll5: Double

    // ── Lag features ─────────────────────────────────────────────────────────
    let hrMeanPrev1: Double
    let hrMeanNext1: Double
    let motionVmStdPrev1: Double
    let motionVmStdNext1: Double

    // ── Time features ────────────────────────────────────────────────────────
    let relativeTimeInNight: Double
    let timeFromStartMinutes: Double
}

// MARK: - CoreML Input Conversion

extension EpochData {
    /// Converte a época diretamente no input exato esperado pelo CoreML.
    func toCoreMLInput() -> SleepStageClassifier3ClassesInput {
        SleepStageClassifier3ClassesInput(
            hr_mean:                   hrMean,
            hr_std:                    hrStd,
            hr_min:                    hrMin,
            hr_max:                    hrMax,
            hr_count:                  hrCount,
            motion_vm_mean:            motionVmMean,
            motion_vm_std:             motionVmStd,
            motion_vm_max:             motionVmMax,
            motion_vm_min:             motionVmMin,
            motion_enorm_mean:         motionEnormMean,
            motion_enorm_std:          motionEnormStd,
            motion_enorm_max:          motionEnormMax,
            motion_x_std:              motionXStd,
            motion_y_std:              motionYStd,
            motion_z_std:              motionZStd,
            motion_active_count:       motionActiveCount,
            motion_sample_count:       motionSampleCount,
            motion_active_ratio:       motionActiveRatio,
            hr_diff_night_mean:        hrDiffNightMean,
            hr_ratio_night_mean:       hrRatioNightMean,
            hr_diff_night_min:         hrDiffNightMin,
            hr_mean_roll3:             hrMeanRoll3,
            hr_mean_roll5:             hrMeanRoll5,
            hr_std_roll5:              hrStdRoll5,
            motion_enorm_mean_roll3:   motionEnormMeanRoll3,
            motion_enorm_mean_roll5:   motionEnormMeanRoll5,
            motion_active_ratio_roll5: motionActiveRatioRoll5,
            hr_mean_prev1:             hrMeanPrev1,
            hr_mean_next1:             hrMeanNext1,
            motion_vm_std_prev1:       motionVmStdPrev1,
            motion_vm_std_next1:       motionVmStdNext1,
            relative_time_in_night:    relativeTimeInNight,
            time_from_start_minutes:   timeFromStartMinutes
        )
    }
}
