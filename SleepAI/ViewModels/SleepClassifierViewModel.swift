//
//  SleepClassifierViewModel.swift
//  SleepAI
//
//  Created by Isaque da Silva Targino on 09/08/26.
//

import Foundation
import CoreML
import Combine

@MainActor
final class SleepClassifierViewModel: ObservableObject {
    @Published var heartRate: Double = 62.0
    @Published var hrStd: Double = 1.5
    @Published var motionActivity: Double = 0.02
    @Published var activeRatio: Double = 0.05
    @Published var nightProgress: Double = 0.4
    
    // Mark: - Resultados da predição do modelo CoreML
    @Published private(set) var predictedStage: SleepStage = .light
    @Published private(set) var probabilities: [SleepStage: Double] = [
        .heavy: 0.15,
        .light: 0.80,
        .wake: 0.05
    ]
    
    var confidence: Double {
        probabilities[predictedStage] ?? 0
    }
    
    // Mark: - Inferência Direta via CoreML
    func predictSleepStage() {
        do {
            let model = try SleepStageClassifier3Classes(configuration: MLModelConfiguration())
            
            // Mock de 15 features que não puderam ser medidas em um Apple Watch real (fora do escopo da POC)
            let nightMeanHR: Double = 60.0
            
            let estimatedHRmin = heartRate - 3.0
            let estimatedHRmax = heartRate + 3.0
            
            let estimatedMotionVMmax = 1.0 + motionActivity * 3.0
            let estimatedMotionVMmin = max(0.9, 1.0 - motionActivity)
            let estimatedMotionVMstd = motionActivity * 0.5
            let estimatedMotionEnormStd = motionActivity * 0.4
            let estimatedMotionEnormMax = motionActivity * 3.0
            let estimatedMotionAxisStd = motionActivity * 0.4
            
            let estimatedSampleCount: Double = 1500
            let estimatedActiveCount = activeRatio * estimatedSampleCount
            
            let input = SleepStageClassifier3ClassesInput(
                hr_mean:                    heartRate,
                hr_std:                     hrStd,
                hr_min:                     estimatedHRmin,
                hr_max:                     estimatedHRmax,
                hr_count:                   5.0,
                motion_vm_mean:             1.0 + motionActivity,
                motion_vm_std:              estimatedMotionVMstd,
                motion_vm_max:              estimatedMotionVMmax,
                motion_vm_min:              estimatedMotionVMmin,
                motion_enorm_mean:          motionActivity,
                motion_enorm_std:           estimatedMotionEnormStd,
                motion_enorm_max:           estimatedMotionEnormMax,
                motion_x_std:               estimatedMotionAxisStd,
                motion_y_std:               estimatedMotionAxisStd,
                motion_z_std:               estimatedMotionAxisStd,
                motion_active_count:        estimatedActiveCount,
                motion_sample_count:        estimatedSampleCount,
                motion_active_ratio:        activeRatio,
                hr_diff_night_mean:         heartRate - nightMeanHR,
                hr_ratio_night_mean:        heartRate / nightMeanHR,
                hr_diff_night_min:          heartRate - 50.0,
                hr_mean_roll3:              heartRate,
                hr_mean_roll5:              heartRate,
                hr_std_roll5:               hrStd,
                motion_enorm_mean_roll3:    motionActivity,
                motion_enorm_mean_roll5:    motionActivity,
                motion_active_ratio_roll5:  activeRatio,
                hr_mean_prev1:              heartRate,
                hr_mean_next1:              heartRate,
                motion_vm_std_prev1:        estimatedMotionVMstd,
                motion_vm_std_next1:        estimatedMotionVMstd,
                relative_time_in_night:     nightProgress,
                time_from_start_minutes:    nightProgress * 480.0
            )
            
            let output = try model.prediction(input: input)
            
            if let stage = SleepStage(rawValue: output.sleep_stage) {
                predictedStage = stage
            }
            
            var parsedProbs: [SleepStage: Double] = [:]
            for (key, val) in output.sleep_stageProbability {
                if let stage = SleepStage(rawValue: key) {
                    parsedProbs[stage] = val
                }
            }
            if !parsedProbs.isEmpty {
                probabilities = parsedProbs
            }
            
        } catch {
            print("Erro de predição do CoreML: \(error)")
        }
    }
}
