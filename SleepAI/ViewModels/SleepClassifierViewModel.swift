//
//  SleepClassifierViewModel.swift
//  SleepAI
//
//  Created by Isaque da Silva Targino on 09/08/26.
//

import Foundation
import CoreML
import Combine

// MARK: - Simulation Mode

enum SimulationMode: String, CaseIterable, Identifiable {
    case manual      = "Manual"
    case nightDataset = "Noite de Sono"
    var id: String { rawValue }
}

// MARK: - ViewModel

@MainActor
final class SleepClassifierViewModel: ObservableObject {

    // MARK: Simulation Mode
    @Published var simulationMode: SimulationMode = .manual

    // MARK: Manual Mode — slider inputs (unchanged)
    @Published var heartRate: Double = 62.0
    @Published var hrStd: Double = 1.5
    @Published var motionActivity: Double = 0.02
    @Published var activeRatio: Double = 0.05
    @Published var nightProgress: Double = 0.4

    // MARK: Night Dataset Mode — state
    @Published var epochs: [EpochData] = []
    @Published var selectedEpochIndex: Int = 0
    @Published var isLoadingDataset: Bool = false
    @Published var datasetError: String? = nil

    var currentEpoch: EpochData? {
        guard !epochs.isEmpty else { return nil }
        return epochs[min(selectedEpochIndex, epochs.count - 1)]
    }

    // MARK: Foundation Model State
    @Published var summaryText: String? = nil
    @Published var isGeneratingSummary: Bool = false
    @Published var summaryError: String? = nil

    private let foundationModelService = FoundationModelService()

    // MARK: CoreML Prediction Results
    @Published private(set) var predictedStage: SleepStage = .light
    @Published private(set) var probabilities: [SleepStage: Double] = [
        .heavy: 0.15,
        .light: 0.80,
        .wake: 0.05
    ]

    var confidence: Double {
        probabilities[predictedStage] ?? 0
    }

    // MARK: - Manual Prediction (slider-based, original)

    func predictSleepStage() {
        do {
            let model = try SleepStageClassifier3Classes(configuration: MLModelConfiguration())

            // Mock de features que não podem ser medidas via slider
            let nightMeanHR: Double = 60.0

            let estimatedHRmin = heartRate - 3.0
            let estimatedHRmax = heartRate + 3.0

            let estimatedMotionVMmax   = 1.0 + motionActivity * 3.0
            let estimatedMotionVMmin   = max(0.9, 1.0 - motionActivity)
            let estimatedMotionVMstd   = motionActivity * 0.5
            let estimatedMotionEnormStd = motionActivity * 0.4
            let estimatedMotionEnormMax = motionActivity * 3.0
            let estimatedMotionAxisStd  = motionActivity * 0.4

            let estimatedSampleCount: Double = 1500
            let estimatedActiveCount = activeRatio * estimatedSampleCount

            let input = SleepStageClassifier3ClassesInput(
                hr_mean:                    heartRate,
                hr_std:                     hrStd,
                hr_min:                     estimatedHRmin,
                hr_max:                     estimatedHRmax,
                hr_count:                   5.0,
                motion_vm_mean:             1.0 + motionActivity,
                motion_vm_std:             estimatedMotionVMstd,
                motion_vm_max:             estimatedMotionVMmax,
                motion_vm_min:             estimatedMotionVMmin,
                motion_enorm_mean:         motionActivity,
                motion_enorm_std:          estimatedMotionEnormStd,
                motion_enorm_max:          estimatedMotionEnormMax,
                motion_x_std:              estimatedMotionAxisStd,
                motion_y_std:              estimatedMotionAxisStd,
                motion_z_std:              estimatedMotionAxisStd,
                motion_active_count:       estimatedActiveCount,
                motion_sample_count:       estimatedSampleCount,
                motion_active_ratio:       activeRatio,
                hr_diff_night_mean:        heartRate - nightMeanHR,
                hr_ratio_night_mean:       heartRate / nightMeanHR,
                hr_diff_night_min:         heartRate - 50.0,
                hr_mean_roll3:             heartRate,
                hr_mean_roll5:             heartRate,
                hr_std_roll5:              hrStd,
                motion_enorm_mean_roll3:   motionActivity,
                motion_enorm_mean_roll5:   motionActivity,
                motion_active_ratio_roll5: activeRatio,
                hr_mean_prev1:             heartRate,
                hr_mean_next1:             heartRate,
                motion_vm_std_prev1:       estimatedMotionVMstd,
                motion_vm_std_next1:       estimatedMotionVMstd,
                relative_time_in_night:    nightProgress,
                time_from_start_minutes:   nightProgress * 480.0
            )

            applyPrediction(model: model, input: input)

        } catch {
            print("Erro de predição CoreML (manual): \(error)")
        }
    }

    // MARK: - Dataset Prediction (epoch-based)

    func predictFromEpoch() {
        guard let epoch = currentEpoch else { return }
        do {
            let model = try SleepStageClassifier3Classes(configuration: MLModelConfiguration())
            applyPrediction(model: model, input: epoch.toCoreMLInput())
        } catch {
            print("Erro de predição CoreML (dataset): \(error)")
        }
    }

    // MARK: - Load CSV Dataset

    func loadDataset(from url: URL) {
        isLoadingDataset = true
        datasetError = nil

        Task {
            do {
                // Precisa de acesso seguro ao arquivo selecionado pelo file picker
                let accessed = url.startAccessingSecurityScopedResource()
                defer { if accessed { url.stopAccessingSecurityScopedResource() } }

                let loaded = try NightDatasetLoader.load(from: url)
                self.epochs = loaded
                self.selectedEpochIndex = 0
                self.predictFromEpoch()
            } catch {
                self.datasetError = error.localizedDescription
            }
            self.isLoadingDataset = false
        }
    }

    // MARK: - Private helper

    private func applyPrediction(model: SleepStageClassifier3Classes, input: SleepStageClassifier3ClassesInput) {
        do {
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
            print("Erro ao aplicar predição CoreML: \(error)")
        }
    }

    // MARK: - AI Summary

    func generateSummary() {
        isGeneratingSummary = true
        summaryError = nil
        summaryText = nil

        let probsDict = Dictionary(
            uniqueKeysWithValues: probabilities.map { ($0.key.displayName, $0.value) }
        )

        // Contexto adicional quando há época real selecionada
        let epochTimestamp: String? = currentEpoch.map {
            let f = DateFormatter()
            f.dateFormat = "HH:mm:ss"
            return f.string(from: $0.timestamp)
        }
        let trueStageLabel: String? = currentEpoch?.trueStage?.displayName

        Task {
            do {
                let text = try await foundationModelService.generateSummary(
                    heartRate:       currentEpoch?.hrMean ?? heartRate,
                    hrStd:           currentEpoch?.hrStd ?? hrStd,
                    motionActivity:  currentEpoch?.motionEnormMean ?? motionActivity,
                    nightProgress:   currentEpoch?.relativeTimeInNight ?? nightProgress,
                    predictedStage:  predictedStage.displayName,
                    confidence:      confidence,
                    probabilities:   probsDict,
                    epochTimestamp:  epochTimestamp,
                    trueStage:       trueStageLabel
                )
                self.summaryText = text
            } catch {
                self.summaryError = "Falha ao gerar o resumo. Tente novamente mais tarde."
            }
            self.isGeneratingSummary = false
        }
    }
}
