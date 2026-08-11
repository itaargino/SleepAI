//
//  NightDatasetLoader.swift
//  SleepAI
//
//  Created by Isaque da Silva Targino on 10/08/26.
//
//  Carrega e faz o parse de um arquivo CSV de noite de sono sintético
//  (gerado pelo generate_night_mock.py), retornando um array de EpochData.
//  Compatível com qualquer URL (file picker, bundle, etc.).

import Foundation

// MARK: - Service

enum NightDatasetLoaderError: LocalizedError {
    case unreadableFile
    case emptyDataset
    case malformedHeader(String)
    case missingColumn(String)

    var errorDescription: String? {
        switch self {
        case .unreadableFile:
            return "Não foi possível ler o arquivo CSV."
        case .emptyDataset:
            return "O arquivo não contém dados de sono."
        case .malformedHeader(let h):
            return "Cabeçalho inválido: \(h)"
        case .missingColumn(let col):
            return "Coluna ausente no CSV: \(col)"
        }
    }
}

struct NightDatasetLoader {

    // Colunas obrigatórias esperadas no CSV (mesma ordem do generate_night_mock.py)
    private static let requiredColumns: [String] = [
        "timestamp", "epoch", "true_stage",
        "hr_mean", "hr_std", "hr_min", "hr_max", "hr_count",
        "motion_vm_mean", "motion_vm_std", "motion_vm_max", "motion_vm_min",
        "motion_enorm_mean", "motion_enorm_std", "motion_enorm_max",
        "motion_x_std", "motion_y_std", "motion_z_std",
        "motion_active_count", "motion_sample_count", "motion_active_ratio",
        "hr_diff_night_mean", "hr_ratio_night_mean", "hr_diff_night_min",
        "hr_mean_roll3", "hr_mean_roll5", "hr_std_roll5",
        "motion_enorm_mean_roll3", "motion_enorm_mean_roll5", "motion_active_ratio_roll5",
        "hr_mean_prev1", "hr_mean_next1", "motion_vm_std_prev1", "motion_vm_std_next1",
        "relative_time_in_night", "time_from_start_minutes"
    ]

    // MARK: - Public API

    /// Carrega o CSV a partir de uma URL (arquivo local ou bundle).
    /// Pode ser chamado em background — não faz I/O na main thread.
    static func load(from url: URL) throws -> [EpochData] {
        // Leitura do arquivo como string UTF-8
        let raw: String
        do {
            raw = try String(contentsOf: url, encoding: .utf8)
        } catch {
            throw NightDatasetLoaderError.unreadableFile
        }

        var lines = raw.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        guard !lines.isEmpty else { throw NightDatasetLoaderError.emptyDataset }

        // Cabeçalho
        let header = lines.removeFirst().components(separatedBy: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        let colIndex: [String: Int] = Dictionary(
            uniqueKeysWithValues: header.enumerated().map { ($1, $0) }
        )

        // Verifica colunas obrigatórias (permite "true_stage" ou "sleep_stage")
        for col in requiredColumns {
            if col == "true_stage" && (colIndex["true_stage"] != nil || colIndex["sleep_stage"] != nil) {
                continue
            }
            guard colIndex[col] != nil else {
                throw NightDatasetLoaderError.missingColumn(col)
            }
        }

        // Helper para extrair double de uma linha pelo nome da coluna
        func d(_ row: [String], _ col: String) -> Double {
            guard let idx = colIndex[col], idx < row.count else { return 0 }
            let str = row[idx].trimmingCharacters(in: .whitespacesAndNewlines)
            return Double(str) ?? 0
        }

        // Parse das linhas
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withSpaceBetweenDateAndTime]
        let fallbackFormatter: DateFormatter = {
            let f = DateFormatter()
            f.dateFormat = "yyyy-MM-dd HH:mm:ss"
            f.locale = Locale(identifier: "en_US_POSIX")
            return f
        }()

        var epochs: [EpochData] = []
        epochs.reserveCapacity(lines.count)

        for line in lines {
            let cols = line.components(separatedBy: ",")
            guard cols.count >= header.count else { continue }

            let tsRaw = colIndex["timestamp"].map { cols[$0].trimmingCharacters(in: .whitespacesAndNewlines) } ?? ""
            let timestamp = formatter.date(from: tsRaw)
                ?? fallbackFormatter.date(from: tsRaw)
                ?? Date()

            let stageIdx = colIndex["true_stage"] ?? colIndex["sleep_stage"]
            let trueStageRaw = stageIdx.map { cols[$0].trimmingCharacters(in: .whitespacesAndNewlines) } ?? ""
            let trueStage = SleepStage(rawValue: trueStageRaw)

            let epoch = EpochData(
                id:                        Int(d(cols, "epoch")),
                timestamp:                 timestamp,
                trueStage:                 trueStage,
                hrMean:                    d(cols, "hr_mean"),
                hrStd:                     d(cols, "hr_std"),
                hrMin:                     d(cols, "hr_min"),
                hrMax:                     d(cols, "hr_max"),
                hrCount:                   d(cols, "hr_count"),
                motionVmMean:              d(cols, "motion_vm_mean"),
                motionVmStd:               d(cols, "motion_vm_std"),
                motionVmMax:               d(cols, "motion_vm_max"),
                motionVmMin:               d(cols, "motion_vm_min"),
                motionEnormMean:           d(cols, "motion_enorm_mean"),
                motionEnormStd:            d(cols, "motion_enorm_std"),
                motionEnormMax:            d(cols, "motion_enorm_max"),
                motionXStd:                d(cols, "motion_x_std"),
                motionYStd:                d(cols, "motion_y_std"),
                motionZStd:                d(cols, "motion_z_std"),
                motionActiveCount:         d(cols, "motion_active_count"),
                motionSampleCount:         d(cols, "motion_sample_count"),
                motionActiveRatio:         d(cols, "motion_active_ratio"),
                hrDiffNightMean:           d(cols, "hr_diff_night_mean"),
                hrRatioNightMean:          d(cols, "hr_ratio_night_mean"),
                hrDiffNightMin:            d(cols, "hr_diff_night_min"),
                hrMeanRoll3:               d(cols, "hr_mean_roll3"),
                hrMeanRoll5:               d(cols, "hr_mean_roll5"),
                hrStdRoll5:                d(cols, "hr_std_roll5"),
                motionEnormMeanRoll3:      d(cols, "motion_enorm_mean_roll3"),
                motionEnormMeanRoll5:      d(cols, "motion_enorm_mean_roll5"),
                motionActiveRatioRoll5:    d(cols, "motion_active_ratio_roll5"),
                hrMeanPrev1:               d(cols, "hr_mean_prev1"),
                hrMeanNext1:               d(cols, "hr_mean_next1"),
                motionVmStdPrev1:          d(cols, "motion_vm_std_prev1"),
                motionVmStdNext1:          d(cols, "motion_vm_std_next1"),
                relativeTimeInNight:       d(cols, "relative_time_in_night"),
                timeFromStartMinutes:      d(cols, "time_from_start_minutes")
            )
            epochs.append(epoch)
        }

        guard !epochs.isEmpty else { throw NightDatasetLoaderError.emptyDataset }
        return epochs
    }
}
