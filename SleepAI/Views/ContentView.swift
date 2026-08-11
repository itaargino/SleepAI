//
//  ContentView.swift
//  SleepAI
//
//  Created by Isaque da Silva Targino on 09/08/26.
//

import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var viewModel = SleepClassifierViewModel()
    @State private var isImporterPresented = false

    var body: some View {
        NavigationStack {
            List {
                // ── Picker de Modo ──────────────────────────────────────────
                modeSwitcherSection

                // ── Predição atual (compartilhado pelos dois modos) ─────────
                currentStageSection
                probabilitiesSection

                // ── Controles: condicional por modo ─────────────────────────
                switch viewModel.simulationMode {
                case .manual:
                    sensorControlsSection
                case .nightDataset:
                    nightDatasetSection
                }

                // ── Resumo com IA (compartilhado pelos dois modos) ──────────
                summarySection
            }
            .listStyle(.insetGrouped)
            .navigationTitle("Classificador de Sono")
            .navigationBarTitleDisplayMode(.inline)
            .onAppear {
                viewModel.predictSleepStage()
            }
            // File Importer (iOS 14+ HIG)
            .fileImporter(
                isPresented: $isImporterPresented,
                allowedContentTypes: [UTType.commaSeparatedText],
                allowsMultipleSelection: false
            ) { result in
                switch result {
                case .success(let urls):
                    if let url = urls.first {
                        viewModel.loadDataset(from: url)
                    }
                case .failure(let error):
                    viewModel.datasetError = error.localizedDescription
                }
            }
        }
    }

    // MARK: - Picker de Modo

    private var modeSwitcherSection: some View {
        Section {
            Picker("Modo de Simulação", selection: $viewModel.simulationMode) {
                ForEach(SimulationMode.allCases) { mode in
                    Text(mode.rawValue).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .listRowInsets(.init(top: 8, leading: 16, bottom: 8, trailing: 16))
            .onChange(of: viewModel.simulationMode) { _, newMode in
                if newMode == .manual {
                    viewModel.predictSleepStage()
                } else if !viewModel.epochs.isEmpty {
                    viewModel.predictFromEpoch()
                }
            }
        }
    }

    // MARK: - Estágio atual predito

    private var currentStageSection: some View {
        Section {
            HStack(spacing: 16) {
                Image(systemName: viewModel.predictedStage.icon)
                    .font(.title)
                    .foregroundStyle(viewModel.predictedStage.tint)
                    .frame(width: 44)

                VStack(alignment: .leading, spacing: 2) {
                    Text(viewModel.predictedStage.displayName)
                        .font(.headline)

                    Text("\(Int(viewModel.confidence * 100))% de certeza")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }

                Spacer()
            }
            .padding(.vertical, 4)
        } header: {
            Text("Estágio Atual")
        }
    }

    // MARK: - Probabilidade dos estágios

    private var probabilitiesSection: some View {
        Section {
            ForEach(SleepStage.allCases) { stage in
                let probability = viewModel.probabilities[stage] ?? 0

                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Label(stage.displayName, systemImage: stage.icon)
                            .font(.subheadline)

                        Spacer()

                        Text(String(format: "%.1f%%", probability * 100))
                            .font(.subheadline)
                            .foregroundStyle(stage == viewModel.predictedStage ? .primary : .secondary)
                    }

                    ProgressView(value: probability)
                        .tint(stage.tint)
                }
                .padding(.vertical, 2)
            }
        } header: {
            Text("Probabilidades dos Estágios")
        }
    }

    // MARK: - Controle dos sensores (Modo Manual — inalterado)

    private var sensorControlsSection: some View {
        Section {
            SensorSlider(
                title: "Frequência Cardíaca Média",
                value: $viewModel.heartRate,
                range: 40...110,
                unit: "BPM",
                systemImage: "heart.fill",
                onChange: viewModel.predictSleepStage
            )

            SensorSlider(
                title: "Movimento de Pulso",
                value: $viewModel.motionActivity,
                range: 0.00...0.25,
                unit: "g",
                systemImage: "waveform.path",
                onChange: viewModel.predictSleepStage
            )

            SensorSlider(
                title: "Tempo Decorrido da Noite",
                value: $viewModel.nightProgress,
                range: 0.0...1.0,
                unit: "min",
                systemImage: "clock.fill",
                onChange: viewModel.predictSleepStage,
                displayText: { progress in
                    String(format: "%.0f min", progress * 480)
                }
            )
        } header: {
            Text("Ajustar Parâmetros dos Sensores")
        } footer: {
            Text("Simulação manual")
        }
    }

    // MARK: - Modo Noite de Sono (CSV)

    private var nightDatasetSection: some View {
        Section {
            // Botão de importar
            Button {
                isImporterPresented = true
            } label: {
                Label(
                    viewModel.epochs.isEmpty ? "Importar Noite de Sono" : "Substituir Noite de Sono",
                    systemImage: "square.and.arrow.down"
                )
            }
            .disabled(viewModel.isLoadingDataset)

            // Estado de carregamento
            if viewModel.isLoadingDataset {
                HStack {
                    ProgressView()
                        .padding(.trailing, 8)
                    Text("Lendo dataset…")
                        .foregroundStyle(.secondary)
                        .font(.subheadline)
                }
            }

            // Erro de importação
            if let error = viewModel.datasetError {
                Label(error, systemImage: "exclamationmark.triangle.fill")
                    .font(.subheadline)
                    .foregroundStyle(.red)
            }

            // Controles da época — só quando o dataset está carregado
            if !viewModel.epochs.isEmpty {
                epochSliderRow
                epochDetailsCard
            }

        } header: {
            Text("Noite de Sono")
        } footer: {
            if viewModel.epochs.isEmpty {
                Text("Importe um arquivo CSV gerado pelo night_mock para avaliar a noite completa.")
            } else {
                Text("\(viewModel.epochs.count) épocas de 30 s carregadas")
            }
        }
    }

    // MARK: - Slider de Período da Noite

    private var epochSliderRow: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Período da Noite", systemImage: "clock.fill")
                .font(.subheadline)
                .foregroundStyle(.secondary)

            HStack {
                Slider(
                    value: Binding(
                        get: { Double(viewModel.selectedEpochIndex) },
                        set: { newVal in
                            viewModel.selectedEpochIndex = Int(newVal.rounded())
                            viewModel.predictFromEpoch()
                        }
                    ),
                    in: 0...Double(max(viewModel.epochs.count - 1, 1)),
                    step: 1
                )

                Text(epochTimeLabel)
                    .font(.subheadline.monospacedDigit())
                    .foregroundStyle(.primary)
                    .frame(minWidth: 64, alignment: .trailing)
            }
        }
        .padding(.vertical, 4)
    }

    private var epochTimeLabel: String {
        guard let epoch = viewModel.currentEpoch else { return "--:--:--" }
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss"
        return f.string(from: epoch.timestamp)
    }

    // MARK: - Card de Detalhes da Época

    private var epochDetailsCard: some View {
        VStack(spacing: 0) {
            // BPM
            epochMetricRow(
                icon: "heart.fill",
                tint: .red,
                label: "Frequência Cardíaca",
                value: viewModel.currentEpoch.map {
                    String(format: "%.1f BPM", $0.hrMean)
                } ?? "—"
            )
            Divider().padding(.leading, 44)

            // Movimento
            epochMetricRow(
                icon: "waveform.path",
                tint: .orange,
                label: "Movimento Ativo",
                value: viewModel.currentEpoch.map {
                    String(format: "%.1f%%", $0.motionActiveRatio * 100)
                } ?? "—"
            )
            Divider().padding(.leading, 44)

            // Progresso
            epochMetricRow(
                icon: "moon.stars.fill",
                tint: .indigo,
                label: "Progresso da Noite",
                value: viewModel.currentEpoch.map {
                    String(format: "%.0f%%", $0.relativeTimeInNight * 100)
                } ?? "—"
            )

            // Ground Truth (separado visualmente)
            if let gt = viewModel.currentEpoch?.trueStage {
                Divider().padding(.leading, 44)
                HStack(spacing: 12) {
                    Image(systemName: gt.icon)
                        .foregroundStyle(gt.tint)
                        .frame(width: 28)

                    VStack(alignment: .leading, spacing: 1) {
                        Text("Ground Truth (Gabarito)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Text(gt.displayName)
                            .font(.subheadline)
                    }

                    Spacer()

                    // Badge de concordância
                    if gt == viewModel.predictedStage {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                    } else {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.orange)
                    }
                }
                .padding(.vertical, 8)
            }
        }
    }

    private func epochMetricRow(icon: String, tint: Color, label: String, value: String) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .foregroundStyle(tint)
                .frame(width: 28)

            Text(label)
                .font(.subheadline)

            Spacer()

            Text(value)
                .font(.subheadline.monospacedDigit())
                .foregroundStyle(.primary)
        }
        .padding(.vertical, 8)
    }

    // MARK: - Resumo com IA

    private var summarySection: some View {
        Section {
            Button(action: {
                viewModel.generateSummary()
            }) {
                HStack {
                    Spacer()
                    if viewModel.isGeneratingSummary {
                        ProgressView()
                            .padding(.trailing, 8)
                        Text("Gerando Resumo...")
                    } else {
                        Image(systemName: "sparkles")
                        Text("Gerar Resumo com IA")
                    }
                    Spacer()
                }
            }
            .disabled(viewModel.isGeneratingSummary)
            .buttonStyle(.borderless)
            .padding(.vertical, 4)

            if let error = viewModel.summaryError {
                Text(error)
                    .font(.subheadline)
                    .foregroundStyle(.red)
                    .padding(.vertical, 4)
            } else if let summary = viewModel.summaryText {
                Text(LocalizedStringKey(summary))
                    .font(.subheadline)
                    .foregroundStyle(.primary)
                    .lineSpacing(4)
                    .padding(.vertical, 4)
            }
        } header: {
            Text("Insights")
        }
    }
}
