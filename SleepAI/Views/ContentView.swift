//
//  ContentView.swift
//  SleepAI
//
//  Created by Isaque da Silva Targino on 09/08/26.
//

import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = SleepClassifierViewModel()
    
    var body: some View {
        NavigationStack {
            List {
                currentStageSection
                probabilitiesSection
                sensorControlsSection
                summarySection
            }
            .listStyle(.insetGrouped)
            .navigationTitle("Classificador de Sono")
            .navigationBarTitleDisplayMode(.inline)
            .onAppear {
                viewModel.predictSleepStage() }
        }
    }
    
    // Mark: - Estágio atual predito
    private var currentStageSection: some View {
        Section {
            HStack (spacing: 16){
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
    
    // Mark: - Probabilidade dos estágios
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
    
    // Mark: - Controle dos sensores
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
                onChange: viewModel.predictSleepStage
            )
        } header: {
            Text("Ajustar Parâmetros dos Sensores")
        } footer: {
            Text("Simulação manual")
        }
    }
    
    // Mark: - Resumo com IA
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
