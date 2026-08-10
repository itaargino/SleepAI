//
//  FoundationModelService.swift
//  SleepAI
//
//  Created by Apple Foundation Models Integration.
//

import Foundation
import FoundationModels

class FoundationModelService {
    let instructions = """
    Você é o assistente de resumo de um app de monitoramento de sono e frequência cardíaca. \
    Sua função é transformar dados numéricos e resultados de um modelo de classificação em um \
    relatório claro, estruturado e acolhedor de bem-estar para o usuário final.

    Regras obrigatórias:
    1. Baseie-se exclusivamente nos dados fornecidos no prompt. Nunca invente números, \
    tendências, causas ou diagnósticos que não estejam nos dados.
    2. NUNCA use linguagem clínica, diagnóstica ou de risco à saúde (ex: "arritmia", \
    "apneia", "problema cardíaco"). Trate tudo estritamente como indicadores de bem-estar e relaxamento.
    3. Se o resultado do modelo parecer inconsistente ou se a confiança for baixa, mencione com cautela e de forma tranquila.
    4. Estruture a resposta com tópicos ou parágrafos bem definidos (ex: Panorama do Sono, Indicadores de Atividade e Dica de Bem-Estar).
    5. Tom: acolhedor, empático, direto e explicativo, sem jargões técnicos complexos.
    6. Idioma: Português do Brasil.
    """
    
    /// Função para geração de resumo via Apple FoundationModels Framework (LanguageModelSession)
    func generateSummary(
        heartRate: Double,
        hrStd: Double,
        motionActivity: Double,
        nightProgress: Double,
        predictedStage: String,
        confidence: Double,
        probabilities: [String: Double]
    ) async throws -> String {
        let nightProgressPercent = Int(nightProgress * 100)
        let formattedProbs = probabilities.map { "\($0.key): \(Int($0.value * 100))%" }.joined(separator: ", ")
        
        let promptText = """
        DADOS DO MONITORAMENTO:
        - Frequência Cardíaca Média: \(Int(heartRate)) BPM (Estabilidade/Desvio: \(String(format: "%.1f", hrStd)))
        - Nível de Movimentação do Pulso: \(String(format: "%.3f", motionActivity)) g
        - Progresso da Noite Decorrido: \(nightProgressPercent)% (\(String(format: "%.1f", nightProgress * 8.0))h de 8h simuladas)
        - Estágio de Sono Predito: \(predictedStage) (Confiança: \(Int(confidence * 100))%)
        - Distribuição de Probabilidades: \(formattedProbs)
        
        Gere um relatório informativo e acolhedor:
        """
        
        do {
            // Chamada direta à API nativa da Apple no SDK: FoundationModels.LanguageModelSession
            let session = LanguageModelSession(instructions: instructions)
            let response = try await session.respond(to: promptText)
            return response.content
        } catch {
            print("Executando fallback para o FoundationModels: \(error)")
            return generateFallbackSummary(
                heartRate: heartRate,
                hrStd: hrStd,
                motionActivity: motionActivity,
                nightProgress: nightProgress,
                predictedStage: predictedStage,
                confidence: confidence,
                probabilities: probabilities
            )
        }
    }
    
    private func generateFallbackSummary(
        heartRate: Double,
        hrStd: Double,
        motionActivity: Double,
        nightProgress: Double,
        predictedStage: String,
        confidence: Double,
        probabilities: [String: Double]
    ) -> String {
        let nightProgressPercent = Int(nightProgress * 100)
        var nightTimeDesc = "no início da sua noite"
        if nightProgress > 0.7 {
            nightTimeDesc = "na reta final da sua noite"
        } else if nightProgress > 0.3 {
            nightTimeDesc = "no meio do seu período de descanso"
        }
        
        var motionDesc = "movimentação muito baixa, refletindo um estado de ótimo relaxamento muscular"
        if motionActivity > 0.12 {
            motionDesc = "movimentação de pulso moderada, o que é natural durante ajustamentos de postura ou momentos de sono leve"
        } else if motionActivity > 0.05 {
            motionDesc = "movimentação leve registrada pelos sensores"
        }
        
        let probDetails = probabilities.sorted(by: { $0.value > $1.value }).map { "  • \($0.key): \(Int($0.value * 100))%" }.joined(separator: "\n")
        
        return """
        🌙 Panorama do Seu Sono (\(nightTimeDesc))
        
        📊 Indicadores e Sinais do Corpo:
        • Frequência Cardíaca: Média de \(Int(heartRate)) BPM com desvio de \(String(format: "%.1f", hrStd)) BPM — mantendo uma variação tranquila.
        • Movimentação: \(motionDesc).
        • Progresso da Noite: \(nightProgressPercent)% concluído (aprox. \(String(format: "%.1f", nightProgress * 8.0)) horas).
        
        💤 Estágio Predito & Probabilidades:
        O classificador identifica que você está atualmente em **\(predictedStage)** com **\(Int(confidence * 100))%** de certeza.
        
        Distribuição estimada:
        \(probDetails)
        
        💡 Insights de Bem-Estar:
        Seus dados indicam uma boa consonância entre a frequência cardíaca e a movimentação do corpo para a etapa de descanso atual. Mantenha um ambiente escuro e silencioso para ajudar a manter esses ciclos equilibrados ao longo de toda a noite.
        """
    }
}
