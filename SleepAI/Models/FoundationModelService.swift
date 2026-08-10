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
    relatório claro do momento do sono, estruturado e acolhedor de bem-estar para o usuário final. Analise como se \
    fosse um ponto específico do sono, o modelo da a porcentagem do quão provável é das três fases (acordado, leve, pesado). Explique a relação dos dados de entrada com a saida do modelo coreml.

    Regras obrigatórias:
    1. Baseie-se exclusivamente nos dados fornecidos no prompt. Nunca invente números, \
    tendências, causas ou diagnósticos que não estejam nos dados.
    2. NUNCA use linguagem clínica, diagnóstica ou de risco à saúde (ex: "arritmia", \
    "apneia", "problema cardíaco"). Trate tudo estritamente como indicadores de bem-estar e relaxamento.
    3. Se o resultado do modelo parecer inconsistente ou se a confiança for baixa, mencione com cautela e de forma tranquila.
    4. Estruture a resposta com tópicos ou parágrafos bem definidos (ex: Panorama do Sono, Indicadores de Atividade e Conclusão).
    5. Tom: acolhedor, empático, direto e explicativo, sem jargões técnicos complexos.
    6. Idioma: Português do Brasil.
    7. Não retorne com uma mensagem de "caso houver dúvida, pergunte mais".
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
        - Distribuição de Probabilidades de que sono fase do sono ele está: \(formattedProbs)
        
        Gere um relatório informativo e acolhedor:
        """
        
        // Chamada direta à API nativa da Apple no SDK: FoundationModels.LanguageModelSession
        let session = LanguageModelSession(instructions: instructions)
        let response = try await session.respond(to: promptText)
        return response.content
    }
}
