//
//  SleepStage.swift
//  SleepAI
//
//  Created by Isaque da Silva Targino on 09/08/26.
//
// Mark: - Model
// Reconhece os estados de sono

import SwiftUI

enum SleepStage: String, CaseIterable, Identifiable {
    case heavy = "Heavy"
    case light = "Light"
    case wake = "Wake"
    
    var id: String { rawValue }
    
    var displayName: String {
        switch self {
        case .heavy: return "Sono profundo"
        case .light: return "Sono leve"
        case .wake: return "Acordado"
        }
    }
    
    var icon: String {
        switch self {
        case .heavy: return "bed.double.fill"
        case .light: return "moon.fill"
        case .wake: return "sun.max.fill"
        }
    }
    
    var tint: Color {
        switch self {
        case .heavy: return .indigo
        case .light: return .blue
        case .wake: return .orange
        }
    }
}

