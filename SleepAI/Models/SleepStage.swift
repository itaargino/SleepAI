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
    case wake = "Wake"
    case light = "Light"
    case n3 = "N3"
    case rem = "REM"
    
    var id: String { rawValue }
    
    init?(rawValue: String) {
        let clean = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        for stage in SleepStage.allCases {
            if stage.rawValue.caseInsensitiveCompare(clean) == .orderedSame {
                self = stage
                return
            }
        }
        let lower = clean.lowercased()
        if lower == "heavy" || lower == "deep" {
            self = .n3
            return
        }
        return nil
    }
    
    var displayName: String {
        switch self {
        case .wake: return "Acordado"
        case .light: return "Sono leve"
        case .n3: return "Sono profundo (N3)"
        case .rem: return "Sono REM"
        }
    }
    
    var icon: String {
        switch self {
        case .wake: return "sun.max.fill"
        case .light: return "moon.fill"
        case .n3: return "bed.double.fill"
        case .rem: return "brain.head.profile"
        }
    }
    
    var tint: Color {
        switch self {
        case .wake: return .orange
        case .light: return .blue
        case .n3: return .indigo
        case .rem: return .purple
        }
    }
}

