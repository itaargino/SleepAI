//
//  SensorSlider.swift
//  SleepAI
//
//  Created by Isaque da Silva Targino on 09/08/26.
//

import SwiftUI

struct SensorSlider: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let unit: String
    let systemImage: String
    let onChange: () -> Void
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(title, systemImage: systemImage)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            
            HStack {
                Slider(value: $value, in: range) { _ in
                    onChange()
                }
                
                Text(formattedValue)
                    .font(.subheadline.monospacedDigit())
                    .foregroundStyle(.primary)
                    .frame(minWidth: 64, alignment: .trailing)
            }
        }
        .padding(.vertical, 4)
    }
    
    private var formattedValue: String {
        let format = range.lowerBound < 1.0 ? "%.3f" : "%.1f"
        return String(format: "\(format) %@", value, unit)
    }
}
