# 🌙 SleepAI — Documentação Completa: Do Pré-processamento ao Aplicativo iOS

O **SleepAI** é uma solução completa para classificação de estágios de sono (*Sono Profundo / Heavy*, *Sono Leve / Light* e *Acordado / Wake*) com base em dados de sensores vestíveis (**HealthKit** e **CoreMotion** do Apple Watch). 

O ecossistema do projeto contempla desde o **pré-processamento de sinais fisiológicos brutos**, **engenharia de features temporais**, **treinamento no Apple Create ML**, **simulação sintética calibrada** até a **aplicação iOS nativa em SwiftUI com CoreML e Apple Foundation Models**.

---

## 🗺️ Visão Geral do Pipeline End-to-End

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ ETAPA 1: PRÉ-PROCESSAMENTO & ENGENHARIA DE FEATURES (prepare_sleep_dataset.py)           │
│ • Entrada: Sinais brutos de HR (csv), Acelerômetro x,y,z (csv) e Gabarito POLI (mat)  │
│ • Agregação: Épocas de 30 segundos (médias, desvios, min/máx, razões ativas)           │
│ • Janelas Temporais: Rolling windows (3 e 5 épocas), Lags (t-1, t+1) e Deltas da noite │
│ • Saída: sleep_stage_train.csv e sleep_stage_test.csv (divisão por sujeito)            │
└────────────────────────────────────────────────────────┬───────────────────────────────┘
                                                         │
                                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ ETAPA 2: TREINAMENTO DO MODELO NO APPLE CREATE ML (SleepStageClassifier.mlproj)          │
│ • Framework: Apple Create ML (Tabular Classifier)                                      │
│ • Variável Alvo: sleep_stage (Wake, Light, Heavy)                                     │
│ • Input: 33 features numéricas derivadas de sensores                                   │
│ • Saída: SleepStageClassifier3Classes.mlmodel (~1.1 MB)                                 │
└────────────────────────────────────────────────────────┬───────────────────────────────┘
                                                         │
                                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ ETAPA 3: GERADOR SINTÉTICO & CALIBRAÇÃO DE DECISÃO (generate_night_mock.py)              │
│ • Simulação: 8 horas de sono (960 épocas de 30s) com 50Hz de acelerômetro e HR        │
│ • Sondagem: Ferramentas Swift (probe_model.swift) para extrair as fronteiras do CoreML │
│ • Resultado: Dataset night_mock.csv com acurácia de 91,46% em relação ao modelo       │
└────────────────────────────────────────────────────────┬───────────────────────────────┘
                                                         │
                                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ ETAPA 4: APLICATIVO iOS NATIVO (SleepAI)                                               │
│ • Interface: SwiftUI com Picker segmentado (Modo Manual vs. Modo Noite de Sono)       │
│ • Importação: fileImporter nativo de CSVs e slider de período temporal (HH:mm:ss)     │
│ • Inferência: Modelo CoreML executado localmente em tempo real                          │
│ • Resumo IA: Relatório de bem-estar via Apple Foundation Models API                    │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. 🧪 Preprocessamento e Engenharia de Features

Localizado em `Etapa de Pré-processamento/Scripts/prepare_sleep_dataset.py`.

### Sinais de Entrada:
1. **Frequência Cardíaca (`hr.csv`)**: Timestamps e batimentos por minuto (BPM).
2. **Acelerômetro Triaxial (`motion.csv`)**: Amostras dos eixos $x, y, z$ em gravidade ($g$).
3. **Anotações Polissonográficas (`labels.mat`)**: Rótulos anotados por especialistas em épocas de 30s.

### Pipeline de Agregação por Época de 30 segundos:
- **Norma Vetorial de Aceleração**:
  $$vm = \sqrt{x^2 + y^2 + z^2}$$
  $$enorm = |vm - 1.0|$$
- **Razão de Movimento Ativo (`motion_active_ratio`)**: Proporção de amostras em 30s onde $enorm > 0.05g$.
- **Normalização em Relação à Noite**:
  - `hr_diff_night_mean` = $HR_{época} - \overline{HR}_{noite}$
  - `hr_ratio_night_mean` = $HR_{época} / \overline{HR}_{noite}$
  - `hr_diff_night_min` = $HR_{época} - HR_{mínimo\_noite}$
- **Janelas Móveis e Contexto Temporal**:
  - Média móvel de HR e movimento em janelas de 3 e 5 épocas (`roll3`, `roll5`).
  - Lags de contexto imediato: época anterior ($t-1$) e época posterior ($t+1$).
  - Progresso temporal relativo da noite (`relative_time_in_night` de 0.0 a 1.0).

### Mapeamento dos Estágios de Sono:
```python
STAGE_MAP = {
    0: 'Wake',    # Acordado
    1: 'Light',   # Sono Leve (N1/N2)
    2: 'Light',   
    3: 'Heavy',   # Sono Profundo (N3/REM)
    4: 'Heavy',   
    5: 'Unknown'  # Descartado do treinamento
}
```

---

## 2. 🧠 Treinamento do Modelo CoreML

Localizado em `Modelo Criado com CoreML/SleepStageClassifier.mlproj`.

1. **Plataforma**: Apple Create ML (Xcode).
2. **Algoritmo**: Classificador Tabular (Ensemble/Decision Trees).
3. **Features de Entrada (33 variáveis)**:
   - Frequência Cardíaca: `hr_mean`, `hr_std`, `hr_min`, `hr_max`, `hr_count`
   - Acelerômetro: `motion_vm_mean`, `motion_vm_std`, `motion_vm_max`, `motion_vm_min`, `motion_enorm_mean`, `motion_enorm_std`, `motion_enorm_max`, `motion_x_std`, `motion_y_std`, `motion_z_std`, `motion_active_count`, `motion_sample_count`, `motion_active_ratio`
   - Métricas da Noite: `hr_diff_night_mean`, `hr_ratio_night_mean`, `hr_diff_night_min`
   - Rolling Windows: `hr_mean_roll3`, `hr_mean_roll5`, `hr_std_roll5`, `motion_enorm_mean_roll3`, `motion_enorm_mean_roll5`, `motion_active_ratio_roll5`
   - Lags: `hr_mean_prev1`, `hr_mean_next1`, `motion_vm_std_prev1`, `motion_vm_std_next1`
   - Tempo: `relative_time_in_night`, `time_from_start_minutes`
4. **Artefato Gerado**: `SleepStageClassifier3Classes.mlmodel` com saídas `sleep_stage` (String) e `sleep_stageProbability` (Dicionário `[String: Double]`).

---

## 3. 🎲 Gerador Sintético e Calibração (`generate_night_mock.py`)

Localizado em `Gerador de Noites/generate_night_mock.py`.

Para permitir simulações sem depender do hardware físico do Apple Watch durante o teste da POC, foi construído um gerador estocástico calibrado:
- Simula 960 épocas (8 horas de sono a 30s por época).
- Modela a transição de estágios usando probabilidade ultradiana (sono profundo prevalente no primeiro terço, sono leve e despertares no final da noite).
- **Acurácia de Alinhamento**: As distribuições sintéticas foram calibradas com o espaço de decisão do modelo CoreML pré-treinado, alcançando **91,46% de precisão geral** (878 acertos em 960 épocas).

---

## 4. 📱 Aplicativo iOS (`SleepAI`)

O app iOS foi arquitetado utilizando **SwiftUI**, **Combine** e **CoreML**.

### Principais Componentes em Swift:
- **`EpochData.swift`**: Representação tipada da época de sono de 30s com o método `toCoreMLInput()` que converte o CSV diretamente para a API gerada pelo CoreML.
- **`NightDatasetLoader.swift`**: Parser de arquivo CSV de alto desempenho que lida com codificação UTF-8, validação de 33 colunas obrigatórias e permissões do gerenciador de arquivos do iOS (`startAccessingSecurityScopedResource`).
- **`SleepClassifierViewModel.swift`**: ViewModel reativo que coordena:
  - **Modo Simulação Manual**: Ajuste dinâmico dos sliders com estimativa de features.
  - **Modo Noite de Sono**: Carregamento do CSV e seleção de épocas via slider.
- **`FoundationModelService.swift`**: Integração com a API nativa de IA Generativa da Apple (`FoundationModels.LanguageModelSession`) para compor resumos acolhedores de bem-estar.
- **`ContentView.swift`**: Interface com Picker segmentado de modo, botão nativo de importação `fileImporter`, slider de época com formatação de horário `HH:mm:ss`, card de sensores e indicador de concordância com o Ground Truth.

---

## 📊 Tabela Resumo de Acurácia e Matriz de Confusão

```
=====================================================
 ACURÁCIA DO MODELO COREML NO DATASET DA NOITE: 91.46%
=====================================================
```

| Estágio Sintético | Épocas no Dataset | Predição: Light | Predição: Heavy | Predição: Wake | Acurácia por Classe |
|---|---|---|---|---|---|
| 🌙 **Sono Leve (Light)** | 565 | **507** | 1 | 57 | **89,7%** |
| 🛌 **Sono Profundo (Heavy)** | 235 | 18 | **215** | 2 | **91,5%** |
| ☀️ **Acordado (Wake)** | 160 | 4 | 0 | **156** | **97,5%** |

---

## 📂 Estrutura Completa de Diretórios do Repositório

```
SleepAI/
├── Etapa de Pré-processamento/
│   └── Scripts/
│       └── prepare_sleep_dataset.py     # Script Python original de pré-processamento
├── Modelo Criado com CoreML/
│   ├── SleepStageClassifier.mlproj      # Projeto de treinamento do Xcode Create ML
│   └── SleepStageClassifier3Classes.mlmodel # Modelo treinado exportado
├── Gerador de Noites/
│   ├── generate_night_mock.py           # Gerador sintético calibrado
│   └── night_mock.csv                   # CSV sintético gerado (8h / 960 épocas)
├── SleepAI/
│   ├── Models/
│   │   ├── EpochData.swift              # Struct Swift das 33 features
│   │   ├── NightDatasetLoader.swift     # Leitor/parser de CSV
│   │   ├── SleepStage.swift             # Enum dos estágios de sono
│   │   └── FoundationModelService.swift # Integração com Apple Foundation Models API
│   ├── ViewModels/
│   │   └── SleepClassifierViewModel.swift # ViewModel reativo
│   ├── Views/
│   │   └── ContentView.swift            # UI SwiftUI
│   ├── Components/
│   │   └── SensorSlider.swift           # Slider reutilizável
│   └── Resources/
│       └── night_mock.csv               # Dataset padrão do bundle do app
├── SleepStageClassifier3Classes.mlmodel  # Binário do modelo CoreML na raiz do app
└── README.md                            # Esta documentação completa do projeto
```

---

## 🚀 Como Executar

1. **Abrir o Projeto no Xcode 16+**:
   ```bash
   open SleepAI.xcodeproj
   ```
2. Compilar e rodar no Simulador iOS (`Cmd + R`).
3. Alternar para a aba **Noite de Sono**, clicar em **Importar Noite de Sono** e selecionar o `night_mock.csv`.
