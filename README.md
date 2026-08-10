O app ainda não está integrado a sensores reais. Os sliders controlam 5 valores-base manualmente:
heartRate, hrStd, motionActivity, activeRatio, nightProgress
A partir desses 5 valores, o ViewModel estima as ~25 features que o modelo espera (mín/máx de HR, variância de movimento, contagem de amostras, médias da noite, janelas históricas etc.) usando fórmulas simplificadas — não a partir de sensores ou de um histórico real.
