import pandas as pd
import numpy as np

def transform(df_pac, df_eval):
    # Unir pacientes con evaluaciones
    wide = df_eval.merge(df_pac, on="id_paciente", how="left")

    # Feature engineering: índice de exposición repetitiva (mock)
    wide["IER"] = wide["fuerza_prension_kg"] / (1 + wide["edad"]/50)

    # Score de riesgo compuesto (mock)
    wide["riesgo_score"] = (
        wide["latencia_emg_ms"].rank(pct=True) +
        wide["fuerza_prension_kg"].rank(pct=True)
    )

    # Etiquetas de riesgo
    wide["riesgo_label"] = pd.cut(
        wide["riesgo_score"],
        bins=[-np.inf,
              wide["riesgo_score"].quantile(0.33),
              wide["riesgo_score"].quantile(0.66),
              np.inf],
        labels=["bajo", "medio", "alto"]
    )
    return wide
