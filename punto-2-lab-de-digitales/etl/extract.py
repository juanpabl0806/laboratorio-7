import pandas as pd

def extract():
    # Leer los CSV generados en data/raw
    df_pac = pd.read_csv("data/raw/pacientes.csv")
    df_eval = pd.read_csv("data/raw/evaluaciones_clinicas.csv", parse_dates=["fecha"])
    return df_pac, df_eval
