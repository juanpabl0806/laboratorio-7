def load(wide):
    # Guardar dataset procesado en data/processed
    wide.to_csv("data/processed/dataset_analitico.csv", index=False)
    wide.to_parquet("data/processed/dataset_analitico.parquet", index=False)
    print("Datos procesados guardados en data/processed/")
