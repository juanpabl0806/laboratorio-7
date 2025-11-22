
.PHONY: etl dashboard clean

# Ejecuta el pipeline ETL: extrae, transforma y carga los datos
etl:
	python -c "from etl.extract import extract; from etl.transform import transform; from etl.load import load; load(transform(*extract()))"

# Lanza el dashboard de Streamlit
dashboard:
	streamlit run dashboards/app.py

# Limpia los archivos procesados
clean:
	rm -f data/processed/*.parquet data/processed/*.csv
