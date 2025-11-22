# Punto-2-lab-de-digitales
# Objetivos del proyecto

- **Integración:** Unificar y documentar el proceso ETL sobre la base de datos compartida.  
- **Calidad de datos:** Aplicar validaciones, manejo de nulos, outliers y estandarización.  
- **Feature engineering:** Generar variables relevantes para modelos de prevención con RNA.  
- **Visualización:** Construir un dashboard en Streamlit que comunique métricas, tendencias y riesgos.  
- **Reproducibilidad:** Entregar un pipeline ejecutable (CLI/Makefile) con logs y configuración externa.  

---

# Estructura de datos y supuestos del dominio

## Esquema de origen (ejemplo)

- **Paciente:**  
  `id_paciente`, `edad`, `sexo`, `mano_dominante`, `ocupación`, `años_exposición`

- **Evaluación_clínica:**  
  `id_eval`, `id_paciente`, `fecha`, `síntomas (dolor, parestesias)`, `severidad_nordic`

- **Ergonomía_laboral:**  
  `id_eval`, `postura_promedio`, `fuerza_promedio`, `repetitividad`, `pausas_por_hora`

- **EMG_sensorial:**  
  `id_eval`, `latencia_ms`, `amplitud_mv`, `velocidad_conducción_m_s`

- **Encuesta_hábitos:**  
  `id_eval`, `sueño_horas`, `actividad_física`, `micro-pausas`, `equipo_ergonómico`

---

# Variables derivadas (ejemplo)

- **Índice de exposición repetitiva (IER):** combina repetitividad, pausas y fuerza.  

  

\[
  \text{IER} = \frac{\text{repetitividad} \cdot \text{fuerza\_promedio}}{1 + \text{pausas\_por\_hora}}
  \]



- **Score de riesgo compuesto:** normalización y suma ponderada de postura, EMG y síntomas.  

- **Etiquetas de riesgo:** bajo / medio / alto usando umbrales estadísticos o clínicos.  
