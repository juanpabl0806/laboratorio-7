# Laboratorio-7
# Punto 1: Detector de Emociones con MediaPipe y Random Forest

En este proyecto, configuré todo mi entorno en **Linux** para poder ejecutar modelos basados en **MediaPipe**. Al principio, tenía Python 3.13, pero no era compatible con MediaPipe, así que instalé **Python 3.10** y creé un entorno virtual llamado `emotion-env`. Dentro de este entorno instalé todas las dependencias necesarias: OpenCV, NumPy, scikit-learn, joblib y más. Además, organicé mi proyecto en una carpeta llamada `mi_proyecto_emociones`, donde guardo tanto los scripts de entrenamiento como los de ejecución en tiempo real.

## Preparación del Dataset

Para entrenar el modelo, creé un dataset propio de emociones. Organicé las imágenes en tres carpetas: **alegría**, **enojo** y **tristeza**, con unas 20 imágenes por emoción. Esto me permitió trabajar con un dataset manejable sin necesidad de millones de imágenes ni redes neuronales pesadas.

## Entrenamiento del Modelo

Desarrollé un script llamado `train_emotion_from_landmarks.py` que:

1. Extrae **landmarks faciales** usando MediaPipe FaceMesh.
2. Convierte cada imagen en un vector numérico.
3. Entrena un **Random Forest** para clasificar las emociones.
4. Guarda el modelo entrenado como `emotion_landmark_rf.joblib`.

```
import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# ============================================
# CONFIGURACIÓN
# ============================================
DATASET_PATH = "dataset"

# Las 3 emociones que usas
CLASSES = ["alegria", "enojo", "tristeza"]

# Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# ============================================
# EXTRAER LANDMARKS
# ============================================
def extract_landmarks(image):
    """Devuelve un vector [x1,y1,x2,y2,…] con 468 puntos."""
    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    landmarks = []
    for lm in results.multi_face_landmarks[0].landmark:
        landmarks.append(lm.x)
        landmarks.append(lm.y)

    return np.array(landmarks)


# ============================================
# CARGAR DATASET
# ============================================
X = []
y = []

print("📂 Cargando dataset y extrayendo landmarks...")

for idx, emotion in enumerate(CLASSES):
    emotion_folder = os.path.join(DATASET_PATH, emotion)

    if not os.path.isdir(emotion_folder):
        print(f"❌ No existe la carpeta: {emotion_folder}")
        continue

    for filename in os.listdir(emotion_folder):
        path = os.path.join(emotion_folder, filename)

        image = cv2.imread(path)
        if image is None:
            continue

        landmarks = extract_landmarks(image)
        if landmarks is not None:
            X.append(landmarks)
            y.append(idx)

X = np.array(X)
y = np.array(y)

print(f"✔ Total de muestras cargadas: {len(X)}")

# ============================================
# ENTRENAMIENTO DEL MODELO
# ============================================
print("🚀 Entrenando modelo Random Forest...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=250,
    max_depth=15,
    random_state=42
)

model.fit(X_train, y_train)

# ============================================
# EVALUACIÓN
# ============================================
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print(f"📊 Precisión del modelo: {acc*100:.2f}%")

# ============================================
# GUARDAR MODELO
# ============================================
OUTPUT_MODEL = "emotion_landmark_rf.joblib"
joblib.dump(model, OUTPUT_MODEL)

print("💾 Modelo guardado como:", OUTPUT_MODEL)
print("✅ Entrenamiento finalizado con éxito.")
Gracias a este enfoque, logré un clasificador funcional, eficiente y liviano, ideal para inferencia en tiempo real.
```

## Inferencia en Tiempo Real

Para la ejecución en tiempo real, implementé `realtime_emotion_multithread.py`. Para mantener estabilidad y rendimiento:

* Utilicé **tres hilos** separados: uno captura la cámara, otro procesa y predice emociones, y el tercero muestra la ventana.
* Esto evita la apertura de múltiples ventanas o bloqueos de hilos, mostrando únicamente un **frame actualizado** en pantalla de manera fluida.
```
import cv2
import mediapipe as mp
import numpy as np
import threading
import joblib

# =======================================
# VARIABLES GLOBALES
# =======================================
current_frame = None
processed_frame = None
running = True

# =======================================
# Cargar modelo entrenado
# =======================================
model = joblib.load("emotion_landmark_rf.joblib")

# SOLO tus 3 clases
CLASSES = ["alegria", "enojo", "tristeza"]

# Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =======================================
# HILO 1: Captura de cámara
# =======================================
def camera_thread():
    global current_frame, running

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara.")
        running = False
        return

    while running:
        ret, frame = cap.read()
        if not ret:
            continue
        current_frame = frame

    cap.release()


# =======================================
# HILO 2: Procesamiento e inferencia
# =======================================
def emotion_thread():
    global current_frame, processed_frame, running

    while running:
        if current_frame is None:
            continue

        frame = current_frame.copy()
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]

            # Extraer x,y de 468 puntos
            landmarks = []
            for lm in face.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            landmarks = np.array(landmarks).reshape(1, -1)

            # Predicción
            pred = model.predict(landmarks)[0]
            label = CLASSES[pred]

            # Caja del rostro
            xs = [lm.x for lm in face.landmark]
            ys = [lm.y for lm in face.landmark]
            xmin = int(min(xs) * w)
            xmax = int(max(xs) * w)
            ymin = int(min(ys) * h)
            ymax = int(max(ys) * h)

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)
            cv2.putText(frame, label, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        processed_frame = frame


# =======================================
# HILO 3: Mostrar ventana (solo UNA)
# =======================================
def display_thread():
    global processed_frame, running

    while running:
        if processed_frame is not None:
            cv2.imshow("Detector de Emociones - 3 Clases", processed_frame)

        # Presiona Q para cerrar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cv2.destroyAllWindows()


# =======================================
# MAIN
# =======================================
if __name__ == "__main__":
    print("▶ Iniciando detector de emociones...")

    t1 = threading.Thread(target=camera_thread)
    t2 = threading.Thread(target=emotion_thread)
    t3 = threading.Thread(target=display_thread)

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    print("🟢 Detector finalizado.")
```
<img width="1600" height="568" alt="imagen" src="https://github.com/user-attachments/assets/1eb658fb-6599-4025-a673-579fc091512d" />

## Control de Versiones con Git y GitHub

Todo el proyecto está gestionado con **Git**. Puedo modificar archivos directamente desde GitHub y luego sincronizar los cambios en Linux con `git pull`, manteniendo el proyecto siempre actualizado y versionado.
<img width="762" height="700" alt="imagen" src="https://github.com/user-attachments/assets/682c5243-9c0b-4a7f-8e18-ecf48741b8af" />
<img width="762" height="700" alt="imagen" src="https://github.com/user-attachments/assets/28053c00-7c5a-4bc0-bfae-f0db867b3bb0" />
<img width="762" height="700" alt="imagen" src="https://github.com/user-attachments/assets/786fc5e1-efaf-45d0-8460-3d3c9c214e62" />

# Punto 2: ETL y Dashboard – Prevención del Síndrome de Túnel Carpiano

Este proyecto implementa un flujo ETL (Extract, Transform, Load) y un dashboard en Streamlit para visualizar información relacionada con la prevención del síndrome de túnel carpiano. Se trabaja con datos sintéticos generados en la carpeta `data/raw/`, que permiten simular un escenario real sin necesidad de contar con una base de datos clínica.  

El objetivo principal es que los estudiantes integren conceptos de la materia y desarrollen un pipeline reproducible. El proceso incluye la integración de datos, validaciones de calidad, creación de variables relevantes para modelos de redes neuronales, y la construcción de un dashboard que comunique métricas y tendencias de riesgo. Todo se organiza en carpetas claras para mantener reproducibilidad y facilitar la colaboración.  

---

## Objetivos del proyecto

- **Integración:** unificar y documentar el proceso ETL sobre la base de datos compartida.  
- **Calidad de datos:** aplicar validaciones, manejo de nulos, outliers y estandarización.  
- **Feature engineering:** generar variables relevantes para modelos de prevención con redes neuronales.  
- **Visualización:** construir un dashboard en Streamlit que comunique métricas, tendencias y riesgos.  
- **Reproducibilidad:** entregar un pipeline ejecutable con Makefile y configuración externa.  

---

## Flujo de trabajo

El flujo de trabajo se divide en tres etapas principales. Primero se generan datos sintéticos para poblar la carpeta `data/raw/`. Estos datos simulan pacientes y evaluaciones clínicas, permitiendo probar el pipeline sin necesidad de contar con información real.  

Luego se ejecuta el ETL, que extrae los datos, los transforma aplicando limpieza y creación de variables como el índice de exposición repetitiva y el score de riesgo, y finalmente los carga en la carpeta `data/processed/`.  

Por último, se lanza el dashboard en Streamlit. Este dashboard permite explorar métricas clave como número de pacientes, evaluaciones realizadas y porcentaje de riesgo alto, además de visualizar gráficas interactivas que muestran relaciones entre edad, fuerza de prensión y latencia EMG.  

---

## Estructura del repositorio

El proyecto está organizado en carpetas para mantener claridad y reproducibilidad:

- **etl/**: contiene los módulos de extracción, transformación, carga y el generador de datos sintéticos.  
- **dashboards/**: incluye la aplicación de Streamlit para visualizar resultados.  
- **data/raw/**: almacena los insumos originales (CSVs sintéticos).  
- **data/processed/**: guarda las salidas del ETL (dataset analítico).  
- **Makefile**: define comandos abreviados para ejecutar el ETL, lanzar el dashboard y limpiar salidas.  
- **README.md**: documentación del proyecto.  

<img width="1600" height="1007" alt="imagen" src="https://github.com/user-attachments/assets/b0f61ac8-67ef-4226-9b87-ce3b15f232b1" />

---

## Ejecución paso a paso

1. Clonar el repositorio desde GitHub y entrar en la carpeta del proyecto.  
2. Crear un entorno virtual e instalar las dependencias necesarias (pandas, numpy, streamlit, plotly, faker).  
3. Generar los datos sintéticos para llenar `data/raw/`.  
4. Ejecutar el ETL para producir el dataset analítico en `data/processed/`.  
5. Lanzar el dashboard en Streamlit y explorar las métricas y gráficas.  

<img width="1600" height="1007" alt="imagen" src="https://github.com/user-attachments/assets/d5cd0aa6-c00d-4f34-a74f-06360fc96d6e" />

---

## Buenas prácticas

Se recomienda no subir datos sensibles ni archivos pesados al repositorio. Para ello, se debe usar un archivo `.gitignore` que excluya las carpetas de datos (`data/raw/`, `data/processed/`) y el entorno virtual. De esta manera, solo se versionan los scripts y la documentación, garantizando que el repositorio sea ligero y seguro.  

Además, es importante documentar claramente los pasos de instalación y ejecución para que cualquier persona pueda reproducir el flujo. El proyecto puede extenderse con validaciones de calidad, pruebas automáticas y configuración externa en YAML, lo que lo hace más robusto y adaptable.  

---

## Resultado esperado

Al finalizar, se obtiene un pipeline reproducible que transforma datos en un dataset analítico y un dashboard interactivo. Esto demuestra el proceso completo de integración de datos, limpieza, creación de variables y visualización, aplicando conceptos de la materia en un caso práctico.  

<img width="1600" height="1007" alt="imagen" src="https://github.com/user-attachments/assets/4608a9d1-e611-4c0c-8f87-a3d5b8282755" />

El dashboard permite comunicar de manera clara y visual los hallazgos del ETL, mostrando métricas relevantes y gráficas que ayudan a comprender mejor los factores de riesgo asociados al síndrome de túnel carpiano.  

---

## Evidencia

📂 **Nota final:** mirar la carpeta **Laboratorio 2** para comprobar la evidencia del trabajo realizado.  

# Punto 3: Exploración de Tecnologías Clave en la Nube

## a) Terraform
- **Definición**: Terraform es una herramienta de *Infraestructura como Código (IaC)* desarrollada por HashiCorp. Permite definir, aprovisionar y gestionar infraestructura en múltiples proveedores de nube (AWS, Azure, GCP, OpenStack, etc.) mediante archivos de configuración declarativos.  
- **Características principales**:
  - Lenguaje declarativo (HCL) para describir el estado deseado de la infraestructura.
  - Control de versiones y reproducibilidad de entornos.
  - Compatible con infraestructura híbrida y multi-nube.
  - Genera un *plan de ejecución* que muestra los cambios antes de aplicarlos.  
- **Ventajas**:
  - Escalabilidad y automatización.
  - Reducción de errores humanos.
  - Integración con pipelines DevOps.  

---

## b) Ansible
- **Definición**: Ansible es una plataforma *open source* de automatización desarrollada por Red Hat. Se utiliza para la gestión de configuración, despliegue de aplicaciones y orquestación de sistemas.  
- **Características principales**:
  - **Automatización sin agentes**: no requiere instalar software en los nodos gestionados.
  - Uso de *Playbooks* escritos en YAML para definir tareas.
  - Modularidad: incluye cientos de módulos para gestionar servidores, redes y aplicaciones.
  - Escalable y fácil de integrar en entornos DevOps.  
- **Ventajas**:
  - Simplifica tareas repetitivas.
  - Mejora la seguridad y cumplimiento normativo.
  - Reduce errores en despliegues complejos.  

---

## c) RabbitMQ
- **Definición**: RabbitMQ es un *message broker* de código abierto que implementa el estándar **AMQP (Advanced Message Queuing Protocol)**.  
- **Características principales**:
  - Permite la comunicación asíncrona entre aplicaciones mediante colas de mensajes.
  - Soporta múltiples protocolos (AMQP, MQTT, STOMP).
  - Programado en Erlang, altamente confiable y escalable.
  - Ideal para arquitecturas distribuidas, microservicios e IoT.  
- **Ventajas**:
  - Desacopla servicios y mejora la resiliencia.
  - Escalabilidad horizontal mediante clústeres.
  - Amplio ecosistema y soporte empresarial.  

---

## d) Tecnologías OpenStack para la generación de nubes propias
- **Definición**: OpenStack es una plataforma *open source* que permite crear y gestionar nubes privadas y públicas, similar a montar un AWS en infraestructura propia.  
- **Componentes principales**:
  - **Nova**: gestión de máquinas virtuales.
  - **Neutron**: redes definidas por software.
  - **Swift**: almacenamiento de objetos.
  - **Cinder**: almacenamiento en bloques.
  - **Keystone**: autenticación y autorización.  
- **Ventajas**:
  - Evita dependencia de proveedores (*vendor lock-in*).
  - Altamente personalizable.
  - Escalable y flexible para empresas y universidades.  
- **Casos de uso**:
  - Nubes privadas corporativas.
  - Centros de investigación.
  - Proveedores de servicios locales.  

---

## e) Análisis del Cuadrante Mágico de Gartner sobre tecnologías orientadas a la nube
- **Definición**: El *Cuadrante Mágico de Gartner* es un informe que evalúa proveedores de servicios de nube según dos dimensiones: **capacidad de ejecución** y **completitud de visión**.  
- **Tendencias recientes (2024-2025)**:
  - **Líderes**: AWS, Microsoft Azure y Google Cloud siguen dominando por su capacidad de ejecución y visión estratégica.
  - **Retadores**: IBM Cloud y Oracle Cloud, con fuerte presencia en sectores específicos.
  - **Visionarios**: Proveedores como Huawei Cloud y Alibaba Cloud, con innovación en mercados emergentes.
  - **Jugadores de nicho**: Empresas regionales o especializadas en servicios específicos.  
- **Importancia**:
  - Ayuda a las organizaciones a seleccionar proveedores estratégicos.
  - Refleja la evolución hacia servicios multi-nube y plataformas híbridas.
  - Destaca la relevancia de la seguridad, escalabilidad y soporte global.  

---

# 📌 Conclusión
Estas tecnologías representan pilares fundamentales en la **automatización, comunicación distribuida y gestión de nubes privadas y públicas**.  
- **Terraform y Ansible**: Automatización y reproducibilidad de infraestructura.  
- **RabbitMQ**: Comunicación confiable en arquitecturas distribuidas.  
- **OpenStack**: Alternativa open source para nubes privadas.  
- **Gartner**: Brinda un mapa estratégico para entender el mercado de la nube y sus principales actores.

# Punto 4: Proyecto: IA para la Prevención de Contaminación Atmosférica en Ciudades Colombianas

## 📌 Contexto
La convocatoria **IA para la Sostenibilidad y el Cambio Climático** de MinCiencias busca proyectos que apliquen inteligencia artificial para enfrentar retos ambientales.  
Como ingenieros electrónicos, proponemos un sistema inteligente de monitoreo de calidad del aire en zonas urbanas, aplicando lo aprendido en **Digitales III**.

---

## 🎯 Objetivo del Proyecto
Desarrollar una red de sensores IoT de bajo costo que:
- Midan partículas (PM2.5, PM10), gases (CO₂, NOx) y variables ambientales.
- Procesen las señales en tiempo real con técnicas de **Digitales III** (filtrado digital, FFT, normalización).
- Entrenen un modelo de IA para predecir episodios críticos de contaminación.
- Generen alertas tempranas y recomendaciones para autoridades y ciudadanos.

---

## 🛠️ Metodología
1. **Adquisición de datos**: Sensores IoT desplegados en puntos estratégicos de la ciudad.  
2. **Procesamiento digital**: Filtrado pasa-banda, normalización y extracción de características.  
3. **Modelo de IA**: Redes neuronales para predicción de contaminación.  
4. **Dashboard**: Visualización en tiempo real para ciudadanos y autoridades.  
5. **Infraestructura**: Integración con nube privada (OpenStack) para almacenamiento seguro.  

---

## 🏗️ Arquitectura del Sistema

```mermaid
flowchart TD
    A[Sensores IoT de Calidad del Aire] --> B[Microcontrolador con Procesamiento Digital]
    B --> C[Red de Comunicación LoRa/WiFi]
    C --> D[Servidor en la Nube]
    D --> E[Modelo de IA - Predicción de Contaminación]
    E --> F[Dashboard Ciudadano y Alertas]

---

📌 Este diagrama muestra el flujo completo:
- **Sensores IoT** capturan los datos acústicos.  
- **Microcontrolador con DSP** realiza el procesamiento digital (filtrado y FFT).  
- Se extraen características relevantes y se envían por **red de comunicación**.  
- En el **servidor en la nube**, un modelo de IA predice niveles de ruido.  
- Finalmente, se generan **alertas y visualizaciones** en un dashboard ciudadano.  

## 💡 Ideas de Aplicación
- **Sensores más pequeños y baratos**  
  Que se puedan poner en casas, colegios o fábricas para medir ruido, temperatura o calidad del aire sin necesidad de equipos costosos.

- **Dispositivos que se cargan solos**  
  Sensores que aprovechen la luz solar o el movimiento para no depender de baterías que hay que cambiar todo el tiempo.

- **Aplicaciones móviles de alerta**  
  Apps sencillas que muestren en tiempo real si hay niveles altos de ruido, contaminación o calor en tu zona.

- **Mapas interactivos ciudadanos**  
  Plataformas que permitan ver en un mapa cómo están las condiciones ambientales en tu barrio y recibir recomendaciones.

- **Redes comunitarias de datos**  
  Vecinos o instituciones que compartan la información de sus sensores para tener una visión más completa de la ciudad.

---

## 🎯 Impacto Esperado
- **Social:** Mayor conciencia ciudadana sobre el ambiente y la salud.  
- **Tecnológico:** Uso de herramientas simples y accesibles para todos.  
- **Económico:** Soluciones de bajo costo que pueden escalarse fácilmente.  

---

## 📌 Conclusión
Las tecnologías futuras no siempre tienen que ser complejas. A veces lo más útil es lo más sencillo: sensores accesibles, aplicaciones fáciles de usar y redes comunitarias que permitan mejorar la calidad de vida con información clara y práctica.


