# Laboratorio-7
# Detector de Emociones con MediaPipe y Random Forest

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




