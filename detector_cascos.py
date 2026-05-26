from ultralytics import YOLO
import cv2
from playsound import playsound
import threading
import time

# ==========================================
# CARGAR MODELO YOLO
# ==========================================

model = YOLO("models/my_model.pt")

# ==========================================
# ABRIR CAMARA
# ==========================================

cap = cv2.VideoCapture(0)

# ==========================================
# CONTROL DE SONIDOS
# ==========================================

ultimo_sonido_rojo = 0

intervalo_rojo = 9

# ==========================================
# FUNCIONES DE SONIDO
# ==========================================

def sonido_rojo():
    try:
        playsound("alerta.mp3")
    except Exception as e:
        print("Error sonido rojo:", e)

# ==========================================
# MENSAJE INICIAL
# ==========================================

print("Sistema iniciado... Presiona 'q' para salir.")

# ==========================================
# LOOP PRINCIPAL
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # ==========================================
    # INFERENCIA YOLO
    # ==========================================

    resultados = model(frame, conf=0.5)

    estado = "SIN_DETECCION"

    # ==========================================
    # RECORRER DETECCIONES
    # ==========================================

    for resultado in resultados:

        for box in resultado.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            confianza = float(box.conf[0])

            clase = int(box.cls[0])

            nombre_clase = model.names[clase]

            nombre_clase = nombre_clase.lower()

            # ==========================================
            # CLASE SIN CASCO
            # ==========================================

            if nombre_clase == "sin_casco":

                color = (0, 0, 255)

                estado = "ROJO"

            # ==========================================
            # CLASE CON CASCO
            # ==========================================

            else:

                color = (0, 255, 0)

                if estado != "ROJO":
                    estado = "VERDE"

            # ==========================================
            # DIBUJAR DETECCION
            # ==========================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            texto = f"{nombre_clase} {confianza:.2f}"

            cv2.putText(
                frame,
                texto,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    # ==========================================
    # DIBUJAR SEMAFORO
    # ==========================================

    cv2.rectangle(frame, (20, 20), (120, 320), (40, 40, 40), -1)

    rojo = (50, 50, 50)
    amarillo = (50, 50, 50)
    verde = (50, 50, 50)

    tiempo_actual = time.time()

    # ==========================================
    # ESTADO ROJO
    # ==========================================

    if estado == "ROJO":

        rojo = (0, 0, 255)

        if tiempo_actual - ultimo_sonido_rojo > intervalo_rojo:

            threading.Thread(target=sonido_rojo).start()

            ultimo_sonido_rojo = tiempo_actual

    # ==========================================
    # ESTADO VERDE
    # ==========================================

    elif estado == "VERDE":

        verde = (0, 255, 0)

    # ==========================================
    # SIN DETECCION
    # ==========================================

    else:

        amarillo = (0, 255, 255)

    # ==========================================
    # LUCES DEL SEMAFORO
    # ==========================================

    cv2.circle(frame, (70, 70), 30, rojo, -1)

    cv2.circle(frame, (70, 170), 30, amarillo, -1)

    cv2.circle(frame, (70, 270), 30, verde, -1)

    # ==========================================
    # TEXTO DE ESTADO
    # ==========================================

    cv2.putText(
        frame,
        f"Estado: {estado}",
        (150, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # ==========================================
    # MOSTRAR VENTANA
    # ==========================================q

    cv2.imshow("Semaforo Inteligente", frame)

    # ==========================================
    # SALIR
    # ==========================================

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================================
# LIBERAR RECURSOS
# ==========================================

cap.release()

cv2.destroyAllWindows()