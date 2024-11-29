import numpy as np
import cv2

def ordenar_puntos(puntos):
    puntos = np.concatenate(puntos).tolist()
    puntos_ordenados = sorted(puntos, key=lambda punto: punto[1])
    
    parte_superior = sorted(puntos_ordenados[:2], key=lambda punto: punto[0])
    parte_inferior = sorted(puntos_ordenados[2:], key=lambda punto: punto[0])
    
    return [parte_superior[0], parte_superior[1], parte_inferior[0], parte_inferior[1]]

def roi(image, ancho, alto):
    imagen_alineada = None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Usamos un umbral adaptativo
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY, 11, 2)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
    
    for c in cnts:
        epsilon = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        
        if len(approx) == 4:
            puntos = ordenar_puntos(approx)
            pts1 = np.float32(puntos)
            pts2 = np.float32([[0, 0], [ancho, 0], [0, alto], [ancho, alto]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            imagen_alineada = cv2.warpPerspective(image, M, (ancho, alto))
    return imagen_alineada

cap = cv2.VideoCapture(0)

while True: 
    ret, frame = cap.read()
    if not ret:
        break

    # Reducir ruido en la imagen
    frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
    
    imagen_A4 = roi(frame_blur, ancho=720, alto=509)
    if imagen_A4 is not None:
        # Convertimos a escala de grises y aplicamos Canny
        gray = cv2.cvtColor(imagen_A4, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 200)  # Valores ajustados para mejor detección
        edges = cv2.dilate(edges, None, iterations=1)
        edges = cv2.erode(edges, None, iterations=1)
        
        # Encontramos los contornos
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            if cv2.contourArea(c) < 500 or cv2.contourArea(c) > 50000:
                continue  # Ignorar contornos muy pequeños o grandes

            epsilon = 0.02 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            x, y, w, h = cv2.boundingRect(approx)

            # Clasificación de las figuras
            if len(approx) == 3:
                cv2.putText(imagen_A4, 'Triangulo', (x, y - 10), 1, 1, (0, 255, 0), 2)
            elif len(approx) == 4:
                aspect_ratio = float(w) / h
                if 0.9 <= aspect_ratio <= 1.1:  # Mayor tolerancia para cuadrado
                    cv2.putText(imagen_A4, 'Cuadrado', (x, y - 10), 1, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(imagen_A4, 'Rectangulo', (x, y - 10), 1, 1, (0, 255, 0), 2)
            elif len(approx) == 5:
                cv2.putText(imagen_A4, 'Pentagono', (x, y - 10), 1, 1, (0, 255, 0), 2)
            elif 6 <= len(approx) <= 10:
                cv2.putText(imagen_A4, 'Hexagono', (x, y - 10), 1, 1, (0, 255, 0), 2)
            elif len(approx) > 10:
                cv2.putText(imagen_A4, 'Circulo', (x, y - 10), 1, 1, (0, 255, 0), 2)

            # Dibujar contorno
            cv2.drawContours(imagen_A4, [approx], 0, (0, 255, 0), 2)

        cv2.imshow('Figura Detectada', imagen_A4)

    cv2.imshow('Camara', frame)
    
    # Salir si presionamos la tecla ESC
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
