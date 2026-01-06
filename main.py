# Deformacion de imagenes con malla triangular
# 1. IMPORTACIONES
import cv2
import numpy as np
import math

puntos_originales = []
puntos_malla = []  
arrastrando = False
indice_seleccionado = -1
necesita_actualizar = True

NOMBRE_IMAGEN = "img.jpg"
ANCHO_MAXIMO = 800
RADIO_TOLERANCIA = 10

# 2. FUNCIONES DE CONFIGURACIÓN
def seleccionar_interpolacion():
    print("\n--- CALIDAD DE RENDERIZADO ---")
    print("1. Bilineal (Rápido, recomendado)")
    print("2. Bicúbica (Mejor calidad, más lento)")
    while True:
        opcion = input("Elige [1 o 2]: ").strip()
        if opcion == "1":
            return cv2.INTER_LINEAR
        elif opcion == "2":
            return cv2.INTER_CUBIC

def obtener_configuracion_malla():
    print("\n--- CONFIGURACIÓN DE LA MALLA ---")
    while True:
        try:
            print("Recomendado: 4 a 6 para fluidez óptima.")
            c = int(input("Columnas: "))
            r = int(input("Filas: "))
            if 2 <= c <= 30 and 2 <= r <= 30:
                return r, c
            print("⚠️ Valores entre 2 y 30.")
        except ValueError:
            print("⚠️ Número entero requerido.")

# 3. GENERACIÓN DE MALLA
#                         30    30     2     2
def generar_datos_malla(ancho, alto, filas, cols):
    lista_puntos = []
    ancho_max = max(0, ancho - 1)
    alto_max = max(0, alto - 1)

    for fila in range(filas + 1):
        y = (fila * alto_max) // filas
        for col in range(cols + 1):
            x = (col * ancho_max) // cols
            lista_puntos.append((int(x), int(y)))

    lista_triangulos = []   
    puntos_por_fila = cols + 1 
    for fila in range(filas):
        for col in range(cols):
            esquina_superior_izquierda = fila * puntos_por_fila + col
            esquina_superior_derecha = esquina_superior_izquierda + 1
            esquina_inferior_izquierda = (fila + 1) * puntos_por_fila + col
            esquina_inferior_derecha = esquina_inferior_izquierda + 1

            lista_triangulos.append((esquina_superior_izquierda, esquina_inferior_izquierda, esquina_inferior_derecha))
            lista_triangulos.append((esquina_superior_izquierda, esquina_inferior_derecha, esquina_superior_derecha))

    return lista_puntos, lista_triangulos

# 4. FUNCIONES MATEMATICAS
def area_triangulo(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

# 5. NUCLEO DEL PANDEO
def aplicar_warping_triangular(img_src, tri_src, tri_dst, metodo_interp):
    if abs(area_triangulo(tri_dst[0], tri_dst[1], tri_dst[2])) < 1:
        return None, None, None

    r_src = cv2.boundingRect(np.float32([tri_src]))
    r_dst = cv2.boundingRect(np.float32([tri_dst]))

    x_src, y_src, w_src, h_src = r_src
    x_dst, y_dst, w_dst, h_dst = r_dst

    if w_dst <= 0 or h_dst <= 0:
        return None, None, None

    crop_src = img_src[y_src:y_src + h_src, x_src:x_src + w_src]
    if crop_src.size == 0:
        return None, None, None

    pts_src_rel = []
    pts_dst_rel = []
    for i in range(3):
        pts_src_rel.append((tri_src[i][0] - x_src, tri_src[i][1] - y_src))
        pts_dst_rel.append((tri_dst[i][0] - x_dst, tri_dst[i][1] - y_dst))

    matriz = cv2.getAffineTransform(np.float32(pts_src_rel), np.float32(pts_dst_rel))

    warp_dst = cv2.warpAffine(
        crop_src, matriz, (w_dst, h_dst),
        flags=metodo_interp,
        borderMode=cv2.BORDER_REFLECT_101
    )

    mascara = np.zeros((h_dst, w_dst, 3), dtype=np.uint8)
    cv2.fillConvexPoly(mascara, np.int32(pts_dst_rel), (1, 1, 1), 16, 0)

    return warp_dst * mascara, mascara, r_dst

def renderizar_imagen_completa(img_original, triangulos, interpolacion):
    canvas = np.zeros_like(img_original)
    h_canvas, w_canvas = canvas.shape[:2]

    for tri in triangulos:
        t_src = [puntos_originales[tri[0]], puntos_originales[tri[1]], puntos_originales[tri[2]]]
        t_dst = [puntos_malla[tri[0]], puntos_malla[tri[1]], puntos_malla[tri[2]]]

        trozo_warped, mascara, rect_dst = aplicar_warping_triangular(
            img_original, t_src, t_dst, interpolacion
        )
        if trozo_warped is None:
            continue

        x, y, w, h = rect_dst

        y1, y2 = y, y + h
        x1, x2 = x, x + w

        cy1 = max(0, y1)
        cy2 = min(h_canvas, y2)
        cx1 = max(0, x1)
        cx2 = min(w_canvas, x2)

        if cy1 >= cy2 or cx1 >= cx2:
            continue

        oy1 = cy1 - y1
        oy2 = h - (y2 - cy2)
        ox1 = cx1 - x1
        ox2 = w - (x2 - cx2)

        zona_canvas = canvas[cy1:cy2, cx1:cx2]
        trozo_recortado = trozo_warped[oy1:oy2, ox1:ox2]
        mascara_recortada = mascara[oy1:oy2, ox1:ox2]

        canvas[cy1:cy2, cx1:cx2] = zona_canvas * (1 - mascara_recortada) + trozo_recortado

    return canvas

# 6. INTERFAZ DE USUARIO
def manejador_mouse(event, x, y, flags, param):
    global puntos_malla, arrastrando, indice_seleccionado, necesita_actualizar

    ancho, alto = param
    ancho_limite = ancho - 1
    alto_limite = alto - 1

    if event == cv2.EVENT_LBUTTONDOWN:
        mejor_i = -1
        mejor_d = float("inf")
        for i, (px, py) in enumerate(puntos_malla):
            d = math.hypot(x - px, y - py)
            if d < RADIO_TOLERANCIA and d < mejor_d:
                mejor_d = d
                mejor_i = i

        if mejor_i != -1:
            indice_seleccionado = mejor_i
            arrastrando = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if arrastrando and indice_seleccionado != -1:
            nuevo_x = int(np.clip(x, 0, ancho_limite))
            nuevo_y = int(np.clip(y, 0, alto_limite))
            puntos_malla[indice_seleccionado] = (nuevo_x, nuevo_y)
            necesita_actualizar = True

    elif event == cv2.EVENT_LBUTTONUP:
        if arrastrando:
            necesita_actualizar = True
        arrastrando = False
        indice_seleccionado = -1

def dibujar_interfaz(imagen, triangulos):
    visual = imagen.copy()

    for tri in triangulos:
        pt1 = puntos_malla[tri[0]]
        pt2 = puntos_malla[tri[1]]
        pt3 = puntos_malla[tri[2]]
        cv2.line(visual, pt1, pt2, (200, 255, 0), 1, cv2.LINE_AA)
        cv2.line(visual, pt2, pt3, (200, 255, 0), 1, cv2.LINE_AA)
        cv2.line(visual, pt3, pt1, (200, 255, 0), 1, cv2.LINE_AA)

    for i, p in enumerate(puntos_malla):
        color = (0, 0, 255)
        radio = 4
        if i == indice_seleccionado:
            color = (0, 255, 0)
            radio = 6
        cv2.circle(visual, p, radio, color, -1, cv2.LINE_AA)

    cv2.putText(
        visual,
        "[Mouse] Arrastra puntos   [S] Guardar   [R] Reset   [ESC] Salir",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return visual

# 7. GESTION DE ARCHIVO
def guardar_imagen(imagen):
    nombre = "resultado_warping.jpg"
    cv2.imwrite(nombre, imagen)   
    print(f"Imagen guardada como: {nombre}")
    return nombre

# 8. fUNCION PRINCIPAL
def iniciar_proyecto_final():
    global puntos_malla, puntos_originales, necesita_actualizar

    img = cv2.imread(NOMBRE_IMAGEN)
    if img is None:
        print(f" Error: No se encuentra '{NOMBRE_IMAGEN}'")
        return

    alto, ancho = img.shape[:2]
    if ancho > ANCHO_MAXIMO:
        escala = ANCHO_MAXIMO / float(ancho)
        img = cv2.resize(
            img,
            (int(ancho * escala), int(alto * escala)),
            interpolation=cv2.INTER_AREA
        )
    alto, ancho = img.shape[:2]

    filas, cols = obtener_configuracion_malla()
    metodo_interp = seleccionar_interpolacion()

    puntos_malla, triangulos = generar_datos_malla(ancho, alto, filas, cols) ########---------
    puntos_originales = list(puntos_malla)
    necesita_actualizar = True

    nombre_ventana = "Proyecto Final - Deformacion de imagen con maya triangular"
    cv2.namedWindow(nombre_ventana)
    cv2.setMouseCallback(nombre_ventana, manejador_mouse, param=(ancho, alto))

    print("\n--- PROYECTO INICIADO ---")
    print("Controles:                 ")
    print("  [Mouse] Arrastrar puntos ")
    print("  [  R  ] Eesetear malla   ")
    print("  [  S  ] Guardar resultado")
    print("  [ ESC ] Salir            ")

    imagen_warped = img.copy()

    while True:
        if necesita_actualizar:
            imagen_warped = renderizar_imagen_completa(img, triangulos, metodo_interp)
            necesita_actualizar = False

        imagen_final = dibujar_interfaz(imagen_warped, triangulos)
        cv2.imshow(nombre_ventana, imagen_final)

        k = cv2.waitKey(1) & 0xFF

        if k == 27 or cv2.getWindowProperty(nombre_ventana, cv2.WND_PROP_VISIBLE) < 1:
            break

        if k == ord('r') or k == ord('R'):
            print("Reseteando malla...")
            puntos_malla = list(puntos_originales)
            necesita_actualizar = True

        if k == ord('s') or k == ord('S'):
            guardar_imagen(imagen_warped)

    cv2.destroyAllWindows()
    print("Programa finalizado.")

# 9. ENTRADA DEL PROGRAMA
if __name__ == "__main__":
    iniciar_proyecto_final()