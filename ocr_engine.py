# pre-procesado, YOLO y EasyOCR

import cv2, numpy as np, os, time, torch
import easyocr
from ultralytics import YOLO
from typing import List, Tuple

#-----------------------------------
#---------- Configuración ----------
#-----------------------------------
GPU_FLAG   = os.getenv("GPU_EASYOCR", "1") == "1" and torch.cuda.is_available()
LANGS      = os.getenv("EASYOCR_LANGS", "es,en").split(",")
YOLO_W     = os.getenv("YOLO_WEIGHTS", "yolov8n_tables.pt")
YOLO_CONF  = float(os.getenv("YOLO_CONF_THR", 0.10))
YOLO_IOU   = float(os.getenv("YOLO_IOU_THR", 0.80))
PAD        = 4  # píxeles extra alrededor de la tabla

#-----------------------------------
#---------- Inicialización ---------
#-----------------------------------

# Carga el modelo YOLO y EasyOCR
_reader = easyocr.Reader(LANGS, gpu=GPU_FLAG)
try:
    _yolo = YOLO(YOLO_W)
except Exception as e:
    print(f"[OCR] No se pudo cargar YOLO ({e}); se usará imagen completa")
    _yolo = None


# -----------------------------------------
# ---------- Funciones auxiliares ---------
# -----------------------------------------

# Convierte bytes de imagen a formato RGB
def bytes_to_rgb(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Bytes no representan una imagen válida")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# Preprocesa la imagen para detectar tablas
def _crop_table(img_rgb: np.ndarray) -> np.ndarray:
    if _yolo is None:
        return img_rgb

    res   = _yolo.predict(img_rgb, imgsz=640, conf=YOLO_CONF,
                          iou=YOLO_IOU, verbose=False)
    boxes = res[0].boxes
    if not boxes:
        return img_rgb

    sel = max(boxes, key=lambda b: (b.xyxy[0][2]-b.xyxy[0][0])
                                * (b.xyxy[0][3]-b.xyxy[0][1]))
    x1,y1,x2,y2 = map(int, sel.xyxy[0].tolist())
    h,w,_ = img_rgb.shape
    return img_rgb[max(0,y1-PAD):min(h,y2+PAD),
                   max(0,x1-PAD):min(w,x2+PAD)]


# --------------------------------------
# ---------- Funciones de OCR ----------
# --------------------------------------
def recognise(image_bytes: bytes) -> Tuple[str, float]:
    """Devuelve (texto, latencia_ms) usando EasyOCR local"""
    rgb     = bytes_to_rgb(image_bytes)
    cropped = _crop_table(rgb)
    bgr     = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)

    t0   = time.perf_counter()
    res  = _reader.readtext(bgr, paragraph=True)
    t_ms = (time.perf_counter() - t0) * 1000
    text = " ".join(r[1] for r in res)
    return text, round(t_ms, 2)


def recognise_batch(images: List[bytes]) -> Tuple[List[str], float]:
    """OCR de varias imágenes, devuelve (lista_textos, latencia_ms_total)"""
    out = []
    t0  = time.perf_counter()
    for b in images:
        txt, _ = recognise(b)
        out.append(txt)
    t_ms = (time.perf_counter() - t0) * 1000
    return out, round(t_ms, 2)
