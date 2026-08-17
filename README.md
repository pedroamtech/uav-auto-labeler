# uav-auto-labeler

Herramienta de auto-etiquetado para imagenes aereas/UAV usando un modelo YOLOv8 (`yolov8x-visdrone`) afinado sobre el dataset VisDrone. Genera etiquetas en formato YOLO fusionando las clases `pedestrian` y `people` en una unica clase `person` (id `0`), mediante inferencia por lotes en GPU. El modelo se descarga automaticamente desde Hugging Face Hub si no esta presente localmente.

## Estructura

- [main.py](main.py) — punto de entrada. Descarga el modelo si falta, corre la inferencia y escribe las etiquetas. No requiere modificaciones para uso normal.
- [config.py](config.py) — toda la configuracion editable: rutas del dataset, modelo y parametros de inferencia.
- [requirements.txt](requirements.txt) — dependencias de Python.

## Requisitos

- Python 3.9+
- GPU con CUDA (opcional, ver `DEVICE` en la configuracion para usar CPU)

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Configuracion

Todas las rutas y parametros se ajustan en [config.py](config.py), sin tocar `main.py`:

| Variable     | Descripcion                                                          |
|--------------|-----------------------------------------------------------------------|
| `HF_REPO`    | Repositorio de Hugging Face del que se descarga el modelo             |
| `MODEL_FILE` | Ruta local donde se guarda/busca el archivo `.pt` del modelo          |
| `IMG_DIR`    | Carpeta con las imagenes de entrada a etiquetar                       |
| `LABEL_DIR`  | Carpeta de salida donde se escriben las etiquetas `.txt` (formato YOLO) |
| `CLASSES`    | IDs de clase VisDrone a detectar (`0`=pedestrian, `1`=people)         |
| `CONF`       | Confianza minima de deteccion                                         |
| `IMGSZ`      | Resolucion de entrada del modelo (no cambiar: debe coincidir con el entrenamiento VisDrone) |
| `DEVICE`     | Dispositivo de inferencia (`0` para GPU, `"cpu"` para CPU)            |
| `BATCH`      | Tamano de batch, ajustar segun VRAM disponible                        |

Para etiquetar un dataset distinto, solo hay que editar `IMG_DIR` y `LABEL_DIR` en `config.py`.

## Uso

```bash
python main.py
```

El script:

1. Verifica si el modelo (`MODEL_FILE`) existe localmente; si no, lo descarga desde `HF_REPO`.
2. Crea `LABEL_DIR` si no existe.
3. Corre inferencia en streaming/batch sobre todas las imagenes de `IMG_DIR`, mostrando una barra de progreso (`tqdm`).
4. Por cada imagen, escribe un archivo `.txt` en `LABEL_DIR` con una linea por deteccion en formato YOLO normalizado:

   ```
   0 x_center y_center width height
   ```

   donde `0` es la clase `person` (fusion de `pedestrian` + `people`) y las coordenadas estan normalizadas entre 0 y 1.

La consola muestra el progreso paso a paso (descarga/carga del modelo, conteo de imagenes, barra de avance) y un resumen final con el total de imagenes procesadas, imagenes con detecciones y detecciones totales:

```
[1/3] Modelo local encontrado: yolov8x-visdrone.pt
[2/3] Cargando modelo...
      842 imagenes encontradas en C:/.../train/images
[3/3] Etiquetando imagenes...
Etiquetando: 100%|██████████| 842/842 [02:14<00:00,  6.28img/s]

Etiquetado completo.
  Imagenes procesadas:      842
  Imagenes con detecciones: 731
  Detecciones totales:      5210
  Etiquetas guardadas en:   C:/.../train/labels
```
