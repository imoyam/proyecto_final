import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage import filters
from sklearn.cluster import KMeans
import cv2
from cv2 import findContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE
from scipy import ndimage as ndi
from skimage.morphology import closing, disk, remove_small_objects
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# config

# Obtiene la carpeta donde se encuentra este script
CARPETA = os.path.dirname(os.path.abspath(__file__))
# Luego cambias al directorio (opcional, si usas rutas relativas)
os.chdir(CARPETA)

ARCHIVO_T1 = "mapa_T1_SAX.npy"      
ARCHIVO_T2 = "mapa_T2_SAX.npy"      

#high
N_OTSU_HIGH         = 3
N_TAKE_OTSU_HIGH    = 2
N_KMEANS_HIGH       = 14
N_TAKE_KMEANS_HIGH  = 5

#low
N_OTSU_LOW          = 3
N_TAKE_OTSU_LOW     = 2
N_KMEANS_LOW        = 16
N_TAKE_KMEANS_LOW   = 11

#comunes
SIGMA          = 1.0
MIN_ISLA       = 15
FONDO          = (0,)
RADIO_CIERRE   = 1
MIN_AREA       = 30
MOSTRAR        = True
GUARDAR        = False
CMAP_REGIONES  = "nipy_spectral"

os.chdir(CARPETA)
t1_sax = np.load(ARCHIVO_T1)
t2_sax = np.load(ARCHIVO_T2)

def normalizar(img):
    return img


def mascara_otsu(img, n_otsu, n_take_otsu, HighIntensity=True):
    """
    Máscara booleana que toma las n_take_otsu clases extremas de Otsu.
    """
    if n_otsu < 2:
        raise ValueError("n_otsu debe ser >= 2")
    if not (1 <= n_take_otsu <= n_otsu - 1):
        raise ValueError(f"n_take_otsu debe estar entre 1 y {n_otsu-1}")

    umbrales = filters.threshold_multiotsu(img, classes=n_otsu)
    clases = np.digitize(img, bins=umbrales)

    if HighIntensity:
        mascara = clases >= (n_otsu - n_take_otsu)
    else:
        mascara = clases < n_take_otsu
    return mascara


def segmentar_otsu_kmeans(imagen_2d, n_otsu, n_take_otsu,
                          n_kmeans, n_take_kmeans,
                          sigma, HighIntensity):
    '''
    Segmenta con Otsu y K-means.'''
    img = ndi.gaussian_filter(normalizar(imagen_2d), sigma=sigma)

    mask_otsu = mascara_otsu(img, n_otsu, n_take_otsu, HighIntensity)
    if not np.any(mask_otsu):
        print("Ningún píxel en la zona Otsu seleccionada.")
        return np.zeros_like(imagen_2d, dtype=int)

    valores = img[mask_otsu].reshape(-1, 1)
    k = min(n_kmeans, np.unique(valores).size)
    if k < 1:
        print("Demasiados pocos valores para K‑means.")
        return np.zeros_like(imagen_2d, dtype=int)

    km = KMeans(n_clusters=k, random_state=0, n_init="auto")
    etq = km.fit_predict(valores)

    centros = km.cluster_centers_.flatten()
    orden = np.argsort(centros)
    remap = np.zeros_like(orden)
    for nuevo, viejo in enumerate(orden, start=1):
        remap[viejo] = nuevo

    etiquetas_todos = np.zeros_like(imagen_2d, dtype=int)
    etiquetas_todos[mask_otsu] = remap[etq]

    if HighIntensity:
        umbral = k - n_take_kmeans + 1
        mascara_final = etiquetas_todos >= umbral
    else:
        mascara_final = etiquetas_todos <= n_take_kmeans

    resultado = np.zeros_like(imagen_2d, dtype=int)
    resultado[mascara_final] = etiquetas_todos[mascara_final]

    # Renumerar de 1 a n_take_kmeans
    conservados = np.unique(resultado)
    conservados = conservados[conservados > 0]
    if len(conservados) > 0:
        for nueva, vieja in enumerate(conservados, start=1):
            resultado[resultado == vieja] = nueva
    return resultado


def eliminar_islas_foreground(etiquetas_2d, min_pixeles, fondo=(0,), connectivity=2):
    '''
    Elimina regiones de foreground (no fondo) con menos de min_pixeles píxeles
    '''
    fg = ~np.isin(etiquetas_2d, fondo)
    if not fg.any():
        return etiquetas_2d
    fg_limpio = remove_small_objects(fg, min_size=min_pixeles, connectivity=connectivity)
    salida = etiquetas_2d.copy()
    salida[~fg_limpio] = 0
    return salida


def contornos_por_etiqueta(etiquetas_2d, omitir=(0,)):
    '''
    Devuelve un diccionario {etiqueta: [contornos]} de los contornos de cada etiqueta.'''
    conts = {}
    for etiq in np.unique(etiquetas_2d):
        if int(etiq) in omitir:
            continue
        mascara = (etiquetas_2d == etiq).astype(np.uint8) * 255
        cnts, _ = findContours(mascara, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
        if cnts:
            conts[int(etiq)] = [c.squeeze() for c in cnts if c.shape[0] > 0]
    return conts


def mapa_contornos(etiquetas_2d, conts_etiq):
    '''
    Crea un mapa de contornos a partir de un diccionario de contornos por etiqueta.
    '''
    mapa = np.zeros(etiquetas_2d.shape, dtype=np.uint8)
    for etiq, lista in conts_etiq.items():
        for c in lista:
            pts = c.reshape(-1, 2)
            for x, y in pts:
                x, y = int(x), int(y)
                if 0 <= x < mapa.shape[1] and 0 <= y < mapa.shape[0]:
                    mapa[y, x] = etiq
    return mapa


def contorno_envolvente(etiquetas_2d, fondo=(0,), radio_cierre=1, min_area=50):
    '''
    Devuelve los contornos de la región más grande (envolvente) y su centroide.
    '''
    fg = ~np.isin(etiquetas_2d, fondo)
    if not fg.any():
        return None, (0, 0)
    fg = closing(fg, disk(radio_cierre))
    fg = ndi.binary_fill_holes(fg)
    fg = remove_small_objects(fg, min_size=min_area)
    if not fg.any():
        return None, (0, 0)
    ys, xs = np.nonzero(fg)
    cy, cx = ys.mean(), xs.mean()
    lbl, n = ndi.label(fg)
    if n == 0:
        return None, (cx, cy)
    etiq = lbl[int(round(cy)), int(round(cx))]
    if etiq == 0:
        areas = np.bincount(lbl.ravel()); areas[0] = 0
        etiq = areas.argmax()
    mascara = (lbl == etiq).astype(np.uint8) * 255
    conts, _ = findContours(mascara, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
    return conts, (cx, cy)


def zonas_no_segmentadas(etiq_high, etiq_low):
    """Máscara booleana de píxeles no capturados por ninguna segmentación."""
    return (etiq_high == 0) & (etiq_low == 0)


def segmentar_volumen(volumen, n_otsu, n_take_otsu,
                      n_kmeans, n_take_kmeans, sigma, HighIntensity):
    n = volumen.shape[0]
    etiquetas_vol  = np.zeros_like(volumen, dtype=int)
    conts_vol      = [None] * n
    mapas_cont_vol = np.zeros_like(volumen, dtype=np.uint8)

    for i in range(n):
        print(f"Slice {i+1}/{n} (High={HighIntensity})...")
        etq = segmentar_otsu_kmeans(volumen[i],
                                    n_otsu, n_take_otsu,
                                    n_kmeans, n_take_kmeans,
                                    sigma, HighIntensity)
        etq = eliminar_islas_foreground(etq, MIN_ISLA, FONDO)
        conts = contornos_por_etiqueta(etq, omitir=FONDO)
        mapa  = mapa_contornos(etq, conts)
        etiquetas_vol[i]  = etq
        conts_vol[i]      = conts
        mapas_cont_vol[i] = mapa
    return etiquetas_vol, conts_vol, mapas_cont_vol


def envolventes_volumen(etiquetas_vol, fondo, radio_cierre, min_area):
    n = etiquetas_vol.shape[0]
    conts_env  = [None] * n
    centroides = [None] * n
    mapas_env  = np.zeros_like(etiquetas_vol, dtype=np.uint8)
    for i in range(n):
        conts, (cx, cy) = contorno_envolvente(etiquetas_vol[i], fondo, radio_cierre, min_area)
        conts_env[i]  = conts
        centroides[i] = (cx, cy)
        if conts:
            for c in conts:
                cv2.drawContours(mapas_env[i], [c], -1, 255, thickness=cv2.FILLED)
    return conts_env, centroides, mapas_env


# visor
def inspeccionar_cuatro(volumen, etiq_alta, etiq_baja, nombre):
    n = volumen.shape[0]
    n_etiq_alta = int(etiq_alta.max()) + 1
    n_etiq_baja = int(etiq_baja.max()) + 1
    cmap_alta = plt.get_cmap('plasma', n_etiq_alta)
    cmap_baja = plt.get_cmap('viridis', n_etiq_baja)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle(f"{nombre} — Alta | Baja | No segmentado", fontsize=11)
    plt.subplots_adjust(bottom=0.15, wspace=0.3)

    ax_orig, ax_alta, ax_baja, ax_no = axes

    def dibujar(idx):
        img = normalizar(volumen[idx])
        no_seg = zonas_no_segmentadas(etiq_alta[idx], etiq_baja[idx])

        # Original
        ax_orig.clear(); ax_orig.axis("off")
        ax_orig.imshow(img, cmap="gray")
        ax_orig.set_title(f"Corte {idx} – Original")

        # Alta
        ax_alta.clear(); ax_alta.axis("off")
        ax_alta.imshow(img, cmap="gray")
        ax_alta.imshow(np.ma.masked_equal(etiq_alta[idx], 0), cmap=cmap_alta, alpha=0.6,
                      vmin=0, vmax=max(n_etiq_alta-1,1))
        ax_alta.set_title(f"Alta intensidad\n({N_TAKE_KMEANS_HIGH} seg)")

        # Baja
        ax_baja.clear(); ax_baja.axis("off")
        ax_baja.imshow(img, cmap="gray")
        ax_baja.imshow(np.ma.masked_equal(etiq_baja[idx], 0), cmap=cmap_baja, alpha=0.6,
                      vmin=0, vmax=max(n_etiq_baja-1,1))
        ax_baja.set_title(f"Baja intensidad\n({N_TAKE_KMEANS_LOW} seg)")

        # No segmentado (rojo)
        ax_no.clear(); ax_no.axis("off")
        ax_no.imshow(img, cmap="gray")
        overlay_rojo = np.zeros((*no_seg.shape, 4))
        overlay_rojo[no_seg] = [1, 0, 0, 0.5]
        ax_no.imshow(overlay_rojo)
        ax_no.set_title(f"No segmentado\n({np.sum(no_seg)} px)")

        fig.canvas.draw_idle()

    dibujar(0)
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.04])
    slider = Slider(ax_slider, "Corte", 0, n-1, valinit=0, valstep=1)
    slider.on_changed(lambda val: dibujar(int(slider.val)))
    plt.show()


# main
if __name__ == "__main__":
    print("=== Segmentación ALTA intensidad ===")
    etiq_t1_high, conts_t1_high, _ = segmentar_volumen(
        t1_sax, N_OTSU_HIGH, N_TAKE_OTSU_HIGH,
        N_KMEANS_HIGH, N_TAKE_KMEANS_HIGH, SIGMA, HighIntensity=True)

    print("\n=== Segmentación BAJA intensidad ===")
    etiq_t1_low, conts_t1_low, _ = segmentar_volumen(
        t1_sax, N_OTSU_LOW, N_TAKE_OTSU_LOW,
        N_KMEANS_LOW, N_TAKE_KMEANS_LOW, SIGMA, HighIntensity=False)

    print("\n=== T2 ALTA ===")
    etiq_t2_high, _, _ = segmentar_volumen(
        t2_sax, N_OTSU_HIGH, N_TAKE_OTSU_HIGH,
        N_KMEANS_HIGH, N_TAKE_KMEANS_HIGH, SIGMA, HighIntensity=True)
    print("=== T2 BAJA ===")
    etiq_t2_low, _, _ = segmentar_volumen(
        t2_sax, N_OTSU_LOW, N_TAKE_OTSU_LOW,
        N_KMEANS_LOW, N_TAKE_KMEANS_LOW, SIGMA, HighIntensity=False)

    if MOSTRAR:
        inspeccionar_cuatro(t1_sax, etiq_t1_high, etiq_t1_low, "T1 SAX basal")
        inspeccionar_cuatro(t2_sax, etiq_t2_high, etiq_t2_low, "T2 SAX basal")

    if GUARDAR:
        np.save("etiquetas_t1_high.npy", etiq_t1_high)
        np.save("etiquetas_t1_low.npy", etiq_t1_low)
        np.save("etiquetas_t2_high.npy", etiq_t2_high)
        np.save("etiquetas_t2_low.npy", etiq_t2_low)
        print("Archivos guardados.")