import numpy as np
from matplotlib.colors import Normalize
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

# =============================================================================

CARPETA = os.path.dirname(os.path.abspath(__file__))
os.chdir(CARPETA)

ARCHIVO_T1 = "mapa_T1_SAX.npy"
ARCHIVO_T2 = "mapa_T2_SAX.npy"

N_OTSU_HIGH         = 3
N_TAKE_OTSU_HIGH    = 2
N_KMEANS_HIGH       = 14
N_TAKE_KMEANS_HIGH  = 5

N_OTSU_LOW          = 3
N_TAKE_OTSU_LOW     = 2
N_KMEANS_LOW        = 16
N_TAKE_KMEANS_LOW   = 11

SIGMA          = 1.0
MIN_ISLA       = 15
FONDO          = (0,)
RADIO_CIERRE   = 1
MIN_AREA       = 30
MOSTRAR        = True
GUARDAR        = True
CMAP_REGIONES  = "nipy_spectral"

# =============================================================================

def normalizar(img):
    return img

def mascara_otsu(img, n_otsu, n_take_otsu, HighIntensity=True):
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
    conservados = np.unique(resultado)
    conservados = conservados[conservados > 0]
    if len(conservados) > 0:
        for nueva, vieja in enumerate(conservados, start=1):
            resultado[resultado == vieja] = nueva
    return resultado

def eliminar_islas_foreground(etiquetas_2d, min_pixeles, fondo=(0,), connectivity=2):
    fg = ~np.isin(etiquetas_2d, fondo)
    if not fg.any():
        return etiquetas_2d
    fg_limpio = remove_small_objects(fg, min_size=min_pixeles, connectivity=connectivity)
    salida = etiquetas_2d.copy()
    salida[~fg_limpio] = 0
    return salida

def contornos_por_etiqueta(etiquetas_2d, omitir=(0,)):
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
        ax_orig.clear(); ax_orig.axis("off")
        ax_orig.imshow(img, cmap="gray")
        ax_orig.set_title(f"Corte {idx} – Original")
        ax_alta.clear(); ax_alta.axis("off")
        ax_alta.imshow(img, cmap="gray")
        ax_alta.imshow(np.ma.masked_equal(etiq_alta[idx], 0), cmap=cmap_alta, alpha=0.6,
                      vmin=0, vmax=max(n_etiq_alta-1,1))
        ax_alta.set_title(f"Alta intensidad\n({N_TAKE_KMEANS_HIGH} seg)")
        ax_baja.clear(); ax_baja.axis("off")
        ax_baja.imshow(img, cmap="gray")
        ax_baja.imshow(np.ma.masked_equal(etiq_baja[idx], 0), cmap=cmap_baja, alpha=0.6,
                      vmin=0, vmax=max(n_etiq_baja-1,1))
        ax_baja.set_title(f"Baja intensidad\n({N_TAKE_KMEANS_LOW} seg)")
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


def asignar_aha(mascara_3d, num_slices_nivel=None):
    """
    Asigna los 16 segmentos AHA a una máscara 3D del miocardio.
    """
    n_slices = mascara_3d.shape[0]
    if num_slices_nivel is None:
        basal = n_slices // 3
        medio = n_slices // 3
        apical = n_slices - basal - medio
    else:
        basal, medio, apical = num_slices_nivel

    etiquetas = np.zeros_like(mascara_3d, dtype=int)

    for i in range(n_slices):
        mascara = mascara_3d[i]
        if not np.any(mascara):
            continue

        ys, xs = np.nonzero(mascara)
        cy, cx = ys.mean(), xs.mean()

        if i < basal:
            num_seg = 6
            offset = 0
        elif i < basal + medio:
            num_seg = 6
            offset = 6
        else:
            num_seg = 4
            offset = 12

        y_grid, x_grid = np.ogrid[:mascara.shape[0], :mascara.shape[1]]
        angulo = np.arctan2(y_grid - cy, x_grid - cx)
        angulo_grados = (np.degrees(angulo) + 90) % 360

        sector = np.zeros_like(mascara, dtype=int)
        if num_seg == 6:
            for s in range(num_seg):
                inicio = s * 60
                fin = (s + 1) * 60
                mascara_sector = (angulo_grados >= inicio) & (angulo_grados < fin) & mascara
                sector[mascara_sector] = s + 1
        else:
            for s in range(num_seg):
                inicio = s * 90
                fin = (s + 1) * 90
                mascara_sector = (angulo_grados >= inicio) & (angulo_grados < fin) & mascara
                sector[mascara_sector] = s + 1

        for s in range(1, num_seg + 1):
            etiquetas[i][sector == s] = offset + s

    return etiquetas

def calcular_medias_por_segmento(volumen, etiquetas_aha, segmentos=range(1,17)):
    """Calcula el valor promedio de intensidad para cada segmento AHA."""
    medias = {}
    for seg in segmentos:
        mascara = (etiquetas_aha == seg)
        if np.any(mascara):
            medias[seg] = np.mean(volumen[mascara])
        else:
            medias[seg] = np.nan
    return medias



def plot_bullseye(medias, titulo="Bullseye AHA", cmap='viridis',
                  vmin=None, vmax=None, save_path=None):
    """
    Dibuja el mapa en bullseye de los segmentos AHA con los valores
    promedio de intensidad.
    """

    # Calcular rango automáticamente si no se entrega
    valores = np.array([v for v in medias.values() if not np.isnan(v)])

    if valores.size == 0:
        vmin, vmax = 0, 1
    else:
        if vmin is None:
            vmin = valores.min()
        if vmax is None:
            vmax = valores.max()
        if vmin == vmax:
            vmax = vmin + 1

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_facecolor('white')
    ax.spines['polar'].set_visible(False)

    # ---------- Anillo externo ----------
    radii_ext = [0.8, 1.0]
    for i, seg in enumerate(range(1, 7)):
        theta_start = np.radians(i * 60)
        theta_end = np.radians((i + 1) * 60)

        val = medias.get(seg, np.nan)
        color = cmap_obj(norm(val)) if not np.isnan(val) else 'lightgray'

        ax.bar(theta_start,
               radii_ext[1] - radii_ext[0],
               width=theta_end - theta_start,
               bottom=radii_ext[0],
               color=color,
               edgecolor='black',
               linewidth=0.5)

        ax.text((theta_start + theta_end) / 2,
                np.mean(radii_ext),
                str(seg),
                ha='center',
                va='center',
                fontsize=10,
                color='white',
                weight='bold')

    # ---------- Anillo medio ----------
    radii_mid = [0.5, 0.8]
    for i, seg in enumerate(range(7, 13)):
        theta_start = np.radians(i * 60)
        theta_end = np.radians((i + 1) * 60)

        val = medias.get(seg, np.nan)
        color = cmap_obj(norm(val)) if not np.isnan(val) else 'lightgray'

        ax.bar(theta_start,
               radii_mid[1] - radii_mid[0],
               width=theta_end - theta_start,
               bottom=radii_mid[0],
               color=color,
               edgecolor='black',
               linewidth=0.5)

        ax.text((theta_start + theta_end) / 2,
                np.mean(radii_mid),
                str(seg),
                ha='center',
                va='center',
                fontsize=10,
                color='white',
                weight='bold')

    # ---------- Anillo interno ----------
    radii_ap = [0.2, 0.5]
    for i, seg in enumerate(range(13, 17)):
        theta_start = np.radians(i * 90)
        theta_end = np.radians((i + 1) * 90)

        val = medias.get(seg, np.nan)
        color = cmap_obj(norm(val)) if not np.isnan(val) else 'lightgray'

        ax.bar(theta_start,
               radii_ap[1] - radii_ap[0],
               width=theta_end - theta_start,
               bottom=radii_ap[0],
               color=color,
               edgecolor='black',
               linewidth=0.5)

        ax.text((theta_start + theta_end) / 2,
                np.mean(radii_ap),
                str(seg),
                ha='center',
                va='center',
                fontsize=10,
                color='white',
                weight='bold')

    # ---------- Centro ----------
    if 17 in medias and not np.isnan(medias[17]):
        color = cmap_obj(norm(medias[17]))
        ax.bar(0, 0.2, width=2*np.pi,
               bottom=0,
               color=color,
               edgecolor='black',
               linewidth=0.5)
        ax.text(0, 0.1, "17",
                ha='center',
                va='center',
                fontsize=10,
                color='white',
                weight='bold')
    else:
        ax.bar(0, 0.2, width=2*np.pi,
               bottom=0,
               color='white',
               edgecolor='black',
               linewidth=0.5)
        ax.text(0, 0.1, "?",
                ha='center',
                va='center',
                fontsize=10)

    ax.set_title(titulo, fontsize=14, pad=20)

    # ---------- Barra de color ----------
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax,
                        orientation='vertical',
                        pad=0.1,
                        shrink=0.6)

    ticks = np.linspace(vmin, vmax, 6)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.1f}" for t in ticks])
    cbar.set_label("Intensidad media", fontsize=10)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

# =============================================================================

if __name__ == "__main__":
    t1_sax = np.load(ARCHIVO_T1)
    t2_sax = np.load(ARCHIVO_T2)

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

    print("\n=== Generando segmentos AHA ===")
    mascara_t1 = (etiq_t1_high == 0) & (etiq_t1_low == 0)
    mascara_t2 = (etiq_t2_high == 0) & (etiq_t2_low == 0)

    aha_t1 = asignar_aha(mascara_t1, num_slices_nivel=None)
    aha_t2 = asignar_aha(mascara_t2, num_slices_nivel=None)

    medias_t1 = calcular_medias_por_segmento(t1_sax, aha_t1)
    medias_t2 = calcular_medias_por_segmento(t2_sax, aha_t2)


    medias_t1[17] = np.nan
    medias_t2[17] = np.nan

    # Determinar rango común de intensidades para la barra de color
    all_vals = list(medias_t1.values()) + list(medias_t2.values())
    all_vals = [v for v in all_vals if not np.isnan(v)]
    if all_vals:
        vmin = min(all_vals)
        vmax = max(all_vals)
    else:
        vmin, vmax = 0, 1

    if MOSTRAR:
        plot_bullseye(medias_t1, titulo="Bullseye AHA - T1", cmap='plasma', vmin=vmin, vmax=vmax)
        plot_bullseye(medias_t2, titulo="Bullseye AHA - T2", cmap='viridis', vmin=vmin, vmax=vmax)

    if GUARDAR:
        np.save("segmentos_aha_t1.npy", aha_t1)
        np.save("segmentos_aha_t2.npy", aha_t2)
        plot_bullseye(medias_t1, titulo="Bullseye AHA - T1", cmap='plasma', vmin=vmin, vmax=vmax, save_path="bullseye_T1.png")
        plot_bullseye(medias_t2, titulo="Bullseye AHA - T2", cmap='viridis', vmin=vmin, vmax=vmax, save_path="bullseye_T2.png")
        print("Archivos AHA y bullseyes guardados.")

