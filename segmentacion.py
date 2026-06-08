#Segmentacion de las imagenes del corazón T1 y T2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, morphology, measure
from skimage.segmentation import watershed
import skimage as ski
from cv2 import findContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE

# Imagen final del corazón T1 (formato .png)
t1_image = ski.io.imread("corazon_T1.png", as_gray=True)
#La imgagen se encuentra entre los pixeles 0;0 y 115:115
t1_image = t1_image[0:108, 0:108]
plt.imshow(t1_image, cmap='gray')
plt.show()

# Imagen final del corazón T2 (formato .png)
t2_image = ski.io.imread("corazon_T2.png", as_gray=True)
#La imgagen se encuentra entre los pixeles 115;0 y 230:115
t2_image = t2_image[0:108, 115:230]
plt.imshow(t2_image, cmap='gray')
plt.show()

histogram_t1, bin_edges_t1 = np.histogram(t1_image.flatten(), bins=256, range=(0, 1))
histogram_t2, bin_edges_t2 = np.histogram(t2_image.flatten(), bins=256, range=(0, 1))
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(bin_edges_t1[:-1], histogram_t1, color='blue')
plt.title('Histograma de T1')
plt.xlabel('Intensidad de píxel')
plt.ylabel('Número de píxeles')
plt.subplot(1, 2, 2)
plt.plot(bin_edges_t2[:-1], histogram_t2, color='red')
plt.title('Histograma de T2')
plt.xlabel('Intensidad de píxel')
plt.ylabel('Número de píxeles')
plt.tight_layout()
plt.show()

#Filtro gaussiano para reducir el ruido
t1_image_gaus = filters.gaussian(t1_image, sigma=1)
t2_image_gaus = filters.gaussian(t2_image, sigma=1)
histogram_t1_gaus, bin_edges_t1_gaus = np.histogram(t1_image_gaus.flatten(), bins=256, range=(0, 1))
histogram_t2_gaus, bin_edges_t2_gaus = np.histogram(t2_image_gaus.flatten(), bins=256, range=(0, 1))
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(bin_edges_t1_gaus[:-1], histogram_t1_gaus, color='blue')
plt.title('Histograma de T1 con Filtro Gaussiano')
plt.xlabel('Intensidad de píxel')
plt.ylabel('Número de píxeles')
plt.subplot(1, 2, 2)
plt.plot(bin_edges_t2_gaus[:-1], histogram_t2_gaus, color='red')
plt.title('Histograma de T2 con Filtro Gaussiano')
plt.xlabel('Intensidad de píxel')
plt.ylabel('Número de píxeles')
plt.tight_layout()
plt.show()

# Segmentación utilizando el método de Otsu para T1
threshold_t1 = filters.threshold_otsu(t1_image_gaus)
binary_t1 = t1_image_gaus > threshold_t1

# Segmentación utilizando el método de Otsu para T2
threshold_t2 = filters.threshold_otsu(t2_image_gaus)
binary_t2 = t2_image_gaus > threshold_t2

# Mostrar resultados de segmentación
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(binary_t1, cmap='gray')
axes[0].set_title('Segmentación T1')
axes[0].axis('off')
axes[1].imshow(binary_t2, cmap='gray')
axes[1].set_title('Segmentación T2')
axes[1].axis('off')
plt.show()

# Segmentacion por semillas para T1
seed_t1 = input("Ingrese la coordenadas de la semilla para T1 (formato: x,y): ")
seed_t1 = tuple(map(int, seed_t1.split(',')))  # Convertir a tupla de enteros (x, y)

markers_t1 = np.zeros_like(t1_image_gaus, dtype=int)
markers_t1[seed_t1[1], seed_t1[0]] = 1  # numpy indexa [fila, col] → [y, x]

segmented_t1 = watershed(-t1_image_gaus, markers=markers_t1, mask=binary_t1)

# Segmentacion por semillas para T2
seed_t2 = input("Ingrese la coordenadas de la semilla para T2 (formato: x,y): ")
seed_t2 = tuple(map(int, seed_t2.split(',')))  # Convertir a tupla de enteros (x, y)

markers_t2 = np.zeros_like(t2_image_gaus, dtype=int)
markers_t2[seed_t2[1], seed_t2[0]] = 1  # numpy indexa [fila, col] → [y, x]

segmented_t2 = watershed(-t2_image_gaus, markers=markers_t2, mask=binary_t2)

# Mostrar resultados de segmentación por semillas
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(segmented_t1, cmap='nipy_spectral')
axes[0].set_title('Segmentación por Semillas T1')
axes[0].axis('off')
axes[1].imshow(segmented_t2, cmap='nipy_spectral')
axes[1].set_title('Segmentación por Semillas T2')
axes[1].axis('off')
plt.show()
