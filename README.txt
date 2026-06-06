

PIPELINE SAX - GEOMETRÍA

Archivos entregados:
- mapa_T1_SAX.npy
- mapa_T2_SAX.npy
- preview_SAX.png
- metadata.json

Descripción:
Se aplicó un pipeline semi-manual para reformateo
cardíaco en eje corto (SAX).

Etapas:
1. Suavizado gaussiano
2. Visualización de gradientes
3. Selección semi-manual de landmarks
4. Construcción de eje largo
5. Reformateo SAX mediante interpolación lineal

Importante:
Los volúmenes T1 y T2 comparten exactamente
la misma geometría espacial y orientación.

Los archivos SAX están listos para:
- segmentación
- análisis morfológico
- generación de mapas regionales
