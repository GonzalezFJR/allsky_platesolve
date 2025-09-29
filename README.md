# All-Sky Platesolve Web App

Aplicación web interactiva para calibrar imágenes de cámaras all-sky mediante la identificación de estrellas de referencia y el ajuste de un modelo de distorsión radial.

## Características
- **Carga de imágenes** mediante arrastrar y soltar, con soporte para inversión horizontal si la imagen está orientada con norte arriba.
- **Configuración de observación**: introducción de coordenadas, hora de captura y selección del modelo de distorsión radial.
- **Detección automática de estrellas** con parámetros ajustables para DAOStarFinder y control del número máximo de detecciones.
- **Emparejado manual asistido** entre detecciones y catálogo de 450 estrellas visibles para refinar el modelo de lente.
- **Resumen de parámetros** actualizado en tiempo real y descarga de la configuración ajustada en formato JSON.

## Requisitos
- Python 3.10 o superior
- Dependencias listadas en [`requirements.txt`](requirements.txt)

Instala las dependencias con:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Puesta en marcha
1. Activa el entorno virtual e instala los requisitos como se indica arriba.
2. Inicia la API de FastAPI con:
   ```bash
   uvicorn app.main:app --reload
   ```
3. Abre `http://127.0.0.1:8000/` en tu navegador para acceder a la interfaz.

La aplicación servirá los recursos estáticos del directorio `frontend/` automáticamente.

## Flujo de uso
1. **Sube una imagen** all-sky en la zona principal. Si la imagen ya está orientada con norte arriba y este a la derecha, desmarca la casilla "Imagen en vista nadir" para aplicar el volteo horizontal.
2. **Define tu observación** introduciendo latitud, longitud, elevación y fecha/hora UTC. Selecciona el modelo radial deseado y pulsa "Actualizar proyección" para obtener las posiciones esperadas de las estrellas.
3. **Lanza la detección automática** ajustando los parámetros de DAOStarFinder según la calidad de la imagen. El panel permite repetir o limpiar detecciones en cualquier momento.
4. **Empareja detecciones**: haz clic sobre una estrella detectada y elige su nombre del catálogo (usa la lista desplegable para buscar). Cada identificación refina el modelo y actualiza la proyección de estrellas.
5. **Descarga los parámetros** una vez satisfecho con el ajuste para reutilizarlos en otros procesos.

Puedes alternar la visualización de detecciones, posiciones esperadas y etiquetas mediante los interruptores situados debajo de la imagen.

## Datos
El catálogo de referencia se encuentra en `data/named_stars.csv`. Si añades nuevas entradas o catálogos, reinicia la aplicación para que sean tenidos en cuenta.

## Desarrollo y pruebas
- Ejecuta `python -m compileall calibration app` para validar rápidamente que no existan errores de sintaxis en los módulos Python.
- Ajusta los estilos o la lógica del frontend editando los archivos en `frontend/` y recarga la página para ver los cambios.

## Estructura del proyecto
```
app/            # Backend FastAPI
calibration/    # Lógica de modelos, utilidades y catálogos
frontend/       # Interfaz estática (HTML, CSS, JS)
data/           # Archivos de datos (catálogo de estrellas)
```

¡Felices calibraciones!
