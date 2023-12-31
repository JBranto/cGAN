# cGAN - Generador de Caracteres y Palabras

Este repositorio es la implementación del proyecto final de carrera de Ingeniería Informática. En este proyecto se explora el uso de un Modelo Generativo Adversario (GAN) capaz de crear caracteres individuales y su aplicación en la generación de palabras mediante la concatenación de estos caracteres.

Por motivos de seguridad, los datos de entrenamiento no están disponibles en este repositorio, ni tampoco los modelos pre-entrenados. En su lugar, se recomienda utilizar el conjunto de datos MNIST como conjunto de prueba. Si es necesario obtener el conjunto de datos de letras creado específicamente para este proyecto, por favor, envíe un correo electrónico a la siguiente dirección: [jeassomontenegro@gmail.com](mailto:jeassomontenegro@gmail.com).

El repositorio se divide en dos proyectos principales:

1. **Extracción y Limpieza de Datos:** En esta parte del proyecto, se encuentran los scripts necesarios para recopilar y limpiar los datos requeridos para el entrenamiento.

TODO

2. **Diseño del Modelo y Entrenamiento:** En esta sección se realizan todas las tareas relacionadas con la creación y entrenamiento del modelo GAN.

La estructura del repositorio se organiza de la siguiente manera:

- En la carpeta `main`, se encuentran los archivos principales para iniciar el proceso de optimización y entrenamiento. También, aquí se ubican los scripts necesarios para utilizar los modelos y generar palabras. Para utilizar otro conjunto de datos se recomienda cambiar la variable DATASET_NAME = 'LETTERS' a 'MNIST' en el fichero tune_CDCGAN.
  
- En la carpeta `packages`, se albergan los modelos y los gestores de conjuntos de datos necesarios para el proyecto.
  
- La carpeta `results` contiene los resultados del proceso de entrenamiento del modelo, incluyendo imágenes generadas y el modelo entrenado.

Para ejecutar el código de manera efectiva, se recomienda el uso de Docker para crear un entorno de desarrollo y ejecución aislado.
