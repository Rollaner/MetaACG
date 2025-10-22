#MetaACG

Sistema de configuracion automatica de Metauristicas usando LLMs. 

---
Modulo de definicion

El programa automaticamente utiliza los datos presentes en la carpeta data, sando el formato: Dataset/Tipo_problem/consolidated*.csv.Estos archivos han de contener las descripciones en lenguage natural del problema y la llave que lo identifica. Por ejemplo: hard_6_0. El algoritmo buscara en el directorio donde se encuentre consolidated por un archivo con el mismo nombre para cargar los datos completos de la instancia.

Por defecto el programa corre en modo "batch". En este modo no se realiza la optimizacion de los problemas. Estos son solo definidos. Los resultados se recolectan en el archivo "definicion_log.csv". Estos incluyen el ID de la instancia, la resputesta del modelo, tiempo de generacion y otros datos. 
---
## Preparacion de entorno

Instale las dependencias: 
 - Conda
 - Langchain
 - Clone el repositorio con https://github.com/Rollaner/MetaACG.git
 - Cree un archivo .env con la api de OpenAi. (OPENAI_API_KEY)

---
LICENCIA

This project is licensed under the APGL License.
