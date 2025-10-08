# Modulo integrador, contiene los loops de la arquitectura general
from Instancias import Instancia,DataLoader
from Generador import generador
from Evaluador import *
from PromptSamplerDM import *

#TO-DO modificar promptsamplerDM para que pueda trabajar con los datos de instancia. Cargar LLMs, y ver como guardar las respuestas. 
#La solucion para eso, se puede despues repetir para el evaluador. Puesto que el formato es un CSV, podemos extraer directamente lo que necesitamos.

intancias = DataLoader()
intancias.cargarDatos()
intanciaPrueba= intancias.getDatosInstancia("graph_coloring_random_dataset_in_house_9_8")
## Generar prompts con PromptSamplerDM
#prompt = generateSeedPrompt()
## Preparar enjambre de LLMs (Actualmente solo con ChatGPT)
llms=generador()
llms.cargarLLMs
#llms.generarDefinicion(prompt)
## Falta guardar las respuestas
