# Modulo integrador, contiene los loops de la arquitectura general
from Instancias import Instancia,DataLoader
from Generador import generador
from Evaluador import *
from PromptSamplerDM import *

#TO-DO modificar promptsamplerDM para que pueda trabajar con los datos de instancia. Cargar LLMs, y ver como guardar las respuestas. 
#La solucion para eso, se puede despues repetir para el evaluador. Puesto que el formato es un CSV, podemos extraer directamente lo que necesitamos.
semilla = 1234
instancias = DataLoader()
instancias.cargarDatos()
instanciaPrueba:Instancia = instancias.getDatosInstancia("graph_coloring_random_dataset_in_house_9_8")
## Generar prompts con PromptSamplerDM
prompt = generateSeedPrompt(instanciaPrueba.problem,semilla)
## Prompt con tipo de problema. Este es menos exigente en las capacidades de razocinio de la LLM. Servira para pruebas a futuro
#prompt = generateSeedPromptWithProblemTYpe(instanciaPrueba.problem,instanciaPrueba.problemType,semilla)
## Preparar enjambre de LLMs (Actualmente solo con ChatGPT)
llms=generador()
llms.cargarLLMs
#llms.generarDefinicion(prompt)
## Falta guardar las respuestas
