from Generador import generador
from .PromptSamplerOP import *
import sys
import json

#Recalibración de las LLMS
def reiniciarLLMS():
    llms = generador()
    llms.cargarLLMs()
    return llms

def optimizarProblema():
   return  

def evaluarComponentes():
    return

def generarComponente():
    return

def refinarComponentes():
    return

def guardarResultados(datos, header, path):
    num_encabezados = len(header)
    num_datos = len(datos)
    if num_datos != num_encabezados:
        print(
            f"Error: La cantidad de datos ({num_datos}) no encajan con el numero de encabezados ({num_encabezados})."
            f"El registro NO fue guardado. Use 'None' explícito para campos faltantes."
            ,file=sys.stderr)
        return
    record_dict = dict(zip(header, datos))
    json_line = json.dumps(record_dict, ensure_ascii=False)

    try:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json_line + '\n')
    except Exception as e:
        print(f"Ocurrio un error al momento de escribir en el archivo: {e}", file=sys.stderr)