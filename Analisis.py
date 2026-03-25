from dataclasses import dataclass
from enum import Enum, auto
import json
import pandas as pd
import numpy as np
import ast
import numbers
from Instancias import DataLoader

#Esto causara un fallo immediato si se cambian los outputs del evaluador sin actualizar _clasificarFallo tambien.
class TipoFallo(Enum):
    FAILURE_TO_EVALUATE = auto()
    FAILURE_TO_RUN_TARGET_HEURISTIC = auto()
    FAILURE_TO_LOAD = auto()
    INVALID_MOVE_OR_SCORE = auto()  # Se assume que inf equivale a >=1e3 debido al rango de puntajes promedio. Con otros problemas esta restriccion se tiene que relajar 
    OTRO = auto()

@dataclass
class ResultadoParseado:
    score: float | None
    fallo: TipoFallo | None  

    @property
    def valido(self) -> bool:
        return self.score is not None and self.fallo is None

def _getScore(id_problema: str,instancias: DataLoader):
    inst = instancias.getInstancia(id_problema)
    return inst.objectiveScore if inst is not None else None

def _getSolucion(id_problema: str,instancias: DataLoader):
    inst = instancias.getInstancia(id_problema)
    return inst.parsedSolution if inst is not None else None

def _normalizarResultado(resultado):
    if isinstance(resultado, (list, dict)):
        return resultado
    if isinstance(resultado, str):
        resultado = resultado.strip()
        try:
            return json.loads(resultado)
        except json.JSONDecodeError:
            pass
        try:
            return ast.literal_eval(resultado)
        except (SyntaxError, ValueError):                
            return resultado
    return resultado

def _contarCorrectitud(comparaciones: pd.Series) -> dict:
    return {
        'Soluciones Identicas': int(comparaciones.sum()),
        'Soluciones Distintas': int((~comparaciones).sum()),
    }

def _clasificarFallo(texto: str) -> TipoFallo:
    texto = texto.lower()
    if 'failed_to_evaluate' in texto:
        return TipoFallo.FAILURE_TO_EVALUATE
    if 'failed_to_run_target_heuristic' in texto:
        return TipoFallo.FAILURE_TO_RUN_TARGET_HEURISTIC
    if 'failed_to_load' in texto:
        return TipoFallo.FAILURE_TO_LOAD
    return TipoFallo.OTRO

def _extraerPuntaje(resultado) -> float | None:
    if isinstance(resultado, tuple):
        resultado = list(resultado)
    elif isinstance(resultado, np.ndarray):
        resultado = resultado.tolist()

    if not isinstance(resultado, list):
        return None

    for elem in resultado:
        if isinstance(elem, dict):
            score = elem.get('bestScore') or elem.get('currentScore')
            if isinstance(score, numbers.Real):
                return float(score)

    # En caso de que no se encuentre el la columna correcta, se hace fallback al penultimo valor numerico obtenido
    numeric_vals = [e for e in resultado if isinstance(e, numbers.Real)]
    if numeric_vals:
        return float(numeric_vals[-1])

    return None

def parsearResultado(resultado) -> ResultadoParseado:
    if isinstance(resultado, str):
        return ResultadoParseado(score=None, fallo=_clasificarFallo(resultado))

    score = _extraerPuntaje(resultado)

    if score is None:
        return ResultadoParseado(score=None, fallo=TipoFallo.OTRO)

    if abs(score) >= 1e3:
        return ResultadoParseado(score=None, fallo=TipoFallo.SCORE_INVALIDO)

    return ResultadoParseado(score=score, fallo=None)

def _contarFallos(resultadosParseados: pd.Series) -> dict:
    fallos = {t: 0 for t in TipoFallo}
    exitos = 0

    for resultado in resultadosParseados:
        if resultado.valido:
            exitos += 1
        else:
            fallos[resultado.fallo] += 1

    return {
        'Failure_to_evaluate':             fallos[TipoFallo.FAILURE_TO_EVALUATE],
        'Failure_to_run_target_heuristic': fallos[TipoFallo.FAILURE_TO_RUN_TARGET_HEURISTIC],
        'Failure_to_load':                 fallos[TipoFallo.FAILURE_TO_LOAD],
        'Score_invalido':                  fallos[TipoFallo.SCORE_INVALIDO],
        'Otros':                           fallos[TipoFallo.OTRO],
        'Total Fallos':                    sum(fallos.values()),
        'Total Exitos':                    exitos,
    }

def _compararSoluciones(solucionGenerada, solucionInstancia) -> bool:
    try:
        final_solution = solucionGenerada[2]
        return list(final_solution) == list(solucionInstancia)
    except (TypeError, IndexError):
        return False


def procesarResultados(resultados: pd.DataFrame, instancias: DataLoader):
    pd.set_option('display.float_format', lambda x: '%.0f' % x)
    resultados = resultados.copy()
    resultados['Valor Optimo'] = resultados['ID_Problema'].map(
        lambda id_: _getScore(id_, instancias)
    )
    resultados['Solucion'] = resultados['ID_Problema'].map(
        lambda id_: _getSolucion(id_, instancias)
    )
    resultados['Resultados'] = resultados['Resultados'].apply(_normalizarResultado)

    resultadosParseados = resultados['Resultados'].apply(parsearResultado)

    puntajes = resultadosParseados.apply(lambda r: r.score if r.valido else np.nan)

    dfProcesado = resultados[puntajes.notna()].copy() #mascara filtrado. df_procesado solo tiene puntajes exitosos

    dfProcesado['Puntaje Real'] = puntajes[puntajes.notna()] #copia valores (para que no se me olvide despues)

    dfProcesado['Solucion identica'] = dfProcesado.apply(
        lambda row: _compararSoluciones(row['Resultados'], row['Solucion']), axis=1
    )

    FallosTot = _contarFallos(resultadosParseados)
    
    correctitud = _contarCorrectitud(dfProcesado['Solucion identica'])

    return dfProcesado,FallosTot, resultados,  correctitud

