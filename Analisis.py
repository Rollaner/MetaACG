from dataclasses import dataclass
from enum import Enum, auto
import json
import pandas as pd
import numpy as np
import ast
import numbers
from Instancias import DataLoader
from scipy.stats import t

import Plotter

#Esto causara un fallo immediato si se cambian los outputs del evaluador sin actualizar _clasificarFallo tambien.
class TipoFallo(Enum):
    FAILED_TO_EVALUATE = auto()
    FAILED_TO_RUN_TARGET_HEURISTIC = auto()
    FAILED_TO_LOAD = auto()
    SCORE_INVALIDO = auto()  # Se assume que inf equivale a >=1e3 debido al rango de puntajes promedio. Con otros problemas esta restriccion se tiene que relajar 
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
    texto = texto.lower().replace("_", " ")
    if 'failed to evaluate' in texto:
        return TipoFallo.FAILED_TO_EVALUATE
    if 'failed to run target heuristic' in texto:
        return TipoFallo.FAILED_TO_RUN_TARGET_HEURISTIC
    if 'failed to load' in texto:
        return TipoFallo.FAILED_TO_LOAD
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
    contador = resultadosParseados.apply(lambda x: x.fallo if not x.valido else 'EXITO').value_counts()
    fallos = {t: contador.get(t, 0) for t in TipoFallo}
    exitos = contador.get('EXITO', 0)

    return {
        'Failed to evaluate':             fallos[TipoFallo.FAILED_TO_EVALUATE],
        'Failed to run target heuristic': fallos[TipoFallo.FAILED_TO_RUN_TARGET_HEURISTIC],
        'Failed to load':                 fallos[TipoFallo.FAILED_TO_LOAD],
        'Invalid_score':                   fallos[TipoFallo.SCORE_INVALIDO],
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
    resultadosAux = resultados.copy()
    resultadosAux['Valor Optimo'] = resultadosAux['ID_Problema'].map(
        lambda id_: _getScore(id_, instancias)
    )
    resultadosAux['Solucion'] = resultadosAux['ID_Problema'].map(
        lambda id_: _getSolucion(id_, instancias)
    )

    resultadosAux['Resultados'] = resultadosAux['Resultados'].apply(_normalizarResultado)
    resultadosAuxI = resultadosAux[resultadosAux['ID_Problema'].str.endswith('inverted')]
    resultadosAuxS = resultadosAux[resultadosAux['ID_Problema'].str.endswith('standard')]

    resultadosParseados = resultadosAux['Resultados'].apply(parsearResultado)
    resultadosParseadosI = resultadosAuxI['Resultados'].apply(parsearResultado)
    resultadosParseadosS = resultadosAuxS['Resultados'].apply(parsearResultado)


    puntajes = resultadosParseados.apply(lambda r: r.score if r.valido else np.nan)
    resultadosAux['Puntaje Real'] = puntajes  


    dfProcesado = resultadosAux[puntajes.notna()].copy()  

    dfProcesado['Solucion identica'] = dfProcesado.apply(
        lambda row: _compararSoluciones(row['Resultados'], row['Solucion']),
        axis=1
    )

    dfProcesado['Solucion identica'] = dfProcesado.apply(
        lambda row: _compararSoluciones(row['Resultados'], row['Solucion']), axis=1
    )
    
    tiposFallos = _contarFallos(resultadosParseados)
    tiposFallosI = _contarFallos(resultadosParseadosI)
    tiposFallosS = _contarFallos(resultadosParseadosS)

    dfProcesadoI = dfProcesado[dfProcesado['ID_Problema'].str.endswith('inverted')]
    dfProcesadoS = dfProcesado[dfProcesado['ID_Problema'].str.endswith('standard')]

    correctitud = _contarCorrectitud(dfProcesado['Solucion identica'])
    correctitudI = _contarCorrectitud(dfProcesadoI['Solucion identica'])
    correctitudS = _contarCorrectitud(dfProcesadoS['Solucion identica'])

    dfProcesado['Diferencia Absoluta']    = ( dfProcesado['Puntaje Real'] -  dfProcesado['Valor Optimo']).abs()
    dfProcesado['Diferencia Porcentual']  = ( dfProcesado['Diferencia Absoluta'] /  dfProcesado['Valor Optimo']).abs() * 100 #Valor optimo de EHOP nunca es cero

    resultadosAux['TipoFallo'] = resultadosParseados.apply(lambda r: r.fallo.name if not r.valido else None)

    return dfProcesado,tiposFallos,tiposFallosI,tiposFallosS,resultadosAux,correctitud,correctitudI,correctitudS

def agregarResultados(dfProcesado: pd.DataFrame, resultadosAux: pd.DataFrame) -> dict:
    fallos      = len(resultadosAux) - len(dfProcesado)
    ejecuciones = int((~dfProcesado['Solucion identica']).sum())
    optimos     = int(dfProcesado['Solucion identica'].sum())
    assert fallos + ejecuciones + optimos == len(resultadosAux), "Los totales no cuadran"
    return {'Fallos': fallos, 'Ejecuciones': ejecuciones, 'Optimos': optimos}

def generarFigurasYTablasLatexLocales(dfProcesado: pd.DataFrame, FallosTot: dict,FallosTotI: dict,FallosTotS: dict, resultadosAux: pd.DataFrame, correctitud: dict,correctitudI: dict,correctitudS: dict, output: str, pipeline: str):
    totalExperimentos = len(resultadosAux)
    tasaDeFallos      = FallosTot['Total Fallos'] / totalExperimentos
    metricasRendimiento = _calcularMetricasRendimiento(dfProcesado)
    desempeñoPorSolver  = _desempeñoPorSolver(resultadosAux)

    _imprimirResultados(FallosTot,FallosTotI,FallosTotS, tasaDeFallos, dfProcesado, correctitud,correctitudI,correctitudS, metricasRendimiento, desempeñoPorSolver, pipeline)
    _crearTablasLatex(dfProcesado,resultadosAux,FallosTot, output, pipeline)
    _plots(totalExperimentos, pipeline, dfProcesado, resultadosAux)

    return FallosTot, desempeñoPorSolver, metricasRendimiento

def _calcularMetricasRendimiento(dfProcesado):
    ##  metricas rendimiento
    metricasRendimiento = dfProcesado.groupby('Metaheuristica').agg(
        exitosTotales = ('Puntaje Real', 'count'),
        Promedio_Error_Abs =('Diferencia Absoluta', 'mean'),
        Desviacion_Estandar_Abs = ('Diferencia Absoluta', 'std'),
        Promedio_Error_Porcentual =('Diferencia Porcentual', 'mean'),
        Desviacion_Estandar_Porcentual = ('Diferencia Porcentual', 'std')
    ).reset_index()
    alpha = 0.05
    tCritico = t.ppf(1-alpha/2, metricasRendimiento['exitosTotales']-1)
    margenAbs = metricasRendimiento['Desviacion_Estandar_Abs']/np.sqrt(metricasRendimiento['exitosTotales'])
    margenPor = metricasRendimiento['Desviacion_Estandar_Porcentual']/np.sqrt(metricasRendimiento['exitosTotales'])
    #Para saber que tan generlizables son los resultados, considerando la pequeña muestra
    metricasRendimiento['Intervalo_de_Confianza_Error_Absoluto_Minimo'] = ( metricasRendimiento['Promedio_Error_Abs'] - tCritico * margenAbs)
    metricasRendimiento['Intervalo_de_Confianza_Error_Absoluto_Maximo'] = ( metricasRendimiento['Promedio_Error_Abs'] + tCritico * margenAbs)
    metricasRendimiento['Intervalo_de_Confianza_Error_Porcentual_Minimo'] = ( metricasRendimiento['Promedio_Error_Porcentual'] - tCritico * margenPor )
    metricasRendimiento['Intervalo_de_Confianza_Error_Porcentual_Maximo'] = ( metricasRendimiento['Promedio_Error_Porcentual'] + tCritico * margenPor )
    return metricasRendimiento

def _desempeñoPorSolver(resultadosAux: pd.DataFrame): #Tiene problemas. Revisar
    desempeñoPorSolver = (resultadosAux
        .assign(Exito=resultadosAux['Puntaje Real'].notna()) ## crea una copia y le aplica una mascara para que filtremos todo lo que no sea una ejecucion exitosa
        .groupby('Metaheuristica')['Exito']
        .agg(TotalExperimentos='count', TotalExitos='sum')
        .reset_index()
    )
    desempeñoPorSolver['TotalFallos'] = desempeñoPorSolver['TotalExperimentos'] - desempeñoPorSolver['TotalExitos']
    desempeñoPorSolver['Tasa de Fallo'] = desempeñoPorSolver['TotalFallos'] / desempeñoPorSolver['TotalExperimentos']
    return desempeñoPorSolver

def _imprimirResultados(FallosTot,FallosTotI,FallosTotS, tasaDeFallos, dfProcesado, correctitud,correctitudI,correctitudS, metricasRendimiento, desempeñoPorSolver, pipeline):
    totalExperimentosExitosos = len(dfProcesado)
    print(f"--- Metricas problemas invertidos {pipeline}---")
    print(f"--- Tasa de fallo {pipeline}---")
    for key, count in FallosTotI.items():
        if key not in ['Total Fallos', 'Total Exitos']:
            print(f"* {key}: {count}")
    print(  f"Fallos Totales: {FallosTotI["Total Fallos"]}")
    print(f"---Soluciones identicas al optimo {pipeline}---")
    print(f"Soluciones Identicas: {correctitudI['Soluciones Identicas']} / {totalExperimentosExitosos} "
          f"({correctitudI['Soluciones Identicas']/totalExperimentosExitosos:.2%})")
    print(f"Soluciones Distintas: {correctitudI['Soluciones Distintas']} / {totalExperimentosExitosos} "
          f"({correctitudI['Soluciones Distintas']/totalExperimentosExitosos:.2%})")
    print(f"--- Metricas problemas estandar {pipeline}---")
    print(f"--- Tasa de fallo {pipeline}---")
    for key, count in FallosTotS.items():
        if key not in ['Total Fallos', 'Total Exitos']:
            print(f"* {key}: {count}")
    print(  f"Fallos Totales: {FallosTotS["Total Fallos"]}")
    print(f"---Soluciones identicas al optimo {pipeline}---")
    print(f"Soluciones Identicas: {correctitudS['Soluciones Identicas']} / {totalExperimentosExitosos} "
          f"({correctitudS['Soluciones Identicas']/totalExperimentosExitosos:.2%})")
    print(f"Soluciones Distintas: {correctitudS['Soluciones Distintas']} / {totalExperimentosExitosos} "
          f"({correctitudS['Soluciones Distintas']/totalExperimentosExitosos:.2%})")
    print(f"--- Metricas generales {pipeline}---")
    print(f"--- Tasa de fallo {pipeline}---")
    for key, count in FallosTot.items():
        if key not in ['Total Fallos', 'Total Exitos']:
            print(f"* {key}: {count}")
    print(  f"Fallos Totales: {FallosTot["Total Fallos"]}")
    print(f"--- Metricas estandar de desempeño {pipeline}---")
    print(metricasRendimiento[['Metaheuristica','Promedio_Error_Porcentual', 'Desviacion_Estandar_Porcentual','Intervalo_de_Confianza_Error_Absoluto_Minimo','Intervalo_de_Confianza_Error_Absoluto_Maximo', 'Promedio_Error_Abs','Desviacion_Estandar_Abs', 'Intervalo_de_Confianza_Error_Porcentual_Minimo', 'Intervalo_de_Confianza_Error_Porcentual_Maximo']])
    print(f"Tasa de Fallos {tasaDeFallos:.2%}")
    print(dfProcesado[['Metaheuristica', 'Resultados', 'Puntaje Real', 'Valor Optimo']])
    print(f"--- Desempeño por Solver {pipeline}---")
    print(desempeñoPorSolver)
    print(f"---Soluciones identicas al optimo {pipeline}---")
    print(f"Soluciones Identicas: {correctitud['Soluciones Identicas']} / {totalExperimentosExitosos} "
          f"({correctitud['Soluciones Identicas']/totalExperimentosExitosos:.2%})")
    print(f"Soluciones Distintas: {correctitud['Soluciones Distintas']} / {totalExperimentosExitosos} "
          f"({correctitud['Soluciones Distintas']/totalExperimentosExitosos:.2%})")
    print("--- Tasa De Fallo por solver ---")
    print(desempeñoPorSolver[['Metaheuristica', 'TotalExperimentos', 'Tasa de Fallo']].sort_values(by='Tasa de Fallo'))

def _crearTablasLatex(dfProcesado, resultadosAux,FallosTot, output, pipeline):
    ## escribir latex 
    totales = resultadosAux.groupby("Metaheuristica").size()
    resultadosPorMH = dfProcesado.groupby("Metaheuristica")["Solucion identica"].agg(
        Optimos="sum",
        Ejecuciones=lambda s: (~s).sum(),
    )
    dfTasaExito = totales.rename("Total").to_frame().join(resultadosPorMH, how="left").fillna(0)
    dfTasaExito["Fallos"] = dfTasaExito["Total"] - (dfTasaExito["Optimos"] + dfTasaExito["Ejecuciones"])
    dfTasaExito[r"\% Ópt."] = (dfTasaExito["Optimos"] / dfTasaExito["Total"] * 100).map("{:.1f}\\%".format)
    dfTasaExito[r"\% Ejec."] = (dfTasaExito["Ejecuciones"] / dfTasaExito["Total"] * 100).map("{:.1f}\\%".format)
    dfTasaExito[r"\% Fallo"] = (dfTasaExito["Fallos"] / dfTasaExito["Total"] * 100).map("{:.1f}\\%".format)
    
    mapaValores = {
        'Failure_to_evaluate': 'FAILURE_TO_EVALUATE',
        'Failure_to_run_target_heuristic': 'FAILURE_TO_RUN_TARGET_HEURISTIC',
        'Failure_to_load': 'FAILURE_TO_LOAD',
        'Invalid_score': 'SCORE_INVALIDO',
        'Otros': 'OTRO'
    }
    
    nombresLegibles = list(mapaValores.keys())
    fallosAux = resultadosAux[~resultadosAux.index.isin(dfProcesado.index)]
    if not fallosAux.empty and "TipoFallo" in fallosAux.columns:
        dfFallos = (
            fallosAux.groupby(["Metaheuristica", "TipoFallo"])
            .size()
            .unstack(fill_value=0)
        )
        tiposDeFallo = [mapaValores[k] for k in nombresLegibles]
        dfFallos = dfFallos.reindex(columns=tiposDeFallo, fill_value=0)
        dfFallos = dfFallos.rename(columns={v: k for k, v in mapaValores.items()})
    else:
        metaheuristicas = resultadosAux["Metaheuristica"].unique()
        dfFallos = pd.DataFrame(0, index=metaheuristicas, columns=nombresLegibles)

    ## escribir tablas 
    with open(output, 'a', encoding='utf-8') as f:
         ##tabla automagica en latex 
        f.write("\n")
        cols_to_show = [ r"\% Ópt.", r"\% Ejec.", r"\% Fallo", "Optimos", "Ejecuciones", "Fallos"]
        f.write(dfTasaExito[cols_to_show].style.to_latex(
            hrules=True,
            caption=f"Desglose de resultados: Óptimos, Ejecuciones y Fallos ({pipeline})",
            label=f"tab:{pipeline}_desempeño",
            column_format="lcccccc"
        ))
        f.write("\n")
        f.write("\\begin{table}[H]\n\\centering\n")
        f.write(f"\\caption{{Desglose de tipos de fallo por metaheurística ({pipeline})}}\n")
        f.write(f"\\label{{tab:{pipeline}_modos_de_fallo}}\n")
        dfFallos.columns = [str(c).replace('_', ' ').title() for c in dfFallos.columns]
        f.write("\\begin{adjustbox}{max width=\\textwidth}\n{\n")
        f.write(dfFallos.style.to_latex(hrules=True, column_format="l" + "c" * len(dfFallos.columns)))
        f.write("}\n\\end{adjustbox}\n\\end{table}\n")

    print(f"Tablas de LaTeX generadas exitosamente para {pipeline}")        

## requieren un estado global o que le pase todas las pipelines de una en main. Habra que pensar que es mejor
def plotsGlobales(registroPipelines: dict):
    Plotter.graficos_globales(registroPipelines)

def actualizarDictPipelines(FallosTot: dict, registroPipelines, pipeline: str, dfProcesado: pd.DataFrame, resultadosAux: pd.DataFrame): 
    registroPipelines.append({
        "pipeline":     pipeline,
        "dfProcesado":  dfProcesado,
        "resultadosAux": resultadosAux,
        "FallosTot":    FallosTot,
    })

def _plots(totalExperimentos: int,pipeline: str,dfProcesado: pd.DataFrame,resultadosAux: pd.DataFrame) -> None:
    Plotter.graficos_por_pipeline(pipeline, dfProcesado, resultadosAux)


    


