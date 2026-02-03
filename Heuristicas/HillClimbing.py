import random

def HC(solucionActual, mejorSolucion, mejorCosto,datosProblema, funcionVecindad, funcionPerturbacion, funcionEvaluacion, iteraciones, tasaAceptacion):
    i = 0
    solucionActual, _ = funcionVecindad(solucionActual,datosProblema)
    costoActual = funcionEvaluacion(solucionActual,datosProblema)
    mejorCosto = costoActual
    while i < iteraciones:
        nuevaSolucion, _  = funcionVecindad(solucionActual,datosProblema)
        nuevoCosto = funcionEvaluacion(nuevaSolucion,datosProblema)
        if nuevoCosto < mejorCosto:
            mejorSolucion = nuevaSolucion
            mejorCosto = nuevoCosto
        i = i +1
    return solucionActual,costoActual, mejorSolucion, mejorCosto