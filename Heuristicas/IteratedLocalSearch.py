import random

def ILS(solucionActual, mejorSolucion, mejorCosto, funcionVecindad, funcionPerturbacion, funcionEvaluacion, iteraciones, tasaAceptacion):
    i = 0
    solucionActual, _ = funcionVecindad(solucionActual)
    costoActual = funcionEvaluacion(solucionActual)
    mejorCosto = costoActual
    while i < iteraciones:
        solucionPerturbada = funcionPerturbacion(solucionActual)
        solucionMejorada, _  = funcionVecindad(solucionPerturbada)
        costoMejorado = funcionEvaluacion(solucionMejorada)
        if costoMejorado < mejorCosto:
            mejorSolucion = solucionMejorada
            mejorCosto = costoMejorado
        else: 
            if(random.random() < tasaAceptacion):
                continue
            solucionActual = solucionMejorada
        i = i +1
    return solucionActual,costoActual, mejorSolucion, mejorCosto