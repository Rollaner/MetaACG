import random

def ILS(solucionActual, mejorSolucion, mejorCosto, funcionVecindad, funcionPerturbacion, funcionEvaluacion, iteraciones, tasaRechazo):
    i = 0
    mejorSolucion = solucionActual
    mejorCosto = funcionEvaluacion(mejorSolucion)
    solucionActual = funcionVecindad(solucionActual)
    while i < iteraciones:
        solucionActual = funcionPerturbacion(solucionActual)
        solucionMejorada = funcionVecindad(solucionActual)
        costoMejorado = funcionEvaluacion(solucionMejorada)
        
        if costoMejorado < mejorCosto:
            mejorSolucion = solucionMejorada
            mejorCosto = costoMejorado
        else: 
            if(random.random() > tasaRechazo):
                continue
            solucionActual = solucionMejorada
        i = i +1
    return solucionActual, mejorSolucion, mejorCosto