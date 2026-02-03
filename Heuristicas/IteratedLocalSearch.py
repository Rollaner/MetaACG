import random

def ILS(solucionActual, mejorSolucion, mejorCosto,datosProblema, funcionVecindad, funcionPerturbacion, funcionEvaluacion, iteraciones, tasaAceptacion):
    i = 0
    solucionActual, _ = funcionVecindad(solucionActual,datosProblema)
    costoActual = funcionEvaluacion(solucionActual,datosProblema)
    mejorCosto = costoActual
    while i < iteraciones:
        solucionPerturbada = funcionPerturbacion(solucionActual,datosProblema)
        solucionMejorada, _  = funcionVecindad(solucionPerturbada,datosProblema)
        costoMejorado = funcionEvaluacion(solucionMejorada,datosProblema)
        if costoMejorado < mejorCosto:
            mejorSolucion = solucionMejorada
            mejorCosto = costoMejorado
        else: 
            if(random.random() < tasaAceptacion):
                continue
            solucionActual = solucionMejorada
        i = i +1
    return solucionActual,costoActual, mejorSolucion, mejorCosto