def TS(solucionActual,mejorSolucion,mejorCosto,datosProblema,funcionVencindad, funcionEvaluacion,iteraciones, sizeListaTabu, duracionTabu):
    i = 0
    listaTabu = {}
    mejorCosto = funcionEvaluacion(solucionActual,datosProblema)
    while i < iteraciones:
        vecindad = funcionVencindad(solucionActual,datosProblema)
        mejorVecinoIteracion = None
        costoMejorVecinoIteracion = float('inf')
        movimientoSeleccionado = None
        for vecino, movimiento in vecindad:
            costoVecino = funcionEvaluacion(vecino)
            esTabu = movimiento in listaTabu and listaTabu[movimiento] > i
            if esTabu and costoVecino < mejorCosto:
                esTabu = False  # Aspiration rule overrides tabu
            if not esTabu:
                if costoVecino < costoMejorVecinoIteracion:
                    costoMejorVecinoIteracion = costoVecino
                    mejorVecinoIteracion = vecino
                    movimientoSeleccionado = movimiento

        if mejorVecinoIteracion is None:
            break

        solucionActual = mejorVecinoIteracion
        if costoMejorVecinoIteracion < mejorCosto:
            mejorSolucion = solucionActual
            mejorCosto = costoMejorVecinoIteracion

        if movimientoSeleccionado is not None:
            listaTabu[movimientoSeleccionado] = i + duracionTabu
            if len(listaTabu) > sizeListaTabu:
                listaTabu.pop(0)

        movimientos_a_remover = [m for m, exp in listaTabu.items() if exp <= i]
        for m in movimientos_a_remover:
            del listaTabu[m]

        i += 1
    return solucionActual, funcionEvaluacion(solucionActual,datosProblema), mejorSolucion, mejorCosto

