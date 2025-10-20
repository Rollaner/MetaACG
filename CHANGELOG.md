Alpha 1.0: Modulo de definicion

####Experimentos en curso### 


Modificado Generador para que instancie un Evaluador y Optimizador. Se añadio la capacidad de usar el interpretador de python provisto por la API de openAI. Se planea añadir nuevos prompts de sistema para las API de gemini y otras LLM a futuro si el tiempo da.

Modificados los prompts de systema y cliente para indicar la posibilidad de utilizar el interpretador de la api

Modificado como se guardan los datos para que guarde cada generacion individual, esto permitira reducir el uso de tokens a futuro puesto que se pueden hacer ejecuciones parciales

Se guarda mayor variedad de datos de la generacion: Tiempo, clave, tipo y subtipo. con esto se puede tener referencia para la fase de optimizacion.

Se depreco el sistema de evaluacion interno. Es un problema de seguridad y presenta dificultades al momento de procesar las respuestas generadas.

Se implemento un sistema de argumentos para poder optimizar problemas individuales a futuro, se planea tambien añadir un modo que acepte llaves. 

