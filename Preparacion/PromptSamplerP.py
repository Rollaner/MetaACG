#Generador de prompts y cargado de las maquinas.
from string import Template
templateSeed= Template("""TASK:GENERATE_MATH_DATA.CODE_FUNC.EVAL
PROBLEM_DEF:
---
$problema
---
OUTPUT_FORMAT_STRICT:JSON.MUST_CONTAIN_3_KEYS.
OUTPUT_JSON_SCHEMA:{"MATH_DATA": "...", "SOL_TYPE": "...", "EVAL_CODE": "..."}
MATH_DATA_DEF:CONCISE_MATH_LOGIC_INCLUDE_OBJECTIVE_AND_CONSTRAINTS.USE_SOLVER_NOTATION.
SOLUTION_TYPE_DEF:STRING.SOL_REPRESENTATION_TYPE_MUST_MATCH.
EVAL_CODE_DEF:PYTHON_FUNC.NAME=evaluate_solution.ARGS=1(solution).RET_FITNESS_SCORE.SIG=def evaluate_solution(solution):
CRITICAL_INSTRUCTIONS:
1. CODE_SYNTAX_CORRECT_USE_MATH_MODULE_OTHER_IMPORTS_FORBIDDEN.
2. FOCUS_ON_DATA_PARITY_WITH_PROBLEM_DATA.
3. SOLUTION_ARG_TYPE_MUST_MATCH_GIVEN_SOLUTION_TYPE_DATA_IF_PRESENT.
4. EVAL = OBJ_AND_PROBLEM_CONSTRAINTS.INVALID_SOLUTIONS_MUST_BE_PENALIZED
5. EXEC_ENV_LOCAL_VARIABLE: "solution" == SOL_TYPE 
6. SOLVER_COMPATIBLE_NOTATION_FOR_PROBLEM_DEFINITION_EXPECTED
PROBLEM_TYPE_MATCHES:
$inspirations     
""")

templateUpdate= Template("""TASK:GENERATE_MATH_DATA.CODE_FUNC.EVAL
PROBLEM_DEF:
---
$problema
---
OUTPUT_FORMAT_STRICT:JSON.MUST_CONTAIN_3_KEYS.
OUTPUT_JSON_SCHEMA:{"MATH_DATA": "...", "SOL_TYPE": "...", "EVAL_CODE": "..."}
MATH_DATA_DEF:CONCISE_MATH_LOGIC_INCLUDE_OBJECTIVE_AND_CONSTRAINTS.USE_SOLVER_NOTATION.
SOLUTION_TYPE_DEF:STRING.SOL_REPRESENTATION_TYPE_MUST_MATCH.
EVAL_CODE_DEF:PYTHON_FUNC.NAME=evaluate_solution.ARGS=1(solution).RET_FITNESS_SCORE.SIG=def evaluate_solution(solution):
CRITICAL_INSTRUCTIONS:
1. CODE_SYNTAX_CORRECT_USE_MATH_MODULE_OTHER_IMPORTS_FORBIDDEN.
2. FOCUS_ON_DATA_PARITY_WITH_PROBLEM_DATA.
3. SOLUTION_ARG_TYPE_MUST_MATCH_GIVEN_SOLUTION_TYPE_DATA_IF_PRESENT.
4. EVAL = OBJ_AND_PROBLEM_CONSTRAINTS.INVALID_SOLUTIONS_MUST_BE_PENALIZED
5. EXEC_ENV_LOCAL_VARIABLE: "solution" == SOL_TYPE 
PROBLEM_TYPE_MATCHES:
$inspirations                     
FEEDBACK:
$feedback
""")

feedbackTemplate= Template("""TASK:GENERATE_FEEDBACK 
FEEDBACK_INSTRUCTIONS:
1.GENERATE_CRITICAL_FEEDBACK.FOCUS_ON_WEAKNESSES_AND_IMPROVEMENTS.AVOID_POSITIVE_REINFORCEMENT.USE_CLASIFICATIONS: DATA_ERROR, LOGIC_ERROR, RESULTS_NOT_CONSISTENT
2.FORMAT_AS_KEY_VALUE_PAIRS.EX: "DATA_ERROR: Proposed functions do not match the values given in the problem definition."
3.PINPOINT_SPECIFIC_MATH_FLAWS.EX: "LOGIC_ERROR: Operator in objective function not aligned with problem def, Suggest operator change in line: X."
4.FOCUS_ON_COMMON_ERRORS. EX_1: "RESULTS_NOT_CONSISTENT: Eval and expected result should be the same. EX_2 :LOGIC_ERROR: Eval code operators do not match the problem constraints."
5.ACCESS_GRANTED_TO str("python tool"): EVALUATE_INDEPENDENTLY.
6.LOCAL_SOLVERS_PROVIDE_GROUND_TRUTH: EVALUATION_FUNCTION should provide the means to verify expected result of provided random solution.  Optimality is obtained from local solvers with ast_eval() dynamic compiling   
---
PROBLEM_RAW:
$problema
---
DEFINITION:
$definicion
---                     
EVALUATION_FUNCTION:
$evaluacion                  
---
KNOWN_RANDOM_SOLUTION: $solucionParseada
EXPECTED_RESULT_FROM_SOLUTION: $esperado
OUTPUT_FORMAT_STRICT:
"DEFINITION", "FEEDBACK" """)

#Modificar esto para que carge de las instancias

def generarStrings(dataframe):
    if 'Text' in dataframe.columns:
        return  "\n".join(dataframe['Text'].astype(str).tolist())
    else: return f"{len(dataframe)} items found."

def generateSeedPrompt(problemaSample:str):
    inspiraciones = "NOT AVAILABLE"
    prompt = templateSeed.safe_substitute(problema=problemaSample, inspirations=inspiraciones) 
    return prompt

def generateSeedPromptWithProblemTYpe(problemaSample:str,tipoProblema:str):
    inspiraciones = tipoProblema
    prompt = templateSeed.safe_substitute(problema=problemaSample, inspirations=inspiraciones) 
    return prompt

def updatePrompt(problemaSample:str, tipoProblema:str, solucionParseada, feedback:str):
    ## En curso, este deberia añadir el feedback del evaluador, y los resultados esperados de las funciones objetivo y evaluacion. 
    prompt = templateUpdate.safe_substitute(problema=problemaSample, inspirations=tipoProblema, solucion=solucionParseada, feedback=feedback) 
    return prompt
## Feedback tiene que estar enfocado en los errores mas comunes de las LLM, Y errores que sabemos que son probables. por ejemplo que la funcion de evaluacion no tenga restricciones o bien que los numeros de la funcion objetivo no encajen con los datos de la instancia
## Este tambien nos sirve para clasificar automaticamente los errores que se detecten, en teoria

def generateFeedbackPrompt(problemaSample:str, definicion:str, objetivo, evaluacion,solucionParseada, esperado):
    prompt = feedbackTemplate.safe_substitute(problema=problemaSample, definicion=definicion,objetivo=objetivo, evaluacion=evaluacion, solucionParseada=solucionParseada, esperado=esperado) 
    return prompt