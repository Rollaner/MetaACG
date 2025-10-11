#Generador de promts y cargado de las maquinas.
from string import Template
## To-Do: Revisar templates. (Sanity check)
templateSeed= Template("""TASK:GENERATE_MATH_DEF.CODE_FUNC.EVAL
PROBLEM_DEF:
---
$problema
---
OUTPUT_FORMAT_STRICT:"MATH_DEF","SOL_TYPE","OBJ_CODE","EVAL_CODE"
MATH_DEF_DEF:CONCISE_MATH_LOGIC_DEF_OBJ_CONSTRAINTS.USE_STANDARD_NOTATION.
SOL_TYPE_DEF:STRING.SOL_REPRESENTATION_TYPE.OPTIONS=INDEX_LIST,BINARY_STRING,NODE_SEQUENCE.
OBJ_CODE_DEF:PYTHON_FUNC.NAME=objective_function.ARGS=1(solution).RET_NUM_VALUE.SIG=def objective_function(solution):
EVAL_CODE_DEF:PYTHON_FUNC.NAME=evaluate_solution.ARGS=1(solution).RET_NUM_FITNESS.SIG=def evaluate_solution(solution):
CRITICAL_INSTRUCTIONS:
1. CODE_SYNTAX_CORRECT_FUNC_SELF_CONTAINED.
2. NO_ADD_TEXT_EXPLANATION_FORMAT_OUTSIDE_CSV.
3. SOL_ARG_TYPE_MUST_MATCH_SOL_TYPE_DEF_VALUE.
4. MATH_DEF_MUST_BE_PARSABLE_BY_PYTHON_SYM_PACKAGE (e.g., SymPy).
5. MATCH_PROBLEM-REQUIRED_SOL_TYPE_IF_EXISTS
PROBLEM_TYPE_MATCHES:
$inspirations
""")

templateUpdate= Template("""TASK:GENERATE_MATH_DEF.CODE_FUNC.EVAL
PROBLEM_DEF:
---
$problema
---
OUTPUT_FORMAT_STRICT:"MATH_DEF","SOL_TYPE","OBJ_CODE","EVAL_CODE"
MATH_DEF_DEF:CONCISE_MATH_LOGIC_DEF_OBJ_CONSTRAINTS.USE_STANDARD_NOTATION.
SOL_TYPE_DEF:STRING.SOL_REPRESENTATION_TYPE.OPTIONS=INDEX_LIST,BINARY_STRING,NODE_SEQUENCE.
OBJ_CODE_DEF:PYTHON_FUNC.NAME=objective_function.ARGS=1(solution).RET_NUM_VALUE.SIG=def objective_function(solution):
EVAL_CODE_DEF:PYTHON_FUNC.NAME=evaluate_solution.ARGS=1(solution).RET_NUM_FITNESS.SIG=def evaluate_solution(solution):
CRITICAL_INSTRUCTIONS:
1. CODE_SYNTAX_CORRECT_FUNC_SELF_CONTAINED.
2. NO_ADD_TEXT_EXPLANATION_FORMAT_OUTSIDE_CSV.
3. SOL_ARG_TYPE_MUST_MATCH_SOL_TYPE_DEF_VALUE.
4. MATH_DEF_MUST_BE_PARSABLE_BY_PYTHON_SYM_PACKAGE (e.g., SymPy).
5. MATCH_PROBLEM-REQUIRED_SOL_TYPE_IF_EXISTS
PROBLEM_TYPE_MATCHES:
$inspirations                     
RESULTS
$resultados
EXPECTED
$esperado
FEEDBACK:
$feedback
""")

feedbackTemplate= Template("""TASK:GENERATE_FEEDBACK 
FEEDBACK_INSTRUCTIONS:
1.GENERATE_CRITICAL_FEEDBACK.FOCUS_ON_WEAKNESSES_AND_IMPROVEMENTS.AVOID_POSITIVE_REINFORCEMENT.
2.FORMAT_AS_KEY_VALUE_PAIRS.EX: "EVAL_HAS_NO_CONSTRAINTS:O(n). Add constraints to check validity at line X."
3.PINPOINT_SPECIFIC_MATH_FLAWS.EX: "OBJ_CODE_FAIL_LOCAL_OPT:Operator in objective function not aligned with problem def, Suggest operator change in line: X."
4.SUGGEST_SPECIFIC_IMPROVEMENTS.EX: "R_STR_INADEQUATE:Binary string causing poor exploration. Recommend a permutation in line X."
5.FOCUS_ON_COMMON_ERRORS: EVAL_HAS_NO_CONSTRAINTS, RESULTS_NOT_CONSISTENT:Objective, Eval and expected result shoulñd be the same. LOGIC_ERROR: Eval does not match the problem constraints. ARITHMETIC_ERROR: Objective function does not math the values given in the problem definition.
---
PROBLEM_RAW:
$problema
---
DEFINITION:
$definicion
---                      
OBJECTIVE_FUNCTION:
$objetivo
---
EVALUATION_FUNCTION:
$evaluacion                  
---
RESULTS
$resultados
EXPECTED: $esperado
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

def updatePrompt(problemaSample:str, tipoProblema:str, resultados, esperado, feedback:str):
    ## En curso, este deberia añadir el feedback del evaluador, y los resultados esperados de las funciones objetivo y evaluacion. 
    prompt = templateUpdate.safe_substitute(problema=problemaSample, inspirations=tipoProblema, resultados=resultados, esperado=esperado, feedback=feedback) 
    return prompt
## Feedback tiene que estar enfocado en los errores mas comunes de las LLM, Y errores que sabemos que son probables. por ejemplo que la funcion de evaluacion no tenga restricciones o bien que los numeros de la funcion objetivo no encajen con los datos de la instancia
## Este tambien nos sirve para clasificar automaticamente los errores que se detecten, en teoria
def generateFeedbackPrompt(problemaSample:str, definicion:str, objetivo, evaluacion, resultados, esperado):
    prompt = feedbackTemplate.safe_substitute(problema=problemaSample.iloc[0,1], definicion=definicion,objetivo=objetivo, evaluacion=evaluacion, resultados=resultados, esperado=esperado) 
    return prompt