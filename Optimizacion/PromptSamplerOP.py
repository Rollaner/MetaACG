import random
import itertools
from string import Template
import pandas as pd
import json
## To-Do: Revisar templates. (Sanity check)
templateEval = Template("""TASK:GENERATE_COMPONENT_HEURISTIC_EVAL
PROBLEM_DATACLASS:
---
$schema
---
PROBLEM_OBJECTIVE:
---
$objective
---
PROBLEM_CONSTRAINTS:
---
$constraints
---
SAMPLE_SOL:
$sampleSol
---
OUTPUT_FORMAT_STRICT:JSON.MUST_CONTAIN_2_KEYS.
OUTPUT_JSON_SCHEMA:{"REPRESENTATION": "...", "EVAL_CODE": "..."}

EVAL_CODE_DEF:PYTHON_FUNC.NAME=evaluate_solution.ARGS=2(solution,dataclass).RET_NUM_FITNESS.SIG=def evaluate_solution(solution,pdataclass):

CRITICAL_INSTRUCTIONS:
1.CODE_SYNTAX_CORRECT_FUNC_SELF_CONTAINED_SCHEMA_WILL_BE_GIVEN_AS_INSTANCED_DATACLASS.
2.AVOID_EXPLANATION_OR_COMMENTS
3.EVAL_CODE_SIG_MUST_BE_def evaluate_solution(solution,dataclass):
4.GENERATED_CODE_MUST_BE_COMPATIBLE_WITH_SAMPLE_SOL
5.CODE_MUST_INCLUDE_ALL_NECESARY_IMPORTS_ENVIROMENT_HEAVILY_SANDBOXED.
6.PROBLEM_DATA_MUST_BE_OBTAINED_FROM_DATACLASS_VIA_DIRECT_ACCESS_ONLY.
7.EXTERNAL_DATA_MUST_BE_PASSED_AS_ARGUMENT_DATA_MUST_BE_ENCODED_OR_EMBEDDED.
                        
INSPIRATIONS:
$Inspirations                        
""")

templatePerturb= Template("""TASK:GENERATE_COMPONENT_HEURISTIC_PERTURB                        
PROBLEM_DATACLASS:
---
$schema
---
PROBLEM_OBJECTIVE:
---
$objective
---
PROBLEM_CONSTRAINTS:
---
$constraints
---
SAMPLE_SOL:
$sampleSol
---
OUTPUT_FORMAT_STRICT:JSON.MUST_CONTAIN_2_KEYS.
OUTPUT_JSON_SCHEMA:{"REPRESENTATION": "...", "PERTURB_CODE": "..."}

PERTURB_CODE_DEF:PYTHON_FUNC.NAME=perturb_solution.ARGS=2(solution,dataclass).OP_PERTURBED_SOLUTION.SIG=def perturb_solution(solution,dataclass):

CRITICAL_INSTRUCTIONS:
1.CODE_SYNTAX_CORRECT_FUNC_SELF_CONTAINED_SCHEMA_WILL_BE_GIVEN_AS_INSTANCED_DATACLASS.
2.AVOID_EXPLANATION_OR_COMMENTS
3.PERTURB_CODE_SIG_MUST_BE_def perturb_solution(solution,dataclass):
4.GENERATED_CODE_MUST_BE_COMPATIBLE_WITH_SAMPLE_SOL
5.CODE_MUST_INCLUDE_ALL_NECESARY_IMPORTS_ENVIROMENT_HEAVILY_SANDBOXED.
6.PROBLEM_DATA_MUST_BE_OBTAINED_FROM_DATACLASS_VIA_DIRECT_ACCESS_ONLY.
7.EXTERNAL_DATA_MUST_BE_PASSED_AS_ARGUMENT_DATA_MUST_BE_ENCODED_OR_EMBEDDED.
                        
INSPIRATIONS:
$Inspirations
""")

templateNB= Template("""TASK:GENERATE_COMPONENT_HEURISTIC_NB
PROBLEM_DATACLASS:
---
$schema
---
PROBLEM_OBJECTIVE:
---
$objective
---
PROBLEM_CONSTRAINTS:
---
$constraints
---
SAMPLE_SOL:
$sampleSol
---
OUTPUT_FORMAT_STRICT:JSON.MUST_CONTAIN_2_KEYS.
OUTPUT_JSON_SCHEMA:{"REPRESENTATION": "...", "NB_CODE": "..."}

NB_CODE_DEF:PYTHON_FUNC.NAME=generate_neighbour.ARGS=(solution,dataclass).RETURN=tuple(any,any).REQUIRED_EXIT="return (NB, Movement)"
                     
CRITICAL_INSTRUCTIONS:
1.CODE_SYNTAX_CORRECT_FUNC_SELF_CONTAINED_SCHEMA_WILL_BE_GIVEN_AS_INSTANCED_DATACLASS.
2.AVOID_EXPLANATION_OR_COMMENTS
3.STRICT_RETURN_TYPE: MUST_RETURN_TUPLE_OF_EXACTLY_2_ELEMENTS. 
4.GENERATED_CODE_MUST_BE_COMPATIBLE_WITH_SAMPLE_SOL
5.CODE_MUST_INCLUDE_ALL_NECESARY_IMPORTS_ENVIROMENT_HEAVILY_SANDBOXED.
6.PROBLEM_DATA_MUST_BE_OBTAINED_FROM_DATACLASS_VIA_DIRECT_ACCESS_ONLY.
7.EXTERNAL_DATA_MUST_BE_PASSED_AS_ARGUMENT_DATA_MUST_BE_ENCODED_OR_EMBEDDED.
                        
INSPIRATIONS:
$Inspirations
""")

templateEvalUpdate = Template("""TASK:GENERATE_COMPONENT_HEURISTIC_EVAL
PROBLEM_DATACLASS:
---
$schema
---
PROBLEM_OBJECTIVE:
---
$objective
---
PROBLEM_CONSTRAINTS:
---
$constraints
---
SAMPLE_SOL:
$sampleSol
---
OUTPUT_FORMAT_STRICT:JSON.MUST_CONTAIN_2_KEYS.
OUTPUT_JSON_SCHEMA:{"REPRESENTATION": "...", "EVAL_CODE": "..."}

EVAL_CODE_DEF:PYTHON_FUNC.NAME=evaluate_solution.ARGS=2(solution,dataclass).RET_NUM_FITNESS.SIG=def evaluate_solution(solution,dataclass):

CRITICAL_INSTRUCTIONS:
1.CODE_SYNTAX_CORRECT_FUNC_SELF_CONTAINED_SCHEMA_WILL_BE_GIVEN_AS_INSTANCED_DATACLASS.
2.AVOID_EXPLANATION_OR_COMMENTS
3.EVAL_CODE_SIG_MUST_BE_def evaluate_solution(solution,dataclass):
4.GENERATED_CODE_MUST_BE_COMPATIBLE_WITH_SAMPLE_SOL
5.CODE_MUST_INCLUDE_ALL_NECESARY_IMPORTS_ENVIROMENT_HEAVILY_SANDBOXED.
6.PROBLEM_DATA_MUST_BE_OBTAINED_FROM_DATACLASS_VIA_DIRECT_ACCESS_ONLY.
7.EXTERNAL_DATA_MUST_BE_PASSED_AS_ARGUMENT_DATA_MUST_BE_ENCODED_OR_EMBEDDED.
                        
INSPIRATIONS:
$Inspirations    
RESULTS_FORMAT "{current, currentScore, best, bestScore}"
$Resultados
FEEDBACK:
$Feedback
""")

templatePerturbUpdate = Template("""TASK:GENERATE_COMPONENT_HEURISTIC_PERTURB
PROBLEM_DATACLASS:
---
$schema
---
PROBLEM_OBJECTIVE:
---
$objective
---
PROBLEM_CONSTRAINTS:
---
$constraints
---
SAMPLE_SOL:
$sampleSol
---
OUTPUT_FORMAT_STRICT:JSON.MUST_CONTAIN_2_KEYS.
OUTPUT_JSON_SCHEMA:{"REPRESENTATION": "...", "PERTURB_CODE": "..."}

PERTURB_CODE_DEF:PYTHON_FUNC.NAME=perturb_solution.ARGS=2(solution,dataclass).OP_PERTURBED_SOLUTION.SIG=def perturb_solution(solution,dataclass):

CRITICAL_INSTRUCTIONS:
1.CODE_SYNTAX_CORRECT_FUNC_SELF_CONTAINED_SCHEMA_WILL_BE_GIVEN_AS_INSTANCED_DATACLASS.
2.AVOID_EXPLANATION_OR_COMMENTS
3.PERTURB_CODE_SIG_MUST_BE_def perturb_solution(solution,dataclass):
4.GENERATED_CODE_MUST_BE_COMPATIBLE_WITH_SAMPLE_SOL
5.CODE_MUST_INCLUDE_ALL_NECESARY_IMPORTS_ENVIROMENT_HEAVILY_SANDBOXED.
6.PROBLEM_DATA_MUST_BE_OBTAINED_FROM_DATACLASS_VIA_DIRECT_ACCESS_ONLY.
7.EXTERNAL_DATA_MUST_BE_PASSED_AS_ARGUMENT_DATA_MUST_BE_ENCODED_OR_EMBEDDED.
                        
INSPIRATIONS:
$Inspirations
RESULTS_FORMAT "{current, currentScore, best, bestScore}"
$Resultados
FEEDBACK:
$Feedback
""")

templateNBUpdate = Template("""TASK:GENERATE_COMPONENT_HEURISTIC_NB
PROBLEM_DATACLASS:
---
$schema
---
PROBLEM_OBJECTIVE:
---
$objective
---
PROBLEM_CONSTRAINTS:
---
$constraints
---
SAMPLE_SOL:
$sampleSol
---
OUTPUT_FORMAT_STRICT:JSON.MUST_CONTAIN_2_KEYS.
OUTPUT_JSON_SCHEMA:{"REPRESENTATION": "...", "NB_CODE": "..."}

NB_CODE_DEF:PYTHON_FUNC.NAME=generate_neighbour.ARGS=(solution,dataclass).RETURN=tuple(any,any).REQUIRED_EXIT="return (NB, Movement)"
                     
CRITICAL_INSTRUCTIONS:
1.CODE_SYNTAX_CORRECT_FUNC_SELF_CONTAINED_SCHEMA_WILL_BE_GIVEN_AS_INSTANCED_DATACLASS.
2.AVOID_EXPLANATION_OR_COMMENTS
3.STRICT_RETURN_TYPE: MUST_RETURN_TUPLE_OF_EXACTLY_2_ELEMENTS. 
4.GENERATED_CODE_MUST_BE_COMPATIBLE_WITH_SAMPLE_SOL;STANDARD_FORMAT_IS_0_INDEX_UNLESS_SPECIFIED_OTHERWISE
5.CODE_MUST_INCLUDE_ALL_NECESARY_IMPORTS_ENVIROMENT_HEAVILY_SANDBOXED.
6.PROBLEM_DATA_MUST_BE_OBTAINED_FROM_DATACLASS_VIA_DIRECT_ACCESS_ONLY.
7.EXTERNAL_DATA_MUST_BE_PASSED_AS_ARGUMENT_DATA_MUST_BE_ENCODED_OR_EMBEDDED.
                        
INSPIRATIONS:
$Inspirations
RESULTS_FORMAT "{current, currentScore, best, bestScore}"
$Resultados
FEEDBACK:
$Feedback
""")

feedbackTemplate= Template("""TASK:GENERATE_FEEDBACK_FOR_COMPONENT_REFINEMENT 
FEEDBACK_INSTRUCTIONS:
0.LOCAL_SOLVER_ERRORS_MUST_BE_CORRECTED_FIRST_IF_NO_CRITICAL_ERRORS_OUTPUT: "CODE_STATUS: OPTIMAL"
1.GENERATE_CONSTRUCTIVE_FEEDBACK:ONLY_FLAG_ERRORS_THAT_CRASH_LOCAL_SOLVER
2.FORMAT_AS_KEY_VALUE_PAIRS.EX: "E_CODE_PARSE_ERROR: Solution is string, while E_CODE expects an array"
3.IF_ERRORS_FOUND_PINPOINT_SPECIFIC_FLAW_SUGGEST_SPECIFIC_IMPROVEMENT. MAXIMUM_4_ITEMS_ALLOWED_ONLY_SUGGEST_CHANGES_IF_REDUCES_ERROR_COUNT
4.PROBLEM_DATA,KNOWN_RANDOM_SOLUTION_AND_VALUE_GIVEN: USE_PYTHON_TOOL_TO_EVALUATE_COMPONENTS_TO_ASSERT_CORRECTNESS_OF_COMPONENTS
5.LOCAL_SOLVER_DESIGNED_FOR_EVALUATION_EXTRA_OUTPUTS_ARE_EXPECTED
6.CODE_MUST_BE_COMPATIBLE_WITH_SAMPLE_SOLUTION_AND_LOCAL_SOLVER
7.LOCAL_SOLVER_CANNOT_OBTAIN_PROBLEM_DATA_DIRECTLY_FUNCTIONS_MUST_DO_IT_VIA_DIRECT_ACCESS_ONLY
8.DO_NOT_FIX_SUCCESS: IF_SCORE_BEATS_KNOWN_BEST_WHILE_CONSISTENT_WITH_DATA_AND_CONSTRAINTS_OUTPUT: "CODE_STATUS: OPTIMAL"
PROBLEM_DATA:
---
$schema
---
PROBLEM_OBJECTIVE:
---
$objective
---
PROBLEM_CONSTRAINTS:
---
$constraints
---
TARGET_HEURISTIC_GENERAL_SIGNATURE= "def Heuristic(currentSolution,best, best_score, generate_neighbour, evaluate_solution, perturb_solution, other_params)".                       
COMPONENTS:
"Evaluation Function":
$Eval
"Neigbour Function":
$NB    
"Perturbation Function:"
$perturb
"Sample Solution":
$SampleSol
RESULTS_FROM_LOCAL_SOLVER
Simulated_Annealing
$resultadosSA
Iterated_Local_Search
$resultadosILS
HillClimbing
$resultadosHC
KNOWN_BENCHMARK_SOLUTION
$knownSol
EXPECTED_SCORE_FROM_KNOWN_BENCHMARK_SOLUTION
$knownScore
OUTPUT_FORMAT_STRICT:
"COMPONENT_VERSION" "$version", "FEEDBACK" """)


def sampleProblemaDB(problemaDB,seed):
    return problemaDB.sample(n=1, random_state=seed) 

def generarStrings(dataframe):
    if 'Text' in dataframe.columns:
        return  "\n".join(dataframe['Text'].astype(str).tolist())
    else: return f"{len(dataframe)} items found."

def generateEvalPrompt(schema,objetivo, restricciones, samplesol):
    prompt = templateEval.safe_substitute(
        schema = schema,
        objective = objetivo,
        constraints = restricciones,
        sampleSol = samplesol, 
        Inspirations = 'NA'
    ) 
    return prompt
#Updatea el prompt, assume una sola fila por version. 

def updateEvalPrompt(schema,objetivo, restricciones, resultados, feedback,samplesol):
    prompt = templateEvalUpdate.safe_substitute(
        schema = schema,
        objective = objetivo,
        constraints = restricciones,
        sampleSol = samplesol,
        Inspirations = 'NA',
        Resultados=resultados,
        Feedback = feedback
    )
    return prompt

def generateNBPrompt(schema,objetivo, restricciones,samplesol):
    prompt = templateNB.safe_substitute(
        schema = schema,
        objective = objetivo,
        constraints = restricciones,
        sampleSol = samplesol, 
        Inspirations = 'NA'
    ) 
    return prompt
#Updatea el prompt, assume una sola fila por version. 

def updateNBPrompt(schema,objetivo, restricciones, resultados, feedback,samplesol):
    prompt = templateNBUpdate.safe_substitute(
        schema = schema,
        objective = objetivo,
        constraints = restricciones,
        sampleSol = samplesol,
        Inspirations = 'NA',
        Resultados=resultados,
        Feedback = feedback
    )
    return prompt


def generatePerturbPrompt(schema,objetivo, restricciones, samplesol):
    prompt = templatePerturb.safe_substitute(
        schema = schema,
        objective = objetivo,
        constraints = restricciones,
        sampleSol = samplesol, 
        Inspirations = 'NA'
    ) 
    return prompt
#Updatea el prompt, assume una sola fila por version. 

def updatePerturbPrompt(schema,objetivo, restricciones, resultados, feedback, samplesol):
    prompt = templatePerturbUpdate.safe_substitute(
        schema = schema,
        objective = objetivo,
        constraints = restricciones,
        sampleSol = samplesol,
        Inspirations = 'NA',
        Resultados=resultados,
        Feedback = feedback
    )
    return prompt

# Este requiere que existan las DB de antemano. 
#(Para hacer mix and match se tiene que mezclar antes de que se haga la evaluacion y feedback, con la version siendo dependentiente de la iteracion)
#En consecuencia las versiones son enteros en el rango [0:Iteraciones]
def retornarCoincidentes(df:pd.DataFrame, problemaID, version):
    filaCoincidente = df[
        (df['ID_Problema'] == problemaID) &
        (df['Version'] == version)
    ]
    if filaCoincidente.empty:
        raise ValueError(f"No component set found for Problema ID: {problemaID} and Version: {version}")
    return filaCoincidente.iloc[0]

#Funcion para hacer mix and match. Debe llamarse antes de hacer una evaluacion. Se genera feedback con los componentes mezclados y los resultados de la evalucion
def sampleComponenteDB(componenteDB, problemaID,version,seed):
    subsetProblema = componenteDB[componenteDB['ID_Problema'] == problemaID]
    if subsetProblema.empty:
        return pd.DataFrame()
    random.seed(seed)
    try:
        representaciones = subsetProblema['Representacion'].unique()
        representacionEscogida = random.choice(representaciones)
    except IndexError:
        return pd.DataFrame()
    subsetCompatible = subsetProblema[subsetProblema['Representacion'] == representacionEscogida]
    try:
        eval = random.choice(subsetCompatible['Evaluacion'].unique())
        vecindad = random.choice(subsetCompatible['Vecindad'].unique())
        perturbacion = random.choice(subsetCompatible['Perturbacion'].unique())
    except IndexError:
        return pd.DataFrame()
    datosComponentes = {
        'ID_Problema': problemaID,
        'Representacion': representacionEscogida,
        'Evaluacion': eval,
        'Vecindad': vecindad,
        'Perturbacion': perturbacion,
        'Version': version 
    }
    return pd.DataFrame([datosComponentes])

## Feedback tiene que estar enfocado en un solo set de componentes a la vez. El ultimo que fue generado
def generateFeedbackPrompt(problemData,objetivo, restricciones, Eval, Nb, Perturb, SampleSol, resultadosSA, resultadosILS, resultadosTS, knownSol, knownObj, version):
    prompt = feedbackTemplate.safe_substitute(
        problema=problemData,
        objective = objetivo,
        constraints = restricciones, 
        Eval=Eval, 
        NB=Nb,
        perturb=Perturb, 
        SampleSol=SampleSol,
        resultadosSA=resultadosSA,
        resultadosILS=resultadosILS, 
        resultadosTS=resultadosTS, 
        knownSol=knownSol, 
        knownScore=knownObj,
        version = version
    ) 
    return prompt