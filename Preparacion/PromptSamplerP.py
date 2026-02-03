#Generador de prompts y cargado de las maquinas.
from string import Template

templateExtract = Template(
    """TASK: EXTRACT_PROBLEM_DATA
    PROBLEM_DEF:
    ---
    $problema
    ---
    OUTPUT_FORMAT_STRICT: JSON.MUST_CONTAIN_6_KEYS_AS_DEFINED_IN: PROBLEM_DATA.
    PROBLEM_DATA_DEF:SPECIFIC_PROBLEM_INSTANCE_VALUES_FORMATTED_AS_JSON:{DECISION_VARIABLES,DATA_ROLES,DATA,OBJECTIVE,CONSTRAINTS,SOLUTION_FORMAT}
    DECISION_VARIABLES_DEF: EACH_ENTRY_HAS_SYMBOL_DOMAIN_MEANING
    DATA_ROLES_DEF: EACH_ENTRY_HAS_NAME_INDEXING_MEANING
    INDEXING_DEF: EITHER_"SCALAR"_OR_"LIST"_IF_SINGLE_VALUE_OR_LIST_OF_VALUES_RESPECTIVELY
    DATA_DEF: RAW_VALUES_ONLY_KEYS_DECLARED_IN_DATA_ROLES
    OBJECTIVE_DEF: SENSE,EXPRESSION_SOLVER_COMPATIBLE
    CONSTRAINTS_DEF: LIST_OF_SOLVER_COMPATIBLE_EXPRESSIONS
    SOLUTION_FORMAT_DEF: TYPE,DESCRIPTION
    CRITICAL_INSTRUCTIONS:
    1. FOCUS_ON_DATA_PARITY_WITH_PROBLEM_DEF.
    2. SOLUTION_ARG_TYPE_MUST_MATCH_GIVEN_SOLUTION_TYPE_DATA_IF_PRESENT.
    3. SOLVER_COMPATIBLE_NOTATION_FOR_PROBLEM_DEFINITION_GUROBI_COMPATIBLE_SYNTAX_EXPECTED
    4. ALL_SYMBOLS_MUST_BE_DECLARED_BEFORE_USE_INDEXING_STR_MUST_BE_IN_UPPER
    5. DATA_VALUES_MUST_BE_PARSABLE_WITH_AST.EVAL()_USE_CUSTOM_CLASS
    6. DATA_FROM_DEF_MUST_REMAIN_UNCHANGED_AND_UNMODIFIED_IN_FINAL_SCHEMA
    """)

templateConvert = Template(
    """TASK: CONVERT_TEST_DATA
    TARGET_SCHEMA:
    ----
    $Schema
    ----
    EXAMPLE_INPUT:
    ----
    $Input
    ----
    OUTPUT_FORMAT_STRICT: PYTHON_CODE_SINGLE_FUNCTION, EX: Transform(input) -> Output:Schema
    
    CRITICAL_INSTRUCTIONS:
    1. INPUT_CLASS_STRUCTURE_IS_FINAL_BUT_CONTENTS_VARY
    2. TARGET_SCHEMA_IS_FINAL
    3. CODE_MUST_BE_COMPATIBLE_WITH_PYTHON_EVAL()_or_EXEC()
    """
)

templateUpdate= Template("""TASK: EXTRACT_PROBLEM_DATA
    PROBLEM_DEF:
    ---
    $problema
    ---
    OUTPUT_FORMAT_STRICT: JSON.MUST_CONTAIN_6_KEYS_AS_DEFINED_IN: PROBLEM_DATA.
    PROBLEM_DATA_DEF:SPECIFIC_PROBLEM_INSTANCE_VALUES_FORMATTED_AS_JSON:{DECISION_VARIABLES,DATA_ROLES,DATA,OBJECTIVE,CONSTRAINTS,SOLUTION_FORMAT}
    DECISION_VARIABLES_DEF: EACH_ENTRY_HAS_SYMBOL_DOMAIN_MEANING
    DATA_ROLES_DEF: EACH_ENTRY_HAS_NAME_INDEXING_MEANING
    INDEXING_DEF: EITHER_"SCALAR"_OR_"LIST"_IF_SINGLE_VALUE_OR_LIST_OF_VALUES_RESPECTIVELY
    DATA_DEF: RAW_VALUES_ONLY_KEYS_DECLARED_IN_DATA_ROLES
    OBJECTIVE_DEF: SENSE,EXPRESSION_SOLVER_COMPATIBLE
    CONSTRAINTS_DEF: LIST_OF_SOLVER_COMPATIBLE_EXPRESSIONS
    SOLUTION_FORMAT_DEF: TYPE,DESCRIPTION
    CRITICAL_INSTRUCTIONS:
    1. FOCUS_ON_DATA_PARITY_WITH_PROBLEM_DEF.
    2. SOLUTION_ARG_TYPE_MUST_MATCH_GIVEN_SOLUTION_TYPE_DATA_IF_PRESENT.
    3. SOLVER_COMPATIBLE_NOTATION_FOR_PROBLEM_DEFINITION_GUROBI_COMPATIBLE_SYNTAX_EXPECTED
    4. ALL_SYMBOLS_MUST_BE_DECLARED_BEFORE_USE_INDEXING_STR_MUST_BE_IN_UPPER
    5. DATA_VALUES_MUST_BE_PARSABLE_WITH_AST.EVAL()_USE_CUSTOM_CLASS
    6. DATA_FROM_DEF_MUST_REMAIN_UNCHANGED_AND_UNMODIFIED_IN_FINAL_SCHEMA
    FEEDBACK:
    $feedback
""")

feedbackTemplate= Template("""TASK:GENERATE_FEEDBACK 
FEEDBACK_INSTRUCTIONS:
1.GENERATE_CRITICAL_FEEDBACK.FOCUS_ON_WEAKNESSES_AND_IMPROVEMENTS.AVOID_POSITIVE_REINFORCEMENT.USE_CLASIFICATIONS: DATA_ERROR, LOGIC_ERROR, RESULTS_NOT_CONSISTENT
2.FORMAT_AS_KEY_VALUE_PAIRS.EX: "DATA_ERROR: Proposed functions do not match the values given in the problem definition."
3.PINPOINT_SPECIFIC_MATH_FLAWS.EX: "LOGIC_ERROR: Operator in objective function not aligned with problem def, Suggest operator change in line: X."
4.LOCAL_SOLVERS_PROVIDE_GROUND_TRUTH: EVALUATION_FUNCTION should provide the means to verify expected result of provided random solution.  Optimality is obtained from local solvers with ast_eval() dynamic compiling 
5 EXTRACTED_DATA_MUST_BE_CONSISTENT_WITH_THE_FOLLOWING_RULES_AT_ALL_COSTS:
---
OUTPUT_FORMAT_STRICT: JSON.MUST_CONTAIN_6_KEYS_AS_DEFINED_IN: PROBLEM_DATA.
PROBLEM_DATA_DEF:SPECIFIC_PROBLEM_INSTANCE_VALUES_FORMATTED_AS_JSON:{DECISION_VARIABLES,DATA_ROLES,DATA,OBJECTIVE,CONSTRAINTS,SOLUTION_FORMAT}
DECISION_VARIABLES_DEF: EACH_ENTRY_HAS_SYMBOL_DOMAIN_MEANING
DATA_ROLES_DEF: EACH_ENTRY_HAS_NAME_INDEXING_MEANING
INDEXING_DEF: EITHER_"SCALAR"_OR_"LIST"; INDEXING_STR_MUST_BE_IN_UPPER
DATA_DEF: RAW_VALUES_ONLY_KEYS_DECLARED_IN_DATA_ROLES
OBJECTIVE_DEF: SENSE,EXPRESSION_SOLVER_COMPATIBLE
CONSTRAINTS_DEF: LIST_OF_SOLVER_COMPATIBLE_EXPRESSIONS
SOLUTION_FORMAT_DEF: TYPE,DESCRIPTION
---
PROBLEM_RAW:
$problema
---
EXTRACTED_DATA:
$extracted
---                                  
KNOWN_RANDOM_SOLUTION: $solucionParseada
EXPECTED_RESULT_FROM_SOLUTION: $esperado
OUTPUT_FORMAT_STRICT:
"FEEDBACK:" """)

#Modificar esto para que carge de las instancias

def generarStrings(dataframe):
    if 'Text' in dataframe.columns:
        return  "\n".join(dataframe['Text'].astype(str).tolist())
    else: return f"{len(dataframe)} items found."

def generateSeedPrompt(problemaSample:str):
    inspiraciones = "NOT AVAILABLE"
    prompt = templateExtract.safe_substitute(problema=problemaSample, inspirations=inspiraciones) 
    return prompt

def generateConverterPrompt(schema,input):
    prompt = templateConvert.safe_substitute(Schema = schema, Input = input)
    return prompt

def updatePrompt(problemaSample:str, tipoProblema:str, solucionParseada, feedback:str):
    ## En curso, este deberia añadir el feedback del evaluador, y los resultados esperados de las funciones objetivo y evaluacion. 
    prompt = templateUpdate.safe_substitute(problema=problemaSample, inspirations=tipoProblema, solucion=solucionParseada, feedback=feedback) 
    return prompt
## Feedback tiene que estar enfocado en los errores mas comunes de las LLM, Y errores que sabemos que son probables. por ejemplo que la funcion de evaluacion no tenga restricciones o bien que los numeros de la funcion objetivo no encajen con los datos de la instancia
## Este tambien nos sirve para clasificar automaticamente los errores que se detecten, en teoria

def generateFeedbackPrompt(problemaSample:str, extractedData:str,solucionParseada, esperado):
    prompt = feedbackTemplate.safe_substitute(problema=problemaSample, extracted=extractedData,  solucionParseada=solucionParseada, esperado=esperado) 
    return prompt