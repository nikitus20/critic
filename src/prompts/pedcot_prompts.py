"""PedCOT prompt templates based on the original paper."""

# Stage 1: Regenerate expected reasoning following pedagogical principles
STAGE1_REGENERATE_TEMPLATE = """You are given a problem and several initial steps.

Question: {question}

Initial steps: {context}

Execute the following instructions sequentially:

1. List the {domain} concepts that need to be REMEMBERED (recalled from memory) for the next step.
Following pedagogical principles:
Remember: {remember_principle}

2. List the key analyses that need to be UNDERSTOOD (comprehended and interpreted) for the next step.
Following pedagogical principles:
Understand: {understand_principle}

3. List the {domain} operations that need to be APPLIED (executed or implemented) for the next step.
Following pedagogical principles:
Apply: {apply_principle}

Based on these pedagogical principles, generate the expected reasoning for the next step of this problem.

Your response should be structured as:

**Remember (Concepts):**
[List key concepts to recall]

**Understand (Analysis):**
[Describe the approach and reasoning strategy]

**Apply (Operations):**
[List specific calculations or implementations]

**Expected Next Step:**
[Generate the complete expected reasoning for the next step]"""

# Stage 2: Compare actual vs expected reasoning
STAGE2_COMPARE_TEMPLATE = """You are given a problem and several initial steps.

Question: {question}

Initial steps: {context}

Current section to analyze: {section}

Your task is to identify incorrectly reasoned sections from the given sections. Execute the following instructions sequentially:

1. First extract the {domain} concepts employed in the current section. Give a detailed comparison between the actual concepts used and the expected concepts from the regenerated reasoning below. Finally, output a label to categorize the correctness of the concepts employed.

Expected concepts from Stage 1: {expected_concepts}

2. First extract the key analyses present in the current section. Give a detailed comparison between the actual reasoning approach and the expected approach from the regenerated reasoning below. Finally, output a label to categorize the correctness of the problem-solving approach.

Expected approach from Stage 1: {expected_approach}

3. First extract the key {domain} operations performed in the current section. Give a detailed comparison between the actual operations and the expected operations from the regenerated reasoning below. Finally, output a label to categorize the correctness of the calculations/implementations.

Expected operations from Stage 1: {expected_operations}

Expected complete reasoning from Stage 1: {expected_reasoning}

Respond in JSON format:
{{
    "remember_analysis": {{
        "actual_concepts": [list of concepts used in current section],
        "expected_concepts": [list of expected concepts],
        "comparison": "detailed comparison between actual and expected concepts",
        "correct": boolean,
        "explanation": "explanation of concept correctness"
    }},
    "understand_analysis": {{
        "actual_approach": "description of actual reasoning approach",
        "expected_approach": "description of expected reasoning approach", 
        "comparison": "detailed comparison between actual and expected approaches",
        "correct": boolean,
        "explanation": "explanation of approach correctness"
    }},
    "apply_analysis": {{
        "actual_operations": [list of operations performed in current section],
        "expected_operations": [list of expected operations],
        "comparison": "detailed comparison between actual and expected operations",
        "correct": boolean,
        "explanation": "explanation of operation correctness"
    }},
    "has_error": boolean,
    "error_types": [list of error types from DeltaBench taxonomy],
    "error_location": "description of where the error occurs",
    "explanation": "overall explanation of the error analysis",
    "confidence": float between 0 and 1,
    "principle_mistakes": [list of principle levels with errors: "remember", "understand", "apply"]
}}"""

# Simplified Stage 1 for domain-specific reasoning
STAGE1_MATH_TEMPLATE = """You are given a math problem and several initial steps.

Question: {question}

Initial steps: {context}

Execute the following instructions sequentially:

1. List the mathematical concepts that need to be recalled for the next step.
2. List the key analyses that need to be understood for the next step.  
3. List the mathematical expressions that need to be applied for the next step.

Following pedagogical principles:
Remember: {concepts}
Understand: {approach}
Apply: {calculations}

Generate the expected reasoning for the next step following these principles.

**Expected Next Step:**
[Generate the complete expected reasoning]"""

# Simplified Stage 2 for math domain
STAGE2_MATH_TEMPLATE = """You are given a math problem and several initial steps.

Question: {question}

Initial steps: {context}

Current section: {section}

Your task is to identify incorrectly reasoned sections. Execute the following instructions sequentially:

1. First extract the mathematical concepts employed in the current section. Give a detailed comparison between the actual concepts and the expected concepts: {expected_concepts}. Finally, output a label to categorize the correctness of the mathematical concepts employed.

2. First extract the key analyses present in the current section. Give a detailed comparison between the actual reasoning and the expected approach: {expected_approach}. Finally, output a label to categorize the correctness of the problem-solving approach.

3. First extract the mathematical expressions used in the current section. Give a detailed comparison between the actual calculations and the expected calculations: {expected_calculations}. Finally, output a label to categorize the correctness of the calculations.

Expected reasoning from Stage 1: {expected_reasoning}

Respond in JSON format:
{{
    "remember_analysis": {{"correct": boolean, "explanation": "explanation"}},
    "understand_analysis": {{"correct": boolean, "explanation": "explanation"}}, 
    "apply_analysis": {{"correct": boolean, "explanation": "explanation"}},
    "has_error": boolean,
    "error_types": [list from DeltaBench taxonomy],
    "error_location": "description",
    "explanation": "overall explanation",
    "confidence": float,
    "principle_mistakes": [list of principle levels with errors]
}}"""

# Template selection based on domain
PEDCOT_TEMPLATES = {
    'math': {
        'stage1': STAGE1_MATH_TEMPLATE,
        'stage2': STAGE2_MATH_TEMPLATE
    },
    'general': {
        'stage1': STAGE1_REGENERATE_TEMPLATE,
        'stage2': STAGE2_COMPARE_TEMPLATE
    }
}