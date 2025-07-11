"""Configuration for DeltaBench."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = "gpt-4o-mini"
TEMPERATURE = 1.0
TOP_P = 0.8

# DeltaBench original prompt
DELTABENCH_PROMPT = """I will provide you with a question and a robot's answer, divided into several sections. Your task is to evaluate each section of the robot's answer for any errors.

**Evaluation Criteria:**
- Evaluate each section independently. Assess each section based solely on the accuracy and logic within that section.
- Do not consider subjective elements such as redundancy or stylistic preferences as errors.
- Do not consider corrections or reflections made in later sections. Even if a later section acknowledges and fixes an earlier mistake, the original section must still be marked as erroneous.
- If a subsequent section contains an error caused by an earlier section's mistake, do not count it as a new error.

**Output Format:**
- If you think all sections of the robot's answer are correct, output in the following format:  
    Conclusion: no error
- If you think any section contains an error, output in the following format:  
    Conclusion: yes
    Error Section Number: [section number]
    Explanation: [explanation for the error in this section]
    Error Section Number: [section number]
    Explanation: [explanation for the error in this section]
    … (repeat for each erroneous section)  

**Input:**
- Question: {question}
- Robot's Answer: {model_output}
- judge result: """

# PedCOT prompt (placeholder - to be discussed and updated)
PEDCOT_PROMPT = """I will provide you with a question and a robot's answer, divided into several sections. Your task is to evaluate each section of the robot's answer for any errors using a pedagogical chain-of-thought approach.

**Evaluation Approach:**
- For each section, first understand what the section is trying to accomplish
- Check if the reasoning in that section is logically sound
- Verify any calculations or factual claims
- Consider whether the approach taken is appropriate for the problem

**Output Format:**
- If you think all sections of the robot's answer are correct, output in the following format:  
    Conclusion: no error
- If you think any section contains an error, output in the following format:  
    Conclusion: yes
    Error Section Number: [section number]
    Explanation: [explanation for the error in this section]
    Error Section Number: [section number]
    Explanation: [explanation for the error in this section]
    … (repeat for each erroneous section)  

**Input:**
- Question: {question}
- Robot's Answer: {model_output}
- judge result: """

# Direct General prompt - custom logic critic
DIRECT_GENERAL_PROMPT = """I will provide you with a question and a robot's answer divided into sections. Your task is to evaluate the robot's reasoning as a logic critic, examining both individual sections and the overall logical flow.

**Evaluation Framework:**

1. **Section-Level Analysis**: Check each section for:
   - Factual or computational errors
   - Invalid assumptions or simplifications
   - Logical inconsistencies within the section

2. **Cross-Section Logic Analysis**: Track how reasoning flows between sections:
   - Does each conclusion logically follow from previous work?
   - Are generalizations properly justified?
   - Do later sections contradict or invalidate earlier assumptions?
   - Is the logical chain of reasoning sound throughout?

3. **Pattern Recognition**: Identify these specific reasoning errors:
   - **Unjustified Generalization**: Drawing broad conclusions from specific cases without proper justification
   - **Invalid Simplification**: Making assumptions that lose essential problem properties
   - **Symmetry Misuse**: Assuming equivalence between non-equivalent cases
   - **Excessive Brute Force**: Using exhaustive enumeration when a general approach is needed
   - **Logical Disconnection**: Conclusions that don't follow from stated premises

**Analysis Process:**
1. Read the entire solution first to understand the overall approach
2. Create a logical dependency map showing how each section builds on previous ones
3. Evaluate each section both independently AND in context of the full solution
4. Identify where the logical chain breaks down

**Output Format:**
- If all reasoning is sound:
    Conclusion: no error
    
- If errors exist:
    Conclusion: yes
    
    Error Type: [specific error pattern, e.g., "Unjustified Generalization"]
    Location: Section [X] → Section [Y]
    Explanation: [detailed explanation of the logical flaw]
    Impact: [how this error affects subsequent reasoning]
    
    [Repeat for each distinct error]

**Important Notes:**
- An error in reasoning flow is distinct from a computational mistake
- If a later section acknowledges and corrects an earlier error, still mark the original error
- Focus on logical structure, not stylistic preferences
- Consider whether brute force approaches are reasonable given the problem constraints

**Input:**
- Question: {question}
- Robot's Answer: {model_output}"""

# Prompt selection
PROMPTS = {
    "deltabench": DELTABENCH_PROMPT,
    "pedcot": PEDCOT_PROMPT,
    "direct_general": DIRECT_GENERAL_PROMPT
}

# Default prompt for backward compatibility
CRITIC_PROMPT = DELTABENCH_PROMPT

# PedCOT-specific configuration
PEDCOT_CONFIG = {
    'pedagogical_principles': True,
    'two_stage_process': True,
    'principle_weighting': {
        'remember': 0.3,
        'understand': 0.4,
        'apply': 0.3
    },
    'error_mapping_strategy': 'weighted_consensus',
    'domain_mapping': {
        'math': ['mathematics', 'mathematical', 'number theory', 'algebra', 'geometry', 'calculus'],
        'programming': ['programming', 'computer science', 'coding', 'algorithm'],
        'pcb': ['physics', 'chemistry', 'biology', 'science'],
        'general': []  # Fallback domain
    },
    'confidence_threshold': 0.5,  # Minimum confidence for error detection
    'max_sections_per_analysis': 10  # Limit for performance
}

# Critic configurations
CRITIC_CONFIGS = {
    'direct': {
        'type': 'DirectCritic',
        'prompt_type': 'deltabench',
        'strategy': 'section_by_section'
    },
    'pedcot': {
        'type': 'PedCoTCritic',
        **PEDCOT_CONFIG
    }
}