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

# Prompt selection
PROMPTS = {
    "deltabench": DELTABENCH_PROMPT,
    "pedcot": PEDCOT_PROMPT
}

# Default prompt for backward compatibility
CRITIC_PROMPT = DELTABENCH_PROMPT