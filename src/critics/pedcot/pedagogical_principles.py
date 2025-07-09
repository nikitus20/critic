"""Pedagogical principles engine for PedCOT critic."""

from typing import Dict, List, Optional
from openai import OpenAI
import httpx
from ... import config


class PedagogicalPrinciples:
    """Manages pedagogical principles based on Bloom's taxonomy."""
    
    # Bloom's taxonomy principles mapped to domains
    PRINCIPLES = {
        'remember': {
            'math': "Mathematical concepts, definitions, formulas, theorems, and properties",
            'programming': "Programming syntax, language constructs, built-in functions, and data structures", 
            'pcb': "Physics laws, chemistry principles, biology concepts, and scientific facts",
            'general': "Relevant facts, definitions, and information needed for the problem"
        },
        'understand': {
            'math': "Problem-solving approaches, strategies, mathematical reasoning patterns, and solution methods",
            'programming': "Algorithm design, logic flow, problem decomposition, and computational thinking",
            'pcb': "Scientific reasoning, methodology, cause-and-effect relationships, and process understanding", 
            'general': "Problem analysis, interpretation, relationships between concepts, and underlying principles"
        },
        'apply': {
            'math': "Mathematical calculations, formula application, step-by-step execution, and computational procedures",
            'programming': "Code implementation, algorithm execution, syntax application, and debugging",
            'pcb': "Application of scientific principles, experimental procedures, and practical implementations",
            'general': "Solution implementation, procedure execution, and verification of results"
        }
    }
    
    def __init__(self, model: str = config.DEFAULT_MODEL):
        self.model = model
        # Create OpenAI client with custom httpx client to avoid proxy issues
        http_client = httpx.Client()
        self.client = OpenAI(api_key=config.OPENAI_API_KEY, http_client=http_client)
    
    def extract_principles_for_domain(self, domain: str) -> Dict[str, str]:
        """Extract appropriate principles for given domain."""
        domain = domain.lower()
        
        # Map domain variations to standard domains
        domain_mapping = {
            'mathematics': 'math',
            'mathematical': 'math',
            'number theory': 'math',
            'algebra': 'math',
            'geometry': 'math',
            'calculus': 'math',
            'programming': 'programming',
            'computer science': 'programming',
            'coding': 'programming',
            'physics': 'pcb',
            'chemistry': 'pcb',
            'biology': 'pcb',
            'science': 'pcb'
        }
        
        # Get the standardized domain
        standardized_domain = domain_mapping.get(domain, 'general')
        
        return {
            'remember': self.PRINCIPLES['remember'][standardized_domain],
            'understand': self.PRINCIPLES['understand'][standardized_domain],
            'apply': self.PRINCIPLES['apply'][standardized_domain]
        }
    
    def get_concepts_for_question(self, question: str, domain: str) -> List[str]:
        """Extract relevant concepts from question using LLM."""
        prompt = f"""Given the following question from the {domain} domain, identify the key concepts that need to be REMEMBERED (known facts, definitions, formulas, etc.) to solve this problem.

Question: {question}

List the key concepts that a student would need to recall from memory to approach this problem. Focus on factual knowledge, definitions, formulas, theorems, or established facts.

Format your response as a simple list of concepts, one per line."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            output = response.choices[0].message.content
            if output:
                # Extract concepts from the response
                concepts = []
                for line in output.strip().split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Remove bullet points and numbering
                        line = line.lstrip('•-*123456789. ')
                        if line:
                            concepts.append(line)
                return concepts
            
        except Exception as e:
            print(f"Error extracting concepts: {e}")
        
        return []
    
    def get_approach_for_section(self, section: str, domain: str) -> str:
        """Identify problem-solving approach in section."""
        prompt = f"""Given the following reasoning section from a {domain} problem, identify the problem-solving approach being used.

Section: {section}

What approach or strategy is being employed in this section? Focus on the METHOD or STRATEGY being used to solve the problem (e.g., "direct calculation", "substitution method", "proof by contradiction", "algorithmic approach", etc.).

Provide a brief description of the approach in 1-2 sentences."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            output = response.choices[0].message.content
            if output:
                return output.strip()
            
        except Exception as e:
            print(f"Error identifying approach: {e}")
        
        return "Unknown approach"
    
    def get_calculations_for_section(self, section: str, domain: str) -> List[str]:
        """Extract calculations or applications from section."""
        prompt = f"""Given the following reasoning section from a {domain} problem, identify the specific calculations, applications, or implementations being performed.

Section: {section}

List the concrete calculations, formula applications, or step-by-step procedures being executed in this section. Focus on the APPLIED actions (calculations, substitutions, implementations, etc.).

Format your response as a simple list, one per line."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400
            )
            
            output = response.choices[0].message.content
            if output:
                # Extract calculations from the response
                calculations = []
                for line in output.strip().split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Remove bullet points and numbering
                        line = line.lstrip('•-*123456789. ')
                        if line:
                            calculations.append(line)
                return calculations
            
        except Exception as e:
            print(f"Error extracting calculations: {e}")
        
        return []
    
    def extract_domain_from_question(self, question: str) -> str:
        """Determine domain from question content."""
        question_lower = question.lower()
        
        # Math indicators
        math_keywords = ['equation', 'formula', 'calculate', 'solve', 'prove', 'theorem', 'number', 'integer', 'fraction', 'decimal', 'algebra', 'geometry', 'calculus', 'mathematical', 'function', 'variable', 'expression', 'graph', 'coordinate', 'triangle', 'circle', 'polynomial', 'derivative', 'integral', 'matrix', 'vector', 'probability', 'statistics']
        
        # Programming indicators  
        programming_keywords = ['code', 'program', 'algorithm', 'function', 'variable', 'array', 'loop', 'condition', 'class', 'object', 'method', 'syntax', 'compile', 'debug', 'implementation', 'programming', 'software', 'computer']
        
        # Science indicators
        science_keywords = ['physics', 'chemistry', 'biology', 'experiment', 'hypothesis', 'theory', 'law', 'reaction', 'molecule', 'atom', 'cell', 'organism', 'force', 'energy', 'mass', 'velocity', 'acceleration', 'temperature', 'pressure', 'scientific']
        
        # Count keyword matches
        math_count = sum(1 for keyword in math_keywords if keyword in question_lower)
        programming_count = sum(1 for keyword in programming_keywords if keyword in question_lower)
        science_count = sum(1 for keyword in science_keywords if keyword in question_lower)
        
        # Determine domain based on highest count
        if math_count >= programming_count and math_count >= science_count:
            return 'math'
        elif programming_count >= science_count:
            return 'programming'
        elif science_count > 0:
            return 'pcb'
        else:
            return 'general'