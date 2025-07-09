"""Two-Stage Interaction Process (TIP) processor for PedCOT."""

import json
import re
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
import httpx

from ... import config
from ...prompts.pedcot_prompts import PEDCOT_TEMPLATES
from .pedagogical_principles import PedagogicalPrinciples


class TIPProcessor:
    """Implements the Two-Stage Interaction Process for PedCOT."""
    
    def __init__(self, model: str = config.DEFAULT_MODEL, principles: Optional[PedagogicalPrinciples] = None):
        self.model = model
        # Create OpenAI client with custom httpx client to avoid proxy issues
        http_client = httpx.Client()
        self.client = OpenAI(api_key=config.OPENAI_API_KEY, http_client=http_client)
        self.principles = principles or PedagogicalPrinciples(model)
    
    def stage1_regenerate(self, question: str, section: str, context: List[str], domain: str) -> Dict:
        """
        Stage 1: Regenerate expected reasoning following pedagogical principles.
        
        Args:
            question: The original question
            section: The current section to analyze
            context: Previous sections for context
            domain: Domain of the problem (math, programming, etc.)
            
        Returns:
            Dictionary with regenerated reasoning and pedagogical components
        """
        try:
            # Get domain-specific principles
            principles = self.principles.extract_principles_for_domain(domain)
            
            # Extract pedagogical components
            concepts = self.principles.get_concepts_for_question(question, domain)
            approach = self.principles.get_approach_for_section(section, domain)
            calculations = self.principles.get_calculations_for_section(section, domain)
            
            # Format context
            context_str = "\n".join([f"Section {i+1}: {ctx}" for i, ctx in enumerate(context)])
            
            # Select appropriate template
            template_key = 'math' if domain == 'math' else 'general'
            template = PEDCOT_TEMPLATES[template_key]['stage1']
            
            # Create prompt
            if domain == 'math':
                prompt = template.format(
                    question=question,
                    context=context_str,
                    concepts="; ".join(concepts) if concepts else "No specific concepts identified",
                    approach=approach,
                    calculations="; ".join(calculations) if calculations else "No specific calculations identified"
                )
            else:
                prompt = template.format(
                    question=question,
                    context=context_str,
                    domain=domain,
                    remember_principle=principles['remember'],
                    understand_principle=principles['understand'],
                    apply_principle=principles['apply']
                )
            
            # Get LLM response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            
            output = response.choices[0].message.content
            
            # Parse the regenerated reasoning
            expected_reasoning = self._extract_expected_reasoning(output)
            
            return {
                'regenerated_reasoning': expected_reasoning,
                'concepts': concepts,
                'approach': approach,
                'calculations': calculations,
                'principles': principles,
                'raw_output': output,
                'token_info': {
                    'total_tokens': response.usage.total_tokens,
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'stage': 'stage1'
                }
            }
            
        except Exception as e:
            print(f"Error in Stage 1 regeneration: {e}")
            return {
                'regenerated_reasoning': "Error in regeneration",
                'concepts': [],
                'approach': "Unknown approach",
                'calculations': [],
                'principles': {},
                'raw_output': f"Error: {str(e)}",
                'token_info': {'total_tokens': 0, 'stage': 'stage1'}
            }
    
    def stage2_compare(self, question: str, section: str, context: List[str], 
                      stage1_output: Dict, domain: str) -> Dict:
        """
        Stage 2: Compare actual vs expected reasoning, identify mistakes.
        
        Args:
            question: The original question
            section: The current section to analyze
            context: Previous sections for context
            stage1_output: Output from Stage 1 regeneration
            domain: Domain of the problem
            
        Returns:
            Dictionary with comparison analysis and error detection
        """
        try:
            # Format context
            context_str = "\n".join([f"Section {i+1}: {ctx}" for i, ctx in enumerate(context)])
            
            # Extract Stage 1 components
            expected_reasoning = stage1_output.get('regenerated_reasoning', '')
            expected_concepts = stage1_output.get('concepts', [])
            expected_approach = stage1_output.get('approach', '')
            expected_calculations = stage1_output.get('calculations', [])
            
            # Select appropriate template
            template_key = 'math' if domain == 'math' else 'general'
            template = PEDCOT_TEMPLATES[template_key]['stage2']
            
            # Create prompt
            if domain == 'math':
                prompt = template.format(
                    question=question,
                    context=context_str,
                    section=section,
                    expected_concepts="; ".join(expected_concepts) if expected_concepts else "No specific concepts",
                    expected_approach=expected_approach,
                    expected_calculations="; ".join(expected_calculations) if expected_calculations else "No specific calculations",
                    expected_reasoning=expected_reasoning
                )
            else:
                prompt = template.format(
                    question=question,
                    context=context_str,
                    section=section,
                    domain=domain,
                    expected_concepts="; ".join(expected_concepts) if expected_concepts else "No specific concepts",
                    expected_approach=expected_approach,
                    expected_operations="; ".join(expected_calculations) if expected_calculations else "No specific operations",
                    expected_reasoning=expected_reasoning
                )
            
            # Get LLM response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500
            )
            
            output = response.choices[0].message.content
            
            # Parse JSON response
            analysis = self._parse_stage2_output(output)
            
            # Add token info
            analysis['token_info'] = {
                'total_tokens': response.usage.total_tokens,
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'stage': 'stage2'
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error in Stage 2 comparison: {e}")
            return {
                'has_error': False,
                'error_types': [],
                'error_location': '',
                'explanation': f"Error in analysis: {str(e)}",
                'confidence': 0.0,
                'principle_mistakes': [],
                'remember_analysis': {'correct': True, 'explanation': 'Error in analysis'},
                'understand_analysis': {'correct': True, 'explanation': 'Error in analysis'},
                'apply_analysis': {'correct': True, 'explanation': 'Error in analysis'},
                'token_info': {'total_tokens': 0, 'stage': 'stage2'}
            }
    
    def parse_section_into_steps(self, section: str) -> List[str]:
        """Parse section content into individual reasoning steps."""
        # Split by common step indicators
        step_patterns = [
            r'\n\d+\.',  # "1.", "2.", etc.
            r'\nStep \d+',  # "Step 1", "Step 2", etc.
            r'\n•',  # Bullet points
            r'\n-',  # Dashes
            r'\n\([a-zA-Z]\)',  # "(a)", "(b)", etc.
        ]
        
        steps = [section]  # Start with the full section
        
        for pattern in step_patterns:
            new_steps = []
            for step in steps:
                parts = re.split(pattern, step)
                if len(parts) > 1:
                    new_steps.extend([part.strip() for part in parts if part.strip()])
                else:
                    new_steps.append(step)
            steps = new_steps
        
        # Filter out very short steps
        steps = [step for step in steps if len(step.strip()) > 10]
        
        return steps if steps else [section]
    
    def _extract_expected_reasoning(self, output: str) -> str:
        """Extract the expected reasoning from Stage 1 output."""
        # Look for "Expected Next Step:" section
        if "Expected Next Step:" in output:
            parts = output.split("Expected Next Step:")
            if len(parts) > 1:
                return parts[1].strip()
        
        # Look for "Expected reasoning:" section
        if "Expected reasoning:" in output:
            parts = output.split("Expected reasoning:")
            if len(parts) > 1:
                return parts[1].strip()
        
        # If no specific section found, return the whole output
        return output.strip()
    
    def _parse_stage2_output(self, output: str) -> Dict:
        """Parse the JSON output from Stage 2."""
        try:
            # Try to extract JSON from the output
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                analysis = json.loads(json_str)
                
                # Ensure all required fields are present
                required_fields = {
                    'has_error': False,
                    'error_types': [],
                    'error_location': '',
                    'explanation': '',
                    'confidence': 0.5,
                    'principle_mistakes': [],
                    'remember_analysis': {'correct': True, 'explanation': ''},
                    'understand_analysis': {'correct': True, 'explanation': ''},
                    'apply_analysis': {'correct': True, 'explanation': ''}
                }
                
                for field, default_value in required_fields.items():
                    if field not in analysis:
                        analysis[field] = default_value
                
                return analysis
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
        except Exception as e:
            print(f"Error parsing Stage 2 output: {e}")
        
        # Fallback parsing if JSON parsing fails
        return self._fallback_parse_stage2(output)
    
    def _fallback_parse_stage2(self, output: str) -> Dict:
        """Fallback parsing for Stage 2 when JSON parsing fails."""
        # Simple heuristic parsing
        has_error = any(keyword in output.lower() for keyword in ['error', 'incorrect', 'wrong', 'mistake'])
        
        return {
            'has_error': has_error,
            'error_types': ['Understanding Error'] if has_error else [],
            'error_location': 'Unable to parse specific location',
            'explanation': output[:500] + '...' if len(output) > 500 else output,
            'confidence': 0.3,  # Low confidence for fallback parsing
            'principle_mistakes': ['understand'] if has_error else [],
            'remember_analysis': {'correct': not has_error, 'explanation': 'Fallback analysis'},
            'understand_analysis': {'correct': not has_error, 'explanation': 'Fallback analysis'},
            'apply_analysis': {'correct': not has_error, 'explanation': 'Fallback analysis'}
        }