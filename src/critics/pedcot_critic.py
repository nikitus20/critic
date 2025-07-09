"""PedCOT (Pedagogical Chain-of-Thought) critic implementation."""

import re
from typing import Dict, List, Optional, Tuple

from .. import config
from ..utils import CriticResult
from .base_critic import BaseCritic
from .pedcot.pedagogical_principles import PedagogicalPrinciples
from .pedcot.tip_processor import TIPProcessor
from .pedcot.error_mapping import PedCoTErrorMapper


class PedCoTCritic(BaseCritic):
    """PedCOT critic using two-stage interaction process with pedagogical principles."""
    
    def __init__(self, model: str = config.DEFAULT_MODEL, config_dict: Optional[Dict] = None):
        super().__init__(model, config_dict)
        self.principles = PedagogicalPrinciples(model)
        self.tip_processor = TIPProcessor(model, self.principles)
        self.error_mapper = PedCoTErrorMapper()
        
        # Configuration
        self.principle_weighting = self.config.get('principle_weighting', {
            'remember': 0.3,
            'understand': 0.4,
            'apply': 0.3
        })
        self.error_mapping_strategy = self.config.get('error_mapping_strategy', 'weighted_consensus')
    
    def evaluate_reasoning(self, question: str, model_output: str) -> Tuple[str, Dict]:
        """
        Evaluate reasoning using PedCOT two-stage process.
        
        Args:
            question: The original question
            model_output: The model's reasoning output to evaluate
            
        Returns:
            Tuple of (critic_output, token_info)
        """
        try:
            # Parse sections from the model output
            sections = self._parse_sections(model_output)
            
            if not sections:
                return "No sections found to analyze", {'total_tokens': 0, 'stage': 'parsing_error'}
            
            # Extract domain from question
            domain = self.principles.extract_domain_from_question(question)
            
            # Analyze each section using two-stage process
            section_analyses = []
            total_tokens = 0
            
            for i, (section_num, section_content) in enumerate(sections):
                # Get context (previous sections)
                context = [content for _, content in sections[:i]]
                
                # Stage 1: Regenerate expected reasoning
                stage1_output = self.tip_processor.stage1_regenerate(
                    question, section_content, context, domain
                )
                total_tokens += stage1_output.get('token_info', {}).get('total_tokens', 0)
                
                # Stage 2: Compare and analyze
                stage2_output = self.tip_processor.stage2_compare(
                    question, section_content, context, stage1_output, domain
                )
                total_tokens += stage2_output.get('token_info', {}).get('total_tokens', 0)
                
                # Store analysis for this section
                section_analyses.append({
                    'section_num': section_num,
                    'stage1': stage1_output,
                    'stage2': stage2_output,
                    'has_error': stage2_output.get('has_error', False)
                })
            
            # Generate final critic output
            critic_output = self._generate_final_output(section_analyses, domain)
            
            token_info = {
                'total_tokens': total_tokens,
                'prompt_tokens': 0,  # Aggregated across stages
                'completion_tokens': 0,  # Aggregated across stages
                'prompt_type': 'pedcot',
                'num_stages': len(section_analyses) * 2,  # Two stages per section
                'domain': domain
            }
            
            return critic_output, token_info
            
        except Exception as e:
            print(f"Error in PedCOT evaluation: {e}")
            return f"Error in PedCOT evaluation: {str(e)}", {'total_tokens': 0, 'error': str(e)}
    
    def parse_output(self, critic_output: str, true_error_sections: List[int], max_valid_section: Optional[int] = None) -> CriticResult:
        """
        Parse PedCOT critic output into structured result.
        
        Args:
            critic_output: Raw output from the PedCOT critic
            true_error_sections: Ground truth error sections
            max_valid_section: Maximum valid section number (total section count)
            
        Returns:
            CriticResult object with predictions and metrics
        """
        try:
            # Extract conclusion
            result = critic_output.split("Error Section Number:")[0].split("Conclusion:")[-1].strip()
            has_errors = "yes" in result.lower()
            
            predicted_errors = []
            explanations = []
            
            if has_errors:
                # Parse error sections
                sections = critic_output.split("Error Section Number:")[1:]
                for section in sections:
                    # Extract error number
                    number_match = re.search(r'\d+', section.split("Explanation:")[0])
                    if number_match:
                        error_num = int(number_match.group())
                        predicted_errors.append(error_num)
                    
                    # Extract explanation
                    if "Explanation:" in section:
                        explanation = section.split("Explanation:")[-1].strip()
                        explanations.append(explanation)
            
            # Filter predictions to valid range based on actual section count or ground truth range
            if max_valid_section is not None:
                # Use actual section count if provided
                predicted_errors = [x for x in predicted_errors if 1 <= x <= max_valid_section]
            elif true_error_sections:
                # Fallback: use ground truth range (but this may not capture all valid sections)
                max_section = max(true_error_sections + [1])  # At least section 1 should exist
                predicted_errors = [x for x in predicted_errors if 1 <= x <= max_section]
            else:
                # No filtering if we don't know the valid range
                predicted_errors = [x for x in predicted_errors if x >= 1]
            
            # Calculate metrics
            tp = len(set(predicted_errors) & set(true_error_sections))
            fp = len(set(predicted_errors) - set(true_error_sections))
            fn = len(set(true_error_sections) - set(predicted_errors))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return CriticResult(
                predicted_error_sections=predicted_errors,
                explanations=explanations,
                precision=precision,
                recall=recall,
                f1_score=f1,
                raw_output=critic_output
            )
            
        except Exception as e:
            print(f"Error parsing PedCOT output: {e}")
            return None
    
    def analyze_section(self, question: str, section: str, context: List[str]) -> Dict:
        """
        Analyze a single section using PedCOT two-stage process.
        
        Args:
            question: The original question
            section: The section content to analyze
            context: Previous sections for context
            
        Returns:
            Dictionary with PedCOT analysis results
        """
        domain = self.principles.extract_domain_from_question(question)
        
        # Stage 1: Regenerate expected reasoning
        stage1_output = self.tip_processor.stage1_regenerate(question, section, context, domain)
        
        # Stage 2: Compare and analyze
        stage2_output = self.tip_processor.stage2_compare(question, section, context, stage1_output, domain)
        
        return {
            'domain': domain,
            'stage1': stage1_output,
            'stage2': stage2_output,
            'has_error': stage2_output.get('has_error', False),
            'principle_mistakes': stage2_output.get('principle_mistakes', []),
            'confidence': stage2_output.get('confidence', 0.5)
        }
    
    def _parse_sections(self, model_output: str) -> List[Tuple[int, str]]:
        """Parse sections from model output."""
        sections = []
        
        # Match patterns like "section1:", "section 2:", etc.
        pattern = r'section\s*(\d+)\s*:\s*'
        parts = re.split(pattern, model_output, flags=re.IGNORECASE)
        
        # parts[0] is text before first section, then alternating section numbers and content
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                section_num = int(parts[i])
                content = parts[i + 1].strip()
                sections.append((section_num, content))
        
        return sections
    
    def _generate_final_output(self, section_analyses: List[Dict], domain: str) -> str:
        """Generate final critic output in DeltaBench-compatible format."""
        # Find sections with errors
        error_sections = []
        
        for analysis in section_analyses:
            if analysis['has_error']:
                section_num = analysis['section_num']
                stage2 = analysis['stage2']
                
                # Map principle errors to DeltaBench taxonomy
                principle_mistakes = stage2.get('principle_mistakes', [])
                error_types = self.error_mapper.map_principle_errors_to_deltabench(
                    principle_mistakes, domain
                )
                
                # Generate explanation
                explanation = self.error_mapper.get_error_explanation(
                    principle_mistakes, 
                    {k: v for k, v in stage2.items() if k.endswith('_analysis')},
                    domain
                )
                
                error_sections.append({
                    'section_num': section_num,
                    'explanation': explanation,
                    'error_types': error_types,
                    'confidence': stage2.get('confidence', 0.5)
                })
        
        # Generate output in DeltaBench format
        if not error_sections:
            return "Conclusion: no error"
        else:
            output_lines = ["Conclusion: yes"]
            
            for error in error_sections:
                output_lines.append(f"Error Section Number: {error['section_num']}")
                output_lines.append(f"Explanation: {error['explanation']}")
            
            return "\n".join(output_lines)
    
    def _extract_domain(self, question: str) -> str:
        """Extract domain from question (delegated to principles engine)."""
        return self.principles.extract_domain_from_question(question)