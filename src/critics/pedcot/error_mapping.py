"""Error taxonomy mapping for PedCOT to DeltaBench compatibility."""

from typing import Dict, List, Optional


class PedCoTErrorMapper:
    """Maps pedagogical principle errors to DeltaBench error taxonomy."""
    
    # Mapping from principle errors to DeltaBench categories
    PRINCIPLE_TO_DELTABENCH = {
        'remember': {
            'math': ['Knowledge Error', 'Understanding Error'],
            'programming': ['Knowledge Error', 'Programming Error'],
            'pcb': ['Knowledge Error', 'Understanding Error'],
            'general': ['Knowledge Error', 'Understanding Error']
        },
        'understand': {
            'math': ['Understanding Error', 'Logical Error'], 
            'programming': ['Understanding Error', 'Logical Error'],
            'pcb': ['Understanding Error', 'Logical Error'],
            'general': ['Understanding Error', 'Logical Error']
        },
        'apply': {
            'math': ['Calculation Error', 'Logical Error'],
            'programming': ['Programming Error', 'Formal Error'],
            'pcb': ['Calculation Error', 'Formal Error'],
            'general': ['Logical Error', 'Completeness Error']
        }
    }
    
    # DeltaBench error taxonomy (inferred from the dataset)
    DELTABENCH_ERROR_TYPES = [
        'Knowledge Error',
        'Understanding Error',
        'Logical Error',
        'Calculation Error',
        'Programming Error',
        'Formal Error',
        'Completeness Error',
        'Conceptual Error',
        'Procedural Error',
        'Reasoning Error'
    ]
    
    # Confidence weights for different principle combinations
    PRINCIPLE_WEIGHTS = {
        'remember': 0.3,
        'understand': 0.4,
        'apply': 0.3
    }
    
    def __init__(self):
        pass
    
    def map_principle_errors_to_deltabench(self, principle_mistakes: List[str], 
                                         domain: str, 
                                         confidence_scores: Optional[Dict[str, float]] = None) -> List[str]:
        """
        Map pedagogical principle errors to DeltaBench error taxonomy.
        
        Args:
            principle_mistakes: List of principle levels with errors ('remember', 'understand', 'apply')
            domain: Domain of the problem (math, programming, pcb, general)
            confidence_scores: Optional confidence scores for each principle
            
        Returns:
            List of DeltaBench error types
        """
        if not principle_mistakes:
            return []
        
        # Normalize domain
        domain = self._normalize_domain(domain)
        
        # Collect all possible error types
        error_types = set()
        
        for principle in principle_mistakes:
            if principle in self.PRINCIPLE_TO_DELTABENCH:
                domain_errors = self.PRINCIPLE_TO_DELTABENCH[principle].get(domain, [])
                error_types.update(domain_errors)
        
        # Sort by priority based on principle importance
        error_list = list(error_types)
        error_list.sort(key=lambda x: self._get_error_priority(x, principle_mistakes))
        
        return error_list
    
    def calculate_confidence(self, principle_analysis: Dict[str, Dict], 
                           principle_mistakes: List[str]) -> float:
        """
        Calculate confidence score based on principle agreement.
        
        Args:
            principle_analysis: Dictionary with remember/understand/apply analysis
            principle_mistakes: List of principle levels with errors
            
        Returns:
            Confidence score between 0 and 1
        """
        total_weight = 0
        error_weight = 0
        
        for principle, weight in self.PRINCIPLE_WEIGHTS.items():
            analysis_key = f"{principle}_analysis"
            if analysis_key in principle_analysis:
                total_weight += weight
                if principle in principle_mistakes:
                    error_weight += weight
        
        if total_weight == 0:
            return 0.5  # Default confidence
        
        # Higher confidence when fewer principles have errors
        confidence = 1.0 - (error_weight / total_weight)
        
        # Adjust confidence based on number of principles analyzed
        num_principles = len([p for p in ['remember', 'understand', 'apply'] 
                            if f"{p}_analysis" in principle_analysis])
        
        if num_principles < 3:
            confidence *= 0.8  # Lower confidence if not all principles analyzed
        
        return max(0.1, min(1.0, confidence))
    
    def get_error_explanation(self, principle_mistakes: List[str], 
                            principle_analysis: Dict[str, Dict], 
                            domain: str) -> str:
        """
        Generate error explanation based on principle analysis.
        
        Args:
            principle_mistakes: List of principle levels with errors
            principle_analysis: Analysis results for each principle
            domain: Domain of the problem
            
        Returns:
            Comprehensive error explanation
        """
        if not principle_mistakes:
            return "No errors detected in pedagogical principle analysis."
        
        explanations = []
        
        for principle in principle_mistakes:
            analysis_key = f"{principle}_analysis"
            if analysis_key in principle_analysis:
                analysis = principle_analysis[analysis_key]
                explanation = analysis.get('explanation', '')
                
                if explanation:
                    principle_name = principle.capitalize()
                    explanations.append(f"{principle_name}: {explanation}")
        
        if explanations:
            return "Pedagogical analysis reveals errors in: " + "; ".join(explanations)
        else:
            return f"Errors detected in {', '.join(principle_mistakes)} principles but no detailed explanations available."
    
    def _normalize_domain(self, domain: str) -> str:
        """Normalize domain to standard categories."""
        domain = domain.lower()
        
        # Map variations to standard domains
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
        
        return domain_mapping.get(domain, 'general')
    
    def _get_error_priority(self, error_type: str, principle_mistakes: List[str]) -> int:
        """Get priority for error type based on principle mistakes."""
        # Priority order for error types
        priority_order = {
            'Knowledge Error': 1,
            'Understanding Error': 2,
            'Logical Error': 3,
            'Calculation Error': 4,
            'Programming Error': 5,
            'Formal Error': 6,
            'Completeness Error': 7,
            'Conceptual Error': 8,
            'Procedural Error': 9,
            'Reasoning Error': 10
        }
        
        base_priority = priority_order.get(error_type, 999)
        
        # Adjust priority based on which principles have errors
        if 'remember' in principle_mistakes and error_type in ['Knowledge Error', 'Understanding Error']:
            base_priority -= 10
        elif 'understand' in principle_mistakes and error_type in ['Understanding Error', 'Logical Error']:
            base_priority -= 5
        elif 'apply' in principle_mistakes and error_type in ['Calculation Error', 'Programming Error']:
            base_priority -= 3
        
        return base_priority
    
    def get_error_type_description(self, error_type: str) -> str:
        """Get description for a specific error type."""
        descriptions = {
            'Knowledge Error': 'Error in recalling or using fundamental knowledge, concepts, or facts',
            'Understanding Error': 'Error in comprehending or interpreting the problem or concepts',
            'Logical Error': 'Error in logical reasoning or inference',
            'Calculation Error': 'Error in mathematical computations or numerical operations',
            'Programming Error': 'Error in code implementation or programming logic',
            'Formal Error': 'Error in formal procedures or structured approaches',
            'Completeness Error': 'Error due to incomplete analysis or missing steps',
            'Conceptual Error': 'Error in understanding fundamental concepts',
            'Procedural Error': 'Error in following correct procedures',
            'Reasoning Error': 'General error in reasoning process'
        }
        
        return descriptions.get(error_type, 'Unknown error type')