# PedCOT Implementation Todo List

## Phase 1: Core Architecture Setup

### 1.1 Base Critic Interface (`src/critics/base_critic.py`)
- [ ] Create abstract `BaseCritic` class with standard interface
- [ ] Define common methods: `analyze_section()`, `analyze_full_cot()`, `get_predictions()`
- [ ] Standardize return format for both DeltaBench and PedCOT critics
- [ ] Add configuration management for different critic types

```python
class BaseCritic(ABC):
    def __init__(self, model: str, config: Dict):
        self.model = model
        self.config = config
    
    @abstractmethod
    def analyze_section(self, question: str, section: str, context: List[str]) -> Dict:
        """Returns: {has_error, error_types, error_location, explanation, confidence}"""
        pass
    
    @abstractmethod
    def analyze_full_cot(self, question: str, sections: List[str]) -> Dict:
        """Analyze complete reasoning chain"""
        pass
```

### 1.2 Update DirectCritic (`src/critics/direct_critic.py`)
- [ ] Refactor `DirectCritic` to inherit from `BaseCritic`
- [ ] Ensure compatibility with existing DeltaBench evaluation pipeline
- [ ] Test that current functionality still works after refactoring

## Phase 2: PedCOT-Specific Components

### 2.1 Pedagogical Principles Manager (`src/critics/pedcot/pedagogical_principles.py`)
- [ ] Implement Bloom's taxonomy mapping for all domains
- [ ] Create domain-specific principle extraction functions
- [ ] Handle Math, Programming, PCB, and General Reasoning domains

```python
class PedagogicalPrinciples:
    PRINCIPLES = {
        'remember': {
            'math': "Mathematical concepts and definitions",
            'programming': "Programming syntax and language constructs", 
            'pcb': "Physics, chemistry, biology concepts and laws",
            'general': "Relevant facts and information"
        },
        'understand': {
            'math': "Problem-solving approaches and strategies",
            'programming': "Algorithm design and logic flow",
            'pcb': "Scientific reasoning and methodology", 
            'general': "Problem analysis and interpretation"
        },
        'apply': {
            'math': "Calculation and mathematical execution",
            'programming': "Code implementation and execution",
            'pcb': "Application of scientific principles",
            'general': "Solution implementation and verification"
        }
    }
    
    def extract_principles_for_domain(self, domain: str) -> Dict[str, str]:
        """Extract appropriate principles for given domain"""
        
    def get_concepts_for_question(self, question: str, domain: str) -> List[str]:
        """Extract relevant concepts from question using LLM"""
        
    def get_approach_for_section(self, section: str, domain: str) -> str:
        """Identify problem-solving approach in section"""
```

### 2.2 Two-Stage Interaction Process (`src/critics/pedcot/tip_processor.py`)
- [ ] Implement Stage 1: Regenerate expected reasoning
- [ ] Implement Stage 2: Extract-Compare analysis
- [ ] Handle section-to-step parsing within sections
- [ ] Manage context flow between stages

```python
class TIPProcessor:
    def __init__(self, model: str, principles: PedagogicalPrinciples):
        self.model = model
        self.principles = principles
    
    def stage1_regenerate(self, question: str, section: str, domain: str) -> Dict:
        """
        Stage 1: Regenerate expected reasoning following pedagogical principles
        Returns: {regenerated_reasoning, concepts, approach, calculations}
        """
        
    def stage2_compare(self, question: str, section: str, expected_reasoning: str, domain: str) -> Dict:
        """
        Stage 2: Compare actual vs expected reasoning, identify mistakes
        Returns: {has_error, error_types, error_location, explanation, confidence, principle_mistakes}
        """
        
    def parse_section_into_steps(self, section: str) -> List[str]:
        """Parse section content into individual reasoning steps"""
```

### 2.3 PedCOT Prompts (`src/prompts/pedcot_prompts.py`)
- [ ] Implement exact prompt templates from PedCOT paper (Table 1)
- [ ] Create domain-specific prompt variations
- [ ] Add structured JSON output formatting
- [ ] Include error taxonomy integration

```python
# Stage 1 Templates (from PedCOT Table 1)
STAGE1_REGENERATE_TEMPLATE = """
You are given a math problem and several initial steps.
Question: {question}
Initial steps: [Previous sections: {context}]
Execute the following instructions sequentially:
1. List the mathematical concepts that... in the next step.
2. List the key analyses present... Give a detailed comparison between the...
3. List the mathematical expressions that... in the next step.

Following pedagogical principles:
Remember: Recall relevant {concepts}
Understand: Identify the {approach}  
Apply: Execute the {calculations}

Generate the expected reasoning for this section following these principles.
"""

STAGE2_COMPARE_TEMPLATE = """
You are given a math problem and several initial steps.
Question: {question}
Initial steps: [Previous sections: {context}]
Current section: {section}

Your task is to identify the incorrectly reasoned sections from the given sections. Execute the following instructions sequentially:

1. First extract the mathematical concepts employed... Give a detailed comparison between the... [Stage-1 Output Context G^(1)]. Finally, output a label... to categorize the three correctness of the mathematical concepts employed...

2. First extract the key analyses present... Give a detailed comparison between the... [Stage-1 Output Context G^(2)]. Finally, output a label to categorize the correctness of the problem solving approach for the actual next step.

3. First extract the key mathematical expressions... Give a detailed comparison between the... [Stage-1 Output Context G^(3)]. Finally, output a label... to categorize the correctness of the calculations for the actual next step.

Expected reasoning from Stage 1: {expected_reasoning}

Respond in JSON format:
{
    "remember_analysis": {"correct": bool, "explanation": str},
    "understand_analysis": {"correct": bool, "explanation": str}, 
    "apply_analysis": {"correct": bool, "explanation": str},
    "has_error": bool,
    "error_types": [list from DeltaBench taxonomy],
    "error_location": str,
    "explanation": str,
    "confidence": float,
    "principle_mistakes": [list of principle levels with errors]
}
"""
```

### 2.4 Main PedCOT Critic (`src/critics/pedcot_critic.py`)
- [ ] Implement `PedCoTCritic` inheriting from `BaseCritic`
- [ ] Integrate TIP processor and pedagogical principles
- [ ] Handle section-level analysis using two-stage process
- [ ] Map principle-level errors to DeltaBench error taxonomy
- [ ] Implement confidence scoring based on principle agreement

```python
class PedCoTCritic(BaseCritic):
    def __init__(self, model: str = "gpt-4o", config: Dict = None):
        super().__init__(model, config)
        self.principles = PedagogicalPrinciples()
        self.tip_processor = TIPProcessor(model, self.principles)
        
    def analyze_section(self, question: str, section: str, context: List[str]) -> Dict:
        """
        Analyze section using PedCOT two-stage process
        1. Extract domain from question/context
        2. Run Stage 1: Regenerate expected reasoning
        3. Run Stage 2: Compare and identify errors
        4. Map principle errors to DeltaBench taxonomy
        """
        
    def analyze_full_cot(self, question: str, sections: List[str]) -> Dict:
        """Analyze complete reasoning using progressive section analysis"""
        
    def _extract_domain(self, question: str) -> str:
        """Determine domain (math/programming/pcb/general) from question"""
        
    def _map_principle_errors_to_deltabench(self, principle_mistakes: List[str], 
                                          domain: str) -> List[str]:
        """Map pedagogical principle errors to DeltaBench error taxonomy"""
```

## Phase 3: Integration & Configuration

### 3.1 Error Taxonomy Mapping (`src/critics/pedcot/error_mapping.py`)
- [ ] Create mapping between pedagogical principle errors and DeltaBench categories
- [ ] Handle domain-specific error type priorities
- [ ] Implement confidence weighting based on principle agreement

```python
class PedCoTErrorMapper:
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
```

### 3.2 Configuration Updates (`configs/experiment_config.yaml`)
- [ ] Add PedCOT-specific configuration section
- [ ] Include pedagogical principle settings
- [ ] Add two-stage process parameters

```yaml
critics:
  direct:
    type: "DirectCritic"
    strategy: "section_by_section"
    
  pedcot:
    type: "PedCoTCritic"
    pedagogical_principles: true
    two_stage_process: true
    principle_weighting:
      remember: 0.3
      understand: 0.4  
      apply: 0.3
    error_mapping_strategy: "weighted_consensus"
```

### 3.3 Factory Pattern (`src/critics/critic_factory.py`)
- [ ] Implement factory for creating different critic types
- [ ] Handle configuration loading and validation
- [ ] Support easy switching between DirectCritic and PedCoTCritic

```python
class CriticFactory:
    @staticmethod
    def create_critic(critic_type: str, model: str, config: Dict) -> BaseCritic:
        if critic_type == "direct":
            return DirectCritic(model, config)
        elif critic_type == "pedcot":
            return PedCoTCritic(model, config)
        else:
            raise ValueError(f"Unknown critic type: {critic_type}")
```

## Phase 4: Evaluation Integration

### 4.1 Update Evaluation Pipeline (`src/evaluation/evaluator.py`)
- [ ] Ensure `DeltaBenchEvaluator` works with both critic types
- [ ] Add PedCOT-specific logging and analysis
- [ ] Track principle-level performance metrics
- [ ] Compare two-stage vs single-stage performance

### 4.2 Additional Metrics (`src/evaluation/pedcot_metrics.py`)
- [ ] Implement principle-level accuracy tracking
- [ ] Add two-stage process performance analysis
- [ ] Create pedagogical principle confusion matrices
- [ ] Track Stage 1 vs Stage 2 contribution to final decisions

```python
class PedCoTMetrics:
    def principle_level_accuracy(self, predictions: List[Dict]) -> Dict:
        """Track accuracy for Remember/Understand/Apply principles"""
        
    def stage_contribution_analysis(self, predictions: List[Dict]) -> Dict:
        """Analyze how Stage 1 vs Stage 2 contributes to final predictions"""
        
    def pedagogical_error_distribution(self, predictions: List[Dict]) -> Dict:
        """Distribution of errors across pedagogical principles"""
```

## Phase 5: Experimental Setup

### 5.1 Update Experiment Scripts (`experiments/run_baseline.py`)
- [ ] Add `--critic pedcot` option to experiment runner
- [ ] Include comparative evaluation between DirectCritic and PedCoTCritic
- [ ] Add ablation studies (Stage 1 only, Stage 2 only, full TIP)

### 5.2 PedCOT-Specific Experiments (`experiments/pedcot_experiments.py`)
- [ ] Implement domain-specific performance analysis
- [ ] Create principle-level ablation studies
- [ ] Add pedagogical prompt variation experiments
- [ ] Compare against PedCOT paper baselines (Zero-shot CoT, Plan-and-Solve, SelfCheck)

### 5.3 Comparative Analysis (`experiments/comparative_analysis.py`)
- [ ] Direct vs PedCOT performance comparison
- [ ] Error type detection comparison across methods
- [ ] Computational cost analysis (single-stage vs two-stage)
- [ ] Domain transfer analysis

## Phase 6: Testing & Validation

### 6.1 Unit Tests (`tests/critics/`)
- [ ] Test `PedagogicalPrinciples` class functionality
- [ ] Test `TIPProcessor` stage implementations
- [ ] Test error mapping accuracy
- [ ] Test integration with existing evaluation pipeline

### 6.2 Integration Tests (`tests/integration/`)
- [ ] End-to-end PedCOT critic testing
- [ ] Compare outputs with expected PedCOT paper behavior
- [ ] Test on small DeltaBench subset for correctness
- [ ] Validate JSON output format consistency

### 6.3 Performance Validation (`experiments/validation/`)
- [ ] Reproduce PedCOT results on original datasets (if accessible)
- [ ] Validate pedagogical principle extraction
- [ ] Check two-stage process coherence
- [ ] Test domain adaptation effectiveness

## Phase 7: Documentation & Finalization

### 7.1 Documentation Updates
- [ ] Update README with PedCOT usage instructions
- [ ] Document pedagogical principle design decisions
- [ ] Add examples of PedCOT vs DirectCritic outputs
- [ ] Create troubleshooting guide for common issues

### 7.2 Code Quality
- [ ] Add comprehensive type hints throughout PedCOT components
- [ ] Implement proper error handling for API calls
- [ ] Add logging for debugging two-stage process
- [ ] Code review and refactoring for maintainability

## Success Criteria

### Functionality
- [ ] PedCOT critic produces valid DeltaBench-compatible outputs
- [ ] Two-stage interaction process works as described in paper
- [ ] Seamless integration with existing evaluation pipeline
- [ ] Comparable performance to paper claims on mathematical domains

### Performance Expectations
- [ ] PedCOT shows improvement over DirectCritic on DeltaBench Math domain
- [ ] Pedagogical principle analysis provides interpretable error insights
- [ ] Reasonable computational overhead (2x calls, manageable latency)
- [ ] Consistent performance across different DeltaBench domains

### Architecture Quality
- [ ] Clean separation between DirectCritic and PedCOT implementations
- [ ] Easy to extend for additional critic methods
- [ ] Maintainable codebase with good documentation
- [ ] Comprehensive test coverage for new components

## Implementation Priority

**Week 1 (High Priority):**
- Phase 1: Base architecture and critic interface
- Phase 2.1-2.2: Core PedCOT components (principles & TIP)
- Phase 2.3: Basic prompt templates

**Week 2 (Medium Priority):**  
- Phase 2.4: Main PedCoTCritic implementation
- Phase 3: Integration and configuration
- Phase 4.1: Evaluation pipeline updates

**Week 3 (Lower Priority):**
- Phase 4.2: Additional metrics
- Phase 5: Experimental setup
- Phase 6: Testing and validation

This implementation plan provides a pathway to integrate PedCOT methodology while maintaining compatibility with your existing DeltaBench-focused architecture and evaluation framework.