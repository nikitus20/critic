"""Analysis utilities for DeltaBench exploration."""

from typing import Dict, List, Optional, Tuple
import textwrap


def display_example(example: Dict, dataset: 'DeltaBenchDataset', show_sections: bool = True) -> None:
    """Display a single example in a readable format."""
    print("=" * 80)
    print(f"Example ID: {example.get('id', 'N/A')}")
    print(f"Task Type: {example.get('task_l1', 'N/A')} / {example.get('task_l2', 'N/A')}")
    print(f"Origin: {example.get('origin', 'N/A')}")
    print()
    
    # Question
    print("QUESTION:")
    print("-" * 40)
    print(example.get('question', ''))
    print()
    
    # Answer
    print("FINAL ANSWER:")
    print("-" * 40)
    print(example.get('answer', 'N/A'))
    print(f"Correct: {example.get('final_correct', 'N/A')}")
    print()
    
    # Error information
    error_sections = example.get('reason_error_section_numbers', [])
    unuseful_sections = example.get('reason_unuseful_section_numbers', [])
    
    print("ERRORS:")
    print("-" * 40)
    print(f"Error sections: {error_sections}")
    print(f"Unuseful sections: {unuseful_sections}")
    print()
    
    # Sections
    if show_sections:
        sections_content = example.get('sections_content', '') or example.get('section_content', '')
        if sections_content:
            print("REASONING SECTIONS:")
            print("-" * 40)
            sections = dataset.parse_sections(sections_content)
            
            # Get human annotations (ground truth)
            sections_labeled_info = example.get('sections_labeled_info', [])
            annotations_dict = {info['section_number']: info for info in sections_labeled_info}
            
            for section_num, content in sections:
                is_error = section_num in error_sections
                is_unuseful = section_num in unuseful_sections
                
                status = ""
                if is_error:
                    status = " [ERROR]"
                elif is_unuseful:
                    status = " [UNUSEFUL]"
                
                print(f"\nSection {section_num}{status}:")
                
                # Add human annotations (marked ground truth)
                if section_num in annotations_dict:
                    annotation = annotations_dict[section_num]
                    if annotation.get('reasoning_correctness') == '1':  # Error marked by human
                        print(f"  [HUMAN ANNOTATION - GROUND TRUTH]: Error detected")
                        if annotation.get('explanation'):
                            print(f"    Explanation: {annotation['explanation']}")
                        if annotation.get('correction'):
                            print(f"    Correction: {annotation['correction']}")
                    elif annotation.get('reasoning_usefulness') == '0':  # Unuseful marked by human
                        print(f"  [HUMAN ANNOTATION - GROUND TRUTH]: Unuseful section")
                
                # Truncate long sections
                if len(content) > 500:
                    content = content[:500] + "..."
                print(textwrap.fill(content, width=80, initial_indent="  ", subsequent_indent="  "))
    
    print("=" * 80)


def display_critic_comparison(example: Dict, results: Dict[str, Dict], dataset: 'DeltaBenchDataset') -> None:
    """Display critic predictions for different prompt types."""
    print("=" * 80)
    print(f"Example ID: {example.get('id', 'N/A')}")
    print()
    
    # Show question
    print("QUESTION:")
    print(example.get('question', ''))
    print()
    
    # Ground truth
    error_sections = example.get('reason_error_section_numbers', [])
    unuseful_sections = example.get('reason_unuseful_section_numbers', [])
    all_errors = sorted(set(error_sections + unuseful_sections))
    
    print(f"GROUND TRUTH ERRORS: {all_errors}")
    print()
    
    # Compare predictions
    print("CRITIC PREDICTIONS:")
    print("-" * 40)
    
    for prompt_type, result in results.items():
        if result and 'result' in result and result['result']:
            critic_result = result['result']
            predicted = sorted(critic_result.predicted_error_sections)
            precision = critic_result.precision
            recall = critic_result.recall
            f1 = critic_result.f1_score
            
            print(f"\n{prompt_type.upper()}:")
            print(f"  Predicted: {predicted}")
            print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
            
            # Show explanations for errors
            if critic_result.explanations:
                print("  Explanations:")
                for i, (sec, exp) in enumerate(zip(critic_result.predicted_error_sections, 
                                                   critic_result.explanations)):
                    print(f"    Section {sec}: {exp[:100]}...")
        else:
            print(f"\n{prompt_type.upper()}: Failed to evaluate")
    
    print("=" * 80)


def summarize_results(results_df) -> None:
    """Print summary statistics of evaluation results."""
    if len(results_df) == 0:
        print("No results to summarize")
        return
    
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total examples evaluated: {len(results_df)}")
    print()
    
    # Overall metrics
    print("Overall Performance:")
    print(f"  Precision: {results_df['precision'].mean():.3f} ± {results_df['precision'].std():.3f}")
    print(f"  Recall:    {results_df['recall'].mean():.3f} ± {results_df['recall'].std():.3f}")
    print(f"  F1 Score:  {results_df['f1_score'].mean():.3f} ± {results_df['f1_score'].std():.3f}")
    print()
    
    # Error detection
    print("Error Detection:")
    print(f"  Total true errors:      {results_df['true_errors'].sum()}")
    print(f"  Total predicted errors: {results_df['predicted_errors'].sum()}")
    print(f"  Examples with perfect score: {len(results_df[results_df['f1_score'] == 1.0])}")
    print(f"  Examples with zero score:    {len(results_df[results_df['f1_score'] == 0.0])}")
    print()
    
    # Token usage
    print("Token Usage:")
    print(f"  Average per example: {results_df['tokens_used'].mean():.0f}")
    print(f"  Total tokens used:   {results_df['tokens_used'].sum()}")
    print("=" * 50)


def generate_dataset_insights(dataset: 'DeltaBenchDataset', 
                            evaluation_results: Optional[Dict] = None,
                            output_file: str = "dataset_insights.md") -> str:
    """Generate comprehensive dataset insights markdown report."""
    
    from collections import Counter
    import datetime
    
    # Get dataset statistics
    stats = dataset.get_statistics()
    
    # Categorize errors
    error_categories = {
        'early_errors': [],
        'late_errors': [],
        'middle_errors': [],
        'single_errors': [],
        'multiple_errors': [],
        'by_task_type': {}
    }
    
    error_counts = []
    section_errors = []
    
    for ex in dataset.data:
        error_sections = ex.get('reason_error_section_numbers', [])
        unuseful_sections = ex.get('reason_unuseful_section_numbers', [])
        all_errors = error_sections + unuseful_sections
        
        error_counts.append(len(all_errors))
        section_errors.extend(all_errors)
        
        if all_errors:
            total_sections = len(ex.get('reason_steps', []))
            
            # Categorize by position
            for err_sec in all_errors:
                if err_sec <= 3:
                    error_categories['early_errors'].append(ex)
                elif err_sec > total_sections - 3:
                    error_categories['late_errors'].append(ex)
                else:
                    error_categories['middle_errors'].append(ex)
            
            # Categorize by count
            if len(all_errors) == 1:
                error_categories['single_errors'].append(ex)
            else:
                error_categories['multiple_errors'].append(ex)
                
            # Categorize by task type
            task_type = ex.get('task_l1', 'Unknown')
            if task_type not in error_categories['by_task_type']:
                error_categories['by_task_type'][task_type] = []
            error_categories['by_task_type'][task_type].append(ex)
    
    # Generate markdown content
    content = f"""# DeltaBench Dataset Insights

Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Overview

- **Total Examples**: {stats['total_examples']:,}
- **Examples with Errors**: {stats['examples_with_errors']:,}
- **Error Rate**: {stats['error_rate']:.1%}
- **Average Sections per Example**: {sum(len(ex.get('reason_steps', [])) for ex in dataset.data) / len(dataset.data):.1f}

## Task Distribution

| Task Type | Count | Percentage |
|-----------|-------|------------|
"""
    
    # Add task distribution table
    for task, count in sorted(stats['task_l1_distribution'].items(), key=lambda x: x[1], reverse=True):
        percentage = count / stats['total_examples'] * 100
        content += f"| {task} | {count:,} | {percentage:.1f}% |\n"
    
    content += f"""
## Error Analysis

### Error Count Distribution

| Error Count | Examples | Percentage |
|-------------|----------|------------|
"""
    
    # Add error count distribution
    error_dist = Counter(error_counts)
    for n_errors, count in sorted(error_dist.items()):
        percentage = count / len(dataset.data) * 100
        content += f"| {n_errors} | {count:,} | {percentage:.1f}% |\n"
    
    content += f"""
### Error Position Analysis

- **Early Errors** (sections 1-3): {len(error_categories['early_errors'])} examples
- **Middle Errors**: {len(error_categories['middle_errors'])} examples  
- **Late Errors**: {len(error_categories['late_errors'])} examples

### Error Complexity

- **Single Error Examples**: {len(error_categories['single_errors'])} ({len(error_categories['single_errors'])/stats['examples_with_errors']*100:.1f}% of error examples)
- **Multiple Error Examples**: {len(error_categories['multiple_errors'])} ({len(error_categories['multiple_errors'])/stats['examples_with_errors']*100:.1f}% of error examples)

### Most Common Error Sections

"""
    
    # Add most common error sections
    section_dist = Counter(section_errors)
    for section, count in section_dist.most_common(10):
        content += f"- **Section {section}**: {count} errors\n"
    
    content += f"""
### Errors by Task Type

| Task Type | Error Examples | Error Rate |
|-----------|----------------|------------|
"""
    
    # Add errors by task type
    for task, examples in error_categories['by_task_type'].items():
        total_task_examples = stats['task_l1_distribution'].get(task, 0)
        error_rate = len(examples) / total_task_examples * 100 if total_task_examples > 0 else 0
        content += f"| {task} | {len(examples)} | {error_rate:.1f}% |\n"
    
    # Add evaluation results if provided
    if evaluation_results:
        content += f"""
## Evaluation Results

### Critic Performance Summary
"""
        for critic_name, results in evaluation_results.items():
            if hasattr(results, 'mean'):  # DataFrame
                content += f"""
#### {critic_name.title()} Critic

- **F1 Score**: {results['f1_score'].mean():.3f} ± {results['f1_score'].std():.3f}
- **Precision**: {results['precision'].mean():.3f} ± {results['precision'].std():.3f}  
- **Recall**: {results['recall'].mean():.3f} ± {results['recall'].std():.3f}
- **Examples Evaluated**: {len(results)}
- **Perfect Predictions**: {len(results[results['f1_score'] == 1.0])} ({len(results[results['f1_score'] == 1.0])/len(results)*100:.1f}%)
- **Failed Predictions**: {len(results[results['f1_score'] == 0.0])} ({len(results[results['f1_score'] == 0.0])/len(results)*100:.1f}%)
- **Average Token Usage**: {results['tokens_used'].mean():.0f}
"""

    content += f"""
## Key Insights

### Dataset Characteristics
1. **Diverse Task Coverage**: The dataset spans {len(stats['task_l1_distribution'])} different mathematical domains
2. **Balanced Error Distribution**: {stats['error_rate']:.1%} of examples contain reasoning errors
3. **Error Complexity**: {len(error_categories['multiple_errors'])/stats['examples_with_errors']*100:.1f}% of error examples have multiple mistakes

### Error Patterns
1. **Position Bias**: {'Early' if len(error_categories['early_errors']) > len(error_categories['late_errors']) else 'Late'} errors are more common
2. **Section Hotspots**: Sections {', '.join(map(str, [s for s, _ in section_dist.most_common(3)]))} contain the most errors
3. **Task-Specific Errors**: {max(error_categories['by_task_type'].items(), key=lambda x: len(x[1]))[0]} has the highest error count

### Recommendations
1. **Focus Areas**: Prioritize error detection in {max(error_categories['by_task_type'].items(), key=lambda x: len(x[1]))[0]} problems
2. **Model Training**: Include examples with multiple errors for robust critic training
3. **Evaluation Strategy**: Use stratified sampling across error types for fair assessment

---
*Report generated using DeltaBench analysis framework*
"""
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"Dataset insights saved to: {output_file}")
    return content