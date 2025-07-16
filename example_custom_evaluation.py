#!/usr/bin/env python3
"""
Example script demonstrating how to use ProcessBench methodology
on a custom dataset with section-by-section annotations.

This script shows how to:
1. Load custom data in the required format
2. Run the evaluation with and without voting
3. Calculate ProcessBench metrics
4. Analyze results

Usage:
    python example_custom_evaluation.py --api_key YOUR_API_KEY --data_file your_data.json
"""

import argparse
import json
import re
from collections import Counter
from openai import OpenAI
from tqdm import tqdm
import time

# Template for process-level error detection
CRITIQUE_TEMPLATE = """The following is a problem and a solution (split into paragraphs, enclosed with tags and indexed from 0):

[Problem]

{problem}

[Solution]

{tagged_response}

Your task is to review and critique the solution paragraph by paragraph. Once you identify an error in a paragraph, return the index of the paragraph where the earliest error occurs. Otherwise, return the index of -1 (which typically denotes "not found").

Please put your final answer (i.e., the index) in \\boxed{{}}.
"""

def extract_answer(response_text):
    """Extract the boxed answer from the model's response."""
    boxed_pattern = r'\\boxed\\{([^}]*)\\}'
    matches = re.findall(boxed_pattern, response_text)
    if matches:
        try:
            return int(matches[-1].strip())
        except ValueError:
            return None
    return None

def prepare_input(template, input_data):
    """Prepare input for the model by formatting problem and solution."""
    problem = input_data['problem']
    steps = input_data['steps']
    
    # Tag each step with paragraph indices
    tagged_response = ''
    for idx, step in enumerate(steps):
        tagged_response += f'<paragraph_{idx}>\\n{step}\\n</paragraph_{idx}>\\n\\n'
    tagged_response = tagged_response.strip()
    
    # Format the prompt
    prompt = template.format(problem=problem, tagged_response=tagged_response)
    messages = [{'role': 'user', 'content': prompt}]
    return messages

def call_model_api(client, messages, model="gpt-4o-mini", temperature=0.0, max_tokens=4096, max_retries=3):
    """Call the OpenAI API with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=42  # For reproducibility
            )
            return response
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = (attempt + 1) * 2
            print(f"API error, retrying in {wait_time} seconds: {e}")
            time.sleep(wait_time)

def process_single_example(client, template, example, model="gpt-4o-mini", use_voting=False, n_votes=8):
    """Process a single example with optional voting."""
    messages = prepare_input(template, example)
    
    if not use_voting:
        # Single prediction
        response = call_model_api(client, messages, model, temperature=0.0)
        generated_text = response.choices[0].message.content
        prediction = extract_answer(generated_text)
        
        return {
            'prediction': prediction,
            'generated_text': generated_text,
            'votes': None
        }
    else:
        # Voting mechanism
        votes = []
        generated_texts = []
        
        for _ in range(n_votes):
            response = call_model_api(client, messages, model, temperature=0.7)
            generated_text = response.choices[0].message.content
            generated_texts.append(generated_text)
            
            prediction = extract_answer(generated_text)
            if prediction is not None:
                votes.append(prediction)
        
        # Determine final prediction by majority vote
        if votes:
            vote_counts = Counter(votes)
            final_prediction = vote_counts.most_common(1)[0][0]
        else:
            final_prediction = None
        
        return {
            'prediction': final_prediction,
            'generated_text': generated_texts,
            'votes': votes,
            'vote_distribution': dict(vote_counts) if votes else {}
        }

def evaluate_dataset(data, client, template, model="gpt-4o-mini", use_voting=False, n_votes=8):
    """Evaluate a dataset using ProcessBench methodology."""
    results = []
    
    for example in tqdm(data, desc="Processing examples"):
        result = process_single_example(
            client, template, example, model, use_voting, n_votes
        )
        
        # Add ground truth and match information
        result.update({
            'id': example['id'],
            'problem': example['problem'],
            'steps': example['steps'],
            'label': example['label'],
            'match': result['prediction'] == example['label'] if result['prediction'] is not None else False
        })
        
        results.append(result)
    
    return results

def calculate_metrics(results):
    """Calculate ProcessBench metrics."""
    # Split into error and correct examples
    error_examples = [r for r in results if r['label'] != -1]
    correct_examples = [r for r in results if r['label'] == -1]
    
    # Calculate accuracies
    if error_examples:
        error_acc = sum(r['match'] for r in error_examples) / len(error_examples) * 100
    else:
        error_acc = 0.0
    
    if correct_examples:
        correct_acc = sum(r['match'] for r in correct_examples) / len(correct_examples) * 100
    else:
        correct_acc = 0.0
    
    # Calculate F1 score
    if error_acc + correct_acc > 0:
        f1_score = 2 * error_acc * correct_acc / (error_acc + correct_acc)
    else:
        f1_score = 0.0
    
    return {
        'error_accuracy': error_acc,
        'correct_accuracy': correct_acc,
        'f1_score': f1_score,
        'error_count': len(error_examples),
        'correct_count': len(correct_examples),
        'total_count': len(results)
    }

def analyze_results(results, metrics):
    """Analyze and print detailed results."""
    print("\\n=== ProcessBench Evaluation Results ===")
    print(f"Error Accuracy: {metrics['error_accuracy']:.1f}%")
    print(f"Correct Accuracy: {metrics['correct_accuracy']:.1f}%")
    print(f"F1 Score: {metrics['f1_score']:.1f}")
    print(f"Total Examples: {metrics['total_count']}")
    print(f"Error Examples: {metrics['error_count']}")
    print(f"Correct Examples: {metrics['correct_count']}")
    
    # Error analysis
    error_predictions = [r for r in results if r['label'] != -1]
    correct_predictions = [r for r in results if r['label'] == -1]
    
    print("\\n=== Error Analysis ===")
    if error_predictions:
        error_matches = sum(r['match'] for r in error_predictions)
        print(f"Error detection: {error_matches}/{len(error_predictions)} correct")
        
        # Show some examples
        print("\\nError detection examples:")
        for i, r in enumerate(error_predictions[:3]):
            status = "✓" if r['match'] else "✗"
            print(f"{status} {r['id']}: predicted={r['prediction']}, actual={r['label']}")
    
    if correct_predictions:
        correct_matches = sum(r['match'] for r in correct_predictions)
        print(f"Correct recognition: {correct_matches}/{len(correct_predictions)} correct")
        
        # Show some examples
        print("\\nCorrect recognition examples:")
        for i, r in enumerate(correct_predictions[:3]):
            status = "✓" if r['match'] else "✗"
            print(f"{status} {r['id']}: predicted={r['prediction']}, actual={r['label']}")

def validate_data_format(data):
    """Validate that data follows the required format."""
    required_fields = ['id', 'problem', 'steps', 'label']
    
    for i, example in enumerate(data):
        for field in required_fields:
            if field not in example:
                raise ValueError(f"Example {i} missing required field: {field}")
        
        if not isinstance(example['steps'], list):
            raise ValueError(f"Example {i}: 'steps' must be a list")
        
        if not isinstance(example['label'], int):
            raise ValueError(f"Example {i}: 'label' must be an integer")
        
        if example['label'] >= len(example['steps']) and example['label'] != -1:
            raise ValueError(f"Example {i}: label {example['label']} out of range for {len(example['steps'])} steps")
    
    print(f"✓ Data format validation passed for {len(data)} examples")

def create_sample_data():
    """Create sample data for testing."""
    return [
        {
            "id": "sample_1",
            "problem": "A rectangle has length 8 and width 5. What is its area?",
            "steps": [
                "The area of a rectangle is length × width.",
                "Given: length = 8, width = 5",
                "Area = 8 × 5 = 40",
                "Therefore, the area is 40 square units."
            ],
            "label": -1  # No errors
        },
        {
            "id": "sample_2",
            "problem": "John has 20 marbles. He gives 1/4 to his sister. How many marbles does he have left?",
            "steps": [
                "John starts with 20 marbles.",
                "He gives 1/4 of them to his sister: 20 × (1/4) = 5 marbles.",
                "He has 20 - 5 = 15 marbles left.",
                "Actually, let me recalculate: 20 × (1/4) = 6 marbles given away.",
                "So he has 20 - 6 = 14 marbles left."
            ],
            "label": 3  # Error in step 3 (recalculation is wrong)
        },
        {
            "id": "sample_3",
            "problem": "What is 15% of 80?",
            "steps": [
                "To find 15% of 80, I multiply 80 by 0.15.",
                "80 × 0.15 = 12",
                "Therefore, 15% of 80 is 12."
            ],
            "label": -1  # No errors
        }
    ]

def main():
    parser = argparse.ArgumentParser(description="ProcessBench evaluation on custom dataset")
    parser.add_argument('--api_key', required=True, help='OpenAI API key')
    parser.add_argument('--data_file', help='Path to JSON file with data')
    parser.add_argument('--model', default='gpt-4o-mini', help='OpenAI model to use')
    parser.add_argument('--use_voting', action='store_true', help='Use voting mechanism')
    parser.add_argument('--n_votes', type=int, default=8, help='Number of votes')
    parser.add_argument('--output_file', help='Output file for results')
    parser.add_argument('--use_sample_data', action='store_true', help='Use built-in sample data')
    
    args = parser.parse_args()
    
    # Initialize OpenAI client
    client = OpenAI(api_key=args.api_key)
    
    # Load data
    if args.use_sample_data:
        print("Using built-in sample data")
        data = create_sample_data()
    elif args.data_file:
        print(f"Loading data from {args.data_file}")
        with open(args.data_file, 'r') as f:
            data = json.load(f)
    else:
        raise ValueError("Must provide either --data_file or --use_sample_data")
    
    # Validate data format
    validate_data_format(data)
    
    # Run evaluation
    print(f"Running evaluation with {args.model}")
    if args.use_voting:
        print(f"Using voting mechanism with {args.n_votes} votes")
    
    results = evaluate_dataset(
        data=data,
        client=client,
        template=CRITIQUE_TEMPLATE,
        model=args.model,
        use_voting=args.use_voting,
        n_votes=args.n_votes
    )
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Analyze results
    analyze_results(results, metrics)
    
    # Save results
    output_data = {
        'results': results,
        'metrics': metrics,
        'config': {
            'model': args.model,
            'use_voting': args.use_voting,
            'n_votes': args.n_votes if args.use_voting else None
        }
    }
    
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\\nResults saved to {args.output_file}")
    
    # Show voting analysis if applicable
    if args.use_voting:
        print("\\n=== Voting Analysis ===")
        voting_results = [r for r in results if r.get('votes')]
        if voting_results:
            avg_votes = sum(len(r['votes']) for r in voting_results) / len(voting_results)
            print(f"Average valid votes per example: {avg_votes:.1f}")
            
            # Show vote distribution example
            for r in voting_results[:2]:
                if r.get('vote_distribution'):
                    print(f"{r['id']}: {r['vote_distribution']} → {r['prediction']}")

if __name__ == "__main__":
    main()