"""Dataset loading for DeltaBench."""

import json
import pandas as pd
import re
from typing import Dict, List, Optional, Tuple


class DeltaBenchDataset:
    """Simple dataset loader for DeltaBench."""
    
    def __init__(self, file_path: Optional[str] = None):
        self.data = None
        self.file_path = file_path
        
    def load_jsonl(self, file_path: str) -> List[Dict]:
        """Load dataset from JSONL format."""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            self.data = data
            self.file_path = file_path
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []
    
    def load_csv(self, file_path: str) -> List[Dict]:
        """Load dataset from CSV format."""
        try:
            df = pd.read_csv(file_path)
            data = df.to_dict('records')
            self.data = data
            self.file_path = file_path
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []
    
    def get_sample(self, n: int = 10) -> List[Dict]:
        """Get first n examples."""
        if self.data is None:
            return []
        return self.data[:n]
    
    def parse_sections(self, sections_content: str) -> List[Tuple[int, str]]:
        """Parse sections from content string into list of (section_num, content)."""
        sections = []
        # Match patterns like "section1:", "section 2:", etc.
        pattern = r'section\s*(\d+)\s*:\s*'
        parts = re.split(pattern, sections_content, flags=re.IGNORECASE)
        
        # parts[0] is text before first section, then alternating section numbers and content
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                section_num = int(parts[i])
                content = parts[i + 1].strip()
                sections.append((section_num, content))
        
        return sections
    
    def get_examples_with_errors(self, limit: Optional[int] = None) -> List[Dict]:
        """Get examples that have errors."""
        if self.data is None:
            return []
        
        error_examples = []
        for example in self.data:
            error_sections = example.get('reason_error_section_numbers', [])
            unuseful_sections = example.get('reason_unuseful_section_numbers', [])
            if error_sections or unuseful_sections:
                error_examples.append(example)
                if limit and len(error_examples) >= limit:
                    break
        
        return error_examples
    
    def filter_by_task_type(self, task_l1: Optional[str] = None, task_l2: Optional[str] = None) -> List[Dict]:
        """Filter examples by task type."""
        if self.data is None:
            return []
        
        filtered = self.data
        if task_l1:
            filtered = [ex for ex in filtered if ex.get('task_l1') == task_l1]
        if task_l2:
            filtered = [ex for ex in filtered if ex.get('task_l2') == task_l2]
        
        return filtered
    
    def get_statistics(self) -> Dict:
        """Get basic dataset statistics."""
        if self.data is None:
            return {}
        
        total = len(self.data)
        with_errors = len(self.get_examples_with_errors())
        
        # Count task types
        task_l1_counts = {}
        task_l2_counts = {}
        
        for ex in self.data:
            l1 = ex.get('task_l1', 'unknown')
            l2 = ex.get('task_l2', 'unknown')
            task_l1_counts[l1] = task_l1_counts.get(l1, 0) + 1
            task_l2_counts[l2] = task_l2_counts.get(l2, 0) + 1
        
        return {
            'total_examples': total,
            'examples_with_errors': with_errors,
            'error_rate': with_errors / total if total > 0 else 0,
            'task_l1_distribution': task_l1_counts,
            'task_l2_distribution': task_l2_counts
        }