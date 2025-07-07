"""Dataset loading for DeltaBench."""

import json
import pandas as pd
from typing import Dict, List, Optional


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