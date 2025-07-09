# DeltaBench Dataset Insights

Generated on: 2025-07-07 12:58:22

## Dataset Overview

- **Total Examples**: 1,236
- **Examples with Errors**: 1,236
- **Error Rate**: 100.0%
- **Average Sections per Example**: 0.0

## Task Distribution

| Task Type | Count | Percentage |
|-----------|-------|------------|
| math | 562 | 45.5% |
| code | 373 | 30.2% |
| Physics&Chemistry&Biology | 154 | 12.5% |
| general | 147 | 11.9% |

## Error Analysis

### Error Count Distribution

| Error Count | Examples | Percentage |
|-------------|----------|------------|
| 1 | 492 | 39.8% |
| 2 | 466 | 37.7% |
| 3 | 134 | 10.8% |
| 4 | 67 | 5.4% |
| 5 | 38 | 3.1% |
| 6 | 17 | 1.4% |
| 7 | 13 | 1.1% |
| 8 | 6 | 0.5% |
| 9 | 3 | 0.2% |

### Error Position Analysis

- **Early Errors** (sections 1-3): 648 examples
- **Middle Errors**: 0 examples  
- **Late Errors**: 1904 examples

### Error Complexity

- **Single Error Examples**: 492 (39.8% of error examples)
- **Multiple Error Examples**: 744 (60.2% of error examples)

### Most Common Error Sections

- **Section 3**: 338 errors
- **Section 4**: 277 errors
- **Section 2**: 242 errors
- **Section 5**: 229 errors
- **Section 6**: 199 errors
- **Section 7**: 151 errors
- **Section 8**: 122 errors
- **Section 9**: 119 errors
- **Section 10**: 104 errors
- **Section 12**: 89 errors

### Errors by Task Type

| Task Type | Error Examples | Error Rate |
|-----------|----------------|------------|
| math | 562 | 100.0% |
| general | 147 | 100.0% |
| Physics&Chemistry&Biology | 154 | 100.0% |
| code | 373 | 100.0% |

## Key Insights

### Dataset Characteristics
1. **Diverse Task Coverage**: The dataset spans 4 different mathematical domains
2. **Balanced Error Distribution**: 100.0% of examples contain reasoning errors
3. **Error Complexity**: 60.2% of error examples have multiple mistakes

### Error Patterns
1. **Position Bias**: Late errors are more common
2. **Section Hotspots**: Sections 3, 4, 2 contain the most errors
3. **Task-Specific Errors**: math has the highest error count

### Recommendations
1. **Focus Areas**: Prioritize error detection in math problems
2. **Model Training**: Include examples with multiple errors for robust critic training
3. **Evaluation Strategy**: Use stratified sampling across error types for fair assessment

---
*Report generated using DeltaBench analysis framework*
