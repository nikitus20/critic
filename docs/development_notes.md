# DeltaBench Development Notes

## Current Status (2025-01-08)

### ✅ Completed Tasks

#### 1. Framework Implementation & Fixes
- **Error Section Handling**: Fixed to combine both `reason_error_section_numbers` AND `reason_unuseful_section_numbers` as ground truth (matching original paper methodology)
- **API Retry Mechanism**: Added 3-retry logic to `critic.py` for robustness
- **Macro/Micro Metrics**: Implemented both macro (per-example mean) and micro (aggregated TP/FP/FN) metrics calculation matching original paper
- **Evaluation Pipeline**: Updated `evaluator.py` with paper-compliant metrics calculation

#### 2. Comprehensive Experiment Script
- **Created `reproduce_deltabench_paper.py`**: Full experiment script with:
  - Multiprocessing support (configurable process count)
  - Checkpoint/resume capability to handle interruptions
  - Paper-compliant metrics calculation (macro/micro)
  - JSONL output format matching original evaluation
  - CSV metrics summary by task type
  - Cost and runtime estimation

#### 3. Testing & Validation
- **Small-scale test**: Successfully validated on 10 examples
  - 100% success rate
  - F1 macro: 0.414, F1 micro: 0.429
  - Performance by task: Math (0.273), Code (0.583), General (0.667)
  - Cost: ~$0.0008 per example

### 🔄 Current Work

#### Full Dataset Evaluation
- **Target**: Reproduce DeltaBench paper results on all 1,236 examples
- **Model**: GPT-4o-mini (cost-effective choice)
- **Status**: Started but hit API rate limits
- **Progress**: 21/1,236 examples completed (~1.7%)
- **Issue**: 200K TPM rate limit exceeded with 8 processes

### 🚨 Critical Differences Fixed

Identified and resolved key differences from original paper:
1. **Ground Truth**: Now combines error + unuseful sections ✅
2. **Retry Logic**: Added 3-attempt retry mechanism ✅  
3. **Metrics**: Both macro/micro calculation implemented ✅
4. **API Parameters**: Matches original (temp=1.0, top_p=0.8) ✅
5. **Filtering**: Uses max section number for validation ✅

### 📊 Expected Results

Based on test run extrapolation:
- **Estimated Total Cost**: ~$1.00-$1.50 for full dataset
- **Runtime**: ~2-4 hours (depending on rate limits)
- **Expected F1**: 0.3-0.5 (based on preliminary results)

### 🔧 Technical Implementation

#### Key Files Modified:
- `src/evaluator.py`: Fixed error section handling, added macro/micro metrics
- `src/critic.py`: Added retry mechanism for API robustness
- `reproduce_deltabench_paper.py`: Main experiment script
- `evaluate_100_samples.py`: Comprehensive analysis script

#### Data Flow:
1. Load dataset (1,236 examples)
2. Skip already processed examples (resume capability)
3. Multiprocess evaluation with rate limit handling
4. Save results to JSONL (raw data) + CSV (metrics)
5. Calculate paper-compliant macro/micro metrics by task type

### 📁 Output Structure
```
results/
├── Deltabench_v1_gpt-4o-mini_deltabench.jsonl  # Raw evaluation results
├── Deltabench_v1_gpt-4o-mini_deltabench_metrics.csv  # Summary metrics
└── evaluation_100_samples_results.csv  # 100-sample analysis
```

### 🎯 Next Steps

1. **Complete Full Experiment**: Run with reduced process count (2-4) to avoid rate limits
2. **Results Analysis**: Compare with original DeltaBench paper results
3. **Performance Optimization**: Fine-tune multiprocessing for rate limit compliance
4. **Documentation**: Update README with paper reproduction instructions

### 💡 Key Insights

- **Rate Limiting**: GPT-4o-mini has 200K TPM limit, requires careful process management
- **Resume Capability**: Critical for long-running experiments (handles interruptions)
- **Metrics Alignment**: Macro vs Micro metrics show different performance perspectives
- **Task Variation**: Math tasks consistently show lower performance than other domains

### 🛠️ For Other Developers

#### To reproduce paper results:
```bash
# Full dataset (est. $1.50, 2-4 hours)
python reproduce_deltabench_paper.py --processes 2

# Test subset
python reproduce_deltabench_paper.py --test_size 50 --processes 2

# Resume interrupted run
python reproduce_deltabench_paper.py --processes 2  # Automatically resumes
```

#### To run analysis:
```bash
# 100-sample comprehensive analysis
python evaluate_100_samples.py

# Basic 5-sample test
python example.py
```

### 📋 Dependencies
- OpenAI API key required in `.env` file
- All dependencies in `requirements.txt`
- Results directory created automatically

---
*Last updated: 2025-01-08*
*Status: Ready for full experiment execution*