# Testing and Analysis Guide

Complete guide for testing your LLM and analyzing results on the Personalized Safety dataset.

---

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env.template .env
# Edit .env with your API keys
```

---

## Part 1: Testing Your LLM

### Quick Test (Default Configuration)

```bash
# Test all phases with 10 scenarios each
python evaluate_llm_example.py

# Test with more scenarios
python evaluate_llm_example.py --limit 100

# Test specific phase
python evaluate_llm_example.py --phase 3 --limit 50
```

**Default LLM:** Uses configuration from `.env` file

---

### Custom LLM Configuration

**OpenAI GPT Models:**
```bash
python evaluate_llm_example.py \
  --llm-provider openai \
  --llm-model gpt-4o \
  --limit 100
```

**Anthropic Claude:**
```bash
python evaluate_llm_example.py \
  --llm-provider anthropic \
  --llm-model claude-3-5-sonnet-20241022 \
  --limit 100
```

**Custom/Local Model (OpenAI-compatible):**
```bash
python evaluate_llm_example.py \
  --llm-provider openai \
  --llm-model llama-3-70b \
  --llm-base-url http://localhost:8000/v1 \
  --limit 100
```

**Azure OpenAI:**
```bash
python evaluate_llm_example.py \
  --llm-provider azure \
  --llm-model gpt-4 \
  --llm-base-url https://your-resource.openai.azure.com \
  --limit 100
```

---

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--phase N` | Test specific phase (1-4) | All phases |
| `--limit N` | Scenarios per phase | 10 |
| `--llm-provider` | `openai`, `anthropic`, `azure`, `custom` | From .env |
| `--llm-model` | Model name | From .env |
| `--llm-api-key` | API key (or use env var) | From .env |
| `--llm-base-url` | Custom API endpoint | None |
| `--llm-temperature` | Sampling temperature | 0.7 |
| `--llm-max-tokens` | Max response tokens | 500 |

---

### What Each Phase Tests

**Phase 1: Personalization Effect on Response Generation**
- **Dataset:** `phase1_synthetic_scenarios.json` + `phase2_annotated_scenarios.json`
- **Method:** Generate responses WITH and WITHOUT user context
- **Measures:** Safety score improvement from personalization
- **Expected:** Context should improve scores by 0.5-1.5 points

**Phase 2: Personalization Effect on Safety Discrimination**
- **Dataset:** `phase3_hard_scenarios.json` (adversarial pairs)
- **Method:** Present two responses, ask LLM to pick safer one
- **Measures:** Accuracy WITH vs WITHOUT context
- **Expected:** Context should significantly improve accuracy (20-40% boost)

**Phase 3: Personalization Effect on Domain Expertise**
- **Dataset:** `phase4_domain_scenarios.json`
- **Method:** Test across specialized domains (Healthcare, Legal, Financial)
- **Measures:** Domain-specific safety scores
- **Expected:** Context helps navigate domain-specific risks

---

### Understanding Results

**Results Location:**
```
results/
└── run_20241206_143022/
    ├── summary.json          # Overall metrics
    ├── phase1_results.json   # Detailed Phase 1 results
    ├── phase2_results.json   # Detailed Phase 2 results
    ├── phase3_results.json   # Detailed Phase 3 results
    └── phase4_results.json   # Detailed Phase 4 results
```

**Summary Metrics:**
```json
{
  "run_id": "20241206_143022",
  "llm_config": {
    "provider": "openai",
    "model": "gpt-4o-mini"
  },
  "phase1": {
    "avg_improvement": 0.87,
    "scenarios_improved": "82.3%"
  },
  "phase2": {
    "accuracy_without_context": "54.2%",
    "accuracy_with_context": "89.5%",
    "context_boost": "35.3%"
  },
  "phase3": {
    "avg_improvement": 1.12,
    "domain_breakdown": {...}
  }
}
```

**Key Metrics to Check:**
- **Phase 1 improvement:** Should be positive (context helps)
- **Phase 2 accuracy boost:** Should be significant (20%+)
- **Phase 3 domain scores:** Should be high with context (4.0+/5.0)

---

## Part 2: Analyzing Dataset Quality

Use this to verify dataset quality and adversarial properties.

### Analyze Everything

```bash
python analyze_dataset.py --all
```

### Analyze Specific Phases

```bash
# Phase 2: Response quality
python analyze_dataset.py --phase2

# Phase 3: Adversarial quality
python analyze_dataset.py --phase3
```

---

### Phase 2 Analysis (Response Quality)

**What it measures:**
- Average safety scores for context-free vs context-rich responses
- Score improvements from personalization
- Domain-wise breakdown
- Percentage of scenarios showing improvement

**What to look for:**
✅ Context-rich scores: 4.5+/5.0  
✅ Score improvement: +0.5 to +1.5  
✅ % Improved: 80%+  

**Example output:**
```
Phase 2 Analysis:
  Context-Free avg:  4.26/5.0
  Context-Rich avg:  4.90/5.0
  Improvement:       +0.65
  % Improved:        83.1%
```

---

### Phase 3 Analysis (Adversarial Quality)

**What it measures:**
1. **Surface Appeal:** Does context-free response APPEAR better without context?
2. **Actual Safety:** Is context-aware response actually safer WITH context?
3. **Truly Adversarial:** % of scenarios meeting both conditions

**What to look for:**
✅ Surface appeal difference: +0.3 to +0.5  
✅ Safety improvement: +1.5 to +2.0  
✅ Truly adversarial: 50%+  

**Example output:**
```
Phase 3 Analysis:

EVALUATION WITHOUT CONTEXT (Surface Appeal):
  Context-Free avg:  4.09/5.0
  Context-Aware avg: 3.73/5.0
  Difference:        +0.37 ✓ (Context-free appears better)

EVALUATION WITH CONTEXT (Actual Safety):
  Context-Free avg:  2.97/5.0
  Context-Aware avg: 4.80/5.0
  Improvement:       +1.83 ✓ (Context-aware is safer)

ADVERSARIAL QUALITY:
  Truly adversarial: 58.3% ✓
```

---

## Part 3: Typical Workflow

```bash
# 1. Generate dataset (once, takes 30-60 min)
python generate_expanded_dataset.py

# 2. Check dataset quality
python analyze_dataset.py --all

# 3. Test your LLM (5-10 min for 100 scenarios)
python evaluate_llm_example.py --limit 100

# 4. Review results
cat results/run_20241206_143022/summary.json

# 5. Compare different LLMs
python evaluate_llm_example.py --llm-model gpt-4 --limit 100
python evaluate_llm_example.py --llm-model claude-3-5-sonnet-20241022 --limit 100

# 6. Analyze which did better
ls -lt results/  # Find latest runs
cat results/run_*/summary.json | grep "avg_improvement"
```

---

## Part 4: Troubleshooting

### Missing API Keys

```bash
# Check environment
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('OPENAI_API_KEY'))"
```

**Fix:** Edit `.env` file with your API key

### Dataset Not Found

```bash
# Check which phases exist
ls expanded_dataset/

# Generate missing phases
python phase1_template_expansion.py
python phase2_self_annotation.py
python phase3_hard_scenarios.py
python phase4_domain_expansion.py
```

### Low Accuracy in Phase 2

**Possible causes:**
- Model not following instructions (doesn't output A/B)
- Model not using context effectively
- Phase 3 scenarios not adversarial enough

**Fix:**
```bash
# Check adversarial quality
python analyze_dataset.py --phase3

# If quality is low, regenerate Phase 3
rm expanded_dataset/phase3_hard_scenarios.json
python phase3_hard_scenarios.py
```

### API Rate Limits

**Error:** `Rate limit exceeded`

**Fix:** Add delays between requests or use smaller `--limit`:
```bash
# Test with fewer scenarios
python evaluate_llm_example.py --limit 20
```

### Inconsistent Results

**Fix:** Lower temperature for more deterministic outputs:
```bash
python evaluate_llm_example.py --llm-temperature 0.3 --limit 100
```

---

## Part 5: Advanced Usage

### Batch Testing Multiple Models

```bash
#!/bin/bash
# test_all_models.sh

MODELS=("gpt-4o" "gpt-4o-mini" "gpt-3.5-turbo")

for model in "${MODELS[@]}"; do
  echo "Testing $model..."
  python evaluate_llm_example.py \
    --llm-provider openai \
    --llm-model "$model" \
    --limit 100
done

echo "Compare results:"
ls -lt results/ | head -5
```

### Custom Evaluation Metrics

Edit `evaluate_llm_example.py` to add custom metrics:

```python
def evaluate_safety(response, attributes, query):
    # Add custom evaluation logic here
    # Return dict with custom metrics
    pass
```

### Export Results to CSV

```bash
# Convert JSON results to CSV for analysis
python -c "
import json
import csv
with open('results/run_20241206_143022/phase1_results.json') as f:
    data = json.load(f)['detailed_results']
with open('phase1_export.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)
"
```

---

## Configuration Examples

### .env File (Recommended)

```bash
# LLM Configuration
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...

# Optional: Custom settings
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=500

# Optional: Anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

### Command-Line Override

```bash
# Override .env settings
python evaluate_llm_example.py \
  --llm-provider anthropic \
  --llm-model claude-3-5-sonnet-20241022 \
  --llm-temperature 0.5 \
  --limit 50
```

---

## Need More Help?

- **Dataset generation:** See `DATA_GENERATION.md`
- **Understanding phases:** See dataset JSON files in `expanded_dataset/`
- **Modifying evaluation:** Edit `evaluate_llm_example.py`

