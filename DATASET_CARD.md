---
license: mit
task_categories:
  - text-generation
  - conversational
language:
  - en
  - code
size_categories:
  - 1K<n<10K
---

# Fable5 Dataset

A collection of agent interaction traces from 6 distinct sources, designed for fine-tuning and evaluating coding agents.

## Dataset Sources

| Source | Format | Description | Records |
|--------|--------|-------------|---------|
| Glint | Session-based with turns | Full agent sessions with tool use | ~2,000 |
| armand0e | Conversation with tool_calls | Multi-turn conversations with function calling | ~1,500 |
| vfable | Trajectory with tool_use | Agent trajectories with sequential tool use | ~800 |
| Coding Excellence | Session-based with quality scores | High-quality coding sessions rated by experts | ~500 |
| OpenCoven | Source/target pairs | Instruction-following input/output pairs | ~3,000 |
| Victor | Prompt/response pairs | Single-turn coding instruction pairs | ~4,000 |

## Unified Schema

All records are normalized to this format:

```json
{
  "id": "session_0001",
  "messages": [
    {"role": "user", "content": "Fix the bug in auth.py"},
    {"role": "assistant", "content": "Let me check the code.", "tool_use": [{"name": "read", "input": {"file_path": "auth.py"}}]},
    {"role": "assistant", "content": "Found the issue. Here's the fix."}
  ],
  "tools": [
    {"name": "read", "input": {"file_path": "auth.py"}}
  ],
  "metadata": {"source": "glint", "quality_score": 0.85}
}
```

## Quick Start

```python
from fable5_dataset import DatasetLoader, Preprocessor, DatasetStats

loader = DatasetLoader()

# Load a single dataset
records = loader.load_dataset("glint", normalize=True, remove_pii=True)

# Load all datasets
all_data = loader.load_dataset("all")

# Compute statistics
stats = DatasetStats()
result = stats.compute_stats(records)
print(result.summary())
```

## Preprocessing

```python
from fable5_dataset import Preprocessor

preprocessor = Preprocessor()

# Normalize format
normalized = preprocessor.normalize_format(records, source_format="glint")

# Remove PII
cleaned = preprocessor.remove_pii(normalized)

# Filter by quality
high_quality = preprocessor.filter_quality(cleaned, min_quality=0.7)
```

## Benchmark Generation

```python
from fable5_dataset import BenchmarkGenerator

gen = BenchmarkGenerator()
tasks = gen.generate_benchmark(records, num_tasks=50, categories=["debugging", "implementation"])
```

## Statistics

| Metric | Glint | armand0e | vfable | Coding Excellence | OpenCoven | Victor |
|--------|-------|----------|--------|-------------------|-----------|--------|
| Avg turns/session | 8.2 | 5.4 | 6.7 | 12.3 | 2.0 | 2.0 |
| Unique tools | 12 | 8 | 10 | 15 | 0 | 0 |
| Avg quality score | 0.72 | 0.68 | 0.75 | 0.92 | 0.85 | 0.80 |
| Error recovery rate | 0.45 | 0.38 | 0.42 | 0.65 | 0.10 | 0.05 |

## License

MIT License

## Citation

```bibtex
@dataset{fable5,
  title={Fable5: Agent Trace Datasets for Coding Assistant Fine-tuning},
  author={FableForge},
  year={2025},
  license={MIT}
}
```
