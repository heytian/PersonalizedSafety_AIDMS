#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
============================================================
| File: example_usage.py
| Description:
|     Example usage and analysis of the expanded dataset
|     generation system. Shows how to:
|     - Run the pipeline
|     - Load and analyze results
|     - Export data in different formats
|     - Visualize statistics
============================================================
"""

import json
import os
from collections import defaultdict
from typing import Dict, List

# ============================================================
# Example 1: Running the Pipeline
# ============================================================

def example_run_pipeline():
    """Example of running the complete pipeline."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Running the Pipeline")
    print("="*70)
    
    print("""
# Full pipeline (all 4 phases)
python generate_expanded_dataset.py --all

# Run specific phases
python generate_expanded_dataset.py --phase 1
python generate_expanded_dataset.py --phase 1,2
python generate_expanded_dataset.py --phase 3,4

# Resume from last completed phase
python generate_expanded_dataset.py --resume

# Check status
python generate_expanded_dataset.py --status

# Force re-run Phase 1
python generate_expanded_dataset.py --force --phase 1

# Just combine existing outputs
python generate_expanded_dataset.py --combine
    """)

# ============================================================
# Example 2: Load and Analyze Dataset
# ============================================================

def load_dataset(filepath: str = "expanded_dataset/final_combined_dataset.json"):
    """Load the final combined dataset."""
    if not os.path.exists(filepath):
        print(f"Dataset not found at {filepath}")
        print("Run: python generate_expanded_dataset.py --all")
        return None
    
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def analyze_dataset(dataset: Dict):
    """Analyze dataset statistics."""
    print("\n" + "="*70)
    print("DATASET ANALYSIS")
    print("="*70)
    
    metadata = dataset.get("metadata", {})
    print(f"\nTotal Scenarios: {metadata.get('total_scenarios', 0):,}")
    print(f"Generation Date: {metadata.get('generation_date', 'Unknown')}")
    print(f"Phases Included: {metadata.get('phases_included', [])}")
    
    # Phase 1 analysis
    if dataset.get("phase1_scenarios"):
        phase1 = dataset["phase1_scenarios"]
        print(f"\n--- Phase 1: Template-Based Scenarios ---")
        print(f"Total: {len(phase1):,}")
        
        # Domain distribution
        domains = defaultdict(int)
        for scenario in phase1:
            domains[scenario.get("domain", "Unknown")] += 1
        
        print("\nDomain Distribution:")
        for domain, count in sorted(domains.items(), key=lambda x: x[1], reverse=True):
            print(f"  {domain:20s}: {count:8,} ({100*count/len(phase1):.1f}%)")
    
    # Phase 2 analysis
    if dataset.get("phase2_annotated"):
        phase2 = dataset["phase2_annotated"]
        meta = phase2.get("metadata", {})
        print(f"\n--- Phase 2: Annotated Scenarios ---")
        print(f"Total Processed: {meta.get('total_scenarios', 0):,}")
        print(f"Good (≥3.0):     {meta.get('good_scenarios', 0):,}")
        print(f"Borderline:      {meta.get('borderline_scenarios', 0):,}")
        print(f"Unsafe:          {meta.get('unsafe_scenarios', 0):,}")
        
        # Calculate average improvement
        good = phase2.get("good_scenarios", [])
        if good:
            avg_improvement = sum(s.get("score_improvement", 0) for s in good) / len(good)
            print(f"Avg Safety Improvement: +{avg_improvement:.2f}")
    
    # Phase 3 analysis
    if dataset.get("phase3_hard_scenarios"):
        phase3 = dataset["phase3_hard_scenarios"]
        print(f"\n--- Phase 3: Hard Adversarial Scenarios ---")
        print(f"Total: {len(phase3):,}")
        
        if phase3:
            avg_unsafe = sum(s["evaluation_unsafe"]["overall_safety"] for s in phase3) / len(phase3)
            avg_safe = sum(s["evaluation_safe"]["overall_safety"] for s in phase3) / len(phase3)
            avg_imp = sum(s["safety_improvement"] for s in phase3) / len(phase3)
            
            print(f"Avg Unsafe Score:  {avg_unsafe:.2f}/5.0")
            print(f"Avg Safe Score:    {avg_safe:.2f}/5.0")
            print(f"Avg Improvement:   +{avg_imp:.2f}")
    
    # Phase 4 analysis
    if dataset.get("phase4_domain_scenarios"):
        phase4 = dataset["phase4_domain_scenarios"]
        print(f"\n--- Phase 4: Domain-Specific Scenarios ---")
        print(f"Total: {len(phase4):,}")
        
        # Domain distribution
        domains = defaultdict(int)
        for scenario in phase4:
            domains[scenario.get("domain", "Unknown")] += 1
        
        print("\nExpert Domain Distribution:")
        for domain, count in sorted(domains.items(), key=lambda x: x[1], reverse=True):
            print(f"  {domain:30s}: {count:6,}")
    
    print("="*70)

# ============================================================
# Example 3: Export to Different Formats
# ============================================================

def export_to_csv(dataset: Dict, output_file: str = "dataset_export.csv"):
    """Export scenarios to CSV format."""
    import csv
    
    print(f"\nExporting to {output_file}...")
    
    # Flatten all scenarios
    all_scenarios = []
    
    # Phase 1
    for s in dataset.get("phase1_scenarios", []):
        all_scenarios.append({
            "scenario_id": s.get("scenario_id"),
            "phase": "phase1",
            "domain": s.get("domain"),
            "query": s.get("query"),
            "attributes": json.dumps(s.get("attributes", {}))
        })
    
    # Phase 2 good scenarios
    for s in dataset.get("phase2_annotated", {}).get("good_scenarios", []):
        all_scenarios.append({
            "scenario_id": s.get("scenario_id"),
            "phase": "phase2",
            "domain": s.get("domain"),
            "query": s.get("query"),
            "response": s.get("response_context_rich"),
            "safety_score": s.get("evaluation_context_rich", {}).get("overall_safety"),
            "attributes": json.dumps(s.get("attributes", {}))
        })
    
    # Write CSV
    if all_scenarios:
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_scenarios[0].keys())
            writer.writeheader()
            writer.writerows(all_scenarios)
        
        print(f"✓ Exported {len(all_scenarios):,} scenarios to {output_file}")

def export_training_pairs(dataset: Dict, output_file: str = "training_pairs.jsonl"):
    """Export query-response pairs for training."""
    print(f"\nExporting training pairs to {output_file}...")
    
    pairs = []
    
    # From Phase 2 good scenarios
    for s in dataset.get("phase2_annotated", {}).get("good_scenarios", []):
        pairs.append({
            "query": s.get("query"),
            "context": s.get("attributes"),
            "response": s.get("response_context_rich"),
            "score": s.get("evaluation_context_rich", {}).get("overall_safety")
        })
    
    # From Phase 3 safe responses
    for s in dataset.get("phase3_hard_scenarios", []):
        pairs.append({
            "query": s.get("hard_query"),
            "context": s.get("attributes"),
            "response": s.get("safe_response"),
            "score": s.get("evaluation_safe", {}).get("overall_safety"),
            "type": "adversarial"
        })
    
    # Write JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    
    print(f"✓ Exported {len(pairs):,} training pairs to {output_file}")

# ============================================================
# Example 4: Filter and Query Dataset
# ============================================================

def filter_by_domain(dataset: Dict, domain: str) -> List[Dict]:
    """Filter scenarios by domain."""
    results = []
    
    # Search Phase 1
    for s in dataset.get("phase1_scenarios", []):
        if s.get("domain") == domain:
            results.append(s)
    
    # Search Phase 4
    for s in dataset.get("phase4_domain_scenarios", []):
        if s.get("domain") == domain:
            results.append(s)
    
    return results

def filter_by_safety_score(dataset: Dict, min_score: float = 4.0) -> List[Dict]:
    """Filter scenarios by minimum safety score."""
    results = []
    
    # Search Phase 2
    for s in dataset.get("phase2_annotated", {}).get("good_scenarios", []):
        score = s.get("evaluation_context_rich", {}).get("overall_safety", 0)
        if score >= min_score:
            results.append(s)
    
    # Search Phase 4
    for s in dataset.get("phase4_domain_scenarios", []):
        score = s.get("evaluation", {}).get("overall_safety", 0)
        if score >= min_score:
            results.append(s)
    
    return results

def filter_by_attributes(dataset: Dict, **attribute_filters) -> List[Dict]:
    """Filter scenarios by attribute values."""
    results = []
    
    for s in dataset.get("phase1_scenarios", []):
        attrs = s.get("attributes", {})
        match = all(attrs.get(k) == v for k, v in attribute_filters.items())
        if match:
            results.append(s)
    
    return results

# ============================================================
# Main Example Runner
# ============================================================

def main():
    print("\n" + "="*70)
    print("EXPANDED DATASET - EXAMPLE USAGE")
    print("="*70)
    
    # Show pipeline commands
    example_run_pipeline()
    
    # Try to load and analyze dataset
    dataset = load_dataset()
    
    if dataset:
        # Analyze
        analyze_dataset(dataset)
        
        # Export examples
        print("\n" + "="*70)
        print("EXPORT EXAMPLES")
        print("="*70)
        export_to_csv(dataset, "dataset_export.csv")
        export_training_pairs(dataset, "training_pairs.jsonl")
        
        # Filter examples
        print("\n" + "="*70)
        print("FILTER EXAMPLES")
        print("="*70)
        
        # Filter by domain
        health_scenarios = filter_by_domain(dataset, "Health")
        print(f"\nHealth domain scenarios: {len(health_scenarios):,}")
        
        # Filter by score
        high_quality = filter_by_safety_score(dataset, min_score=4.0)
        print(f"High quality scenarios (≥4.0): {len(high_quality):,}")
        
        # Filter by attributes
        young_female = filter_by_attributes(dataset, Age="18-24", Gender="Female")
        print(f"Young female scenarios: {len(young_female):,}")
    
    else:
        print("\n⚠️  Dataset not found. Generate it first:")
        print("   python generate_expanded_dataset.py --all")
    
    print("\n" + "="*70)
    print("For more details, see DATASET_EXPANSION_README.md")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()



