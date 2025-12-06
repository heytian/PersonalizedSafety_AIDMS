#!/usr/bin/env python3
"""
Analyze dataset statistics for Phase 2 and Phase 3.

Phase 2: Analyzes annotated scenarios (context-free vs context-rich responses)
Phase 3: Analyzes adversarial discrimination scenarios

Usage:
    python analyze_dataset.py --all
    python analyze_dataset.py --phase2
    python analyze_dataset.py --phase3
"""

import json
import argparse
import numpy as np
from collections import defaultdict

def analyze_phase2():
    """Analyze Phase 2 annotated scenarios."""
    print("\n" + "="*70)
    print("PHASE 2 ANNOTATED SCENARIOS ANALYSIS")
    print("="*70)
    
    try:
        with open('expanded_dataset/phase2_annotated_scenarios.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ Phase 2 file not found: expanded_dataset/phase2_annotated_scenarios.json")
        return
    
    metadata = data.get('metadata', {})
    good_scenarios = data.get('good_scenarios', [])
    borderline = data.get('borderline_scenarios', [])
    unsafe = data.get('unsafe_scenarios', [])
    
    print(f"\nTotal processed: {metadata.get('total_scenarios', 0):,}")
    print(f"Good (≥3.0):     {len(good_scenarios):,} ({metadata.get('good_percentage', 0):.1f}%)")
    print(f"Borderline:      {len(borderline):,} ({metadata.get('borderline_percentage', 0):.1f}%)")
    print(f"Unsafe (<2.5):   {len(unsafe):,} ({metadata.get('unsafe_percentage', 0):.1f}%)")
    
    if good_scenarios:
        print("\n" + "="*70)
        print("GOOD SCENARIOS STATISTICS")
        print("="*70)
        
        # Score distributions
        context_free_scores = [s['evaluation_context_free']['overall_safety'] for s in good_scenarios]
        context_rich_scores = [s['evaluation_context_rich']['overall_safety'] for s in good_scenarios]
        improvements = [s.get('score_improvement', 0) for s in good_scenarios]
        
        print(f"\nContext-Free Response Scores:")
        print(f"  Mean:   {np.mean(context_free_scores):.2f}/5.0")
        print(f"  Median: {np.median(context_free_scores):.2f}/5.0")
        print(f"  Std:    {np.std(context_free_scores):.2f}")
        
        print(f"\nContext-Rich Response Scores:")
        print(f"  Mean:   {np.mean(context_rich_scores):.2f}/5.0")
        print(f"  Median: {np.median(context_rich_scores):.2f}/5.0")
        print(f"  Std:    {np.std(context_rich_scores):.2f}")
        
        print(f"\nScore Improvements:")
        print(f"  Mean:   {np.mean(improvements):+.2f}")
        print(f"  Median: {np.median(improvements):+.2f}")
        print(f"  % Improved: {100*sum(1 for i in improvements if i > 0)/len(improvements):.1f}%")
        
        # Domain breakdown
        print("\n" + "="*70)
        print("BY DOMAIN")
        print("="*70)
        
        domain_stats = defaultdict(lambda: {'scores': [], 'improvements': []})
        for scenario in good_scenarios:
            domain = scenario.get('domain', 'Unknown')
            domain_stats[domain]['scores'].append(scenario['evaluation_context_rich']['overall_safety'])
            domain_stats[domain]['improvements'].append(scenario.get('score_improvement', 0))
        
        for domain in sorted(domain_stats.keys()):
            stats = domain_stats[domain]
            print(f"\n{domain} (n={len(stats['scores'])}):")
            print(f"  Avg Score:       {np.mean(stats['scores']):.2f}/5.0")
            print(f"  Avg Improvement: {np.mean(stats['improvements']):+.2f}")

def analyze_phase3():
    """Analyze Phase 3 hard adversarial scenarios."""
    print("\n" + "="*70)
    print("PHASE 3 HARD SCENARIOS ANALYSIS")
    print("="*70)
    
    try:
        with open('expanded_dataset/phase3_hard_scenarios.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ Phase 3 file not found: expanded_dataset/phase3_hard_scenarios.json")
        return
    
    scenarios = data['hard_scenarios']
    print(f"\nTotal scenarios: {len(scenarios):,}")
    
    # Extract scores
    context_free_with_ctx = []
    context_aware_with_ctx = []
    context_free_no_ctx = []
    context_aware_no_ctx = []
    
    surface_differences = []
    safety_improvements = []
    
    for scenario in scenarios:
        # With context evaluations
        unsafe_score = scenario['evaluation_unsafe']['overall_safety']
        safe_score = scenario['evaluation_safe']['overall_safety']
        
        context_free_with_ctx.append(unsafe_score)
        context_aware_with_ctx.append(safe_score)
        
        # Without context evaluations
        unsafe_no_ctx = scenario['evaluation_unsafe_no_context']['overall_safety']
        safe_no_ctx = scenario['evaluation_safe_no_context']['overall_safety']
        
        context_free_no_ctx.append(unsafe_no_ctx)
        context_aware_no_ctx.append(safe_no_ctx)
        
        # Calculate differences
        surface_diff = unsafe_no_ctx - safe_no_ctx  # Positive = context-free appears better
        safety_imp = safe_score - unsafe_score  # Positive = context-aware is actually safer
        
        surface_differences.append(surface_diff)
        safety_improvements.append(safety_imp)
    
    print("\n" + "="*70)
    print("EVALUATION WITH CONTEXT (Which is actually safer?)")
    print("="*70)
    print(f"Context-Free (unsafe) avg:  {np.mean(context_free_with_ctx):.2f}/5.0")
    print(f"Context-Aware (safe) avg:   {np.mean(context_aware_with_ctx):.2f}/5.0")
    print(f"Safety Improvement:         {np.mean(safety_improvements):+.2f}")
    print(f"  Std Dev:                  {np.std(safety_improvements):.2f}")
    print(f"  Min/Max:                  {min(safety_improvements):+.2f} / {max(safety_improvements):+.2f}")
    
    safer_with_context = sum(1 for s in safety_improvements if s > 0)
    print(f"\nContext-aware is safer in {safer_with_context}/{len(scenarios)} scenarios ({100*safer_with_context/len(scenarios):.1f}%)")
    
    print("\n" + "="*70)
    print("EVALUATION WITHOUT CONTEXT (Which appears better?)")
    print("="*70)
    print(f"Context-Free (unsafe) avg:  {np.mean(context_free_no_ctx):.2f}/5.0")
    print(f"Context-Aware (safe) avg:   {np.mean(context_aware_no_ctx):.2f}/5.0")
    print(f"Surface Appeal Difference:  {np.mean(surface_differences):+.2f}")
    print(f"  Std Dev:                  {np.std(surface_differences):.2f}")
    print(f"  Min/Max:                  {min(surface_differences):+.2f} / {max(surface_differences):+.2f}")
    
    appears_better_without_ctx = sum(1 for s in surface_differences if s > 0)
    print(f"\nContext-free APPEARS better in {appears_better_without_ctx}/{len(scenarios)} scenarios ({100*appears_better_without_ctx/len(scenarios):.1f}%)")
    
    print("\n" + "="*70)
    print("ADVERSARIAL QUALITY CHECK")
    print("="*70)
    
    # Check if truly adversarial
    truly_adversarial = sum(1 for i in range(len(scenarios)) 
                           if surface_differences[i] > 0 and safety_improvements[i] > 0)
    
    print(f"Truly adversarial scenarios: {truly_adversarial}/{len(scenarios)} ({100*truly_adversarial/len(scenarios):.1f}%)")
    print("  (Context-free appears better WITHOUT context")
    print("   AND context-aware is safer WITH context)")
    
    # Distribution analysis
    print("\n" + "="*70)
    print("SCORE DISTRIBUTIONS")
    print("="*70)
    
    print("\nSurface Appeal Difference (should be positive):")
    print(f"  Mean:   {np.mean(surface_differences):+.2f}")
    print(f"  Median: {np.median(surface_differences):+.2f}")
    print(f"  > 0:    {100*appears_better_without_ctx/len(scenarios):.1f}%")
    print(f"  > 0.5:  {100*sum(1 for s in surface_differences if s > 0.5)/len(scenarios):.1f}%")
    print(f"  > 1.0:  {100*sum(1 for s in surface_differences if s > 1.0)/len(scenarios):.1f}%")
    
    print("\nSafety Improvement (should be positive):")
    print(f"  Mean:   {np.mean(safety_improvements):+.2f}")
    print(f"  Median: {np.median(safety_improvements):+.2f}")
    print(f"  > 0:    {100*safer_with_context/len(scenarios):.1f}%")
    print(f"  > 0.5:  {100*sum(1 for s in safety_improvements if s > 0.5)/len(scenarios):.1f}%")
    print(f"  > 1.0:  {100*sum(1 for s in safety_improvements if s > 1.0)/len(scenarios):.1f}%")
    
    print("\n" + "="*70)
    print("ASSESSMENT")
    print("="*70)
    
    if np.mean(surface_differences) > 0 and np.mean(safety_improvements) > 0:
        print("✅ GOOD: Scenarios are adversarial!")
        print(f"   - Context-free appears {np.mean(surface_differences):.2f} points better without context")
        print(f"   - Context-aware is {np.mean(safety_improvements):.2f} points safer with context")
    elif np.mean(surface_differences) <= 0:
        print("❌ PROBLEM: Context-free doesn't appear better without context")
        print("   - Need to make context-free responses more appealing on surface")
    elif np.mean(safety_improvements) <= 0:
        print("❌ PROBLEM: Context-aware isn't actually safer with context")
        print("   - Need to make context-aware responses more safety-focused")
    
    # Domain breakdown
    print("\n" + "="*70)
    print("BREAKDOWN BY DOMAIN")
    print("="*70)
    
    domain_stats = defaultdict(lambda: {'surface_diff': [], 'safety_imp': []})
    for idx, scenario in enumerate(scenarios):
        domain = scenario.get('domain', 'Unknown')
        domain_stats[domain]['surface_diff'].append(surface_differences[idx])
        domain_stats[domain]['safety_imp'].append(safety_improvements[idx])
    
    for domain in sorted(domain_stats.keys()):
        stats = domain_stats[domain]
        print(f"\n{domain} (n={len(stats['surface_diff'])}):")
        print(f"  Surface Appeal:     {np.mean(stats['surface_diff']):+.2f}")
        print(f"  Safety Improvement: {np.mean(stats['safety_imp']):+.2f}")

def main():
    parser = argparse.ArgumentParser(
        description='Analyze dataset statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze both Phase 2 and Phase 3
  python analyze_dataset.py --all
  
  # Analyze only Phase 2
  python analyze_dataset.py --phase2
  
  # Analyze only Phase 3
  python analyze_dataset.py --phase3
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Analyze all phases')
    parser.add_argument('--phase2', action='store_true',
                       help='Analyze Phase 2 annotated scenarios')
    parser.add_argument('--phase3', action='store_true',
                       help='Analyze Phase 3 hard scenarios')
    
    args = parser.parse_args()
    
    # Default to all if nothing specified
    if not (args.all or args.phase2 or args.phase3):
        args.all = True
    
    try:
        if args.all or args.phase2:
            analyze_phase2()
        
        if args.all or args.phase3:
            analyze_phase3()
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70 + "\n")
    
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

