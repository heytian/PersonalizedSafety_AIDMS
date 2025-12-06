#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze diversity of Phase 1 generated scenarios
"""

import json
from collections import defaultdict, Counter
from typing import Dict, List

def load_dataset(filepath: str = "expanded_dataset/phase1_synthetic_scenarios.json"):
    """Load Phase 1 scenarios."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def analyze_diversity(scenarios: List[Dict]):
    """Comprehensive diversity analysis."""
    
    print("\n" + "="*80)
    print("PHASE 1 DIVERSITY ANALYSIS")
    print("="*80)
    
    # Basic stats
    total = len(scenarios)
    print(f"\n📊 Total Scenarios: {total:,}")
    
    # Domain distribution
    print("\n" + "-"*80)
    print("🌍 DOMAIN DISTRIBUTION")
    print("-"*80)
    domains = Counter(s["domain"] for s in scenarios)
    for domain, count in sorted(domains.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / total
        bar = "█" * int(pct / 2)
        print(f"  {domain:20s}: {count:6,} ({pct:5.1f}%) {bar}")
    
    # Template distribution
    print("\n" + "-"*80)
    print("📋 TEMPLATE DISTRIBUTION")
    print("-"*80)
    templates = Counter(s["template"][:80] + "..." for s in scenarios)
    print(f"  Unique templates: {len(templates)}")
    for template, count in list(templates.most_common(10)):
        print(f"  {count:6,}x: {template}")
    
    # Query diversity
    print("\n" + "-"*80)
    print("💬 QUERY DIVERSITY")
    print("-"*80)
    queries = [s["query"] for s in scenarios]
    unique_queries = set(queries)
    query_counts = Counter(queries)
    
    print(f"  Total queries:         {len(queries):,}")
    print(f"  Unique queries:        {len(unique_queries):,}")
    print(f"  Diversity ratio:       {len(unique_queries)/len(queries):.2%}")
    print(f"  Most common query:     '{list(query_counts.most_common(1))[0][0][:80]}...'")
    print(f"  Times repeated:        {list(query_counts.most_common(1))[0][1]:,}x")
    
    print("\n  Top 10 most common queries:")
    for query, count in query_counts.most_common(10):
        print(f"    {count:6,}x: {query[:70]}...")
    
    # Attribute diversity
    print("\n" + "-"*80)
    print("👤 ATTRIBUTE DIVERSITY")
    print("-"*80)
    
    attribute_stats = defaultdict(lambda: defaultdict(int))
    
    for scenario in scenarios:
        attrs = scenario.get("attributes", {})
        for attr_name, attr_value in attrs.items():
            attribute_stats[attr_name][attr_value] += 1
    
    for attr_name in sorted(attribute_stats.keys()):
        values = attribute_stats[attr_name]
        unique_count = len(values)
        print(f"\n  {attr_name}:")
        print(f"    Unique values: {unique_count}")
        
        # Show distribution
        for value, count in sorted(values.items(), key=lambda x: x[1], reverse=True):
            pct = 100 * count / total
            bar = "▓" * int(pct / 5)
            print(f"      {value:30s}: {count:6,} ({pct:5.1f}%) {bar}")
    
    # Query length distribution
    print("\n" + "-"*80)
    print("📏 QUERY LENGTH DISTRIBUTION")
    print("-"*80)
    
    query_lengths = [len(s["query"]) for s in scenarios]
    avg_length = sum(query_lengths) / len(query_lengths)
    min_length = min(query_lengths)
    max_length = max(query_lengths)
    
    print(f"  Average length: {avg_length:.1f} characters")
    print(f"  Min length:     {min_length} characters")
    print(f"  Max length:     {max_length} characters")
    
    # Length buckets
    length_buckets = defaultdict(int)
    for length in query_lengths:
        bucket = (length // 20) * 20
        length_buckets[bucket] += 1
    
    print("\n  Length distribution:")
    for bucket in sorted(length_buckets.keys()):
        count = length_buckets[bucket]
        pct = 100 * count / total
        bar = "█" * int(pct / 2)
        print(f"    {bucket:3d}-{bucket+19:3d} chars: {count:6,} ({pct:5.1f}%) {bar}")
    
    # Diversity score
    print("\n" + "-"*80)
    print("🎯 OVERALL DIVERSITY SCORE")
    print("-"*80)
    
    query_diversity = len(unique_queries) / len(queries)
    domain_diversity = len(domains) / 7  # 7 total domains
    
    # Calculate attribute diversity (average uniqueness per attribute)
    attr_diversity_scores = []
    for attr_name, values in attribute_stats.items():
        attr_diversity_scores.append(len(values) / total)
    avg_attr_diversity = sum(attr_diversity_scores) / len(attr_diversity_scores)
    
    overall_score = (query_diversity * 0.5 + domain_diversity * 0.25 + avg_attr_diversity * 0.25)
    
    print(f"  Query diversity:     {query_diversity:.1%}")
    print(f"  Domain diversity:    {domain_diversity:.1%}")
    print(f"  Attribute diversity: {avg_attr_diversity:.1%}")
    print(f"  Overall score:       {overall_score:.1%}")
    
    if query_diversity < 0.5:
        print("\n  ⚠️  WARNING: Low query diversity detected!")
        print("  Expected: Each scenario should have a unique, context-specific query")
        print("  Action: Check if API was properly configured during generation")
    
    print("\n" + "="*80)

def main():
    scenarios = load_dataset()
    analyze_diversity(scenarios)

if __name__ == "__main__":
    main()

