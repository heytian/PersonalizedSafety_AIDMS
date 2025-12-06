#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
============================================================
| File: generate_expanded_dataset.py
| Description:
|     MASTER ORCHESTRATION SCRIPT
|     
|     Runs all 4 phases in sequence to generate the complete
|     expanded dataset:
|     
|     Phase 1: Template-based scenario expansion (100k-400k)
|     Phase 2: LLM self-annotation and scoring
|     Phase 3: Hard scenario generation via self-critique
|     Phase 4: Domain-breadth expansion via expert personas
|     
|     Final Output: Combined dataset with full annotations,
|                   multi-turn variants, and domain coverage
|     
|     Usage:
|       python generate_expanded_dataset.py --all
|       python generate_expanded_dataset.py --phase 1
|       python generate_expanded_dataset.py --phase 1,2,3
|       python generate_expanded_dataset.py --resume
============================================================
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from dataset_config import (
    OUTPUT_DIR, PHASE1_OUTPUT, PHASE2_OUTPUT, PHASE3_OUTPUT,
    PHASE4_OUTPUT, FINAL_OUTPUT
)

# Import phase modules
try:
    from phase1_template_expansion import run_phase1
    from phase2_self_annotation import run_phase2
    from phase3_hard_scenarios import run_phase3
    from phase4_domain_expansion import run_phase4
except ImportError as e:
    print(f"[ERROR] Failed to import phase modules: {e}")
    print("Make sure all phase scripts are in the same directory.")
    sys.exit(1)

# ============================================================
# Pipeline State Management
# ============================================================

STATE_FILE = f"{OUTPUT_DIR}/pipeline_state.json"

def load_pipeline_state() -> Dict[str, Any]:
    """Load pipeline execution state."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "completed_phases": [],
        "start_time": None,
        "last_phase_time": None,
        "phase_timings": {}
    }

def save_pipeline_state(state: Dict[str, Any]):
    """Save pipeline execution state."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def mark_phase_complete(phase_num: int, duration_seconds: float):
    """Mark a phase as completed."""
    state = load_pipeline_state()
    if phase_num not in state["completed_phases"]:
        state["completed_phases"].append(phase_num)
    state["completed_phases"].sort()
    state["last_phase_time"] = datetime.now().isoformat()
    state["phase_timings"][f"phase{phase_num}"] = duration_seconds
    save_pipeline_state(state)

# ============================================================
# Phase Execution with Error Handling
# ============================================================

def execute_phase1(force: bool = False):
    """Execute Phase 1 with error handling."""
    state = load_pipeline_state()
    
    if 1 in state["completed_phases"] and not force:
        print("\n[SKIP] Phase 1 already completed. Use --force to re-run.")
        return True
    
    print("\n" + "="*70)
    print("EXECUTING PHASE 1: TEMPLATE-BASED SCENARIO EXPANSION")
    print("="*70)
    
    start_time = time.time()
    try:
        run_phase1()
        duration = time.time() - start_time
        mark_phase_complete(1, duration)
        print(f"\n✓ Phase 1 completed in {duration/60:.1f} minutes")
        return True
    except Exception as e:
        print(f"\n✗ Phase 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def execute_phase2(force: bool = False):
    """Execute Phase 2 with error handling."""
    state = load_pipeline_state()
    
    if 2 in state["completed_phases"] and not force:
        print("\n[SKIP] Phase 2 already completed. Use --force to re-run.")
        return True
    
    if not os.path.exists(PHASE1_OUTPUT):
        print("\n✗ Phase 2 requires Phase 1 output. Run Phase 1 first.")
        return False
    
    print("\n" + "="*70)
    print("EXECUTING PHASE 2: LLM SELF-ANNOTATION")
    print("="*70)
    
    start_time = time.time()
    try:
        run_phase2()
        duration = time.time() - start_time
        mark_phase_complete(2, duration)
        print(f"\n✓ Phase 2 completed in {duration/60:.1f} minutes")
        return True
    except Exception as e:
        print(f"\n✗ Phase 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def execute_phase3(force: bool = False):
    """Execute Phase 3 with error handling."""
    state = load_pipeline_state()
    
    if 3 in state["completed_phases"] and not force:
        print("\n[SKIP] Phase 3 already completed. Use --force to re-run.")
        return True
    
    if not os.path.exists(PHASE2_OUTPUT):
        print("\n✗ Phase 3 requires Phase 2 output. Run Phase 2 first.")
        return False
    
    print("\n" + "="*70)
    print("EXECUTING PHASE 3: HARD SCENARIO GENERATION")
    print("="*70)
    
    start_time = time.time()
    try:
        run_phase3()
        duration = time.time() - start_time
        mark_phase_complete(3, duration)
        print(f"\n✓ Phase 3 completed in {duration/60:.1f} minutes")
        return True
    except Exception as e:
        print(f"\n✗ Phase 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def execute_phase4(force: bool = False):
    """Execute Phase 4 with error handling."""
    state = load_pipeline_state()
    
    if 4 in state["completed_phases"] and not force:
        print("\n[SKIP] Phase 4 already completed. Use --force to re-run.")
        return True
    
    print("\n" + "="*70)
    print("EXECUTING PHASE 4: DOMAIN-BREADTH EXPANSION")
    print("="*70)
    
    start_time = time.time()
    try:
        run_phase4()
        duration = time.time() - start_time
        mark_phase_complete(4, duration)
        print(f"\n✓ Phase 4 completed in {duration/60:.1f} minutes")
        return True
    except Exception as e:
        print(f"\n✗ Phase 4 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================
# Final Dataset Combination
# ============================================================

def combine_all_phases():
    """Combine outputs from all phases into final dataset."""
    print("\n" + "="*70)
    print("COMBINING ALL PHASES INTO FINAL DATASET")
    print("="*70)
    
    final_dataset = {
        "metadata": {
            "generation_date": datetime.now().isoformat(),
            "phases_included": [],
            "total_scenarios": 0,
            "breakdown": {}
        },
        "phase1_scenarios": [],
        "phase2_annotated": {},
        "phase3_hard_scenarios": [],
        "phase4_domain_scenarios": []
    }
    
    # Load Phase 1
    if os.path.exists(PHASE1_OUTPUT):
        print(f"\n✓ Loading Phase 1 data...")
        with open(PHASE1_OUTPUT, "r", encoding="utf-8") as f:
            phase1_data = json.load(f)
        final_dataset["phase1_scenarios"] = phase1_data
        final_dataset["metadata"]["phases_included"].append(1)
        final_dataset["metadata"]["breakdown"]["phase1"] = len(phase1_data)
        print(f"  - {len(phase1_data):,} template-based scenarios")
    
    # Load Phase 2
    if os.path.exists(PHASE2_OUTPUT):
        print(f"\n✓ Loading Phase 2 data...")
        with open(PHASE2_OUTPUT, "r", encoding="utf-8") as f:
            phase2_data = json.load(f)
        final_dataset["phase2_annotated"] = phase2_data
        final_dataset["metadata"]["phases_included"].append(2)
        total_p2 = phase2_data["metadata"]["total_scenarios"]
        final_dataset["metadata"]["breakdown"]["phase2_total"] = total_p2
        final_dataset["metadata"]["breakdown"]["phase2_good"] = phase2_data["metadata"]["good_scenarios"]
        print(f"  - {total_p2:,} annotated scenarios")
        print(f"  - {phase2_data['metadata']['good_scenarios']:,} good quality")
    
    # Load Phase 3
    if os.path.exists(PHASE3_OUTPUT):
        print(f"\n✓ Loading Phase 3 data...")
        with open(PHASE3_OUTPUT, "r", encoding="utf-8") as f:
            phase3_data = json.load(f)
        final_dataset["phase3_hard_scenarios"] = phase3_data["hard_scenarios"]
        final_dataset["metadata"]["phases_included"].append(3)
        final_dataset["metadata"]["breakdown"]["phase3"] = len(phase3_data["hard_scenarios"])
        print(f"  - {len(phase3_data['hard_scenarios']):,} hard adversarial scenarios")
    
    # Load Phase 4
    if os.path.exists(PHASE4_OUTPUT):
        print(f"\n✓ Loading Phase 4 data...")
        with open(PHASE4_OUTPUT, "r", encoding="utf-8") as f:
            phase4_data = json.load(f)
        final_dataset["phase4_domain_scenarios"] = phase4_data["all_scenarios"]
        final_dataset["metadata"]["phases_included"].append(4)
        final_dataset["metadata"]["breakdown"]["phase4"] = len(phase4_data["all_scenarios"])
        print(f"  - {len(phase4_data['all_scenarios']):,} domain-specific scenarios")
    
    # Calculate total
    total = sum(final_dataset["metadata"]["breakdown"].values())
    final_dataset["metadata"]["total_scenarios"] = total
    
    # Save combined dataset
    print(f"\n✓ Saving combined dataset to {FINAL_OUTPUT}...")
    with open(FINAL_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*70)
    print("FINAL DATASET SUMMARY")
    print("="*70)
    for key, value in final_dataset["metadata"]["breakdown"].items():
        print(f"  {key:30s}: {value:10,} scenarios")
    print("="*70)
    print(f"  {'TOTAL':30s}: {total:10,} scenarios")
    print("="*70)
    
    return final_dataset

# ============================================================
# Command Line Interface
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate expanded personalized safety dataset across 4 phases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_expanded_dataset.py --all              # Run all phases
  python generate_expanded_dataset.py --phase 1          # Run only Phase 1
  python generate_expanded_dataset.py --phase 1,2        # Run Phases 1 and 2
  python generate_expanded_dataset.py --resume           # Resume from last phase
  python generate_expanded_dataset.py --combine          # Just combine existing outputs
  python generate_expanded_dataset.py --status           # Show pipeline status
  python generate_expanded_dataset.py --force --phase 1  # Force re-run Phase 1
        """
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all 4 phases in sequence"
    )
    
    parser.add_argument(
        "--phase",
        type=str,
        help="Run specific phase(s), e.g. '1' or '1,2,3'"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last completed phase"
    )
    
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Only combine existing phase outputs into final dataset"
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current pipeline status"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run of phases (ignore completion state)"
    )
    
    args = parser.parse_args()
    
    # Show status
    if args.status:
        state = load_pipeline_state()
        print("\n" + "="*70)
        print("PIPELINE STATUS")
        print("="*70)
        print(f"Completed phases: {state['completed_phases']}")
        print(f"Last run: {state.get('last_phase_time', 'Never')}")
        if state.get("phase_timings"):
            print("\nPhase timings:")
            for phase, duration in state["phase_timings"].items():
                print(f"  {phase}: {duration/60:.1f} minutes")
        print("="*70)
        return
    
    # Just combine
    if args.combine:
        combine_all_phases()
        return
    
    # Initialize state
    state = load_pipeline_state()
    if not state["start_time"]:
        state["start_time"] = datetime.now().isoformat()
        save_pipeline_state(state)
    
    # Determine which phases to run
    phases_to_run = []
    
    if args.all:
        phases_to_run = [1, 2, 3, 4]
    elif args.phase:
        phases_to_run = [int(p.strip()) for p in args.phase.split(",")]
    elif args.resume:
        completed = state["completed_phases"]
        if not completed:
            phases_to_run = [1, 2, 3, 4]
        else:
            next_phase = max(completed) + 1
            if next_phase <= 4:
                phases_to_run = list(range(next_phase, 5))
            else:
                print("\n✓ All phases already completed!")
                combine_all_phases()
                return
    else:
        parser.print_help()
        return
    
    print("\n" + "="*70)
    print("EXPANDED DATASET GENERATION PIPELINE")
    print("="*70)
    print(f"Phases to execute: {phases_to_run}")
    print(f"Force re-run: {args.force}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*70)
    
    # Execute phases
    pipeline_start = time.time()
    
    for phase in phases_to_run:
        if phase == 1:
            success = execute_phase1(force=args.force)
        elif phase == 2:
            success = execute_phase2(force=args.force)
        elif phase == 3:
            success = execute_phase3(force=args.force)
        elif phase == 4:
            success = execute_phase4(force=args.force)
        else:
            print(f"\n✗ Invalid phase number: {phase}")
            success = False
        
        if not success:
            print(f"\n✗ Pipeline stopped at Phase {phase} due to error.")
            sys.exit(1)
    
    # Combine all phases
    print("\n")
    combine_all_phases()
    
    # Final summary
    pipeline_duration = time.time() - pipeline_start
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"Total execution time: {pipeline_duration/60:.1f} minutes")
    print(f"Final dataset saved to: {FINAL_OUTPUT}")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review the dataset quality")
    print("  2. Run analysis on scenario distributions")
    print("  3. Use for model training/evaluation")
    print("="*70 + "\n")

# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()

