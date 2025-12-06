#!/bin/bash
# Script to regenerate Phase 1 with unique attributes per scenario

echo "=========================================="
echo "Regenerating Phase 1 with Unique Attributes"
echo "=========================================="
echo ""

# Step 1: Clear Python cache
echo "Step 1: Clearing Python cache..."
rm -rf __pycache__
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
echo "✓ Cache cleared"
echo ""

# Step 2: Backup existing data (optional)
if [ -f "expanded_dataset/phase1_synthetic_scenarios.json" ]; then
    echo "Step 2: Backing up existing Phase 1 data..."
    timestamp=$(date +%Y%m%d_%H%M%S)
    cp "expanded_dataset/phase1_synthetic_scenarios.json" "expanded_dataset/phase1_synthetic_scenarios_backup_${timestamp}.json"
    echo "✓ Backup created: phase1_synthetic_scenarios_backup_${timestamp}.json"
    echo ""
fi

# Step 3: Regenerate Phase 1
echo "Step 3: Regenerating Phase 1 with unique attributes..."
echo "This will take approximately 6-10 minutes..."
echo ""

python3 generate_expanded_dataset.py --force --phase 1

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Phase 1 Regeneration Complete!"
    echo "=========================================="
    echo ""
    
    # Step 4: Analyze diversity
    echo "Step 4: Analyzing new diversity..."
    echo ""
    python3 analyze_diversity.py
    
    echo ""
    echo "=========================================="
    echo "Expected Improvements:"
    echo "=========================================="
    echo "  Attribute diversity: 0.1% → 99.8%"
    echo "  Overall score: 74.9% → 99.8%"
    echo "  Unique attribute sets: ~1,050 → ~10,500"
    echo "=========================================="
    echo ""
else
    echo ""
    echo "❌ Regeneration failed!"
    echo "Check error messages above."
    exit 1
fi



