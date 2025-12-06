#!/bin/bash
# Quick diagnostic script to check API configuration

echo "========================================"
echo "API Configuration Diagnostic"
echo "========================================"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ PROBLEM: .env file not found"
    echo "   FIX: cp env.template .env"
    exit 1
fi

echo "✓ .env file exists"
echo ""

# Check for required variables
echo "Checking configuration..."
echo ""

if grep -q "^LLM_BACKEND=" .env 2>/dev/null; then
    BACKEND=$(grep "^LLM_BACKEND=" .env | cut -d'=' -f2)
    echo "✓ LLM_BACKEND = $BACKEND"
else
    echo "❌ LLM_BACKEND not set in .env"
    exit 1
fi

if [ "$BACKEND" = "openai" ]; then
    if grep -q "^OPENAI_API_KEY=" .env 2>/dev/null; then
        KEY=$(grep "^OPENAI_API_KEY=" .env | cut -d'=' -f2)
        if [ "$KEY" = "your-openai-api-key-here" ] || [ -z "$KEY" ]; then
            echo "❌ OPENAI_API_KEY is not set (still has placeholder)"
            echo "   FIX: Edit .env and add your actual API key"
            exit 1
        else
            echo "✓ OPENAI_API_KEY is set (${KEY:0:10}...)"
        fi
    else
        echo "❌ OPENAI_API_KEY not found in .env"
        exit 1
    fi
    
    if grep -q "^OPENAI_MODEL=" .env 2>/dev/null; then
        MODEL=$(grep "^OPENAI_MODEL=" .env | cut -d'=' -f2)
        echo "✓ OPENAI_MODEL = $MODEL"
    else
        echo "⚠️  OPENAI_MODEL not set (will use default)"
    fi
fi

if [ "$BACKEND" = "azure" ]; then
    if grep -q "^AZURE_OPENAI_API_KEY=" .env 2>/dev/null; then
        KEY=$(grep "^AZURE_OPENAI_API_KEY=" .env | cut -d'=' -f2)
        if [ "$KEY" = "your-azure-api-key-here" ] || [ -z "$KEY" ]; then
            echo "❌ AZURE_OPENAI_API_KEY is not set (still has placeholder)"
            exit 1
        else
            echo "✓ AZURE_OPENAI_API_KEY is set"
        fi
    else
        echo "❌ AZURE_OPENAI_API_KEY not found in .env"
        exit 1
    fi
fi

echo ""
echo "========================================"
echo "✅ Configuration looks valid!"
echo "========================================"
echo ""
echo "If you're still getting generic queries,"
echo "make sure you're running with the venv:"
echo "  source venv/bin/activate"
echo "  python3 generate_expanded_dataset.py --force --phase 1"



