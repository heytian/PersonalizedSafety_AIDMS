#!/bin/bash
# Setup script for creating .env file

echo "=========================================="
echo "Environment Setup for Dataset Generation"
echo "=========================================="
echo ""

# Check if .env already exists
if [ -f .env ]; then
    echo "⚠️  .env file already exists!"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing .env file."
        exit 0
    fi
fi

# Copy template
if [ -f env.template ]; then
    cp env.template .env
    echo "✓ Created .env file from template"
else
    echo "✗ env.template not found!"
    exit 1
fi

echo ""
echo "Select your LLM backend:"
echo "1) OpenAI"
echo "2) Azure OpenAI"
echo "3) Offline/Dummy mode (for testing)"
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        # OpenAI setup
        read -p "Enter your OpenAI API key: " api_key
        read -p "Enter model name (default: gpt-4o-mini): " model
        model=${model:-gpt-4o-mini}
        
        # Update .env file
        sed -i.bak "s/USE_API=true/USE_API=true/" .env
        sed -i.bak "s/LLM_BACKEND=openai/LLM_BACKEND=openai/" .env
        sed -i.bak "s/OPENAI_API_KEY=your-openai-api-key-here/OPENAI_API_KEY=$api_key/" .env
        sed -i.bak "s/OPENAI_MODEL=gpt-4o-mini/OPENAI_MODEL=$model/" .env
        rm .env.bak 2>/dev/null
        
        echo ""
        echo "✓ OpenAI configuration saved!"
        ;;
    2)
        # Azure OpenAI setup
        read -p "Enter your Azure OpenAI API key: " api_key
        read -p "Enter your Azure endpoint (e.g., https://xxx.openai.azure.com/): " endpoint
        read -p "Enter deployment name (default: gpt-4o): " deployment
        deployment=${deployment:-gpt-4o}
        
        # Update .env file
        sed -i.bak "s/USE_API=true/USE_API=true/" .env
        sed -i.bak "s/LLM_BACKEND=openai/LLM_BACKEND=azure/" .env
        sed -i.bak "s|AZURE_OPENAI_API_KEY=your-azure-api-key-here|AZURE_OPENAI_API_KEY=$api_key|" .env
        sed -i.bak "s|AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/|AZURE_OPENAI_ENDPOINT=$endpoint|" .env
        sed -i.bak "s/AZURE_OPENAI_DEPLOYMENT=gpt-4o/AZURE_OPENAI_DEPLOYMENT=$deployment/" .env
        rm .env.bak 2>/dev/null
        
        echo ""
        echo "✓ Azure OpenAI configuration saved!"
        ;;
    3)
        # Offline mode
        sed -i.bak "s/USE_API=true/USE_API=false/" .env
        rm .env.bak 2>/dev/null
        
        echo ""
        echo "✓ Offline/Dummy mode configured!"
        echo "  (No API calls will be made - dummy data will be generated)"
        ;;
    *)
        echo "Invalid choice. Please edit .env manually."
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Your .env file is ready. You can now run:"
echo "  python generate_expanded_dataset.py --all"
echo ""
echo "To edit configuration later, edit the .env file directly."
echo ""



