#!/bin/bash
# Backend startup script

# Activate virtual environment
source venv/bin/activate

# Create models cache directory if it doesn't exist
mkdir -p models_cache

# Start the server (production mode - no auto-reload)
echo "🚀 Starting RAG Chatbot server..."
uvicorn main:app --port 8002 --host 0.0.0.0
