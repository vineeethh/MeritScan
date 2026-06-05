#!/bin/bash
# MeritScan pipeline with Ollama embeddings

set -e

OLLAMA_BIN="/Applications/Ollama.app/Contents/MacOS/ollama"
JD_FILE="${1:-data/UBS_Job_Description.txt}"
RESUMES_DIR="${2:-data/resumes/}"
OUTPUT_FILE="${3:-data/results/ubs_screening_results.json}"

echo "========================================="
echo "  MeritScan with Ollama Embeddings"
echo "========================================="
echo ""

# Check if Ollama binary exists
if [ ! -f "$OLLAMA_BIN" ]; then
    echo "❌ Ollama not found at $OLLAMA_BIN"
    echo "Please install Ollama from https://ollama.ai"
    exit 1
fi

# Start Ollama server if not already running
if ! pgrep -x "ollama" > /dev/null; then
    echo "📦 Starting Ollama server..."
    "$OLLAMA_BIN" serve > /tmp/ollama.log 2>&1 &
    OLLAMA_PID=$!
    sleep 3

    if ! ps -p $OLLAMA_PID > /dev/null; then
        echo "❌ Failed to start Ollama server"
        cat /tmp/ollama.log
        exit 1
    fi
    echo "✓ Ollama server started (PID: $OLLAMA_PID)"
else
    echo "✓ Ollama server already running"
fi

# Check if model is available
echo ""
echo "📥 Checking mistral-embed model..."
if ! "$OLLAMA_BIN" list | grep -q "mistral-embed"; then
    echo "📦 Pulling mistral-embed (this may take a few minutes)..."
    "$OLLAMA_BIN" pull mistral-embed
fi

echo "✓ Model ready"
echo ""

# Check if files exist
if [ ! -f "$JD_FILE" ]; then
    echo "❌ Job description file not found: $JD_FILE"
    exit 1
fi

if [ ! -d "$RESUMES_DIR" ]; then
    echo "❌ Resumes directory not found: $RESUMES_DIR"
    exit 1
fi

# Run the pipeline
echo "🚀 Starting MeritScan pipeline..."
echo ""

python3 main.py --jd "$JD_FILE" --resumes "$RESUMES_DIR" --output "$OUTPUT_FILE"

echo ""
echo "✓ Pipeline complete!"
echo "Results saved to: $OUTPUT_FILE"
