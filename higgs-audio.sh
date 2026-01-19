#!/bin/bash
#
# Higgs Audio OpenAI-compatible TTS Server
#
# Usage:
#   ./higgs-audio.sh              # Start with defaults
#   ./higgs-audio.sh --port 8080  # Custom port
#
# Environment variables:
#   HIGGS_PORT          - Server port (default: 8005)
#   HIGGS_VOICES_DIR    - Voice prompts directory (default: ./voices)
#   HIGGS_MODEL_PATH    - Model path (default: bosonai/higgs-audio-v2-generation-3B-base)
#   HIGGS_IDLE_TIMEOUT  - Idle timeout in seconds (default: 300)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
PORT="${HIGGS_PORT:-8005}"
VOICES_DIR="${HIGGS_VOICES_DIR:-$SCRIPT_DIR/voices}"
MODEL_PATH="${HIGGS_MODEL_PATH:-/mnt/fileshare2/AI/models/higgs-audio-v2-generation-3B-base}"
IDLE_TIMEOUT="${HIGGS_IDLE_TIMEOUT:-300}"
HF_HOME="${HF_HOME:-/mnt/cache/huggingface}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --voices-dir)
            VOICES_DIR="$2"
            shift 2
            ;;
        --model-path|--local-dir)
            MODEL_PATH="$2"
            shift 2
            ;;
        --idle-timeout)
            IDLE_TIMEOUT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Higgs Audio OpenAI-compatible TTS Server"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --port PORT           Server port (default: 8005)"
            echo "  --voices-dir DIR      Voice prompts directory (default: ./voices)"
            echo "  --model-path PATH     Model path or HuggingFace ID"
            echo "  --local-dir DIR       Alias for --model-path"
            echo "  --idle-timeout SECS   Unload model after N seconds idle (default: 300)"
            echo "  -h, --help            Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  HIGGS_PORT, HIGGS_VOICES_DIR, HIGGS_MODEL_PATH, HIGGS_IDLE_TIMEOUT"
            echo ""
            echo "Examples:"
            echo "  $0                           # Start with defaults"
            echo "  $0 --port 8080               # Custom port"
            echo "  CUDA_VISIBLE_DEVICES=1 $0    # Use specific GPU"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Export environment variables
export HIGGS_PORT="$PORT"
export HIGGS_VOICES_DIR="$VOICES_DIR"
export HIGGS_MODEL_PATH="$MODEL_PATH"
export HIGGS_IDLE_TIMEOUT="$IDLE_TIMEOUT"
export HF_HOME="$HF_HOME"

# Check if voices directory exists
if [[ ! -d "$VOICES_DIR" ]]; then
    echo "Warning: Voices directory not found: $VOICES_DIR"
    echo "Create the directory and add .wav + .txt pairs for voice cloning."
fi

# Activate virtual environment if it exists
if [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
elif [[ -f "$SCRIPT_DIR/higgs_audio_env/bin/activate" ]]; then
    source "$SCRIPT_DIR/higgs_audio_env/bin/activate"
elif [[ -d "$SCRIPT_DIR/conda_env" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$SCRIPT_DIR/conda_env"
fi

echo "=========================================="
echo "Higgs Audio TTS Server"
echo "=========================================="
echo "Port:         $PORT"
echo "Voices Dir:   $VOICES_DIR"
echo "Model:        $MODEL_PATH"
echo "Idle Timeout: ${IDLE_TIMEOUT}s"
echo "HF Cache:     $HF_HOME"
echo "=========================================="
echo ""

# Start the server
exec python -m boson_multimodal.serve.openai_tts_server
