#!/bin/bash
set -e

# Set up environment
export CUDA_HOME=/usr/local/cuda
export FLASHINFER_CUDA_HOME=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export PYTHONPATH=/app:$PYTHONPATH

cd /app

# Login to wandb if API key is provided
if [ -n "$WANDB_API_KEY" ]; then
    echo "Logging into wandb..."
    wandb login "$WANDB_API_KEY"
fi

# If arguments provided, execute them as a command
if [ $# -gt 0 ]; then
    # Check if first argument is a training function
    case "$1" in
        reinforce|grpo|proma|proma_intra|proma_intra_eig|run-alg)
            source run/run.sh
            "$@"
            ;;
        bash)
            shift
            exec /bin/bash "$@"
            ;;
        *)
            exec "$@"
            ;;
    esac
else
    echo "proma2 Docker container ready."
    echo ""
    echo "Training commands (inject WANDB_API_KEY for logging):"
    echo "  docker run --gpus all --shm-size=16g -e WANDB_API_KEY=\$WANDB_API_KEY proma2 reinforce"
    echo "  docker run --gpus all --shm-size=16g -e WANDB_API_KEY=\$WANDB_API_KEY proma2 grpo"
    echo "  docker run --gpus all --shm-size=16g -e WANDB_API_KEY=\$WANDB_API_KEY proma2 proma"
    echo "  docker run --gpus all --shm-size=16g -e WANDB_API_KEY=\$WANDB_API_KEY proma2 proma_intra"
    echo "  docker run --gpus all --shm-size=16g -e WANDB_API_KEY=\$WANDB_API_KEY proma2 proma_intra_eig"
    echo ""
    echo "Run arbitrary commands:"
    echo "  docker run --gpus all proma2 python -c 'import verl; print(verl)'"
    echo "  docker run --gpus all -it proma2 bash"
    echo ""
    exec /bin/bash
fi
