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
        # Math benchmarks (GSM8K train -> SVAMP/GSM8K val)
        reinforce|grpo|proma|proma_intra|proma_intra_eig|gsm8k_reinforce|gsm8k_grpo|gsm8k_proma)
            source run/run_svamp.sh
            "$@"
            ;;
        # Code benchmarks (MBPP train -> HumanEval val)
        code_reinforce|code_grpo|code_proma|code_proma_intra|code_proma_intra_eig|humaneval_reinforce|humaneval_grpo|humaneval_proma|test_run)
            source run/run_humaneval.sh
            # Strip "code_" prefix if present
            cmd="$1"
            shift
            case "$cmd" in
                code_*) cmd="${cmd#code_}" ;;
            esac
            "$cmd" "$@"
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
    echo "=== Math benchmarks (GSM8K -> SVAMP OOD val) ==="
    echo "  docker run --gpus all --shm-size=16g -e WANDB_API_KEY=\$WANDB_API_KEY proma2 reinforce"
    echo "  docker run --gpus all --shm-size=16g -e WANDB_API_KEY=\$WANDB_API_KEY proma2 grpo"
    echo "  docker run --gpus all --shm-size=16g -e WANDB_API_KEY=\$WANDB_API_KEY proma2 proma"
    echo "  docker run --gpus all --shm-size=16g -e WANDB_API_KEY=\$WANDB_API_KEY proma2 proma_intra"
    echo "  docker run --gpus all --shm-size=16g -e WANDB_API_KEY=\$WANDB_API_KEY proma2 proma_intra_eig"
    echo ""
    echo "=== Code benchmarks (MBPP -> HumanEval OOD val) ==="
    echo "  docker run --gpus all --shm-size=16g -e WANDB_API_KEY=\$WANDB_API_KEY proma2 code_reinforce"
    echo "  docker run --gpus all --shm-size=16g -e WANDB_API_KEY=\$WANDB_API_KEY proma2 code_grpo"
    echo "  docker run --gpus all --shm-size=16g -e WANDB_API_KEY=\$WANDB_API_KEY proma2 code_proma"
    echo "  docker run --gpus all --shm-size=16g -e WANDB_API_KEY=\$WANDB_API_KEY proma2 code_proma_intra"
    echo "  docker run --gpus all --shm-size=16g -e WANDB_API_KEY=\$WANDB_API_KEY proma2 code_proma_intra_eig"
    echo ""
    echo "Run arbitrary commands:"
    echo "  docker run --gpus all proma2 python -c 'import verl; print(verl)'"
    echo "  docker run --gpus all -it proma2 bash"
    echo ""
    exec /bin/bash
fi
