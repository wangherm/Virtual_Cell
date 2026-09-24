"""Check the AutoDL PyTorch/CUDA environment without starting training."""
import sys


def main():
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed in this Python environment.", file=sys.stderr)
        return 1
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("CUDA is required for the GPU smoke test.", file=sys.stderr)
        return 1
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
