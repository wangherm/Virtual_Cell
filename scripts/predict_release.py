"""Predict using a checksum-verified published research model; no LLM API calls."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    qwen = commands.add_parser('qwen', help='Use the released adapter with a verified offline base snapshot')
    method = commands.add_parser('method', help='Refit a frozen algorithm in the protected Linux worker')
    for sub in (qwen, method):
        sub.add_argument('--release-dir', required=True)
        sub.add_argument('--input', required=True)
        sub.add_argument('--output', required=True)
        sub.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    qwen.add_argument('--model-dir', required=True)
    qwen.add_argument('--batch-size', type=int, default=16)
    qwen.add_argument('--dtype', choices=['float32', 'bfloat16'])
    method.add_argument('--model', required=True, help='Exact manifest model ID or SHA-256')
    method.add_argument('--steps', type=int, default=500)
    method.add_argument('--seconds', type=int, default=120)
    args = parser.parse_args()
    from vcell.release_inference import qwen_predict, method_predict
    if args.command == 'qwen':
        result = qwen_predict(args.release_dir, args.input, args.model_dir, args.output,
                              args.device, args.batch_size, args.dtype)
    else:
        result = method_predict(args.release_dir, args.model, args.input, args.output,
                                args.device, args.steps, args.seconds)
    print('PREDICTION:', result)


if __name__ == '__main__':
    main()
