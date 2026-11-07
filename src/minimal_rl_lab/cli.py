"""One lazy CLI; argument inspection does not initialize Torch, models or directories."""
import argparse
import importlib
import os
from .config import Workspace, load_config

def build_parser():
    parser = argparse.ArgumentParser(prog='rl-lab', description='Minimal PPO/GRPO/DPO lab')
    parser.add_argument(
        '--workspace',
        default='./workspace',
        help='Runtime root; default: ./workspace',
    )
    parser.add_argument('--config', help='Optional JSON experiment configuration')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('download', help='Fetch the fixed model and dataset revisions')
    commands.add_parser('prepare-data', help='Filter and split GSM8K using the v2 prompt')
    commands.add_parser('prepare-preferences', help='Generate training-only DPO preferences')
    train = commands.add_parser('train', help='Train a fresh adapter')
    train.add_argument('method', choices=['ppo', 'grpo', 'dpo'])
    train.add_argument('--mode', required=True, choices=['smoke', 'full'])
    train.add_argument('--out', help='Output directory relative to the workspace')
    train.add_argument('--pairs', help='DPO pair file relative to the workspace')
    evaluate = commands.add_parser('evaluate', help='Greedy evaluation; no adapter means base model')
    evaluate.add_argument('--split', default='test', choices=['test', 'dev'])
    evaluate.add_argument('--limit', type=int, default=0, help='0 uses the full split')
    evaluate.add_argument('--adapter', help='Adapter directory relative to the workspace')
    evaluate.add_argument('--out', required=True, help='Output directory relative to the workspace')
    summary = commands.add_parser('summarize', help='Recompute results and check matching ID order')
    summary.add_argument(
        '--runs',
        nargs='+',
        required=True,
        help='Evaluation directories relative to workspace',
    )
    summary.add_argument('--out', default='results/metrics.csv')
    return parser

def main(argv=None):
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    paths = Workspace.create_paths(args.workspace, config)
    os.environ['HF_HOME'] = str(paths.cache / 'huggingface')
    if getattr(args, 'out', None) and args.command != 'summarize':
        args.out = str(paths.within(args.out))
    if args.command == 'train':
        args.pairs = str(paths.within(args.pairs)) if args.pairs else str(paths.data / 'v2' / 'dpo_pairs.jsonl')
        target = 'training.' + args.method
    elif args.command == 'evaluate':
        args.splits_dir = str(paths.data / 'splits_v2')
        args.protocol = 'v2'
        if args.adapter:
            args.adapter = str(paths.within(args.adapter))
        target = 'evaluation'
    else:
        target = {
            'download': 'data.download',
            'prepare-data': 'data.splits',
            'prepare-preferences': 'data.preferences',
            'summarize': 'summary',
        }[args.command]
    module = importlib.import_module('minimal_rl_lab.' + target)
    return module.run(config, paths, args)
