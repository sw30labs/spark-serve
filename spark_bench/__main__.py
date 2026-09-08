"""Run `python -m spark_bench --help` on the Spark or in synthetic mode locally."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('concurrency', help='isolated per-node YuE container concurrency sweep')
    run.add_argument('--workload', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--concurrency', default='1,2,3,4')
    run.add_argument('--jobs-per-level', type=int, default=8)
    run.add_argument('--iterations', type=int, default=2)
    run.add_argument('--warmup', type=int, default=1)
    run.add_argument('--cooldown', type=float, default=60)
    run.add_argument('--timeout', type=float, default=14400)
    run.add_argument('--wait-idle', type=float, default=14400)
    run.add_argument('--seed', type=int, default=42)
    run.add_argument('--telemetry-interval', type=float, default=2)
    run.add_argument('--factory-root', type=Path, default=Path('~/.local/share/artist-twin/yue-factory'))
    run.add_argument('--service', default='yue-icl.service')
    run.add_argument('--worker', help='expected factory identity on this physical node')
    run.add_argument('--endpoint', help='recorded Artist Twin-facing origin (inference executes node-locally)')
    run.add_argument('--min-free-gb', type=float, default=12)
    run.add_argument('--max-temperature-c', type=float, default=85)
    run.add_argument('--min-temperature-margin-c', type=float, default=5,
                     help='stop at this NVIDIA T.Limit headroom when the device exposes it')
    run.add_argument('--max-swap-growth-gb', type=float, default=1)
    run.add_argument('--synthetic', action='store_true', help='no GPU/production access; timings are not capacity evidence')
    run.add_argument('--ssh-host', help='deploy/run on this configured Spark, holding the Mac controller lock')
    run.add_argument('--config', type=Path, default=Path('models.toml'))
    analysis = commands.add_parser('analyze', help='rebuild reports from preserved raw results')
    analysis.add_argument('output', type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'analyze':
            from .analysis import analyze
            analyze(args.output)
        else:
            if args.ssh_host:
                if args.synthetic:
                    raise ValueError('Use synthetic mode locally without --ssh-host')
                from .remote import remote_run
                print(remote_run(args))
            else:
                from .experiment import run_experiment
                print(run_experiment(args))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f'Benchmark stopped: {exc}\n')


if __name__ == '__main__':
    main()
