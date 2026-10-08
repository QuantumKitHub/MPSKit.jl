"""Run serialized jld requests against prepared, manifest-matched main/latest snapshots.

See dmrg_main_comparison_results.md for snapshot/input preparation. Warm both
processes before --phase measure. Numerical requests always execute sequentially.
"""
import argparse
import csv
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/tmp/dmrg-main-comparison'))
    parser.add_argument('--julia', default='/tmp/dmrg-julia-existing')
    parser.add_argument('--blas-threads', type=int, default=4)
    parser.add_argument('--cpus', default='0-3')
    parser.add_argument('--dims', type=int, nargs='+', default=[128, 256])
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--solve-sweeps', type=int, default=4)
    parser.add_argument('--continuous-reps', type=int, default=3)
    parser.add_argument('--continuous-sweeps', type=int, default=6)
    parser.add_argument('--phase', choices=['warm', 'measure', 'diagnostics', 'verify', 'all'], default='all')
    args = parser.parse_args()
    root = args.root.resolve()
    results = root / 'results'
    results.mkdir(exist_ok=True)
    source = Path(__file__).with_suffix('.jl').resolve()

    def request(branch, code):
        name = 'dmrg-main-baseline' if branch == 'main' else 'dmrg-main-latest'
        command = [
            'jld', '--project=' + str(root / branch / 'test'), '--name=' + name,
            '--julia=' + args.julia, '--idle-timeout=2h', '--timeout=2400',
            '--max-output=8k', 'eval', code,
        ]
        print('REQUEST', branch, code, flush=True)
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(completed.stdout, flush=True)
        completed.check_returncode()
        return completed.stdout

    for branch in ('main', 'latest'):
        output = request(branch, f'if !isdefined(Main, :comparison_warm); include({json.dumps(str(source))}); end; '
                f'if !isdefined(Main, :comparison_data); comparison_data = deserialize({json.dumps(str(root / "input.jls"))}); end; '
                f'BLAS.set_num_threads({args.blas_threads}); println("DMRG_PID=", getpid()); '
                f'comparison_inventory({json.dumps(str(results / (branch + "_dependencies.csv")))}); nothing')
        pid = next(line.split('=', 1)[1] for line in output.splitlines() if line.startswith('DMRG_PID='))
        subprocess.run(['taskset', '-apc', args.cpus, pid], check=True, stdout=subprocess.DEVNULL)
    inventories = [list(csv.DictReader((results / f'{branch}_dependencies.csv').open())) for branch in ('main', 'latest')]
    assert len(inventories[0]) == len(inventories[1])
    assert all(a == b for a, b in zip(*inventories) if a['name'] != 'MPSKit')
    assert (root / 'main' / 'Manifest.toml').read_bytes() == (root / 'latest' / 'Manifest.toml').read_bytes()

    def prefix(branch, chi, suffix):
        return json.dumps(str(results / f'{branch}_{chi}_{suffix}'))

    if args.phase in ('warm', 'all'):
        for chi in args.dims:
            for branch in ('main', 'latest'):
                request(branch, f'comparison_warm(comparison_data, {chi}, {prefix(branch, chi, "warmup")}); nothing')
    if args.phase in ('measure', 'all'):
        for branch in ('main', 'latest'):
            request(branch, f'comparison_prime(comparison_data, {json.dumps(str(results / (branch + "_prime")))}); nothing')
        for chi in args.dims:
            for sample in range(1, args.samples + 1):
                order = ('main', 'latest') if sample % 2 else ('latest', 'main')
                for branch in order:
                    request(branch, f'comparison_public_sample(comparison_data, {chi}, {prefix(branch, chi, "solve_" + str(sample))}; sweeps={args.solve_sweeps}); nothing')
            for sample in range(1, args.continuous_reps + 1):
                order = ('latest', 'main') if sample % 2 else ('main', 'latest')
                for branch in order:
                    request(branch, f'comparison_continuous(comparison_data, {chi}, {prefix(branch, chi, "continuous_" + str(sample))}; sweeps={args.continuous_sweeps}); nothing')
    if args.phase in ('diagnostics', 'all'):
        for chi in args.dims:
            for branch in ('main', 'latest'):
                request(branch, f'comparison_stages(comparison_data, {chi}, {prefix(branch, chi, "stages")}); nothing')

    if args.phase in ('verify', 'all'):
        request('latest', f'validate_comparison_files({json.dumps(str(results))}; samples={args.samples}, continuous_reps={args.continuous_reps}); nothing')


if __name__ == '__main__':
    main()
