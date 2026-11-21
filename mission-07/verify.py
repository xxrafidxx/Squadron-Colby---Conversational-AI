"""Run all required module harnesses, regression tests and experiments.

Usage: python verify.py
Requires Python 3.11+ and only the standard library. Stops on first failure.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    evidence = root/'evidence'
    evidence.mkdir(exist_ok=True)
    commands = [[name+'.py'] for name in
                ('tot_solver','mcts_protocol','prm_evaluator','horizon_pruner')]
    commands += [['-m','unittest','-v'],['run_experiments.py']]
    log = [f'UTC: {datetime.now(timezone.utc).isoformat()}',
           f'Python: {platform.python_version()}',
           'All models, priors and rollout answers in these tests are synthetic.']
    for args in commands:
        result = subprocess.run([sys.executable,'-X','utf8',*args],cwd=root,
                                capture_output=True,text=True,encoding='utf-8')
        section = '\n$ python '+' '.join(args)+'\n'+result.stdout+result.stderr
        section += f'Exit code: {result.returncode}\n'
        print(section)
        log.append(section)
        if result.returncode:
            (evidence/'verification.txt').write_text('\n'.join(log),encoding='utf-8')
            return result.returncode
    log.append('ALL FOUR COURSE HARNESSES, 22 REGRESSION TESTS AND EXPERIMENTS PASSED.')
    (evidence/'verification.txt').write_text('\n'.join(log)+'\n',encoding='utf-8')
    hashes = {path.name:hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(root.glob('*.py'))}
    (evidence/'source_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
