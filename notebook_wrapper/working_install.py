import pathlib
import subprocess
import sys

print('=== Installing ARC runtime from the offline competition wheelhouse ===', flush=True)

arc_wheel_dir = pathlib.Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels')
if not arc_wheel_dir.is_dir():
    raise FileNotFoundError('ARC-AGI-3 competition wheels are missing: ' + str(arc_wheel_dir))
subprocess.check_call([
    sys.executable,
    '-m',
    'pip',
    'install',
    '--no-index',
    '--find-links',
    str(arc_wheel_dir),
    'arc-agi',
])

print('ARC runtime installed. The Tufa vLLM wheelhouse is installed into an isolated target only in Phase B.', flush=True)
