"""Display stage and recent worker progress without importing torch."""
from pathlib import Path
import runpy
import sys

if len(sys.argv)==1:
    sys.argv.append('/root/autodl-tmp/vcell-work/runs/round11_01')
runpy.run_path(str(Path(__file__).with_name('round9_status.py')),run_name='__main__')
