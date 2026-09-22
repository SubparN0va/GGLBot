import os
import subprocess
import sys

executable, runtime, models = sys.argv[1:]
env = os.environ.copy()
variable = 'PATH' if os.name == 'nt' else 'LD_LIBRARY_PATH'
env[variable] = runtime + os.pathsep + env.get(variable, '')
sys.exit(subprocess.run([executable, models], env=env, timeout=120).returncode)
