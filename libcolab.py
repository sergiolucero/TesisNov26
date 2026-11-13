from IPython.utils.sysinfo import subprocess
import yaml

def modrun(*vals):
  NEW_FILE = 'configs/opt/benchmark_new.yaml'
  data = yaml.safe_load(open('configs/opt/benchmark_small.yaml'))
  nx, ny = [data['domain'][k] for k in ('nx', 'ny')]
  print('WAS', nx,ny, 'NOW:', vals)
  nx, ny = vals
  for ix, k in enumerate(('nx', 'ny')):
        data['domain'][k] = vals[ix]
  yaml.dump(data, open(NEW_FILE, 'w'))
  print('RUNNING...')
  subprocess.run(['python', 'run_2d.py', '--mode', 'optimize', 
                  '--config', NEW_FILE])

# was run_2d.py --mode optimize --config 
modrun(16,16)
