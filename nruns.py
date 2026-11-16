import subprocess

N=5

for ix in range(N):
    seed = ix+10
    subprocess.run(['python3','run_2d.py','--quick',f'--seed={seed}'])
    print('-'*79)
