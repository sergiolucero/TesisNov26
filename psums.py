import time
psum = lambda p: sum(pow(ix,-p) for ix in range(1,1000000))

t0=time.time()
ps=[(p,psum(p)) for p in range(2,50)]
tf=time.time()

print(ps)
print('DT=', round(tf-t0,2), 'secs')
