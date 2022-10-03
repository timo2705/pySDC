import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
name = 'masterwork_timo/pickle/float_A.pickle'
with open(name, 'rb') as f:
    data = pickle.load(f)
Ns = data['Ns']
times_cpu_32 = data['times-cpu-32']
times_gpu_32 = data['times-gpu-32']
times_cpu_64 = data['times-cpu-64']
times_gpu_64 = data['times-gpu-64']

plt.plot(Ns, times_cpu_32, color="orange", ls=":", label="float32 CPU")
plt.plot(Ns, times_gpu_32, color="orange", ls="-", label="float32 GPU")
plt.plot(Ns, times_cpu_64, color="blue", ls=":", label="float64 CPU")
plt.plot(Ns, times_gpu_64, color="blue", ls="-", label="float64 GPU")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Freiheitsgrade')
plt.ylabel('Zeit in s')
plt.legend()
plt.show()
print(times_cpu_32)
print(times_gpu_32)
print(times_cpu_64)
print(times_gpu_64)

