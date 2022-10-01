import pickle
import numpy as np
import matplotlib.pyplot as plt

name_cpu = f'/Users/timolenz/PycharmProjects/pySDC/pySDC/playgrounds/GPU/masterwork_timo/pickle/ac-pySDC-cpu.pickle'
name_gpu = '/Users/timolenz/PycharmProjects/pySDC/pySDC/playgrounds/GPU/masterwork_timo/pickle/ac-pySDC-gpu.pickle'
with open(name_cpu, 'rb') as f:
   data_cpu = pickle.load(f)
Ns = data_cpu['Ns']
D = data_cpu['D']
Ns_plot = Ns**D
schritte = data_cpu['schritte']
dt = data_cpu['dt']
iteration = data_cpu['iteration']
tol = data_cpu['Tolerance']
times_CPU = data_cpu['times']
setup_CPU = data_cpu['setup']
cg_CPU = data_cpu['cg-time']-0.08*data_cpu['cg-time']
cg_Count_CPU = data_cpu['cg-count']
f_im_CPU = data_cpu['f-time-imp']
f_ex_CPU = data_cpu['f-time-exp']
with open(name_gpu, 'rb') as f:
   data_gpu = pickle.load(f)
times_GPU = data_gpu['times']
setup_GPU = data_gpu['setup']
cg_GPU = data_gpu['cg-time']-0.08*data_gpu['cg-time']
cg_Count_GPU = data_gpu['cg-count']
f_im_GPU = data_gpu['f-time-imp']
f_ex_GPU = data_gpu['f-time-exp']

# Start Plotting Time Marching
##############################################################################
plt.scatter(Ns_plot, times_GPU, color="orange", label="Laufzeit")
plt.plot(Ns_plot, times_GPU, color="orange", ls="-", label="GPU")
plt.scatter(Ns_plot, times_CPU, color="orange")
plt.plot(Ns_plot, times_CPU, color="orange", ls=":", label="CPU")
plt.scatter(Ns_plot, setup_GPU, color="blue", label="Konfig.")
plt.plot(Ns_plot, setup_GPU, color="blue", ls="-")
plt.scatter(Ns_plot, setup_CPU, color="blue")
plt.plot(Ns_plot, setup_CPU, color="blue", ls=":")
plt.xscale('log')
plt.yscale('log')
# plt.title("pySDC Allen-Cahn 2D:\nGPU vs CPU only time_marching")
plt.xlabel('Freiheitsgrade')
plt.ylabel('Zeit in s')
plt.legend()
# plt.savefig('pdfs/allen-cahn_jusuf_tm_log2.pdf')
plt.show()
plt.clf()
# Start Plotting Factors
##############################################################################
plt.scatter(Ns_plot, times_CPU/times_GPU, label="Laufzeit")
plt.plot(Ns_plot, times_CPU/times_GPU)
print(times_CPU/times_GPU)
plt.scatter(Ns_plot, setup_CPU/setup_GPU, label="Konfig.")
plt.plot(Ns_plot, setup_CPU/setup_GPU)
plt.scatter(Ns_plot, cg_CPU/cg_GPU-0.08*(cg_CPU/cg_GPU), label="Löser")
plt.plot(Ns_plot, cg_CPU/cg_GPU-0.08*(cg_CPU/cg_GPU))
print(cg_CPU/cg_GPU)
plt.scatter(Ns_plot, f_im_CPU/f_im_GPU, label="F Implizit")
plt.plot(Ns_plot, f_im_CPU/f_im_GPU)
plt.scatter(Ns_plot, f_ex_CPU/f_ex_GPU, label="F Explizit")
plt.plot(Ns_plot, f_ex_CPU/f_ex_GPU)
plt.xscale('log')
plt.yscale('log')
# plt.title("pySDC Allen-Cahn 2D:\nCPU / GPU")
plt.xlabel('Freiheitsgrade')
plt.ylabel('Faktor')
plt.legend()
# plt.savefig('pdfs/allen-cahn_jusuf_factors_log2.pdf')
plt.show()
plt.clf()
# Start Plotting All Times
##############################################################################
plt.scatter(Ns_plot, times_GPU, label="Laufzeit")
plt.plot(Ns_plot, times_GPU)
# plt.scatter(Ns_plot, setup_GPU, label="Konfig.")
# plt.plot(Ns_plot, setup_GPU)
plt.scatter(Ns_plot, cg_GPU, label="Löser")
plt.plot(Ns_plot, cg_GPU)
plt.scatter(Ns_plot, f_im_GPU, label="F Implizit")
plt.plot(Ns_plot, f_im_GPU)
plt.scatter(Ns_plot, f_ex_GPU, label="F Explizit")
plt.plot(Ns_plot, f_ex_GPU)
plt.xscale('log')
plt.yscale('log')
# plt.title("pySDC Allan-Cahn 2D:\nGPU All Times")
plt.ylim([f_ex_CPU[0]-0.5*f_ex_CPU[0], times_CPU[-1]+0.25*times_CPU[-1]])
plt.xlabel('Freiheitsgrade')
plt.ylabel('Zeit in s')
plt.legend()
plt.show()
plt.scatter(Ns_plot, times_CPU, label="Laufzeit")
plt.plot(Ns_plot, times_CPU)
# plt.scatter(Ns_plot, setup_CPU, label="Konfig.")
# plt.plot(Ns_plot, setup_CPU)
plt.scatter(Ns_plot, cg_CPU, label="Löser")
plt.plot(Ns_plot, cg_CPU)
plt.scatter(Ns_plot, f_im_CPU, label="F Implizit")
plt.plot(Ns_plot, f_im_CPU)
plt.scatter(Ns_plot, f_ex_CPU, label="F Explizit")
plt.plot(Ns_plot, f_ex_CPU)
plt.xscale('log')
plt.yscale('log')
# plt.title("pySDC Allan-Cahn 2D:\nCPU All Times")
plt.ylim([f_ex_CPU[0]-0.5*f_ex_CPU[0], times_CPU[-1]+0.25*times_CPU[-1]])
plt.xlabel('Freiheitsgrade')
plt.ylabel('Zeit in s')
plt.legend()
plt.show()
