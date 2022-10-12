import pickle
import matplotlib.pyplot as plt
from pySDC.helpers.plot_helper import setup_mpl, savefig
setup_mpl(12)

name_cpu = '/Users/timolenz/PycharmProjects/pySDC/pySDC/playgrounds/GPU/masterwork_timo/pickle/heat-pySDC-cpu-1.pickle'
name_gpu = '/Users/timolenz/PycharmProjects/pySDC/pySDC/playgrounds/GPU/masterwork_timo/pickle/heat-pySDC-gpu-1.pickle'
# name_gpu = '/Users/timolenz/PycharmProjects/pySDC/pySDC/playgrounds/GPU/masterwork_timo/pickle/heat-pySDC-gpu-1-odtype.pickle'
with open(name_cpu, 'rb') as f:
   data_cpu = pickle.load(f)
Ns = data_cpu['Ns']
D = data_cpu['D']
Ns_plot = Ns**D
schritte = data_cpu['schritte']
dt = data_cpu['dt']
iteration = data_cpu['iteration']
tol = data_cpu['Tolerance']
# times_CPU = data_cpu['times']
setup_CPU = data_cpu['setup']
cg_CPU = data_cpu['cg-time']
# cg_Count_CPU = data_cpu['cg-count']
f_im_CPU = data_cpu['f-time-imp']
f_ex_CPU = data_cpu['f-time-exp']
with open(name_gpu, 'rb') as f:
   data_gpu = pickle.load(f)
# times_GPU = data_gpu['times']
setup_GPU = data_gpu['setup']
cg_GPU = data_gpu['cg-time']
# cg_Count_GPU = data_gpu['cg-count']
f_im_GPU = data_gpu['f-time-imp']
f_ex_GPU = data_gpu['f-time-exp']
# cg_CPU = times_CPU - (f_im_CPU+f_ex_CPU)
# cg_GPU = times_GPU - (f_im_GPU+f_ex_GPU)
times_CPU = cg_CPU+f_im_CPU+f_ex_CPU
times_GPU = cg_GPU+f_im_GPU+f_ex_GPU
# Start Plotting Time Marching and Setup
##############################################################################
plt.plot(Ns_plot, times_GPU, color="dodgerblue", ls="-", marker="v", label="Laufzeit GPU")
plt.plot(Ns_plot, times_CPU, color="orange", ls="-", marker="D", label=" Laufzeit CPU")
plt.plot(Ns_plot, setup_GPU, color="dodgerblue", ls=":", marker="s", label="Konfig. GPU")
plt.plot(Ns_plot, setup_CPU, color="orange", ls=":", marker="X", label="Konfig. CPU")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Freiheitsgrade')
plt.ylabel('Zeit in s')
plt.legend()
plt.grid()
# plt.savefig('pdfs/allen-cahn_jusuf_tm_log2.pdf')
plt.show()
plt.clf()
# Start Plotting Factors
##############################################################################
plt.plot(Ns_plot, times_CPU / times_GPU, color="dodgerblue", marker="d", label="Laufzeit")
print(times_CPU / times_GPU)
plt.plot(Ns_plot, setup_CPU / setup_GPU, color="darkgoldenrod", marker="^", label="Konfig.")
plt.plot(Ns_plot, cg_CPU / cg_GPU, color="violet", marker="X", label="Löser")
plt.plot(Ns_plot, f_im_CPU / f_im_GPU, color="orange", marker="o", label="F Implizit")
plt.plot(Ns_plot, f_ex_CPU / f_ex_GPU, color="seagreen", marker="s", label="F Explizit")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Freiheitsgrade')
plt.ylabel('Faktor')
plt.legend()
plt.grid()
# plt.savefig('pdfs/allen-cahn_jusuf_factors_log2.pdf')
plt.show()
plt.clf()
# Start Plotting All Times
##############################################################################
"""
plt.plot(Ns_plot, times_GPU, color="dodgerblue", marker="d", label="Laufzeit")
# plt.plot(Ns_plot, setup_GPU, color="darkgoldenrod", marker="^", label="Konfig.")
plt.plot(Ns_plot, cg_GPU, color="violet", marker="X", label="Löser")
plt.plot(Ns_plot, f_im_GPU, color="orange", marker="o", label="F Implizit")
plt.plot(Ns_plot, f_ex_GPU, color="seagreen", marker="s", label="F Explizit")
plt.xscale('log')
plt.yscale('log')
plt.ylim([f_im_CPU[0]-0.25*f_im_CPU[0], times_CPU[-1]+0.25*times_CPU[-1]])
# plt.xlabel('Freiheitsgrade')
plt.ylabel('Zeit in s')
# plt.legend()
plt.grid()
plt.show()
# plt.savefig('pdfs/allen-cahn_jusuf_tm_log2.pdf')
plt.clf()
plt.plot(Ns_plot, times_CPU, color="dodgerblue", marker="d", label="Laufzeit")
# plt.plot(Ns_plot, setup_CPU, color="darkgoldenrod", marker="^", label="Konfig.")
plt.plot(Ns_plot, cg_CPU, color="violet", marker="X", label="Löser")
plt.plot(Ns_plot, f_im_CPU, color="orange", marker="o", label="F Implizit")
plt.plot(Ns_plot, f_ex_CPU, color="seagreen", marker="s", label="F Explizit")
plt.xscale('log')
plt.yscale('log')
plt.ylim([f_im_CPU[0]-0.25*f_im_CPU[0], times_CPU[-1]+0.25*times_CPU[-1]])
plt.xlabel('Freiheitsgrade')
plt.ylabel('Zeit in s')
plt.legend()
plt.grid()
plt.show()
# plt.savefig('pdfs/allen-cahn_jusuf_tm_log2.pdf')
plt.clf()
"""
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(11, 5))
ax1.plot(Ns_plot, times_GPU, color="dodgerblue", marker="d", label="Laufzeit")
ax1.plot(Ns_plot, cg_GPU, color="violet", marker="X", label="Löser")
ax1.plot(Ns_plot, f_im_GPU, color="orange", marker="o", label="F Implizit")
ax1.plot(Ns_plot, f_ex_GPU, color="seagreen", marker="s", label="F Explizit")
ax1.set_title('GPU')
ax1.set_xlabel('Freiheitsgrade')
ax1.set_ylabel('Zeit in s')
ax1.legend()
ax1.grid()
ax2.plot(Ns_plot, times_CPU, color="dodgerblue", marker="d", label="Laufzeit")
ax2.plot(Ns_plot, cg_CPU, color="violet", marker="X", label="Löser")
ax2.plot(Ns_plot, f_im_CPU, color="orange", marker="o", label="F Implizit")
ax2.plot(Ns_plot, f_ex_CPU, color="seagreen", marker="s", label="F Explizit")
ax2.set_title('CPU')
ax2.grid()
plt.xscale('log')
plt.yscale('log')
# plt.ylim([f_im_CPU[0]-0.3*f_im_CPU[0], times_CPU[-1]+0.3*times_CPU[-1]])
# plt.xlabel('Freiheitsgrade')
# plt.ylabel('Zeit in s')
# plt.legend()
fig.tight_layout()
plt.show()
# plt.savefig('pdfs/allen-cahn_jusuf_tm_log2.pdf')
plt.clf()
