import pickle
import numpy as np
import matplotlib.pyplot as plt
from pySDC.helpers.plot_helper import setup_mpl, savefig
setup_mpl(12)

name_cpu = f'pickle/ac-pySDC-cpu.pickle'
name_gpu = 'pickle/ac-pySDC-gpu.pickle'
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
cg_Count_CPU = data_cpu['cg-count']
f_im_CPU = data_cpu['f-time-imp']
f_ex_CPU = data_cpu['f-time-exp']
with open(name_gpu, 'rb') as f:
   data_gpu = pickle.load(f)
# times_GPU = data_gpu['times']
setup_GPU = data_gpu['setup']
cg_GPU = data_gpu['cg-time']
cg_Count_GPU = data_gpu['cg-count']
f_im_GPU = data_gpu['f-time-imp']
f_ex_GPU = data_gpu['f-time-exp']
times_CPU = cg_CPU+f_im_CPU+f_ex_CPU
times_GPU = cg_GPU+f_im_GPU+f_ex_GPU

# Start Plotting Time Marching and Setup
##############################################################################
plt.plot(Ns_plot, times_GPU, color="dodgerblue", ls="-", marker="v", label="Zeitschritt GPU")
plt.plot(Ns_plot, times_CPU, color="orange", ls="-", marker="D", label="Zeitschritt CPU")
plt.plot(Ns_plot, setup_GPU, color="dodgerblue", ls=":", marker="s", label="Konfig. GPU")
plt.plot(Ns_plot, setup_CPU, color="orange", ls=":", marker="X", label="Konfig. CPU")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Freiheitsgrade')
plt.ylabel('Zeit in s')
plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
plt.grid()
# savefig('pdfs/ac-ndtype', save_pgf=False, save_png=False)
# plt.show()
plt.clf()
# Start Plotting Factors
##############################################################################
plt.plot(Ns_plot, times_CPU / times_GPU, color="dodgerblue", marker="d", label="Zeitschritt")
print("times:", times_CPU / times_GPU)
print("times:", times_CPU)
plt.plot(Ns_plot, setup_CPU / setup_GPU, color="darkgoldenrod", marker="^", label="Konfig.")
print("setup:", setup_CPU / setup_GPU)
plt.plot(Ns_plot, cg_CPU / cg_GPU, color="violet", marker="X", label="Löser")
print("Löser:", cg_CPU / cg_GPU)
plt.plot(Ns_plot, f_im_CPU / f_im_GPU, color="orange", marker="o", label="F Implizit")
print("f im:", f_im_CPU / f_im_GPU)
plt.plot(Ns_plot, f_ex_CPU / f_ex_GPU, color="seagreen", marker="s", label="F Explizit")
print("f ex:", f_ex_CPU / f_ex_GPU)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Freiheitsgrade')
plt.ylabel('Faktor')
plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
plt.grid()
# savefig('pdfs/ac-factor-ndytpe', save_pgf=False, save_png=False)
# plt.show()
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
ax1.plot(Ns_plot, times_GPU, color="dodgerblue", marker="d", label="Zeitschritt")
ax1.plot(Ns_plot, cg_GPU, color="violet", marker="X", label="Löser")
ax1.plot(Ns_plot, f_im_GPU, color="orange", marker="o", label="F Implizit")
ax1.plot(Ns_plot, f_ex_GPU, color="seagreen", marker="s", label="F Explizit")
ax1.set_title('GPU')
ax1.set_xlabel('Freiheitsgrade')
ax1.set_ylabel('Zeit in s')
# ax1.legend()
ax1.grid()
ax2.plot(Ns_plot, times_CPU, color="dodgerblue", marker="d", label="Zeitschritt")
ax2.plot(Ns_plot, cg_CPU, color="violet", marker="X", label="Löser")
ax2.plot(Ns_plot, f_im_CPU, color="orange", marker="o", label="F Implizit")
ax2.plot(Ns_plot, f_ex_CPU, color="seagreen", marker="s", label="F Explizit")
ax2.set_title('CPU')
ax2.set_xlabel('Freiheitsgrade')
ax2.grid()
ax2.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
plt.xscale('log')
plt.yscale('log')
# plt.ylim([f_im_CPU[0]-0.3*f_im_CPU[0], times_CPU[-1]+0.3*times_CPU[-1]])
# plt.xlabel('Freiheitsgrade')
# plt.ylabel('Zeit in s')
# plt.legend()
fig.tight_layout()
# plt.show()
# savefig('pdfs/ac-times-ndtype', save_pgf=False, save_png=False)
plt.clf()
