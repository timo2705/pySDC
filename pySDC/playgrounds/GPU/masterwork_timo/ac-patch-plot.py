import pickle
import numpy as np
import matplotlib.pyplot as plt
from pySDC.helpers.plot_helper import setup_mpl, savefig
setup_mpl(12)

name = f'pickle/ac-fft-pySDC-patch.pickle'
with open(name, 'rb') as f:
   data = pickle.load(f)
Ns = data['Ns']
D = data['D']
Ns_plot = Ns**D
schritte = data['schritte']
dt = data['dt']
iteration = data['iteration']
# times_CPU = data_cpu['times']
setup_CPU = data['setup-cpu']
cg_CPU = data['cg-time-cpu']
f_im_CPU = data['f-time-imp-cpu']
f_ex_CPU = data['f-time-exp-cpu']
# times_GPU = data_gpu['times']
setup_GPU = data['setup-gpu']
cg_GPU = data['cg-time-gpu']
f_im_GPU = data['f-time-imp-gpu']
f_ex_GPU = data['f-time-exp-gpu']
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
# savefig('pdfs/ac-fft-patch-ndtype', save_pgf=False, save_png=False)
# plt.show()
plt.clf()
# Start Plotting Factors
##############################################################################
plt.plot(Ns_plot, times_CPU / times_GPU, color="dodgerblue", marker="d", label="Zeitschritt")
print("times:", times_CPU / times_GPU)
print("times CPU:", times_CPU)
print("times GPU:", times_GPU)
plt.plot(Ns_plot, setup_CPU / setup_GPU, color="darkgoldenrod", marker="^", label="Konfig.")
print("setup:", setup_CPU / setup_GPU)
plt.plot(Ns_plot, cg_CPU / cg_GPU, color="violet", marker="X", label="Löser")
print("Löser:", cg_CPU / cg_GPU)
plt.plot(Ns_plot, f_im_CPU / f_im_GPU, color="orange", marker="o", label="$f$ implizit")
print("f im:", f_im_CPU / f_im_GPU)
plt.plot(Ns_plot, f_ex_CPU / f_ex_GPU, color="seagreen", marker="s", label="$f$ explizit")
print("f ex:", f_ex_CPU / f_ex_GPU)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Freiheitsgrade')
plt.ylabel('Faktor')
plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
plt.grid()
savefig('pdfs/ac-fft-patch-factor-ndytpe', save_pgf=False, save_png=False)
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
ax1.plot(Ns_plot, f_im_GPU, color="orange", marker="o", label="$f$ implizit")
ax1.plot(Ns_plot, f_ex_GPU, color="seagreen", marker="s", label="$f$ explizit")
ax1.set_title('GPU')
ax1.set_xlabel('Freiheitsgrade')
ax1.set_ylabel('Zeit in s')
# ax1.legend()
ax1.grid()
ax2.plot(Ns_plot, times_CPU, color="dodgerblue", marker="d", label="Zeitschritt")
ax2.plot(Ns_plot, cg_CPU, color="violet", marker="X", label="Löser")
ax2.plot(Ns_plot, f_im_CPU, color="orange", marker="o", label="$f$ implizit")
ax2.plot(Ns_plot, f_ex_CPU, color="seagreen", marker="s", label="$f$ explizit")
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
savefig('pdfs/ac-fft-patch-times-ndtype', save_pgf=False, save_png=False)
plt.clf()
