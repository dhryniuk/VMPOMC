import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.style.use('seaborn-v0_8')z



### MAGNETIZATIONS:

mx = np.load("data/observables/mag_x_N16_2.npy")
my = np.load("data/observables/mag_y_N16_2.npy")
mz = np.load("data/observables/mag_z_N16_2.npy")
#mx = np.load("data/observables/mag_x.npy")
#my = np.load("data/observables/mag_y.npy")
#mz = np.load("data/observables/mag_z.npy")


plt.figure(figsize=(6,4),dpi=600)

plt.plot(mx[1:],label="M_x", color='C0')
#plt.hlines(mx[0],1,len(mx), colors='C0')

plt.plot(my[1:],label="M_y", color='C1')
#plt.hlines(my[0],1,len(my), colors='C1')

plt.plot(mz[1:],label="M_z", color='C2')
#plt.hlines(mz[0],1,len(mz), colors='C2')

plt.xscale("log")
plt.xlim(1,len(mx))
plt.xlabel("Optimisation step")
plt.tight_layout()
plt.legend()
plt.title(r"Ising with decay, $N=16$, $g/\gamma=2, \chi=2 \rightarrow 4 \rightarrow 8$")
plt.tight_layout()
string = r"$M_x = 0.277\pm 0.001$" + "\n" + r"$M_y = 0.377\pm 0.001$" + "\n" + r"$M_z = -0.242\pm 0.001$"
plt.text(1.5,0,string)
plt.savefig("magnetizations.png")
plt.show()



#Calculate final values:
mx_final = np.real(mx[-100:])
my_final = np.real(my[-100:])
mz_final = np.real(mz[-100:])

#mx = 0.277+-0.001
#my = 0.377+-0.001
#mz = -0.242+-0.001


### PURITY:

purity = np.load("data/observables/purity.npy")
entropy = np.load("data/observables/entropy.npy")


plt.figure(figsize=(6,4),dpi=600)

plt.plot(purity[1:],label="Purity", color='C0')
plt.hlines(purity[0],1,len(purity), colors='C0')

plt.xscale("log")
plt.xlim(1,len(mx))
plt.ylim(0,1)
plt.xlabel("Optimisation step")
plt.tight_layout()
plt.legend()
plt.savefig("purity.png")
plt.show()

plt.figure(figsize=(6,4),dpi=600)

plt.plot(entropy[1:],label="Renyi Entropy", color='C0')
plt.hlines(entropy[0],1,len(entropy), colors='C0')

plt.xscale("log")
plt.xlim(1,len(mx))
plt.xlabel("Optimisation step")
plt.tight_layout()
plt.legend()
plt.savefig("entropy.png")
plt.show()



### FIDELITY:

density_matrices = np.load