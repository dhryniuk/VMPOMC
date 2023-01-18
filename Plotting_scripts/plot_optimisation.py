import numpy as np
import matplotlib.pyplot as plt

#SR = np.load("MPOMC_L_SR.npy")
#SGD = np.load("MPOMC_L_SGD.npy")
#SR = np.load("MPOMC_L_SR_χ=2.npy")
#SGD = np.load("MPOMC_L_SGD_χ=2.npy")

SGD = np.load("data/MPOMC_L_SGD_χ=4.npy")
SR = np.load("data/MPOMC_L_SR_χ=4.npy")


plt.figure(figsize=(6,3),dpi=600)
plt.plot(SGD,label="SGD")
plt.plot(SR,label="SR")
plt.xscale("log")
plt.yscale("log")
yticks_array = [10.0**(-i) for i in [0,1,2]]
plt.yticks(yticks_array)
plt.ylabel(r"$\langle\mathcal{L}^\dagger\mathcal{L}\rangle$")
plt.xlabel("Optimisation step")
plt.tight_layout()
plt.legend()
plt.savefig("test.png")
plt.show()