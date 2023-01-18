import numpy as np
import matplotlib.pyplot as plt

rho_real = np.load("rho_real.npy")
rho_imag = np.load("rho_imag.npy")

MPOMC_rho_real_100 = np.load("MPOMC_rho_real_χ=4_step100.npy")
MPOMC_rho_imag_100 = np.load("MPOMC_rho_imag_χ=4_step100.npy")

MPOMC_rho_real_500 = np.load("MPOMC_rho_real_χ=4_step500.npy")
MPOMC_rho_imag_500 = np.load("MPOMC_rho_imag_χ=4_step500.npy")

#print(rho_real)
#print(rho_imag)

#plt.imshow(rho_real, interpolation='nearest')
#plt.show()

fig, ((ax, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12, 16),dpi=600)
im = ax.imshow(MPOMC_rho_real_100)
im2 = ax2.imshow(MPOMC_rho_imag_100)

im3 = ax3.imshow(MPOMC_rho_real_500)
im4 = ax4.imshow(MPOMC_rho_imag_500)

im5 = ax5.imshow(rho_real)
im6 = ax6.imshow(rho_imag)

# Show all ticks and label them with the respective list entries
N=4
ax.set_xticks(range(N**2))
ax.set_yticks(range(N**2))
ax2.set_xticks(range(N**2))
ax2.set_yticks(range(N**2))
ax3.set_xticks(range(N**2))
ax3.set_yticks(range(N**2))
ax4.set_xticks(range(N**2))
ax4.set_yticks(range(N**2))
ax5.set_xticks(range(N**2))
ax5.set_yticks(range(N**2))
ax6.set_xticks(range(N**2))
ax6.set_yticks(range(N**2))

ax.set_title("MPOMC Real, step=100")
ax2.set_title("MPOMC Imaginary, step=100")

ax3.set_title("MPOMC Real, step=500")
ax4.set_title("MPOMC Imaginary, step=500")

ax5.set_title("ED Real")
ax6.set_title("ED Imaginary")

plt.colorbar(im,ax=ax)
plt.colorbar(im2,ax=ax2)
plt.colorbar(im3,ax=ax3)
plt.colorbar(im4,ax=ax4)
plt.colorbar(im5,ax=ax5)
plt.colorbar(im6,ax=ax6)

#fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#fig.colorbar(im, cax=cbar_ax)

fig.tight_layout()
plt.savefig("test.png")
plt.show()


#Calculate fidelity:
import scipy.linalg as sp
a=rho_real+1j*rho_imag
b=MPOMC_rho_real_100+1j*MPOMC_rho_imag_100
#print(a)
#print(b)

sqrt_a=sp.sqrtm(a)
#print(sqrt_a)

f=np.trace(sp.sqrtm(np.matmul(sqrt_a,np.matmul(b,sqrt_a))))**2

print(f)