import numpy as np
import matplotlib.pyplot as plt

rho_real = np.load("data/rho_real.npy")
rho_imag = np.load("data/rho_imag.npy")

MPOMC_rho_real = np.load("data/MPOMC_rho_real_χ=4.npy")
MPOMC_rho_imag = np.load("data/MPOMC_rho_imag_χ=4.npy")

print(rho_real)
print(rho_imag)

#plt.imshow(rho_real, interpolation='nearest')
#plt.show()

fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
im = ax.imshow(MPOMC_rho_real)
im2 = ax2.imshow(MPOMC_rho_imag)

im3 = ax3.imshow(rho_real)
im4 = ax4.imshow(rho_imag)

# Show all ticks and label them with the respective list entries
N=3
ax.set_xticks(range(N**2))
ax.set_yticks(range(N**2))
ax2.set_xticks(range(N**2))
ax2.set_yticks(range(N**2))
ax3.set_xticks(range(N**2))
ax3.set_yticks(range(N**2))
ax4.set_xticks(range(N**2))
ax4.set_yticks(range(N**2))

ax.set_title("MPOMC Real")
ax2.set_title("MPOMC Imaginary")

ax3.set_title("ED Real")
ax4.set_title("ED Imaginary")

fig.tight_layout()
plt.colorbar(im,ax=ax)
plt.colorbar(im2,ax=ax2)
plt.colorbar(im3,ax=ax3)
plt.colorbar(im4,ax=ax4)

#fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#fig.colorbar(im, cax=cbar_ax)

plt.show()


#Calculate fidelity:
import scipy.linalg as sp
a=rho_real+1j*rho_imag
b=MPOMC_rho_real+1j*MPOMC_rho_imag
#print(a)
#print(b)

sqrt_a=sp.sqrtm(a)
#print(sqrt_a)
f=np.trace(sp.sqrtm(np.matmul(sqrt_a,np.matmul(b,sqrt_a))))**2
print(f)

sqrt_b=sp.sqrtm(b)
#print(sqrt_a)
f=np.trace(sp.sqrtm(np.matmul(sqrt_b,np.matmul(a,sqrt_b))))**2
print(f)