import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.style.use('seaborn-v0_8')



#Define constants:
J=0.5 #interaction strength
h=1.0 #transverse field strength
gamma=1.0 #spin decay rate
alpha=0
N=10
chi=8


### MAGNETIZATIONS:

mx = np.load("data/LR_Ising/Mx_N={}_χ={}_J={}_h={}_γ={}_α={}.npy".format(N,chi,J,h,gamma,alpha) )
my = np.load("data/LR_Ising/My_N={}_χ={}_J={}_h={}_γ={}_α={}.npy".format(N,chi,J,h,gamma,alpha) )
mz = np.load("data/LR_Ising/Mz_N={}_χ={}_J={}_h={}_γ={}_α={}.npy".format(N,chi,J,h,gamma,alpha) )



ED_rho_real = np.load("data/LR_Ising/ED_rho_real_N={}_χ={}_J={}_h={}_γ={}_α={}.npy".format(N,chi,J,h,gamma,alpha) )
ED_rho_imag = np.load("data/LR_Ising/ED_rho_imag_N={}_χ={}_J={}_h={}_γ={}_α={}.npy".format(N,chi,J,h,gamma,alpha) )


MPOMC_rho_real = np.load("data/LR_Ising/MPOMC_rho_real_N={}_χ={}_J={}_h={}_γ={}_α={}.npy".format(N,chi,J,h,gamma,alpha) )
MPOMC_rho_imag = np.load("data/LR_Ising/MPOMC_rho_imag_N={}_χ={}_J={}_h={}_γ={}_α={}.npy".format(N,chi,J,h,gamma,alpha) )





plt.figure(figsize=(8,6),dpi=600)

plt.plot(mx[1:],label="M_x", color='C0')
plt.hlines(mx[0],1,len(mx), colors='C0')

plt.plot(my[1:],label="M_y", color='C1')
plt.hlines(my[0],1,len(my), colors='C1')

plt.plot(mz[1:],label="M_z", color='C2')
plt.hlines(mz[0],1,len(mz), colors='C2')

plt.xscale("log")
plt.xlim(10,len(mx))
plt.xlabel("Optimisation step")
plt.tight_layout()
plt.legend()
plt.title(r"LR Dissipative Ising, $N=10$, $J=0.5, h=1, \gamma=1, \alpha=\infty, \chi=4 \rightarrow 8$")
plt.tight_layout()
plt.savefig("magnetizations.png")
plt.show()



#Calculate final values:
mx_final = np.real(mx[-100:])
my_final = np.real(my[-100:])
mz_final = np.real(mz[-100:])

#mx = 0.277+-0.001
#my = 0.377+-0.001
#mz = -0.242+-0.001



#Calculate fidelity:
import scipy.linalg as sp
a=np.array(ED_rho_real+1j*ED_rho_imag)
b=np.array(MPOMC_rho_real+1j*MPOMC_rho_imag)
print((a))
print((b))
print(np.conjugate(np.transpose(b))==b)

sqrt_a=sp.sqrtm(a)
#print(sqrt_a)
f=(np.trace(sp.sqrtm(np.matmul(sqrt_a,np.matmul(b,sqrt_a)))))**2
print(f)

sqrt_b=sp.sqrtm(b)
#print(sqrt_a)
f=(np.trace(sp.sqrtm(np.matmul(sqrt_b,np.matmul(a,sqrt_b)))))**2
print(f)
