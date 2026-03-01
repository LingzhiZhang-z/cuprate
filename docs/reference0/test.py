import sys
import time
from math import comb
from Clusters_Square import *
from Hubbard_SingleBand import *


N=int(sys.argv[1])


params={}
params['N']=N
params['U']=1
params['t']=0.1
params['sz']= 0 if params['N']%2==0 else 0.5

Nup=int((params['N']+2*params['sz'])/2)
Ndo=int((params['N']-2*params['sz'])/2)
dim = comb(params['N'], Nup) * comb(params['N'], Ndo)
dim_eff = comb(params['N'], Nup)

t0=time.time()
model = Hubbard_SingleBand(params['N'], params['U'], params['t'])
model.set_states(nsites=params['N'], nelec=params['N'], sz=params['sz'])
model.set_states_sort()
model.S2_basis = compute_S2_matrix(model.states[0:dim_eff])
S2_basis_eigvals, S2_basis_eigvecs = np.linalg.eigh(model.S2_basis)
np.savetxt(f"S2/S2_basis_eigvals_spin_N{params['N']}_sz{params['sz']}.npy", S2_basis_eigvals)
np.savetxt(f"S2/S2_basis_eigvecs_spin_N{params['N']}_sz{params['sz']}.npy", S2_basis_eigvecs)

print(f"Finish the calculation of N={params['N']}, sz={params['sz']} in {time.time()-t0:.6f} seconds")