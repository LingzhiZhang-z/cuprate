import time
import os
import math
import itertools
import numpy as np
from cuprate.mpi import rank
from cuprate.spin import (
    comb_safe, compute_S2_matrix,
    solve_S2_blocks,
    canonicalize_eigenpairs, canonicalize_vector_phases,
    sort_s2,
    spin_matrix_J_ij, spin_matrix_JJ_ij,
    spin_matrix_JJJ_ij, spin_matrix_square,
)
from cuprate.io import (
    load_array_compat,
    loadfile,
    nonnegative_sz_values,
    savefile,
    setup_work_environment_previous,
)

# ============================================================
# Fock Space States
# ============================================================

def sum_elec(state, index):
    return sum([abs(state[i]) for i in range(index)])

def total_mag(state):
    return sum([spin for spin in state if abs(spin)==1])

def find_index(lst, value):
    try:
        return lst.index(value)
    except ValueError:
        return -1

def is_half_filled(state):
    return all(s in (-1, 1) for s in state)

def sign_fermi(n):
    return 1 if n % 2 == 0 else -1

def sign_state(state, index):
    return sign_fermi(sum_elec(state,index))

def judge_state_same(state1, state2):
    return np.all((state1-state2) == 0)

def judge_state_diff(state1, state2):
    return np.any((state1-state2))

def Model_States_All(N, *args):
    if len(args) == 1 and isinstance(args[0], list):
        s_values = args[0]
    else:
        s_values = args

    return list(itertools.product(s_values, repeat=N))

def Model_States_Nele(Nsite, Nele, *args):
    states = Model_States_All(Nsite, *args)
    filtered_states = [state for state in states if sum_elec(state, Nsite)==Nele]

    return filtered_states

def Model_States_Nele_Sz(Nsite, Nele, Sz, *args):
    states = Model_States_All(Nsite, *args)
    filtered_states = [state for state in states if sum_elec(state, Nsite)==Nele and math.fabs(total_mag(state)-2*Sz) < 1e-6]
    return filtered_states

def sort_states(state_list):
    sorted_list = sorted(state_list, key=lambda s: (abs(total_mag(s)), -total_mag(s)))
    sorted_mags = [total_mag(s) for s in sorted_list]
    counts = {mag: len(list(group)) for mag, group in itertools.groupby(sorted_mags)}
    return sorted_list, counts

def Model_State_Sort(states):
    states_by_double = {}
    for state in states:
        double_occ = calc_double_occupation(state)
        states_by_double.setdefault(double_occ, []).append(state)

    sorted_states = []
    for double_occ in sorted(states_by_double.keys()):
        grouped_states = states_by_double[double_occ]
        sorted_group, _ = sort_states(grouped_states)
        sorted_states.extend(sorted_group)

    return sorted_states

def Model_States_Spin(N):
    return list(itertools.product([1,-1], repeat=2*N))

def calc_double_occupation(state):
    return state.count(2)

def calc_double_occupation_matrix(states):
    DoubleOccupation = [0]*len(states)
    for i, state in enumerate(states):
        DoubleOccupation[i] = calc_double_occupation(state)
    return np.diag(DoubleOccupation)

# ============================================================
# Math Utilities
# ============================================================

def objective_function(x, A, b):
    non_zero_indices = np.abs(b) > 1e-6
    residual = (A @ x - b)[non_zero_indices]
    mse = np.mean(residual**2)
    return np.sqrt(mse)

def derivative_objective_function(x, A, b):
    relative_error = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
    residual = np.linalg.norm(A @ x - b)
    r2 = 1 - relative_error**2
    return (relative_error, residual, r2)

def norm_matrix(M):
    return np.linalg.norm(M.flatten())

# ============================================================
# Hubbard Model
# ============================================================

def compute_t11m1_norm(eigvecs_selected, dimspin):
    """计算投影质量度量 ||T₁₁ - I||。

    T₁₁ = UΣU† 是选中本征态在自旋模型空间的投影矩阵。
    T₁₁ 越接近单位阵，说明选中的态越好地张成自旋模型子空间。

    eigvecs_selected: 选中本征态的全部分量 (dim_full × n_selected)
    dimspin: 自旋模型子空间维度（半填充态数目）
    """
    try:
        # S_BD: 选中本征态在半填充子空间的投影 (dimspin × n_selected)
        S_BD = eigvecs_selected[0:dimspin, :]
        U, Sigma, VH = np.linalg.svd(S_BD)
        T11m1 = U @ np.diag(Sigma) @ U.conj().T - np.eye(dimspin)
        return norm_matrix(T11m1)
    except Exception:
        return np.inf

class Hubbard_SingleBand:
    """单带 Hubbard 模型。

    核心流程: 构建态空间 → 构建哈密顿量 → 对角化 → 选态 → SVD 提取有效自旋哈密顿量
    支持全对角化、Sz 分块、Sz+S² 分块三种模式。
    """
    def __init__(self, N,  U, *args):
        self.N = N
        self.U = U
        if len(args) == 1 and isinstance(args[0], list):
            self.t_values = args[0]
        else:
            self.t_values = args

        # Initialize empty containers
        self._init_containers()
        self.dimspin = 2**N
        self.dimspin_s2 = -1
        
        # Temporary file prefix (can be updated externally)
        self.tmp_dir = None

    def _init_containers(self):
        """Initialize empty containers for data storage."""
        self.bonds = []    # i, j: index1, index2
        self.states = []
        self.hoppings = [] # t=<000...i_{\sigma}...000|H|000...j_{\sigma}...000>

        self.sz = None

        self.Heff = None
        self.error = False
        self.T11m1 = None
        self.T11m1_norm = np.inf
        self.overlap = None
        self.t11_selected_indices = None
        self.t11_selected_occupation = None
        self.double_occupation_expectation = None

        self.S2 = None

        self.Ham = None
        self.eigvals = None
        self.eigvecs = None

        self.sz_Hams = None
        self.sz_eigvals = None
        self.sz_eigvecs = None
        
        self.szs2_Hams = None
        self.szs2_eigvals = None
        self.szs2_eigvecs = None

        # Block case
        self.if_block = False
        self.if_block_sz = False
        self.if_block_szs2 = False
        self.reconstruct_full = False
        self.select_mode = "full"  # "full" or "block"
        self.mode = "full"
        self.result_kind = "spin_couplings"
        self.supports_spin_couplings = True
        self.match_spin_sectors = False
        self.target_sz = None
        self.target_s = None
        self.target_sz_idx = None
        self.target_s_idx = None

        # Sz case
        self.sz_list = None
        self.sz_states = None
        self.sz_dimspin = None
        self.sz_global_indices = None

        # S^2 basis case
        self.szs2_basis_eigvals = None
        self.szs2_basis_eigvecs = None

        # S^2 case
        self.szs2_list = None
        self.szs2_Us = None
        self.szs2_dimspin = None
        self.szs2_global_indices = None

    def clear(self):
        # Clear numpy arrays
        for attr in ['Ham', 'eigvals', 'eigvecs', 'Heff']:
            if hasattr(self, attr) and isinstance(getattr(self, attr), np.ndarray):
                delattr(self, attr)
                setattr(self, attr, None)
            
        # Clear lists
        self.bonds = []
        self.states = []
        self.hoppings = []
        
        # Reset dimensions
        self.dimspin = 0
        self.dimspin_s2 = -1
        

    def __del__(self):
        self.clear()

    def set_block(self, if_block):
        self.if_block = if_block
        self.reconstruct_full = if_block

    def set_block_sz(self, if_block_sz):
        self.if_block_sz = if_block_sz

    def set_block_szs2(self, if_block_szs2):
        self.if_block_szs2 = if_block_szs2
        if if_block_szs2:
            self.if_block_sz = True

    def set_mode_spec(self, mode_spec):
        self.mode = mode_spec.mode
        self.result_kind = mode_spec.result_kind
        self.supports_spin_couplings = mode_spec.supports_spin_couplings
        self.reconstruct_full = mode_spec.reconstruct_full
        self.target_sz = mode_spec.sz
        self.target_s = mode_spec.s
        self.target_sz_idx = mode_spec.sz_idx
        self.target_s_idx = mode_spec.s_idx
        self.if_block = mode_spec.reconstruct_full and mode_spec.block_sz
        self.if_block_sz = mode_spec.block_sz
        self.if_block_szs2 = mode_spec.block_s2

    def set_select_mode(self, select_mode):
        self.select_mode = select_mode if select_mode is not None else "full"

    def enable_match_spin_sectors(self):
        self.match_spin_sectors = True
        self.if_block_sz = True
        self.if_block_szs2 = True
        self.reconstruct_full = True

    def set_bonds_func(self, func, *args):
        self.bonds, self.hoppings=func(*args)

    def set_bonds(self, bonds, hoppings):
        self.bonds = bonds
        self.hoppings = hoppings

    def set_bonds_by_class(self, bond_classes, class_hoppings):
        # Handle single class case
        if class_hoppings is None:
            return
        elif isinstance(class_hoppings, (int, float)):
            self.bonds.extend(bond_classes)
            self.hoppings.extend([class_hoppings for _ in bond_classes])
            return
            
        # Handle multiple classes case
        self.bonds.extend([bond for bond_class in bond_classes for bond in bond_class])
        for bond_class, hopping in zip(bond_classes, class_hoppings):
            self.hoppings.extend([hopping for _ in bond_class])


    def set_states(self, nsites, nelec, sz_set=None):
        full_states = Model_State_Sort(Model_States_Nele(nsites, nelec, [0, 1,-1, 2]))

        if self.reconstruct_full or sz_set is None:
            self.states = full_states
        elif isinstance(sz_set, (int, float)):
            self.states = Model_State_Sort(Model_States_Nele_Sz(nsites, nelec, float(sz_set), [0, 1, -1, 2]))
        else:
            states_selected = []
            for sz in sz_set:
                states_selected.extend(Model_States_Nele_Sz(nsites, nelec, sz, [0, 1, -1, 2]))
            self.states = Model_State_Sort(states_selected)

        self.dimspin = len([s for s in self.states if is_half_filled(s)])

        if self.if_block_sz or self.if_block_szs2 or sz_set is not None:
            if sz_set is None:
                self.sz_list = nonnegative_sz_values(nsites)
            elif isinstance(sz_set, (int, float)):
                self.sz_list = [float(sz_set)]
            else:
                self.sz_list = list(sz_set)
            self.sz_states = [Model_State_Sort(Model_States_Nele_Sz(nsites, nelec, sz, [0, 1,-1, 2])) for sz in self.sz_list]
            self.sz_dimspin = [len([s for s in states if is_half_filled(s)]) for states in self.sz_states]


    def calc_ham_t_ij(self, state1_ref, state2_ref):
        state1=np.array(state1_ref)
        state2=np.array(state2_ref)
        if(sum_elec(state1, self.N) != sum_elec(state2, self.N)):
            return 0.0+0.0j

        res=0.0+0.0j
        for m in range(self.N):
            for n in range(self.N):
                # Here we only consider bond list considering each bond once in the form of [m<n]
                index=find_index(self.bonds, (m,n) if m<=n else (n,m))
                if(index<0): continue
                if(m<n):
                    t=self.hoppings[index]
                else:
                    t=(self.hoppings[index]).conjugate()


                sign=1.0+0.0j
                spin_m=state1[m]
                spin_n=state2[n]
                
                # c^{\dagger}_{m\up}c_{n\up}
                # Remove Up spin and create Up spin
                if(spin_m==2):                     # Up, Down
                    sign_m=sign_state(state1,m)    # Before this site
                    state1[m]=-1                   # 0 , Down
                elif(spin_m==1):                   # Up, 0
                    sign_m=sign_state(state1,m)    # Before this site
                    state1[m]=0                    # 0 , 0
                else:                              # No Up spin
                    sign_m=0.0                     # 0 , 0 or 0, Down

                if(spin_n==2):                     # Up, Down
                    sign_n=sign_state(state2,n)    # Before this site
                    state2[n]=-1                   # 0 , Down
                elif(spin_n==1):                   # Up, 0
                    sign_n=sign_state(state2,n)    # Before this site
                    state2[n]=0                    # 0 , 0
                else:                              # No Up spin
                    sign_n=0.0                     # 0 , 0 or 0, Down

                if(judge_state_same(state1, state2)):
                    sign=sign_m*sign_n
                    res-=sign*t
                state1[m]=spin_m
                state2[n]=spin_n

                ## c^{\dagger}_{m\down}c_{n\down}
                # Remove Down spin and create Down spin
                if(spin_m==2):                        # Up, Down
                    sign_m=sign_state(state1,m)*(-1)  # Before this site * Up
                    state1[m]=1                       # Up, 0
                elif(spin_m==-1):                     # 0 , Down
                    sign_m=sign_state(state1,m)       # Before this site
                    state1[m]=0                       # 0 , 0
                else:                                 # No Down spin
                    sign_m=0.0                        # 0 , 0 or Up, 0

                if(spin_n==2):                        # Up, Down
                    sign_n=sign_state(state2,n)*(-1)  # Before this site * Up
                    state2[n]=1                       # Up, 0
                elif(spin_n==-1):                     # 0 , Down
                    sign_n=sign_state(state2,n)       # Before this site
                    state2[n]=0                       # 0 , 0
                else:                                 # No Down spin
                    sign_n=0.0                        # 0 , 0 or Up, 0

                if(judge_state_same(state1, state2)):
                    sign=sign_m*sign_n
                    res-=sign*t
                state1[m]=spin_m
                state2[n]=spin_n

        return res
                
    def calc_ham_U_ij(self, state1_ref, state2_ref):
        state1=np.array(state1_ref)
        state2=np.array(state2_ref)

        if(judge_state_diff(state1, state2)):
            return 0.0+0.0j
        
        res=0.0+0.0j
        for m in range(self.N):
            if(state2[m]==2):
                res+=self.U

        return res
    
    def calc_ham_ij(self, state1, state2):
        return self.calc_ham_t_ij(state1, state2) + self.calc_ham_U_ij(state1, state2)
    
    def set_heff(self, Heff):
        self.Heff=Heff

    def restart(self, base_filename):
        self.eigvals = load_array_compat(base_filename + "_eigvals.npy", text_dtype=complex)
        self.eigvecs = load_array_compat(base_filename + "_eigvecs.npy", text_dtype=complex)

    def calc_double_occupation_expectation(self):
        double_occupation_matrix=calc_double_occupation_matrix(self.states)
        return np.diag(self.eigvecs.conj().T @ double_occupation_matrix @ self.eigvecs)

    def calc_heff_halffilled(self, params, params_cluster):
        """从全对角化结果中提取自旋模型的有效哈密顿量。

        流程: 计算双占据期望 → 选态 → SVD 提取 Heff
        核心公式: Heff = U V^H Λ V^H† U†
          - S_BD = eigvecs[半填充行, 选中列]，即选中态在自旋子空间的投影
          - SVD(S_BD) = U Σ V^H
          - Λ = diag(选中态的本征值)
          - T₁₁ = UΣU† 衡量投影质量，T₁₁ ≈ I 时 Heff 可靠
        """
        self.double_occupation_expectation = self.calc_double_occupation_expectation()
        self.t11_selected_indices, best_norm, self.overlap = self.select_eigvecs(params, params_cluster)

        self.t11_selected_occupation = self.double_occupation_expectation[self.t11_selected_indices]

        S_BD = self.eigvecs[0:self.dimspin, self.t11_selected_indices]
        U, Sigma, VH = np.linalg.svd(S_BD)
        self.T11m1 = U @ np.diag(Sigma) @ U.conj().T - np.eye(self.dimspin)
        self.T11m1_norm = norm_matrix(self.T11m1)

        Lambda = np.diag(self.eigvals[self.t11_selected_indices])
        self.Heff = U @ VH @ Lambda @ VH.conj().T @ U.conj().T

    def _prepare_selection_blocks(self, params):
        """将本征态按对称性分块，供选态算法在每个块内独立选取。

        分块策略（优先级从高到低）：
        - match_spin_sectors: 按 (Sz, S²) 分块，每块选 dimspin 个态
        - select_mode="block" + szs2: 按 S² 分块
        - select_mode="block" + sz: 按 Sz 分块
        - 默认: 不分块，所有态在一个块内选取
        """
        double_occ = self.double_occupation_expectation

        if params.get("match_spin_sectors"):
            if self.mode not in ("full", "block_sz_full", "block_szs2_full"):
                raise RuntimeError(
                    "MATCH_SPIN_SECTORS is only supported for full, block_sz_full, and block_szs2_full."
                )
            global_indices = self.szs2_global_indices
            dimspin_blocks = self._expanded_szs2_dimspin()
        elif self.select_mode == "block" and self.if_block_szs2:
            global_indices = self.szs2_global_indices
            dimspin_blocks = self._expanded_szs2_dimspin()
        elif self.select_mode == "block" and self.if_block_sz:
            global_indices = self.sz_global_indices
            dimspin_blocks = self._expanded_sz_dimspin()
        elif params.get('s2') is not None and params.get('s2_fix') and self.szs2_global_indices is not None:
            global_indices = self.szs2_global_indices
            dimspin_blocks = self._expanded_szs2_dimspin()
        else:
            global_indices = [np.arange(len(self.eigvals))]
            dimspin_blocks = [self.dimspin]

        eigvals_blocks = [self.eigvals[idx] for idx in global_indices]
        eigvecs_blocks = [self.eigvecs[:, idx] for idx in global_indices]
        double_occ_blocks = [double_occ[idx] for idx in global_indices]

        return global_indices, eigvals_blocks, eigvecs_blocks, double_occ_blocks, dimspin_blocks

    def select_eigvecs(self, params, params_cluster):
        """选取 dimspin 个本征态使 ||T₁₁ - I|| 最小。

        可用策略:
        - None/occ: 按双占据排序，取最小的 dimspin 个（失败时回退到 single）
        - energy: 按能量排序，取最低的 dimspin 个
        - single: 贪心交换，从 occ 初始解出发逐个尝试替换
        - multi: 随机重启贪心，多次 single 取最优
        - adiabatic: 与前一参数点的选中态做重叠度匹配
        """
        method = params['type']
        global_indices, eigvals_blocks, eigvecs_blocks, double_occ_blocks, dimspin_blocks = \
            self._prepare_selection_blocks(params)

        overlap = None
        best_norm = np.inf

        if method is None:
            selected_blocks, best_norm = self.min_t11_occ(double_occ_blocks, eigvecs_blocks, dimspin_blocks)
            if best_norm is np.inf:
                selected_blocks, best_norm = self.min_t11_single(double_occ_blocks, eigvecs_blocks, dimspin_blocks, ratio=8)
        elif method.lower() == "occ":
            selected_blocks, best_norm = self.min_t11_occ(double_occ_blocks, eigvecs_blocks, dimspin_blocks)
        elif method.lower() == "energy":
            selected_blocks, best_norm = self.min_t11_energy(double_occ_blocks, eigvecs_blocks, dimspin_blocks, eigvals_blocks)
        elif method.lower() == 'single':
            selected_blocks, best_norm = self.min_t11_single(double_occ_blocks, eigvecs_blocks, dimspin_blocks, ratio=8)
        elif method.lower() == 'multi':
            selected_flat, best_norm = self.min_t11_multi(
                double_occ_blocks[0], eigvecs_blocks[0], dimspin_blocks[0],
                ratio=8, n_restarts=40, max_iters_rand=4,
            )
            selected_blocks = [selected_flat]
        elif method.lower() == 'adiabatic':
            dir1, dir2, dir3 = setup_work_environment_previous(params)
            if params['restart']:
                dir2 = f"{dir2}_restart"
            filename_indices = f"{dir1}/{dir2}/hole{params_cluster['hole']}_class{params_cluster['class_idx']}"
            filename_eigvecs = f"{dir1}/{dir3}/hole{params_cluster['hole']}_class{params_cluster['class_idx']}"
            selected_blocks, best_norm, overlap = self.max_overlap_adiabatic(
                eigvecs_blocks, dimspin_blocks, global_indices, filename_eigvecs, filename_indices,
            )
        else:
            raise ValueError(f"Invalid method: {method}")

        # Convert per-block local indices to global indices
        indices_selected = []
        for blk_idx, sel in enumerate(selected_blocks):
            indices_selected.extend(global_indices[blk_idx][i] for i in sel)

        return indices_selected, best_norm, overlap

    def _expanded_sz_dimspin(self):
        if self.sz_global_indices is None or self.sz_dimspin is None:
            return self.sz_dimspin
        if len(self.sz_global_indices) == len(self.sz_dimspin):
            return self.sz_dimspin

        expanded = []
        for sz, dimspin in zip(self.sz_list, self.sz_dimspin):
            expanded.append(dimspin)
            if sz > 0:
                expanded.append(dimspin)

        if len(expanded) != len(self.sz_global_indices):
            raise RuntimeError(
                f"Expanded Sz dimensions do not match reconstructed Sz sectors: "
                f"{len(expanded)} != {len(self.sz_global_indices)}"
            )
        return expanded

    def _expanded_szs2_dimspin(self):
        if self.szs2_global_indices is None or self.szs2_dimspin is None:
            return self.szs2_dimspin
        if len(self.szs2_global_indices) == len(self.szs2_dimspin):
            return self.szs2_dimspin

        expanded = []
        for (sz, _s, _idx_sz), dimspin in zip(self.szs2_list, self.szs2_dimspin):
            expanded.append(dimspin)
            if sz > 0:
                expanded.append(dimspin)

        if len(expanded) != len(self.szs2_global_indices):
            raise RuntimeError(
                f"Expanded Sz/S^2 dimensions do not match reconstructed sectors: "
                f"{len(expanded)} != {len(self.szs2_global_indices)}"
            )
        return expanded


    def _compute_block_t11_norm(self, eigvecs_blocks, selected_blocks, dimspin_blocks):
        dimspin = sum(dimspin_blocks)
        selected = np.concatenate(
            [ev[:, idx] for ev, idx in zip(eigvecs_blocks, selected_blocks)], axis=1
        )
        return compute_t11m1_norm(selected, dimspin)

    def min_t11_occ(self, double_occ_blocks, eigvecs_blocks, dimspin_blocks):
        """occ 策略: 取双占据最小的 dimspin 个态。小 t/U 时最有效。"""
        sorted_blocks = [np.argsort(do) for do in double_occ_blocks]
        selected = [si[:ds] for si, ds in zip(sorted_blocks, dimspin_blocks)]
        return selected, self._compute_block_t11_norm(eigvecs_blocks, selected, dimspin_blocks)

    def min_t11_energy(self, double_occ_blocks, eigvecs_blocks, dimspin_blocks, eigvals_blocks):
        """energy 策略: 取能量最低的 dimspin 个态，再按双占据排序。"""
        sorted_blocks = [np.argsort(ev) for ev in eigvals_blocks]
        selected = [si[:ds] for si, ds in zip(sorted_blocks, dimspin_blocks)]
        sorted_by_occ = [np.argsort(do[sel]) for do, sel in zip(double_occ_blocks, selected)]
        selected = [sel[sbo] for sel, sbo in zip(selected, sorted_by_occ)]
        return selected, self._compute_block_t11_norm(eigvecs_blocks, selected, dimspin_blocks)

    def greedy_swap(self, init_blocks, others_blocks, eigvecs_blocks, dimspin_blocks, f=None):
        """贪心交换: 对每个已选态，尝试与候选池中的态交换，保留使 ||T₁₁-I|| 减小的交换。"""
        selected = [idx.copy() for idx in init_blocks]
        swap = [oth.copy() for oth in others_blocks]

        best_norm = self._compute_block_t11_norm(eigvecs_blocks, selected, dimspin_blocks)
        if best_norm is np.inf and f is not None:
            f.write("Failure in SVD, norm is set to inf\n")

        for blk in range(len(dimspin_blocks)):
            for i in range(len(selected[blk])):
                norm_list = []
                for j in range(len(swap[blk])):
                    selected[blk][i], swap[blk][j] = swap[blk][j], selected[blk][i]

                    t0 = time.time()
                    norm = self._compute_block_t11_norm(eigvecs_blocks, selected, dimspin_blocks)
                    if f is not None and norm is not np.inf:
                        f.write(f"Time cost in SVD: {(time.time()-t0)*1000:.4f}ms, norm: {norm:.12f}\n")
                    elif f is not None and norm is np.inf:
                        f.write(f"Time cost in SVD: {(time.time()-t0)*1000:.4f}ms, failure in SVD\n")
                    norm_list.append(norm)

                    selected[blk][i], swap[blk][j] = swap[blk][j], selected[blk][i]

                index_min = np.argmin(norm_list)
                if norm_list[index_min] < best_norm:
                    best_norm = norm_list[index_min]
                    selected[blk][i], swap[blk][index_min] = swap[blk][index_min], selected[blk][i]

        if best_norm is np.inf:
            if f is not None:
                f.write("All svd failed, norm is set to inf, return the initial indices\n")
            return [idx.copy() for idx in init_blocks], [oth.copy() for oth in others_blocks], best_norm

        return selected, swap, best_norm

    def min_t11_single(self, double_occ_blocks, eigvecs_blocks, dimspin_blocks, ratio=5):
        """single 策略: 从 occ 排序初始解出发，做一轮贪心交换。候选池大小 = ratio × dimspin。"""
        t0 = time.time()
        sorted_blocks = [np.argsort(do) for do in double_occ_blocks]
        sizes = [min(ratio * ds, len(do)) for ds, do in zip(dimspin_blocks, double_occ_blocks)]
        init_blocks = [si[:ds].copy() for si, ds in zip(sorted_blocks, dimspin_blocks)]
        others_blocks = [si[ds:ss].copy() for si, ss, ds in zip(sorted_blocks, sizes, dimspin_blocks)]

        f_svd = None
        if self.tmp_dir is not None:
            suffix = f"_sz{self.sz:.4f}" if self.sz is not None else ""
            f_svd = open(f"{self.tmp_dir}{suffix}_rank{rank}_single.txt", "w")
            f_svd.write("Start the greedy algorithm to find the best t11 indices\n")

        selected, _, best_norm = self.greedy_swap(init_blocks, others_blocks, eigvecs_blocks, dimspin_blocks, f_svd)

        if f_svd is not None:
            f_svd.write(f"Best |T11-1| = {best_norm:.12f}\n")
            f_svd.write(f"Total time cost: {time.time() - t0:.1f}s\n")
            f_svd.close()

        return selected, best_norm

    def min_t11_multi(self, double_occ, eigvecs, dimspin,
                      ratio=4, rand_frac=0.10, ratio_rand_swap=2,
                      n_restarts=10, max_iters_rand=None):
        """multi 策略: 随机重启贪心搜索。

        第一轮从 occ 初始解做贪心交换。若未改善，随机扰动已选态的高双占据部分
        （取 rand_frac 比例），从扰动解重新贪心。最多 n_restarts 轮，连续
        max_iters_rand 轮无改善则停止。适用于 occ/single 可能陷入局部最优的情况。
        """
        t0=time.time()
        iter_rand = 0
        flag_rand = False
        sorted_indices=np.argsort(double_occ)
        size_space_mint11 = min(ratio*dimspin, len(double_occ))

        if self.sz is None:
            f=open(f"{self.tmp_dir}_rank{rank}_multi.txt", "w")
        else:
            f=open(f"{self.tmp_dir}_sz{self.sz:.4f}_rank{rank}_multi.txt", "w")

        if max_iters_rand is None:
            max_iters_rand=n_restarts

        f.write(f"N={self.N}, U={self.U:.4f}, t={self.t_values[0]:.4f}, rank={rank}\n")
        f.write(f"n_restarts={n_restarts}, ratio={ratio}, rand_frac={rand_frac}, max_iters_rand={max_iters_rand}\n\n")
        f.write(f"Start to find the best t11 indices\n")
        f.flush()

        best_selected_indices=sorted_indices[0:dimspin].copy()
        best_swap_indices=sorted_indices[dimspin:size_space_mint11].copy()
        best_norm = compute_t11m1_norm(eigvecs[:, best_selected_indices], dimspin)
        if best_norm is np.inf:
            f.write(f"Failure in SVD, norm is set to inf\n")

        for iter_restarts in range(n_restarts):
            t1=time.time()
            if self.sz is None:
                f_svd=open(f"{self.tmp_dir}_rank{rank}_multi_iter{iter_restarts}.txt", "w")
            else:
                f_svd=open(f"{self.tmp_dir}_sz{self.sz:.4f}_rank{rank}_multi_iter{iter_restarts}.txt", "w")
            f_svd.write(f"Start the greedy algorithm to find the best t11 indices\n")

            if not flag_rand:
                sel_blocks, swap_blocks, cur_norm = self.greedy_swap(
                    [best_selected_indices], [best_swap_indices], [eigvecs], [dimspin], f_svd
                )
                cur_selected_indices, cur_swap_indices = sel_blocks[0], swap_blocks[0]
            else:
                sorted_selected = best_selected_indices[np.argsort(double_occ[best_selected_indices])].copy()
                sorted_swap = best_swap_indices[np.argsort(double_occ[best_swap_indices])].copy()

                n_rand = max(8, int(dimspin * rand_frac))
                if n_rand > 0 and len(sorted_swap) >= n_rand and dimspin >= n_rand:
                    select_space = np.concatenate([sorted_selected[-n_rand:], sorted_swap[:n_rand*ratio_rand_swap]])
                    selected_indices = np.random.choice(len(select_space), size=n_rand, replace=False)
                    remaining_indices = np.setdiff1d(np.arange(len(select_space)), selected_indices)
                    cur_selected_indices = sorted_selected.copy()
                    cur_swap_indices = sorted_swap.copy()
                    cur_selected_indices[-n_rand:] = select_space[selected_indices]
                    cur_swap_indices[:n_rand*ratio_rand_swap] = select_space[remaining_indices]
                else:
                    cur_selected_indices = sorted_selected.copy()
                    cur_swap_indices = sorted_swap.copy()

                sel_blocks, swap_blocks, cur_norm = self.greedy_swap(
                    [cur_selected_indices], [cur_swap_indices], [eigvecs], [dimspin], f_svd
                )
                cur_selected_indices, cur_swap_indices = sel_blocks[0], swap_blocks[0]

            f_svd.write(f"Best |T11-1| = {cur_norm:.12f}\n")
            f_svd.write(f"Total time cost: {time.time()-t1:.1f}s\n")
            f_svd.close()

            if cur_norm < best_norm and abs(cur_norm-best_norm)>1e-8:
                f.write(f"Succeed to find better solution at iteration {iter_restarts+1} / {n_restarts}, time cost: {time.time()-t1:.1f}s, current norm: {cur_norm:.12f}, best norm: {best_norm:.12f}")
                if not flag_rand:
                    f.write(f"\n")
                else:
                    f.write(f", random iteration {iter_rand + 1} / {max_iters_rand}\n")
                flag_rand=False
                iter_rand=0
                best_norm, best_selected_indices, best_swap_indices = cur_norm, cur_selected_indices.copy(), cur_swap_indices.copy()
            else:
                f.write(f" Failed to find better solution at iteration {iter_restarts+1} / {n_restarts}, time cost: {time.time()-t1:.1f}s, current norm: {cur_norm:.12f}, best norm: {best_norm:.12f}")
                if not flag_rand:
                    f.write(f", random swap starts!\n")
                    flag_rand=True
                    iter_rand=0
                else:
                    f.write(f", random iteration {iter_rand + 1} / {max_iters_rand}\n")
                    iter_rand+=1
                    if iter_rand>=max_iters_rand:
                        f.write(f"Break the loop!\n")
                        break
            f.flush()

        f.write(f"Best |T11-1| = {best_norm:.12f}\n")
        f.write(f"Total time cost: {time.time()-t0:.1f}s\n")
        f.close()

        selected_double_occ = double_occ[best_selected_indices]
        sorted_by_occ = np.argsort(selected_double_occ)
        return best_selected_indices[sorted_by_occ], best_norm

    def max_overlap_adiabatic(self, eigvecs_blocks, dimspin_blocks, global_indices, filename_eigvecs, filename_indices):
        """adiabatic 策略: 选与前一参数点选中态重叠度最大的态。

        读取前一参数点的本征态和选中索引，计算重叠矩阵 |⟨ψ_prev|ψ_curr⟩|²，
        在每个块内选总重叠度最大的 dimspin 个态。用于参数扫描时追踪态的连续演化。
        """
        eigvecs_all = np.concatenate(eigvecs_blocks, axis=1)
        eigvecs_previous = load_array_compat(f"{filename_eigvecs}_eigvecs.npy", text_dtype=complex)
        selected_previous = np.atleast_1d(
            load_array_compat(f"{filename_indices}_t11_selected_indices.npy", text_dtype=int)
        ).astype(int)
        eigvecs_previous = eigvecs_previous[:, selected_previous]

        Norms_Matrix = np.abs(eigvecs_previous.conj().T @ eigvecs_all) ** 2
        Norms_vector = np.sum(Norms_Matrix, axis=0)

        block_sizes = [ev.shape[1] for ev in eigvecs_blocks]
        offsets = np.cumsum([0] + block_sizes)
        norms_blocks = [Norms_vector[offsets[i]:offsets[i+1]] for i in range(len(eigvecs_blocks))]

        selected = [np.argsort(nb)[-ds:] for nb, ds in zip(norms_blocks, dimspin_blocks)]
        overlap = sum(np.sum(nb[sel]) for nb, sel in zip(norms_blocks, selected))
        best_norm = self._compute_block_t11_norm(eigvecs_blocks, selected, dimspin_blocks)
        return selected, best_norm, overlap

    def _ensure_spin_coupling_fit_supported(self):
        if not self.supports_spin_couplings:
            raise RuntimeError(
                f"{self.mode} does not determine unique SU(2)-invariant spin couplings. "
                "Use full, fixed_sz, block_sz_full, or block_szs2_full for coupling extraction."
            )

    def calc_spin_coeff(self, bonds, s2=None):
        """将 Heff 拟合为自旋算符的线性组合: Heff ≈ c₀I + Σ Jᵢⱼ Sᵢ·Sⱼ + ...

        构建自旋算符矩阵 {I, S_i·S_j, (S_i·S_j)(S_k·S_l), ...}，
        用最小二乘求解系数。返回 (coeffs, error)。
        """
        self._ensure_spin_coupling_fit_supported()
        dimspin = self.dimspin
        Ms_all = []
        Ms_all.append(np.eye(dimspin, dtype=complex))
        for bond_group in bonds:
            for bond in bond_group:
                if(len(bond)==2):
                    Ms_all.append(spin_matrix_J_ij(self.states[0:self.dimspin], bond))
                elif(len(bond)==4):
                    Ms_all.append(spin_matrix_JJ_ij(self.states[0:self.dimspin], bond))
                elif(len(bond)==6):
                    Ms_all.append(spin_matrix_JJJ_ij(self.states[0:self.dimspin], bond))
        
        
        b = self.Heff.flatten()
        A_all = np.array([Ms_all[i].flatten() for i in range(len(Ms_all))]).T
        coeffs_temp = self._solve_spin_fit(A_all, b)

        # First element is constant offset
        coeffs = [coeffs_temp[0]]
        index = 1
        for bond_group in bonds:
            if len(bond_group) != 0:
                coeffs.append(list(coeffs_temp[index:index + len(bond_group)]))
                index += len(bond_group)
        error = derivative_objective_function(coeffs_temp, A_all, b)
        return (coeffs, error)

    def _solve_spin_fit(self, A, b):
        """解 Ax=b 的最小二乘: 先尝试正规方程，奇异时回退到 lstsq。"""
        try:
            return np.linalg.solve(A.T @ A, A.T @ b)
        except np.linalg.LinAlgError:
            coeffs_temp, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            return coeffs_temp
    
    def calc_spin_coeff_class(self, bonds):
        """同 calc_spin_coeff，但额外按等价键类求平均系数。返回 (individual, class, ind_error, cls_error)。"""
        self._ensure_spin_coupling_fit_supported()
        Ms_all = []
        Ms_class = []
        
        Ms_all.append(np.eye(self.dimspin, dtype=complex))
        Ms_class.append(np.eye(self.dimspin, dtype=complex))
        for i in range(len(bonds)):
            Ms_class.append(np.zeros([self.dimspin, self.dimspin],dtype=complex))
            for j in range(len(bonds[i])):
                bond=bonds[i][j]
                if(len(bond)==2):
                    Ms_all.append(spin_matrix_J_ij(self.states[0:self.dimspin], bond))
                elif(len(bond)==4):
                    Ms_all.append(spin_matrix_square(self.states[0:self.dimspin], bond))
                Ms_class[-1] += Ms_all[-1]
    
        b = self.Heff.flatten()

        A_all = np.array([Ms_all[i].flatten() for i in range(len(Ms_all))]).T
        individual_coeffs_temp = self._solve_spin_fit(A_all, b)

        # First element is constant offset
        individual_coeffs = [individual_coeffs_temp[0]]
        index = 1
        for bond_group in bonds:
            if len(bond_group) != 0:
                individual_coeffs.append(list(individual_coeffs_temp[index:index + len(bond_group)]))
                index += len(bond_group)

        A_class = np.array([Ms_class[i].flatten() for i in range(len(Ms_class))]).T
        class_coeffs = self._solve_spin_fit(A_class, b)

        individual_error = objective_function(individual_coeffs_temp, A_all, b)
        class_error = objective_function(class_coeffs, A_class, b)

        return (individual_coeffs, class_coeffs, individual_error, class_error)
    
    def calc_s2(self):
        """计算所有本征态的 ⟨S²⟩ 和 ⟨S⁴⟩-⟨S²⟩² (方差)。"""
        self.S2_basis = compute_S2_matrix(self.states)

        self.S2 = self.eigvecs.conj().T @ self.S2_basis @ self.eigvecs
        self.S4 = self.eigvecs.conj().T @ self.S2_basis @ self.S2_basis @ self.eigvecs

        self.S2_diag = np.array([self.S2[i][i].real for i in range(len(self.S2))])
        self.S2_error = np.array([self.S4[i][i] - self.S2[i][i]**2 for i in range(len(self.S2))])

    def save_blocks(self, N):
        if not self.if_block_szs2:
            return
        os.makedirs(f"S2", exist_ok=True)
        for idx, (sz, states, dimspin) in enumerate(zip(self.sz_list, self.sz_states, self.sz_dimspin)):
            eigvals, eigvecs = solve_S2_blocks(states)
            savefile(f"S2/S2_basis_eigvals_N{N}_sz{sz}", eigvals, encoding="npy")
            savefile(f"S2/S2_basis_eigvecs_N{N}_sz{sz}", eigvecs, encoding="npy")
        
    def load_blocks(self, N):
        if not self.if_block_szs2:
            return
        self.szs2_basis_eigvals = []
        self.szs2_basis_eigvecs = []
        for idx, (sz, states) in enumerate(zip(self.sz_list, self.sz_states)):
            eigvals = loadfile(f"S2/S2_basis_eigvals_N{N}_sz{sz}", encoding="npy", dtype=float, format="1d")
            eigvecs = loadfile(f"S2/S2_basis_eigvecs_N{N}_sz{sz}", encoding="npy", dtype=complex, format="2d")
            self.szs2_basis_eigvals.append(eigvals)
            self.szs2_basis_eigvecs.append(eigvecs)

    def construct_transform_matrix(self, N):
        """构建 Sz→(Sz,S²) 的幺正变换矩阵，按 S 值分块存储在 szs2_Us 中。"""
        if not self.if_block_szs2:
            return
        self.szs2_list=[]
        self.szs2_Us=[]
        self.szs2_dimspin=[]
        for idx, (sz, states, dimspin) in enumerate(zip(self.sz_list, self.sz_states, self.sz_dimspin)):
            eigvals = self.szs2_basis_eigvals[idx]
            eigvecs = self.szs2_basis_eigvecs[idx]

            s_list, index_groups, bad = sort_s2(eigvals)
            if bad is not None:
                raise RuntimeError(f"Failed to sort eigvals for N={N}, sz={sz}")

            self.szs2_list.extend([(sz, s, idx) for s in s_list])
            self.szs2_Us.extend([eigvecs[:, indices] for indices in index_groups])
            self.szs2_dimspin.extend([comb_safe(N, int(N*0.5-s)) - comb_safe(N, int(N*0.5-s-1)) for s in s_list])

    def analyze_szs2_U_columns(self, atol=1e-6, check=True, print_out=True):
        if not self.if_block_szs2:
            raise RuntimeError("S2 block is not enabled; call set_block(True) first.")
        if self.szs2_Us is None or self.szs2_list is None:
            raise RuntimeError("S2 transform matrices are not constructed; call construct_transform_matrix first.")

        s2_cache = {}
        diag_cache = {}
        results = []

        if print_out:
            print("blk  col  sz_blk  sz_cal  s_blk  s_cal  double_occ  s2_cal")

        for blk_idx, (sz, s, idx_sz) in enumerate(self.szs2_list):
            if idx_sz not in s2_cache:
                basis = self.sz_states[idx_sz]
                s2_cache[idx_sz] = compute_S2_matrix(basis)
                sz_vals = np.array([total_mag(state) * 0.5 for state in basis], dtype=float)
                do_vals = np.array([calc_double_occupation(state) for state in basis], dtype=float)
                diag_cache[idx_sz] = (sz_vals, do_vals)

            s2_matrix = s2_cache[idx_sz]
            sz_vals, do_vals = diag_cache[idx_sz]
            U = self.szs2_Us[blk_idx]
            target_s2 = s * (s + 1)

            for col in range(U.shape[1]):
                vec = U[:, col]
                s2_exp = float(np.real(np.vdot(vec, s2_matrix @ vec)))
                sz_exp = float(np.real(np.vdot(vec, sz_vals * vec)))
                do_exp = float(np.real(np.vdot(vec, do_vals * vec)))
                s_cal_raw = (-1.0 + np.sqrt(1.0 + 4.0 * s2_exp)) / 2.0
                s_cal = round(s_cal_raw * 2) / 2

                if print_out:
                    print(
                        f"{blk_idx:3d}  {col:3d}  {sz:6.2f}  {sz_exp:6.2f}  "
                        f"{s:5.1f}  {s_cal:5.1f}  {do_exp:10.6f}  {s2_exp:8.6f}"
                    )

                if check:
                    if abs(sz_exp - sz) > atol:
                        raise AssertionError(
                            f"Sz mismatch in block {blk_idx}, col {col}: "
                            f"expected {sz}, got {sz_exp}"
                        )
                    if abs(s2_exp - target_s2) > atol:
                        raise AssertionError(
                            f"S2 mismatch in block {blk_idx}, col {col}: "
                            f"expected {target_s2}, got {s2_exp}"
                        )
                    if abs(do_exp - round(do_exp)) > atol:
                        raise AssertionError(
                            f"Double occupation not integer in block {blk_idx}, col {col}: "
                            f"got {do_exp}"
                        )

                results.append({
                    "block": blk_idx,
                    "col": col,
                    "idx_sz": idx_sz,
                    "s": s,
                    "sz": sz_exp,
                    "s2": s2_exp,
                    "double_occ": do_exp,
                })

        return results

    def summarize_szs2_blocks(self, atol=1e-6, print_out=True, print_do_values=True, print_do_sequence=False):
        if not self.if_block_szs2:
            raise RuntimeError("S2 block is not enabled; call set_block(True) first.")
        if self.szs2_Us is None or self.szs2_list is None:
            raise RuntimeError("S2 transform matrices are not constructed; call construct_transform_matrix first.")

        s2_cache = {}
        diag_cache = {}
        summary = []

        if print_out:
            print("blk  sz_blk  s_blk  ncols  sz_ok  s_ok  do_unique  do_zero  dimspin  do_match  do_sorted  bad_sz  bad_s")

        for blk_idx, (sz, s, idx_sz) in enumerate(self.szs2_list):
            if idx_sz not in s2_cache:
                basis = self.sz_states[idx_sz]
                s2_cache[idx_sz] = compute_S2_matrix(basis)
                sz_vals = np.array([total_mag(state) * 0.5 for state in basis], dtype=float)
                do_vals = np.array([calc_double_occupation(state) for state in basis], dtype=float)
                diag_cache[idx_sz] = (sz_vals, do_vals)

            s2_matrix = s2_cache[idx_sz]
            sz_vals, do_vals = diag_cache[idx_sz]
            U = self.szs2_Us[blk_idx]

            bad_sz = 0
            bad_s = 0
            do_set = set()
            do_zero = 0
            do_seq = []
            basis = self.sz_states[idx_sz]
            nsites = len(basis[0]) if basis else 0
            for col in range(U.shape[1]):
                vec = U[:, col]
                s2_exp = float(np.real(np.vdot(vec, s2_matrix @ vec)))
                sz_exp = float(np.real(np.vdot(vec, sz_vals * vec)))
                do_exp = float(np.real(np.vdot(vec, do_vals * vec)))
                s_cal_raw = (-1.0 + np.sqrt(1.0 + 4.0 * s2_exp)) / 2.0
                s_cal = round(s_cal_raw * 2) / 2

                if abs(sz_exp - sz) > atol:
                    bad_sz += 1
                if abs(s_cal - s) > atol or abs(s2_exp - s * (s + 1)) > atol:
                    bad_s += 1

                do_round = round(do_exp)
                if abs(do_exp - do_round) <= atol:
                    do_int = int(do_round)
                    do_set.add(do_int)
                    if do_int == 0:
                        do_zero += 1
                    do_seq.append(float(do_int))
                else:
                    do_set.add(round(do_exp, 6))
                    do_seq.append(do_exp)

            sz_ok = (bad_sz == 0)
            s_ok = (bad_s == 0)
            do_values = sorted(do_set)
            do_sorted = all(do_seq[i] <= do_seq[i + 1] + atol for i in range(len(do_seq) - 1))
            dimspin_ref = None
            if self.szs2_dimspin is not None and len(self.szs2_dimspin) == len(self.szs2_list):
                dimspin_ref = self.szs2_dimspin[blk_idx]
            if dimspin_ref is None and nsites > 0:
                dimspin_ref = comb_safe(nsites, int(nsites * 0.5 - s)) - comb_safe(nsites, int(nsites * 0.5 - s - 1))
            do_match = (dimspin_ref is None) or (do_zero == dimspin_ref)
            entry = {
                "block": blk_idx,
                "sz_blk": sz,
                "s_blk": s,
                "ncols": U.shape[1],
                "sz_ok": sz_ok,
                "s_ok": s_ok,
                "do_unique": len(do_set),
                "do_values": do_values,
                "do_zero": do_zero,
                "dimspin": dimspin_ref,
                "do_match": do_match,
                "do_sorted": do_sorted,
                "do_sequence": do_seq,
                "bad_sz": bad_sz,
                "bad_s": bad_s,
            }
            summary.append(entry)

            if print_out:
                print(
                    f"{blk_idx:3d}  {sz:6.2f}  {s:5.1f}  {U.shape[1]:5d}  "
                    f"{str(sz_ok):5s}  {str(s_ok):4s}  {len(do_set):9d}  "
                    f"{do_zero:7d}  {str(dimspin_ref):7s}  {str(do_match):8s}  "
                    f"{str(do_sorted):9s}  {bad_sz:6d}  {bad_s:5d}"
                )
                if print_do_values:
                    do_text = ", ".join(str(v) for v in do_values)
                    print(f"     do_values: [{do_text}]")
                if print_do_sequence:
                    do_seq_text = ", ".join(f"{v:.0f}" if abs(v - round(v)) <= atol else f"{v:.6f}" for v in do_seq)
                    print(f"     do_sequence: [{do_seq_text}]")

        return summary

    def calc_hamiltonian(self):
        """构建哈密顿矩阵。根据分块设置可构建全矩阵、Sz 分块或 Sz+S² 分块。"""
        if not self.if_block_sz and self.sz_list is None:
            self.Ham = np.zeros([len(self.states), len(self.states)], dtype=complex)
            for i in range(len(self.states)):
                for j in range(len(self.states)):
                    self.Ham[i][j] = self.calc_ham_ij(self.states[i], self.states[j])
        elif self.sz_list is not None:
            self.sz_Hams = [np.zeros([len(states), len(states)], dtype=complex) for states in self.sz_states]
            for idx, states in enumerate(self.sz_states):
                for i in range(len(states)):
                    for j in range(len(states)):
                        self.sz_Hams[idx][i][j] = self.calc_ham_ij(states[i], states[j])

            if self.if_block_szs2:
                self.szs2_Hams = [None for _ in range(len(self.szs2_list))]
                for idx, (sz, s, idx_sz) in enumerate(self.szs2_list):
                    U = self.szs2_Us[idx]
                    self.szs2_Hams[idx] = U.conj().T @ self.sz_Hams[idx_sz] @ U

    def _spin_flip_state(self, state):
        flipped = []
        for occupation in state:
            if occupation == 1:
                flipped.append(-1)
            elif occupation == -1:
                flipped.append(1)
            else:
                flipped.append(occupation)
        return tuple(flipped)

    def _build_sz_perm(self):
        """Build per-Sz permutation mapping from local Sz basis to global state basis."""
        state_to_global = {tuple(state): idx for idx, state in enumerate(self.states)}
        perm_per_sz = []
        for sz_states in self.sz_states:
            perm = [state_to_global[tuple(state)] for state in sz_states]
            perm_per_sz.append(perm)
        return perm_per_sz

    def _build_mirrored_rows(self, states_local):
        state_to_global = {tuple(state): idx for idx, state in enumerate(self.states)}
        return [state_to_global[self._spin_flip_state(state)] for state in states_local]

    def _build_mirrored_eigvecs(self, states_local, local_eigvecs):
        signs = np.array([(-1) ** calc_double_occupation(state) for state in states_local], dtype=float)
        mirrored = signs[:, np.newaxis] * local_eigvecs
        return canonicalize_vector_phases(mirrored)

    def _assemble_reconstructed_eigensystem(self, sector_entries, attr_name):
        """将各 sector 的本征值/本征态拼接成全局有序的本征系统，记录块索引映射。"""
        n_total = len(self.states)
        eigvals_cat = np.concatenate([entry[0] for entry in sector_entries])
        sort_idx = np.argsort(eigvals_cat)
        self.eigvals = eigvals_cat[sort_idx]

        eigvecs = np.zeros((n_total, len(eigvals_cat)), dtype=complex)
        block_ranges = []
        col_offset = 0
        for eigvals_block, rows, local_eigvecs in sector_entries:
            n_cols = len(eigvals_block)
            for local_i, global_i in enumerate(rows):
                eigvecs[global_i, col_offset:col_offset + n_cols] = local_eigvecs[local_i, :]
            block_ranges.append(np.arange(col_offset, col_offset + n_cols))
            col_offset += n_cols

        self.eigvecs = eigvecs[:, sort_idx]
        inv_sort = np.argsort(sort_idx)
        setattr(self, attr_name, [inv_sort[block_range] for block_range in block_ranges])
        if attr_name == "sz_global_indices":
            self.szs2_global_indices = None
        else:
            self.sz_global_indices = None

    def _reconstruct_from_sz(self):
        """Reconstruct full eigvals/eigvecs from Sz-blocked diagonalization."""
        perm_per_sz = self._build_sz_perm()
        sector_entries = []
        for sz, sz_states, sz_eigvals, sz_eigvecs, perm in zip(
            self.sz_list, self.sz_states, self.sz_eigvals, self.sz_eigvecs, perm_per_sz
        ):
            sector_entries.append((sz_eigvals, perm, sz_eigvecs))
            if sz > 0:
                sector_entries.append((
                    sz_eigvals,
                    self._build_mirrored_rows(sz_states),
                    self._build_mirrored_eigvecs(sz_states, sz_eigvecs),
                ))

        self._assemble_reconstructed_eigensystem(sector_entries, "sz_global_indices")

    def _reconstruct_from_szs2(self):
        """Reconstruct full eigvals/eigvecs from Sz+S^2-blocked diagonalization."""
        perm_per_sz = self._build_sz_perm()
        sector_entries = []
        for idx, (sz, s, idx_sz) in enumerate(self.szs2_list):
            U = self.szs2_Us[idx]
            s2_eigvals = self.szs2_eigvals[idx]
            s2_eigvecs = self.szs2_eigvecs[idx]
            sz_eigvecs = canonicalize_vector_phases(U @ s2_eigvecs)
            sector_entries.append((s2_eigvals, perm_per_sz[idx_sz], sz_eigvecs))
            if sz > 0:
                sector_entries.append((
                    s2_eigvals,
                    self._build_mirrored_rows(self.sz_states[idx_sz]),
                    self._build_mirrored_eigvecs(self.sz_states[idx_sz], sz_eigvecs),
                ))

        self._assemble_reconstructed_eigensystem(sector_entries, "szs2_global_indices")

    def _find_fixed_szs2_block_index(self):
        if self.szs2_list is None:
            raise RuntimeError("S^2 blocks are unavailable; call construct_transform_matrix first.")
        if self.target_s is None and self.target_s_idx is None:
            raise RuntimeError("Target S sector is not configured for fixed_sz_s2 mode.")

        matching_blocks = []
        for idx, (sz, s, idx_sz) in enumerate(self.szs2_list):
            if self.target_sz is not None and abs(sz - self.target_sz) > 1e-8:
                continue
            matching_blocks.append((idx, s))

        if not matching_blocks:
            raise RuntimeError(f"No S^2 blocks found for target Sz={self.target_sz}")

        if self.target_s_idx is not None:
            if self.target_s_idx < 0 or self.target_s_idx >= len(matching_blocks):
                raise RuntimeError(f"Target S index {self.target_s_idx} is out of range for Sz={self.target_sz}")
            return matching_blocks[self.target_s_idx][0]

        for idx, s in matching_blocks:
            if abs(s - self.target_s) < 1e-8:
                return idx
        raise RuntimeError(f"No S^2 block found for target S={self.target_s} within Sz={self.target_sz}")

    def solve(self):
        """对角化哈密顿矩阵。分块模式下按块对角化，再根据需要重构全谱或取单块。"""
        if not self.if_block_sz and self.sz_list is None:
            self.eigvals, self.eigvecs = np.linalg.eigh(self.Ham)
            self.eigvals, self.eigvecs = canonicalize_eigenpairs(self.eigvals, self.eigvecs)
            self.sz_global_indices = None
            self.szs2_global_indices = None
        elif self.sz_list is not None:
            if self.if_block_szs2:
                self.szs2_eigvals = []
                self.szs2_eigvecs = []
                for szs2_Ham in self.szs2_Hams:
                    szs2_eigvals, szs2_eigvecs = np.linalg.eigh(szs2_Ham)
                    szs2_eigvals, szs2_eigvecs = canonicalize_eigenpairs(szs2_eigvals, szs2_eigvecs)
                    self.szs2_eigvals.append(szs2_eigvals)
                    self.szs2_eigvecs.append(szs2_eigvecs)
                if self.reconstruct_full:
                    self._reconstruct_from_szs2()
                else:
                    block_idx = self._find_fixed_szs2_block_index()
                    U = self.szs2_Us[block_idx]
                    self.eigvals = self.szs2_eigvals[block_idx]
                    self.eigvecs = canonicalize_vector_phases(U @ self.szs2_eigvecs[block_idx])
                    self.dimspin = self.szs2_dimspin[block_idx]
                    self.sz_global_indices = None
                    self.szs2_global_indices = None
            elif self.reconstruct_full:
                self.sz_eigvals = []
                self.sz_eigvecs = []
                for sz_Ham in self.sz_Hams:
                    sz_eigvals, sz_eigvecs = np.linalg.eigh(sz_Ham)
                    sz_eigvals, sz_eigvecs = canonicalize_eigenpairs(sz_eigvals, sz_eigvecs)
                    self.sz_eigvals.append(sz_eigvals)
                    self.sz_eigvecs.append(sz_eigvecs)
                self._reconstruct_from_sz()
            else:
                self.eigvals, self.eigvecs = np.linalg.eigh(self.sz_Hams[0])
                self.eigvals, self.eigvecs = canonicalize_eigenpairs(self.eigvals, self.eigvecs)
                self.sz_global_indices = None
                self.szs2_global_indices = None



    def save_data(self, base_filename):

        np.save(f"{base_filename}_Heff.npy", np.asarray(self.Heff))
        np.save(f"{base_filename}_T11m1.npy", np.asarray(self.T11m1))
        np.save(f"{base_filename}_t11_selected_indices.npy", np.asarray(self.t11_selected_indices, dtype=int))
        np.save(f"{base_filename}_t11_selected_occupation.npy", np.asarray(self.t11_selected_occupation))
        np.save(f"{base_filename}_double_occupation_expectation.npy", np.asarray(self.double_occupation_expectation))
        np.save(f"{base_filename}_states.npy", np.asarray(self.states, dtype=int))
        np.save(f"{base_filename}_s2_digonal.npy", np.column_stack([self.S2_diag, self.S2_error]))
        np.save(
            f"{base_filename}_s2_selected.npy",
            np.column_stack([self.S2_diag[self.t11_selected_indices], self.S2_error[self.t11_selected_indices]]),
        )
