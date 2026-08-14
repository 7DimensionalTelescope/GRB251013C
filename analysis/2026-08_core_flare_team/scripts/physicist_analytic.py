import numpy as np

# constants (cgs)
c = 2.99792458e10
mp = 1.6726219e-24
me = 9.10938e-28
qe = 4.80320e-10
sigT = 6.6524587e-25
z = 0.572
DL = 1.059e28

# FLARE-X core
E = 1.08e52
G0 = 136.0
th = 0.760
n = 134.0
p = 2.121
eps_e = 0.034
eps_B = 0.030
xi = 0.31
tau = 46.5
# RS
p_r, eps_e_r, eps_B_r, xi_r = 2.77, 0.108, 0.528, 0.88

def summarize(E, G0, th, n, eps_e, eps_B, xi, tau, tag=""):
    print(f"===== {tag} =====")
    fb = 1 - np.cos(th)
    print(f"theta_c = {th:.3f} rad = {np.degrees(th):.1f} deg ; f_b = 1-cos = {fb:.4f} ; Omega/4pi(2-sided)={fb:.4f}")
    print(f"E_iso = {E:.3e} ; E_true = f_b*E_iso = {fb*E:.3e} erg")
    # Sedov length
    l = (3*E/(4*np.pi*n*mp*c**2))**(1/3.)
    print(f"Sedov length l = {l:.3e} cm")
    # deceleration radius / time (thin shell)
    Rdec = l/G0**(2/3.)
    tdec_obs = (1+z)*Rdec/(2*G0**2*c)
    print(f"R_dec = {Rdec:.3e} cm ; t_dec(obs) = {tdec_obs:.2f} s")
    # thick vs thin: compare tau (lab-frame shell duration, observer) with tdec
    print(f"shell duration tau = {tau:.1f} s -> {'THICK shell (tau > t_dec)' if tau> tdec_obs else 'THIN shell (tau < t_dec)'}")
    # Sari-Piran critical Lorentz factor for thin/thick
    Gc = (3*E/(32*np.pi*n*mp*c**5*(tau/(1+z))**3))**(1/8.)
    print(f"Gamma_crit (thin if G0<Gc) = {Gc:.1f}  -> {'THIN' if G0<Gc else 'THICK'}")
    # RS crossing time (thick shell) = tau ; (thin shell) = t_dec
    tcross = tau if G0>Gc else tdec_obs
    print(f"RS crossing (obs) ~ {tcross:.1f} s")
    # jet break (Sari Piran Halpern 99): t_j = (1+z) (3E/(32 pi n mp c^5))^(1/3) th^(8/3)
    tj = (1+z)*(3*E/(32*np.pi*n*mp*c**5))**(1/3.)*th**(8/3.)
    print(f"t_jet (Gamma=1/theta) = {tj:.3e} s = {tj/86400:.2f} d")
    # non-relativistic transition: t_NR when swept mass energy ~ E ; R_NR = l, t_NR=(1+z) l/c *(...)
    tNR = (1+z)*l/c/ (2*1**2)  # crude: Gamma~1
    print(f"t_NR ~ (1+z) l/c = {(1+z)*l/c:.3e} s = {(1+z)*l/c/86400:.2f} d")
    # Gamma(t) for spherical: Gamma = (17E/(1024 pi n mp c^5))^{1/8} t^{-3/8}
    for tobs in [1e2, 1e3, 1e4, 1e5, 1e6]:
        t = tobs/(1+z)
        G = (17*E/(1024*np.pi*n*mp*c**5))**(1/8.)*t**(-3/8.)
        G = max(G, 1.0)
        print(f"   t_obs={tobs:8.0e} s  Gamma~{G:7.2f}  1/Gamma={1/G:6.3f} rad ({np.degrees(1/G):5.1f} deg)")
    return tdec_obs, tj

summarize(E,G0,th,n,eps_e,eps_B,xi,tau,"FLARE-X core")

print()
print("=== FS break frequencies (slow cooling, Granot&Sari/Sari98 with xi_e) ===")
# with a fraction xi of electrons accelerated, gamma_m scales as eps_e/xi
def breaks(E,n,eps_e,eps_B,p,xi,tobs):
    t = tobs/(1+z)
    E52 = E/1e52
    # Gamma
    G = (17*E/(1024*np.pi*n*mp*c**5))**(1/8.)*t**(-3/8.)
    # gamma_m
    gm = eps_e/xi*(p-2)/(p-1)*mp/me*G
    B = np.sqrt(32*np.pi*mp*eps_B*n)*G*c
    # gamma_c
    gc = 6*np.pi*me*c/(sigT*B**2*G*t)
    num = lambda g: G*g**2*qe*B/(2*np.pi*me*c)/(1+z)   # obs freq
    nu_m = num(gm); nu_c = num(gc)
    # peak flux
    Ne = 4*np.pi/3*(  ( (17*E/(16*np.pi*n*mp*c**2))**(1/4.) * (c*t)**(1/4.) )**3 )*n*xi  # rough R
    Pnu = me*c**2*sigT*G*B/(3*qe)
    Fmax = Ne*Pnu*(1+z)/(4*np.pi*DL**2)*1e26  # mJy
    return G, nu_m, nu_c, Fmax

for tobs in [1e2, 3e2, 1e3, 3e3, 1e4, 1e5, 1e6]:
    G,nm,nc,Fm = breaks(E,n,eps_e,eps_B,p,xi,tobs)
    print(f"t={tobs:8.0e}  G={G:7.2f}  nu_m={nm:9.3e}  nu_c={nc:9.3e}  F_max~{Fm:8.3f} mJy  (nu_opt=4e14, nu_X=1.2e17)")

print()
print("=== closure relations for observed early optical alpha ~0.6-0.8 ===")
for pp in [1.8,2.0,2.121,2.3]:
    print(f" p={pp:5.3f}: nu_m<nu<nu_c alpha={3*(pp-1)/4:5.3f} beta={-(pp-1)/2:6.3f};  nu>nu_c alpha={(3*pp-2)/4:5.3f} beta={-pp/2:6.3f}")
print(" pre-deceleration (ISM, thin shell) rising FS: alpha = -3 (F ~ t^3) for nu>nu_m")
print(" RS post-crossing thin shell, nu_m,r<nu<nu_c,r: alpha=(27p+7)/35 ->", [(27*pp+7)/35 for pp in [2.0,2.5,2.77,3.0]])
print(" RS post-crossing thick shell: alpha=(73p+21)/96 ->", [(73*pp+21)/96 for pp in [2.0,2.5,2.77,3.0]])
print(" RS Lorentz-factor-of-ejecta decay g ~ t^-(2+g)/... ; standard range alpha_RS = 1.5-3")
