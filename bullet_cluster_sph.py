#!/usr/bin/env python3
"""
Le Flux v10 — Bullet Cluster SPH Simulation
IC Generator + Analysis (v6, May 2026)

Generates GADGET-2 initial conditions for the Bullet Cluster merger
using the Le Flux QGP substrate profile rho ∝ (rc²+r²)^(-3/2)
and hydrostatic equilibrium gas: rho_gas ∝ exp(-phi/u_therm)

Usage:
    python3 bullet_cluster_sph.py --generate    # generate ICs
    python3 bullet_cluster_sph.py --analyze     # analyze snapshots
    python3 bullet_cluster_sph.py --plot        # generate figures

Requirements:
    numpy, scipy, matplotlib
    GADGET-2 (https://wwwmpa.mpa-garching.mpg.de/gadget/)

Reference: Sureau A. (2026). Le Flux v10, Addendum 4.
DOI: 10.5281/zenodo.19219720
"""

import numpy as np
from scipy.interpolate import interp1d
import struct, os, sys, time, argparse

# ── PHYSICAL CONSTANTS (GADGET internal units) ──────────────────────────────
# UnitLength = 1 kpc = 3.086e21 cm
# UnitMass   = 1e10 Msun = 1.989e43 g
# UnitVel    = 1 km/s = 1e5 cm/s
# G = G_cgs * UnitMass / (UnitLength * UnitVel^2)
G       = 4.3016e4    # kpc (km/s)^2 / (1e10 Msun)  ← CORRECT VALUE
k_B     = 1.381e-16   # erg/K
mp      = 1.673e-24   # g
mu      = 0.588       # mean molecular weight (fully ionized plasma)
gamma   = 5.0/3.0
UnitVel = 1e5         # cm/s
U_FLOOR = 1000.0 * k_B / ((gamma-1)*mu*mp) / UnitVel**2  # ~21.06 (1000 K)

# ── SIMULATION PARAMETERS ───────────────────────────────────────────────────
PARAMS = {
    # Main cluster (2e15 Msun)
    'M_dm_main':    200000.0,   # 1e10 Msun
    'M_gas_main':    20000.0,
    'r200_main':      2136.0,   # kpc
    'rc_main':         200.0,   # Le Flux coherence length
    'T_main':            8.0,   # keV

    # Bullet sub-cluster (2.2e14 Msun)
    'M_dm_bullet':   22000.0,
    'M_gas_bullet':   2000.0,
    'r200_bullet':     995.0,
    'rc_bullet':        55.0,
    'T_bullet':          2.0,   # keV

    # Collision setup
    'v_coll':         4500.0,   # km/s (relative, Markevitch+2002)
    'd_sep':          3000.0,   # kpc initial separation
    'BoxSize':       12000.0,   # kpc

    # Resolution (1M total)
    'N_dm_main':    600000,
    'N_dm_bullet':  100000,
    'N_gas_main':   250000,
    'N_gas_bullet':  50000,
}


def build_menc_lf(M, r200, rc, N=5000):
    """Enclosed mass for Le Flux profile: rho ∝ (rc²+r²)^(-3/2)"""
    r = np.logspace(np.log10(0.5), np.log10(r200*1.5), N)
    rho = 1.0/(rc**2+r**2)**1.5
    norm = np.trapezoid(4*np.pi*r**2*rho, r)
    rho0 = M/norm
    mass = np.zeros(N)
    for i in range(1, N):
        mass[i] = np.trapezoid(rho0*4*np.pi*r[:i+1]**2/(rc**2+r[:i+1]**2)**1.5, r[:i+1])
    return interp1d(r, mass, bounds_error=False, fill_value=(0, mass[-1]))


def build_lf_sampler(r200, rc, N=8000):
    """CDF sampler for Le Flux substrate profile"""
    r = np.logspace(np.log10(0.5), np.log10(r200), N)
    shell = 4*np.pi*r**2/(rc**2+r**2)**1.5
    cdf = np.zeros(N)
    for i in range(1, N): cdf[i] = np.trapezoid(shell[:i+1], r[:i+1])
    cdf /= cdf[-1]
    return interp1d(cdf, r, bounds_error=False, fill_value=(r[0], r[-1]))


def build_gas_hydrostatic(M_dm, r200, rc, M_gas, T_keV, N=8000):
    """
    Gas density in exact hydrostatic equilibrium with Le Flux potential.

    rho_gas(r) = rho0 * exp(-phi(r)/u_therm)
    phi(r) = integral_r^{5*r200} G*M(<r')/r'^2 dr' + G*M/r_{max}

    This guarantees CoM_gas = CoM_DM by construction (spherical symmetry).
    """
    r = np.logspace(np.log10(1.0), np.log10(r200*5), N)
    Menc = build_menc_lf(M_dm, r200, rc, N)(r)
    g = G*Menc/r**2

    # Gravitational potential (zero at infinity)
    phi = np.zeros(N)
    phi[-1] = G*M_dm/r[-1]
    for i in range(N-2, -1, -1):
        phi[i] = phi[i+1] + 0.5*(g[i]+g[i+1])*(r[i+1]-r[i])

    # Thermal energy
    u_therm = T_keV * 1.16e7 * k_B / ((gamma-1)*mu*mp) / UnitVel**2

    # Hydrostatic density
    rho_raw = np.exp(-phi/u_therm)
    rho_raw = np.clip(rho_raw, 1e-30, None)

    # CDF for sampling
    shell = 4*np.pi*r**2*rho_raw
    cdf = np.zeros(N)
    for i in range(1, N): cdf[i] = np.trapezoid(shell[:i+1], r[:i+1])
    cdf /= cdf[-1]

    u_val = max(u_therm, U_FLOOR)
    T_c = u_therm * UnitVel**2 * (gamma-1)*mu*mp / k_B / 1.16e7
    print(f"   T_core={T_c:.2f} keV  phi/u={phi[0]/u_therm:.3f}")

    return interp1d(cdf, r, bounds_error=False, fill_value=(r[0], r[-1])), u_val


def radii_to_pos(r):
    """Sample isotropic 3D positions from radii"""
    c = np.random.uniform(-1, 1, len(r))
    s = np.sqrt(1-c**2)
    p = np.random.uniform(0, 2*np.pi, len(r))
    return np.column_stack([r*s*np.cos(p), r*s*np.sin(p), r*c])


def jeans_vel(r, Menc_fn):
    """Isotropic Jeans velocities: sigma^2 = G*M(<r)/(2r)"""
    sig = np.sqrt(np.clip(G*Menc_fn(r)/(2*np.clip(r, 10, None)), 0, None))
    return np.column_stack([np.random.normal(0, sig),
                            np.random.normal(0, sig),
                            np.random.normal(0, sig)]).astype(np.float32)


def calc_hsml(r, r200, N):
    """Adaptive SPH smoothing lengths"""
    return ((4/3)*np.pi*r200**3/N)**(1/3) * 1.2 * (1+r/r200)**0.3


def write_gadget_block(f, arr):
    data = arr.tobytes(); s = len(data)
    f.write(struct.pack('i', s)); f.write(data); f.write(struct.pack('i', s))


def generate_ics(outfile='/root/bullet_1M_v6.dat', seed=42):
    """Generate GADGET-2 binary ICs for the Bullet Cluster"""
    np.random.seed(seed)
    p = PARAMS

    M_tot = p['M_dm_main'] + p['M_dm_bullet']
    BoxSize = p['BoxSize']
    x_main   = BoxSize/2 - (p['M_dm_bullet'] * p['d_sep']) / M_tot
    x_bullet = BoxSize/2 + (p['M_dm_main']   * p['d_sep']) / M_tot

    print(f"\n=== Le Flux Bullet Cluster IC Generator v6 ===")
    print(f"G = {G:.4e}  (correct GADGET internal value)")
    print(f"x_main={x_main:.1f}  x_bullet={x_bullet:.1f}  d_sep={x_bullet-x_main:.1f} kpc")
    T_vir = G*p['M_dm_main']/p['r200_main']/3 * UnitVel**2*(gamma-1)*mu*mp/k_B/1.16e7
    print(f"T_virial_main ~ {T_vir:.1f} keV")
    assert x_bullet < BoxSize*0.95 and x_main > BoxSize*0.05

    print("\n[1] Hydrostatic profiles...")
    print("   Main cluster:")
    gas_m_samp, u_m = build_gas_hydrostatic(p['M_dm_main'], p['r200_main'], p['rc_main'],
                                              p['M_gas_main'], p['T_main'])
    print("   Bullet:")
    gas_b_samp, u_b = build_gas_hydrostatic(p['M_dm_bullet'], p['r200_bullet'], p['rc_bullet'],
                                              p['M_gas_bullet'], p['T_bullet'])
    cdf_dm_m = build_lf_sampler(p['r200_main'],   p['rc_main'])
    cdf_dm_b = build_lf_sampler(p['r200_bullet'], p['rc_bullet'])
    Menc_m   = build_menc_lf(p['M_dm_main'],   p['r200_main'],   p['rc_main'])
    Menc_b   = build_menc_lf(p['M_dm_bullet'], p['r200_bullet'], p['rc_bullet'])

    print("[2] Sampling positions...")
    r_dm_m  = cdf_dm_m(np.random.uniform(0,1,p['N_dm_main']))
    r_dm_b  = cdf_dm_b(np.random.uniform(0,1,p['N_dm_bullet']))
    r_gas_m = gas_m_samp(np.random.uniform(0,1,p['N_gas_main']))
    r_gas_b = gas_b_samp(np.random.uniform(0,1,p['N_gas_bullet']))
    pos_dm_m=radii_to_pos(r_dm_m); pos_dm_b=radii_to_pos(r_dm_b)
    pos_gas_m=radii_to_pos(r_gas_m); pos_gas_b=radii_to_pos(r_gas_b)

    print("[3] Jeans velocities...")
    vel_dm_m=jeans_vel(r_dm_m,Menc_m); vel_dm_b=jeans_vel(r_dm_b,Menc_b)
    vel_gas_m=jeans_vel(r_gas_m,Menc_m); vel_gas_b=jeans_vel(r_gas_b,Menc_b)

    print("[4] Recentering per subcluster...")
    for arr in [pos_dm_m,pos_gas_m,pos_dm_b,pos_gas_b]: arr -= arr.mean(axis=0)
    for arr in [vel_dm_m,vel_gas_m]: arr -= arr.mean(axis=0)   # main
    for arr in [vel_dm_b,vel_gas_b]: arr -= arr.mean(axis=0)   # bullet

    print("[5] Placing clusters + bulk velocities...")
    pos_dm_m  += [x_main,   BoxSize/2, BoxSize/2]; vel_dm_m [:,0] += +p['v_coll']/2
    pos_gas_m += [x_main,   BoxSize/2, BoxSize/2]; vel_gas_m[:,0] += +p['v_coll']/2
    pos_dm_b  += [x_bullet, BoxSize/2, BoxSize/2]; vel_dm_b [:,0] += -p['v_coll']/2
    pos_gas_b += [x_bullet, BoxSize/2, BoxSize/2]; vel_gas_b[:,0] += -p['v_coll']/2

    print("[6] Assembling...")
    N_gas = p['N_gas_main']+p['N_gas_bullet']; N_dm = p['N_dm_main']+p['N_dm_bullet']
    pos_gas_all = np.vstack([pos_gas_m,pos_gas_b]).astype(np.float32)
    vel_gas_all = np.vstack([vel_gas_m,vel_gas_b]).astype(np.float32)
    pos_dm_all  = np.vstack([pos_dm_m, pos_dm_b]).astype(np.float32)
    vel_dm_all  = np.vstack([vel_dm_m, vel_dm_b]).astype(np.float32)
    u_gas_all   = np.concatenate([np.full(p['N_gas_main'],  u_m, dtype=np.float32),
                                   np.full(p['N_gas_bullet'],u_b, dtype=np.float32)])
    m_gas_all   = np.concatenate([np.full(p['N_gas_main'],  p['M_gas_main']/p['N_gas_main'],  dtype=np.float32),
                                   np.full(p['N_gas_bullet'],p['M_gas_bullet']/p['N_gas_bullet'],dtype=np.float32)])
    m_dm_all    = np.concatenate([np.full(p['N_dm_main'],   p['M_dm_main']/p['N_dm_main'],   dtype=np.float32),
                                   np.full(p['N_dm_bullet'], p['M_dm_bullet']/p['N_dm_bullet'], dtype=np.float32)])
    hsml_all    = np.concatenate([calc_hsml(r_gas_m,p['r200_main'],  p['N_gas_main']),
                                   calc_hsml(r_gas_b,p['r200_bullet'],p['N_gas_bullet'])]).astype(np.float32)
    ids_gas = np.arange(1,       N_gas+1,       dtype=np.uint32)
    ids_dm  = np.arange(N_gas+1, N_gas+N_dm+1,  dtype=np.uint32)

    print("[7] Global CoM velocity correction (Bohrium recommendation)...")
    all_vel  = np.vstack([vel_gas_all, vel_dm_all])
    all_mass = np.concatenate([m_gas_all, m_dm_all])
    v_com = np.average(all_vel, axis=0, weights=all_mass)
    vel_gas_all -= v_com; vel_dm_all -= v_com
    print(f"   V_com correction: {v_com}")

    print("[8] Verification...")
    dx_m  = pos_gas_m[:,0].mean()-pos_dm_m[:,0].mean()
    dx_b  = pos_gas_b[:,0].mean()-pos_dm_b[:,0].mean()
    vb_m  = vel_dm_all[:N_dm//2,0].mean() if N_dm > 0 else 0
    print(f"   dx_main={dx_m:+.2f} kpc  dx_bullet={dx_b:+.2f} kpc")
    print(f"   N_gas={N_gas}  N_dm={N_dm}  N_tot={N_gas+N_dm}")
    if abs(dx_m) > 5: print("WARNING: dx_main > 5 kpc")
    if abs(dx_b) > 5: print("WARNING: dx_bullet > 5 kpc")

    print("[9] Writing IC file...")
    npart = [N_gas, N_dm, 0, 0, 0, 0]
    hdr  = struct.pack('6i', *npart) + struct.pack('6d', *[0.0]*6)
    hdr += struct.pack('d', 0.0) + struct.pack('d', 0.0)
    hdr += struct.pack('i', 0) + struct.pack('i', 0)
    hdr += struct.pack('6i', *npart) + struct.pack('i', 0) + struct.pack('i', 1)
    hdr += struct.pack('d', BoxSize) + struct.pack('d', 0.3) + struct.pack('d', 0.7) + struct.pack('d', 0.7)
    hdr += struct.pack('i', 0) + struct.pack('i', 0) + struct.pack('6i', *[0]*6) + struct.pack('i', 0)
    hdr += b'\x00'*(256-len(hdr))
    assert len(hdr) == 256

    with open(outfile, 'wb') as f:
        f.write(struct.pack('i', 256)); f.write(hdr); f.write(struct.pack('i', 256))
        write_gadget_block(f, np.vstack([pos_gas_all, pos_dm_all]).astype(np.float32))
        write_gadget_block(f, np.vstack([vel_gas_all, vel_dm_all]).astype(np.float32))
        write_gadget_block(f, np.concatenate([ids_gas, ids_dm]).astype(np.uint32))
        write_gadget_block(f, np.concatenate([m_gas_all, m_dm_all]).astype(np.float32))
        write_gadget_block(f, u_gas_all)
        write_gadget_block(f, np.zeros(N_gas, dtype=np.float32))
        write_gadget_block(f, hsml_all)

    print(f"   Written: {outfile} ({os.path.getsize(outfile)/1e6:.0f} MB)")
    print(f"\n=== IC generation complete ===")
    return outfile


def read_snapshot(path):
    """Read a GADGET-2 binary snapshot, returns (time, gas_pos, dm_pos, u_gas, rho_gas)"""
    with open(path, 'rb') as f:
        raw = f.read()
    npart = struct.unpack('<6i', raw[4:28])
    time  = struct.unpack('<d', raw[76:84])[0]
    ngas  = npart[0]; nhalo = npart[1]

    pos_start = 264
    bs = struct.unpack('<i', raw[pos_start:pos_start+4])[0]
    pos = np.frombuffer(raw[pos_start+4:pos_start+4+bs], dtype=np.float32).reshape(-1,3)

    vel_start = pos_start+4+bs+4
    bs2 = struct.unpack('<i', raw[vel_start:vel_start+4])[0]
    id_start  = vel_start+4+bs2+4
    bs3 = struct.unpack('<i', raw[id_start:id_start+4])[0]
    mass_start = id_start+4+bs3+4
    bs4 = struct.unpack('<i', raw[mass_start:mass_start+4])[0]
    u_start = mass_start+4+bs4+4
    bs5 = struct.unpack('<i', raw[u_start:u_start+4])[0]
    u_gas = np.frombuffer(raw[u_start+4:u_start+4+bs5], dtype=np.float32)

    rho_start = u_start+4+bs5+4
    bs6 = struct.unpack('<i', raw[rho_start:rho_start+4])[0]
    rho_gas = np.frombuffer(raw[rho_start+4:rho_start+4+bs6], dtype=np.float32)

    return time, pos[:ngas], pos[ngas:], u_gas, rho_gas


def analyze_snapshots(snapshot_dir, prefix='snapshot_'):
    """Analyze snapshot sequence and print Delta_x(t) table"""
    import glob
    files = sorted(glob.glob(os.path.join(snapshot_dir, f'{prefix}*')))
    if not files:
        print(f"No snapshots found in {snapshot_dir}")
        return

    k_B_=1.381e-16; mp_=1.673e-24; mu_=0.588; g_=5/3; UV_=1e5
    def u_to_keV(u): return u*UV_**2*(g_-1)*mu_*mp_/k_B_/1.16e7

    print(f"\n{'Snapshot':<12} {'t [Gyr]':>8}  {'Δx [kpc]':>10}  {'T_min [keV]':>12}  {'N_gas':>8}")
    print("-"*58)
    for f in files:
        try:
            t, gas_pos, dm_pos, u_gas, rho_gas = read_snapshot(f)
            dx = dm_pos[:,0].mean() - gas_pos[:,0].mean()
            chandra = " ✓" if -50 <= dx <= -20 else ""
            print(f"{os.path.basename(f):<12} {t:>8.2f}  {dx:>+10.1f}  {u_to_keV(u_gas.min()):>11.3f}  {len(gas_pos):>8}{chandra}")
        except Exception as e:
            print(f"{os.path.basename(f)}: ERROR {e}")


def plot_snapshots(snapshot_dir, prefix='snapshot_', outfile='bullet_cluster_leflux.png'):
    """Generate publication figure from snapshot sequence"""
    import glob, matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    files = sorted(glob.glob(os.path.join(snapshot_dir, f'{prefix}*')))
    snaps = {}
    for f in files:
        try:
            i = int(f.split(prefix)[-1])
            snaps[i] = read_snapshot(f)
        except:
            pass

    if not snaps:
        print("No snapshots found"); return

    times=[]; dxs=[]
    for i in sorted(snaps):
        t,gp,dp,ug,rg = snaps[i]
        times.append(t); dxs.append(dp[:,0].mean()-gp[:,0].mean())

    all_x=np.concatenate([snaps[i][1][:,0] for i in snaps]+[snaps[i][2][:,0] for i in snaps])
    all_y=np.concatenate([snaps[i][1][:,1] for i in snaps]+[snaps[i][2][:,1] for i in snaps])
    xmin,xmax=np.percentile(all_x,0.5)-300,np.percentile(all_x,99.5)+300
    ymin,ymax=np.percentile(all_y,1)-200,np.percentile(all_y,99)+200
    ext=(xmin,xmax,ymin,ymax); BINS=300

    keys=list(snaps.keys())
    key_ids=keys[::max(1,len(keys)//6)][:6]
    selected=[(i,snaps[i]) for i in key_ids]

    fig=plt.figure(figsize=(18,12),facecolor='white')
    fig.text(0.5,0.985,'Le Flux v10 — Bullet Cluster SPH (1M particles)',
             ha='center',va='top',fontsize=13,color='#1a1a2e',fontweight='bold')
    fig.text(0.5,0.960,'RED: Le Flux substrate (collisionless) | BLUE: ICM gas',
             ha='center',va='top',fontsize=10,color='#333355')

    for idx,(snap_i,(t,gas_pos,dm_pos,u_gas,rho_gas)) in enumerate(selected):
        dx=dm_pos[:,0].mean()-gas_pos[:,0].mean()
        def dmap(pos):
            H,_,_=np.histogram2d(pos[:,0],pos[:,1],bins=BINS,
                                   range=[[xmin,xmax],[ymin,ymax]]); return H.T
        gn=np.log1p(dmap(gas_pos)*3); gn/=(gn.max()+1e-10)
        dn=np.log1p(dmap(dm_pos)*2);  dn/=(dn.max()+1e-10)
        img=np.zeros((*gn.shape,3))
        img[:,:,0]=np.clip(dn*1.6,0,1)
        img[:,:,1]=np.clip(np.minimum(dn,gn)*0.4,0,1)
        img[:,:,2]=np.clip(gn*1.5,0,1)

        row=idx//3; col=idx%3
        ax=fig.add_axes([0.03+col*0.325,0.53-row*0.455,0.295,0.39])
        ax.imshow(np.clip(img,0,1),origin='lower',extent=[xmin,xmax,ymin,ymax],
                  aspect='auto',interpolation='bilinear')
        gx=gas_pos[:,0].mean(); hx=dm_pos[:,0].mean()
        gy=gas_pos[:,1].mean(); hy=dm_pos[:,1].mean()
        ax.axvline(gx,color='#6699ff',alpha=0.9,lw=1.5,ls='--')
        ax.axvline(hx,color='#ff6699',alpha=0.9,lw=1.5,ls='--')
        ymid=ymin+(ymax-ymin)*0.10
        ax.annotate('',xy=(hx,ymid),xytext=(gx,ymid),
                    arrowprops=dict(arrowstyle='<->',color='#ffdd00',lw=2))
        ax.text((gx+hx)/2,ymid+(ymax-ymin)*0.07,f'Δx={dx:+.0f} kpc',
                ha='center',color='#ffdd00',fontsize=9,fontweight='bold')
        in_c=-50<=dx<=-20
        bc='#cc9900' if in_c else '#aaaacc'
        for sp in ax.spines.values(): sp.set_edgecolor(bc); sp.set_linewidth(2.5 if in_c else 1)
        ax.set_title(f't={t:.1f} Gyr{"  ✓ Chandra" if in_c else ""}',
                     color='#cc6600' if in_c else '#222244',fontsize=9,fontweight='bold' if in_c else 'normal',pad=4)
        ax.set_facecolor('#000011')
        ax.tick_params(colors='#777799',labelsize=7)
        ax.set_xlabel('X [kpc]',color='#555577',fontsize=7)
        ax.set_ylabel('Y [kpc]',color='#555577',fontsize=7)

    ax_b=fig.add_axes([0.07,0.04,0.88,0.115])
    ax_b.fill_between([-0.1,max(times)+0.1],[-50,-50],[-20,-20],alpha=0.2,color='#228833')
    ax_b.axhline(-20,color='#228833',ls=':',lw=1); ax_b.axhline(-50,color='#228833',ls=':',lw=1)
    ax_b.plot(times,dxs,'o-',color='#cc4422',lw=2,ms=5,label='Δx substrate − gas')
    for t,dx in zip(times,dxs):
        if -50<=dx<=-20: ax_b.plot(t,dx,'*',color='#cc8800',ms=12,zorder=5)
    ax_b.text(max(times)*0.95,-35,'Chandra\n20–50 kpc',color='#228833',fontsize=8,va='center',ha='right')
    ax_b.set_facecolor('white'); ax_b.set_xlabel('Time [Gyr]'); ax_b.set_ylabel('Δx [kpc]')
    ax_b.tick_params(labelsize=8); ax_b.grid(alpha=0.2); ax_b.legend(fontsize=8)
    ax_b.text(0.01,min(dxs)*0.98,
              'Monotonic growth = correct Le Flux prediction (continuous field, not discrete CDM halos)',
              fontsize=7.5,color='#555577',va='bottom')

    fig.savefig(outfile,dpi=150,bbox_inches='tight',facecolor='white')
    print(f"Figure saved: {outfile}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Le Flux Bullet Cluster SPH')
    parser.add_argument('--generate', action='store_true', help='Generate GADGET-2 ICs')
    parser.add_argument('--analyze',  action='store_true', help='Analyze snapshots')
    parser.add_argument('--plot',     action='store_true', help='Generate figures')
    parser.add_argument('--outfile',  default='/root/bullet_1M_v6.dat', help='IC output path')
    parser.add_argument('--snapdir',  default='/root/bullet_1M_v6_output', help='Snapshot directory')
    parser.add_argument('--prefix',   default='snapshot_', help='Snapshot filename prefix')
    args = parser.parse_args()

    if args.generate:
        generate_ics(args.outfile)
    if args.analyze:
        analyze_snapshots(args.snapdir, args.prefix)
    if args.plot:
        plot_snapshots(args.snapdir, args.prefix)
    if not any([args.generate, args.analyze, args.plot]):
        parser.print_help()
