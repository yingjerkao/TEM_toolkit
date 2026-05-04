import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import (map_coordinates, minimum_filter, maximum_filter,
                           label, center_of_mass, gaussian_filter)
from scipy.spatial import KDTree
from sklearn.cluster import KMeans

# ── 1. Load real image ────────────────────────────────────────────────────────
raw = Image.open('/sessions/beautiful-compassionate-curie/mnt/Downloads/unnamed.jpg')
img = np.array(raw.convert('L'), dtype=float) / 255.0
N = img.shape[0]

# ── 2. Detect dark spots (smoothed to suppress JPEG noise) ────────────────────
img_sm  = gaussian_filter(img, sigma=5)
ws      = 55
loc_min = (img_sm == minimum_filter(img_sm, size=ws))
loc_min &= (img_sm < np.percentile(img_sm, 15))
labeled, n_dark = label(loc_min)
dark = np.array(center_of_mass(loc_min, labeled, range(1, n_dark+1)))
print(f"Dark spots detected: {n_dark}")

# ── 3. Estimate NN spacing for Gaussian width ─────────────────────────────────
tree = KDTree(dark)
dists, idxs = tree.query(dark, k=6)
nn_vecs = []
for i in range(len(dark)):
    for j in idxs[i, 1:]:
        d = np.linalg.norm(dark[j] - dark[i])
        if 60 < d < 110:
            nn_vecs.append(d)
spacing = np.median(nn_vecs) if nn_vecs else 87.0
sigma   = spacing / 5.0
print(f"NN spacing: {spacing:.1f} px,  σ={sigma:.1f} px")

# ── 4. Reconstruct image: Gaussians placed AT detected dark spots ─────────────
x = np.arange(N);  y = np.arange(N)
X, Y = np.meshgrid(x, y)

img_recon = np.ones((N, N), dtype=float)
for d in dark:
    cy, cx = d[0], d[1]
    img_recon -= np.exp(-((X-cx)**2 + (Y-cy)**2) / (2*sigma**2))
img_recon = np.clip(img_recon, 0, 1)

# ── 5. Plaquette centres = local maxima of reconstructed image ────────────────
ws2     = int(spacing * 0.7)
loc_max = (img_recon == maximum_filter(img_recon, size=ws2))
loc_max &= (img_recon > 0.5)
lab2, n_bright = label(loc_max)
bright_spots = np.array(center_of_mass(loc_max, lab2, range(1, n_bright+1)))
print(f"Plaquette centres: {n_bright}")

# ── 6. Manually confirmed atom position ───────────────────────────────────────
atom_y = 166   # shifted to nearest plaquette centre
atom_x = 227
print(f"Atom: col={atom_x}, row={atom_y},  real val={img[atom_y,atom_x]:.3f},  recon val={img_recon[atom_y,atom_x]:.3f}")

# ── 7. Angular scan — average over all plaquette centres ─────────────────────
a, da   = 20, 10
n_theta = 720
n_r     = 40
theta_arr = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
r_arr     = np.linspace(a, a+da, n_r)
theta_deg = np.degrees(theta_arr)

# Only use plaquette centres far enough from the image edge that the ring stays inside
margin = a + da + 2
valid_spots = bright_spots[
    (bright_spots[:,0] > margin) & (bright_spots[:,0] < N - margin) &
    (bright_spots[:,1] > margin) & (bright_spots[:,1] < N - margin)
]
print(f"Plaquette centres used for averaging: {len(valid_spots)} / {len(bright_spots)}")

# Compute I(θ) for each valid plaquette centre and accumulate
all_I = np.zeros((len(valid_spots), n_theta))
for k, spot in enumerate(valid_spots):
    sy, sx = int(round(spot[0])), int(round(spot[1]))
    for i, th in enumerate(theta_arr):
        cols = sx + r_arr * np.cos(th)
        rows = sy + r_arr * np.sin(th)
        vals = map_coordinates(img, [rows, cols], order=1, mode='nearest')
        all_I[k, i] = np.trapezoid(vals, r_arr)

I_theta_avg = np.mean(all_I, axis=0)   # ensemble average
I_theta_std = np.std(all_I,  axis=0)   # spread across atoms

# Single-atom profile for the confirmed atom (for comparison)
I_theta_single = np.zeros(n_theta)
for i, th in enumerate(theta_arr):
    cols = atom_x + r_arr * np.cos(th)
    rows = atom_y + r_arr * np.sin(th)
    vals = map_coordinates(img, [rows, cols], order=1, mode='nearest')
    I_theta_single[i] = np.trapezoid(vals, r_arr)

noise_single = np.std(I_theta_single - np.mean(I_theta_single))
noise_avg    = np.std(I_theta_avg    - np.mean(I_theta_avg))
print(f"Noise (single atom): σ={noise_single:.4f}")
print(f"Noise (averaged):    σ={noise_avg:.4f}  ({len(valid_spots)} atoms)")

# ── 8. Plot: image overview + single vs averaged profiles ────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
ring_theta = np.linspace(0, 2*np.pi, 360)

# Left: real image with all valid plaquette centres and rings
ax0 = axes[0]
ax0.imshow(img, cmap='gray', origin='upper', vmin=0, vmax=1)
ax0.plot(dark[:,1],          dark[:,0],          'b+', ms=6, mew=1.2, label='dark spots')
ax0.plot(valid_spots[:,1],   valid_spots[:,0],   'g.', ms=5, alpha=0.8, label=f'atoms averaged ({len(valid_spots)})')
ax0.plot(atom_x, atom_y, 'r*', ms=14, label='reference atom')
for sy, sx in valid_spots:
    for rad in [a, a+da]:
        ax0.plot(sx + rad*np.cos(ring_theta), sy + rad*np.sin(ring_theta),
                 'orange', lw=0.5, alpha=0.3)
ax0.set_title('Real image — all averaged atoms', fontsize=10)
ax0.set_xlabel('x (px)'); ax0.set_ylabel('y (px)')
ax0.legend(fontsize=7)

# Middle: single atom profile
ax1 = axes[1]
ax1.plot(theta_deg, I_theta_single, color='steelblue', lw=1.4,
         label=f'Single atom (227,166)  σ={noise_single:.4f}')
ax1.axhline(I_theta_single.mean(), color='steelblue', ls='--', lw=1)
for angle in [0, 90, 180, 270]:
    ax1.axvline(angle, color='gray', ls=':', lw=0.8)
ax1.set_xlabel('θ  (degrees)', fontsize=11)
ax1.set_ylabel(r'$I(\theta)$', fontsize=11)
ax1.set_title(f'Single atom\n(a={a} px, da={da} px)', fontsize=10)
ax1.set_xlim(0, 360); ax1.set_xticks([0,90,180,270,360])
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

# Right: ensemble-averaged profile with ±1σ band
ax2 = axes[2]
ax2.fill_between(theta_deg, I_theta_avg - I_theta_std,
                             I_theta_avg + I_theta_std,
                 color='steelblue', alpha=0.25, label='±1σ spread')
ax2.plot(theta_deg, I_theta_avg, color='steelblue', lw=1.6,
         label=f'Averaged ({len(valid_spots)} atoms)  σ={noise_avg:.4f}')
ax2.axhline(I_theta_avg.mean(), color='tomato', ls='--', lw=1)
for angle in [0, 90, 180, 270]:
    ax2.axvline(angle, color='gray', ls=':', lw=0.8)
ax2.set_xlabel('θ  (degrees)', fontsize=11)
ax2.set_ylabel(r'$\langle I(\theta) \rangle$', fontsize=11)
ax2.set_title(f'Ensemble average\n(a={a} px, da={da} px)', fontsize=10)
ax2.set_xlim(0, 360); ax2.set_xticks([0,90,180,270,360])
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_png = '/sessions/beautiful-compassionate-curie/mnt/Downloads/angular_scan.png'
out_py  = '/sessions/beautiful-compassionate-curie/mnt/Downloads/angular_scan.py'
plt.savefig(out_png, dpi=150, bbox_inches='tight')
import shutil; shutil.copy('/sessions/beautiful-compassionate-curie/angular_scan.py', out_py)
print(f"Saved → {out_png}")

# ── Extra: sweep over several da values ───────────────────────────────────────
da_values = [2, 5, 10, 15, 20]
fig2, axes2 = plt.subplots(len(da_values), 1, figsize=(10, 14), sharex=True)

for ax, da_try in zip(axes2, da_values):
    r_arr_try = np.linspace(a, a + da_try, max(5, int(da_try*4)))
    I_try = np.zeros(n_theta)
    for i, th in enumerate(theta_arr):
        cols = atom_x + r_arr_try * np.cos(th)
        rows = atom_y + r_arr_try * np.sin(th)
        vals = map_coordinates(img, [rows, cols], order=1, mode='nearest')
        I_try[i] = np.trapezoid(vals, r_arr_try)
    # noise estimate: std of residual after subtracting 4-fold mean
    noise = np.std(I_try - np.mean(I_try))
    ax.plot(theta_deg, I_try, lw=1.4, label=f'da={da_try} px  (σ={noise:.4f})')
    ax.axhline(I_try.mean(), color='tomato', ls='--', lw=1)
    for angle in [0, 90, 180, 270]:
        ax.axvline(angle, color='gray', ls=':', lw=0.7)
    ax.set_ylabel(r'$I(\theta)$', fontsize=9)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.25)

axes2[-1].set_xlabel('θ  (degrees)', fontsize=11)
axes2[-1].set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
axes2[0].set_title(f'Angular profiles for different da values\n(a={a} px, atom at ({atom_x},{atom_y}))', fontsize=11)
fig2.tight_layout()
out2 = '/sessions/beautiful-compassionate-curie/mnt/Downloads/angular_scan_da_sweep.png'
fig2.savefig(out2, dpi=150, bbox_inches='tight')
print(f"Saved → {out2}")
