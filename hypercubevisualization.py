import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # צורך להרשמת תלת־ממד
from matplotlib import cm
from matplotlib.colors import Normalize

IMAGE_PATH = 'calibration_image.jpeg'
CSV_PATH   = 'spectrum.csv'

# ROI  [y1:y2, x1:x2]
x1, x2 = 10, 120   #  (X)
y1, y2 = 28,  48   #     (Y)


pixs = np.array([20.0,  35.6,  55.0,  70.7,  80.6,  92.15,  106.3])
wls  = np.array([700.0, 620.0, 590.0, 570.0, 500.0, 450.0, 400.0])

#  fitting: λ = a·pixel + b
a, b = np.polyfit(pixs, wls, 1)
plt.scatter(pixs, wls, color='red', label='calibration points')

# קו ההתאמה הליניארית
pix_range = np.linspace(min(pixs), max(pixs), 200)
wls_fit   = a*pix_range + b
plt.plot(pix_range, wls_fit, color='blue', label=f'linear fitting: λ = {a:.2f}·p + {b:.2f}')

# עיצוב גרף
plt.xlabel("Pixel")
plt.ylabel("Wavelength [nm]")
plt.title("Spectral Calibration- Linear Fitting")
plt.legend()
plt.grid(True)
plt.show()
print(f"linear calib: λ(p) = {a:.3f}·p + {b:.3f}\n")

# image loading and ROI
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"error {IMAGE_PATH}")
roi = img[y1:y2, x1:x2].astype(float)
H, W = roi.shape

# spectrum loading, filtering 400-700 and then smooth
df = pd.read_csv(CSV_PATH)
df = df[(df['wavelength_nm'] >= 400) & (df['wavelength_nm'] <= 700)].reset_index(drop=True)
df['intensity_smooth'] = df['intensity'].rolling(window=11, center=True, min_periods=1).mean()

wavelengths = df['wavelength_nm'].values       # wavelength vec
inten       = df['intensity_smooth'].values    # intens vec
i_norm      = (inten - inten.min()) / (inten.max() - inten.min())
Nz          = len(wavelengths)

#  hypercube: (H, W, Nz)
hypercube = np.zeros((H, W, Nz), dtype=float)
for k in range(Nz):
    hypercube[:, :, k] = roi * i_norm[k]

print(f"Hypercube shape: {hypercube.shape} (Y, X, spectral bands)\n")

#  pixel→λ
pixels     = np.arange(W)
lambda_map = a * pixels + b   # λ לכל פיקסל X

#  facecolors
#  (X–Y plane ב־λ=0): ROI
roi_norm = (roi - roi.min()) / (roi.max() - roi.min())
face_top = plt.cm.gray(roi_norm)          #  (H, W, 4)

#   (X–Z at Y=0):  X by λ
norm_lambda = (lambda_map - lambda_map.min()) / (lambda_map.max() - lambda_map.min())
cols_lambda = plt.cm.jet(norm_lambda)      #  (W,4)
face_front  = np.tile(cols_lambda[np.newaxis, :, :], (Nz, 1, 1))  # (Nz, W, 4)

#  (Y–Z at X=W-1): colored by intens
mean_yz = hypercube.mean(axis=1).T          # (Nz, H)
norm_yz = (mean_yz - mean_yz.min()) / (mean_yz.max() - mean_yz.min())
face_side = plt.cm.gray(norm_yz)           #  (Nz, H, 4)

# cube generation
fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(111, projection='3d')

#  X–Y (Z=0)
Xg, Yg = np.meshgrid(np.arange(W), np.arange(H))
ax.plot_surface(Xg, Yg, np.zeros_like(Xg),
                facecolors=face_top, shade=False)

#  X–Z (Y=0)
Xf, Zf = np.meshgrid(np.arange(W), wavelengths)
Yf      = np.zeros_like(Xf)
ax.plot_surface(Xf, Yf, Zf,
                facecolors=face_front, shade=False)

#  Y–Z (X=W-1)
Ys, Zs = np.meshgrid(np.arange(H), wavelengths)
Xs      = np.full_like(Ys, W - 1)
ax.plot_surface(Xs, Ys, Zs,
                facecolors=face_side, shade=False)

# colorbar
mappable = cm.ScalarMappable(
    cmap='jet',
    norm=Normalize(vmin=400, vmax=700)
)
mappable.set_array([])
cbar = fig.colorbar(
    mappable,
    ax=ax,
    pad=0.1,
    label='Wavelength (nm)',
    ticks=np.linspace(400, 700, 7)
)

# labels and titles
ax.set_xlabel('X pixel (dispersion)')
ax.set_ylabel('Y pixel (slit)')
ax.set_zticks([])
ax.set_title('Hyperspectral Calibration')
ax.view_init(elev=30, azim=-60)

plt.tight_layout()
plt.show()
