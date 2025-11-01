#  Hybrid Image Creation using Multiresolution Analysis (MRA)
#  Gaussian baseline, Pyramid fusion, Wavelet fusion

!pip install opencv-python pywavelets --quiet

import cv2, numpy as np, pywt, matplotlib.pyplot as plt

def show(img, title="", scale=1.0, cmap=None):
    plt.figure(figsize=(4*scale,4*scale))
    if img.ndim == 2:
        plt.imshow(img, cmap=cmap or "gray")
    else:
        disp = np.clip(img.astype(np.float32)/255.0, 0, 1)
        plt.imshow(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(title)
    plt.show()

print("Libraries loaded")

path_low_freq  = "/content/dog.jpg"   # low-frequency structure image
path_high_freq = "/content/cat.jpg"   # high-frequency detail image

img_low  = cv2.imread(path_low_freq)
img_high = cv2.imread(path_high_freq)
assert img_low is not None and img_high is not None, "Check image paths."

h, w = min(img_low.shape[0], img_high.shape[0]), min(img_low.shape[1], img_high.shape[1])
img_low  = cv2.resize(img_low,  (w, h))
img_high = cv2.resize(img_high, (w, h))

print(f"Loaded and resized images to {w}x{h}")
show(img_low,"Low-frequency Source (Structure)")
show(img_high,"High-frequency Source (Detail)")

def gaussian_hybrid(img_low, img_high, sigma=6):  ## Gaussian Hybrid
    low_blur  = cv2.GaussianBlur(img_low,  (0,0), sigmaX=sigma)
    high_blur = cv2.GaussianBlur(img_high, (0,0), sigmaX=sigma)
    high_pass = cv2.subtract(img_high, high_blur)
    hybrid = cv2.add(low_blur, high_pass)
    return low_blur, high_pass, np.clip(hybrid,0,255).astype(np.uint8)

print("\n Performing Gaussian baseline hybrid.")
low_blur, high_pass, hybrid_gauss = gaussian_hybrid(img_low, img_high)
show(low_blur,"Gaussian Baseline – Low-frequency (blurred)")
show(high_pass+127,"Gaussian Baseline – High-frequency (shifted for visibility)")
show(hybrid_gauss,"Gaussian Baseline – Combined Hybrid")

def gaussian_pyramid(img, levels=4):  ## Gaussian/Laplacian Pyramid (Multiresolution)
    g = [img.astype(np.float32)]
    for _ in range(1, levels):
        img = cv2.pyrDown(img)
        g.append(img.astype(np.float32))
    return g

def laplacian_pyramid(img, levels=4):
    g = gaussian_pyramid(img, levels)
    l = []
    for i in range(levels-1):
        up = cv2.pyrUp(g[i+1], dstsize=(g[i].shape[1], g[i].shape[0]))
        l.append(g[i] - up)
    l.append(g[-1])
    return l

def reconstruct_from_laplacian(lap):
    img = lap[-1]
    for i in range(len(lap)-2, -1, -1):
        up = cv2.pyrUp(img, dstsize=(lap[i].shape[1], lap[i].shape[0]))
        img = up + lap[i]
    return np.clip(img,0,255).astype(np.uint8)

levels = 4
print("\n Building Laplacian pyramids.")
lap_low  = laplacian_pyramid(img_low,  levels)
lap_high = laplacian_pyramid(img_high, levels)

for i,(Llo,Lhi) in enumerate(zip(lap_low,lap_high)):
    disp_lo=(Llo-Llo.min())/(Llo.max()-Llo.min()+1e-8)
    disp_hi=(Lhi-Lhi.min())/(Lhi.max()-Lhi.min()+1e-8)

print("Fusing pyramids (low structure + high detail).")
def fuse_pyramids(lap_low, lap_high):
    fused=[]
    n=len(lap_low)
    for i in range(n):
        if i<n-1:
            fused.append((1.0*lap_high[i]+0.2*lap_low[i]).astype(np.float32))
        else:
            fused.append((1.0*lap_low[i]+0.2*lap_high[i]).astype(np.float32))
    return fused

fused_pyr = fuse_pyramids(lap_low, lap_high)
hybrid_pyr = reconstruct_from_laplacian(fused_pyr)
show(hybrid_pyr,"Hybrid via Multiresolution Pyramid Fusion")

def dwt2_color(img, wavelet="haar", level=2):  ## Wavelet (MRA) Fusion
    chans = cv2.split(img.astype(np.float32))
    coeffs = [pywt.wavedec2(ch, wavelet=wavelet, level=level) for ch in chans]
    return coeffs

def idwt2_color(coeffs, wavelet="haar"):
    recs = [pywt.waverec2(c, wavelet=wavelet) for c in coeffs]
    return np.clip(cv2.merge(recs),0,255).astype(np.uint8)

def fuse_wavelet_coeffs(clow, chigh):
    fused_all=[]
    for cL,cH in zip(clow,chigh):
        fused_ch=[]
        cA_L=cL[0]; cA_H=cH[0]
        cA=cA_L*1.0+0.1*cA_H
        fused_ch.append(cA)
        for lvl in range(1,len(cL)):
            (cHL,cVL,cDL)=cL[lvl]; (cHH,cVH,cDH)=cH[lvl]
            cHf=cHH*1.0+0.1*cHL
            cVf=cVH*1.0+0.1*cVL
            cDf=cDH*1.0+0.1*cDL
            fused_ch.append((cHf,cVf,cDf))
        fused_all.append(fused_ch)
    return fused_all

print("\n Performing Wavelet (Haar) decomposition and fusion.")
coeffs_low  = dwt2_color(img_low,  "haar", level=2)
coeffs_high = dwt2_color(img_high, "haar", level=2)
coeffs_fused = fuse_wavelet_coeffs(coeffs_low, coeffs_high)
hybrid_wavelet = idwt2_color(coeffs_fused, "haar")
show(hybrid_wavelet, "Hybrid via Wavelet / MRA Fusion (Haar)")

def simulate_view(img, scale=0.25, sigma=1.2):
    small=cv2.resize(img,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    small=cv2.GaussianBlur(small,(0,0),sigma)
    return cv2.resize(small,(img.shape[1],img.shape[0]))

print("\n Simulating viewing distance as far vs near.")
near_pyr, far_pyr = hybrid_pyr, simulate_view(hybrid_pyr,0.25)
near_wave, far_wave = hybrid_wavelet, simulate_view(hybrid_wavelet,0.25)

show(near_pyr,"Pyramid Hybrid (Near View)")
show(far_pyr,"Pyramid Hybrid (Far View)")
show(near_wave,"Wavelet Hybrid (Near View)")
show(far_wave,"Wavelet Hybrid (Far View)")