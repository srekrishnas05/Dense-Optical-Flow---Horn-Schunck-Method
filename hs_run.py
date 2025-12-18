import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def to_gray01(bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return g / 255.0

def save_img01(path: str, img01: np.ndarray) -> None:
    img8 = np.clip(img01 * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img8)

def normalize_for_debug(x: np.ndarray) -> np.ndarray:
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if abs(xmax - xmin) < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - xmin) / (xmax - xmin)

def central_diff_x(I: np.ndarray) -> np.ndarray:
    Ip = np.pad(I, ((0, 0), (1, 1)), mode="edge")
    return 0.5 * (Ip[:, 2:] - Ip[:, :-2])

def central_diff_y(I: np.ndarray) -> np.ndarray:
    Ip = np.pad(I, ((1, 1), (0, 0)), mode="edge")
    return 0.5 * (Ip[2:, :] - Ip[:-2, :])

def neighbor_avg_4(U: np.ndarray) -> np.ndarray:
    Up = np.pad(U, ((1, 1), (1, 1)), mode="edge")
    N = Up[:-2, 1:-1]
    S = Up[2:,  1:-1]
    W = Up[1:-1, :-2]
    E = Up[1:-1, 2:]
    return 0.25 * (N + S + W + E)

def flow_to_hsv(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    mag, ang = cv2.cartToPolar(u, v, angleInDegrees=True)
    hsv = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.float32)
    hsv[..., 0] = ang / 2.0 
    hsv[..., 1] = 1.0

    scale = np.percentile(mag, 95) + 1e-9
    hsv[..., 2] = np.clip(mag / scale, 0, 1)

    hsv8 = (hsv * 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(hsv8, cv2.COLOR_HSV2BGR)
    return bgr

def quiver_plot(I: np.ndarray, u: np.ndarray, v: np.ndarray, step: int = 60, title: str = "flow"):
    h, w = I.shape
    yy, xx = np.mgrid[0:h:step, 0:w:step]
    uu = u[0:h:step, 0:w:step]
    vv = v[0:h:step, 0:w:step]

    plt.figure()
    plt.imshow(I, cmap="gray")
    plt.quiver(xx, yy, uu, vv, angles="xy", scale_units="xy", scale=0.15, width=0.003)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.show()

def warp_image(I: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = I.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + u).astype(np.float32)
    map_y = (y + v).astype(np.float32)
    return cv2.remap(I.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def horn_schunck(I1: np.ndarray,
                 I2: np.ndarray,
                 alpha: float,
                 iters: int,
                 preblur_ksize: int = 5):
    """
    this is HS fixed-point iteration derived from:
      u <- ubar - Ix*(Ix*ubar + Iy*vbar + It)/(alpha^2 + Ix^2 + Iy^2)
      v <- vbar - Iy*(Ix*ubar + Iy*vbar + It)/(alpha^2 + Ix^2 + Iy^2)
    """
    if preblur_ksize >= 3 and preblur_ksize % 2 == 1:
        I1s = cv2.GaussianBlur(I1, (preblur_ksize, preblur_ksize), 0)
        I2s = cv2.GaussianBlur(I2, (preblur_ksize, preblur_ksize), 0)
    else:
        I1s, I2s = I1, I2

    Iavg = 0.5 * (I1s + I2s)
    Ix = central_diff_x(Iavg)
    Iy = central_diff_y(Iavg)
    It = I2s - I1s

    u = np.zeros_like(I1, dtype=np.float32)
    v = np.zeros_like(I1, dtype=np.float32)

    a2 = alpha * alpha
    eps = 1e-12

    avg_fn = neighbor_avg_4

    for k in range(iters):
        ubar = avg_fn(u)
        vbar = avg_fn(v)

        residual = Ix * ubar + Iy * vbar + It
        denom = a2 + Ix * Ix + Iy * Iy

        u = ubar - (Ix * residual) / (denom + eps)
        v = vbar - (Iy * residual) / (denom + eps)

        if (k + 1) in (1, 5, 20, iters):
            mean_mag = float(np.mean(np.sqrt(u*u + v*v)))
            print(f"iter {k+1:4d}/{iters}: mean |flow| = {mean_mag:.6f}")

    return u, v, Ix, Iy, It

if __name__ == "__main__":
    os.makedirs("out", exist_ok=True)
    os.makedirs("out/debug", exist_ok=True)

    f1 = cv2.imread("frame1.png")
    f2 = cv2.imread("frame2.png")
    if f1 is None or f2 is None:
        raise RuntimeError("missing frames, run extract_frames.py first")

    I1 = to_gray01(f1)
    I2 = to_gray01(f2)

    iters = 500
    blur = 5

    alphas = [0.1, 0.3, 1.0, 3.0]

    for alpha in alphas:
        print("\n==============================")
        print(f"running HS: alpha={alpha}, iters={iters}, blur={blur}, stencil={'4N'}")

        u, v, Ix, Iy, It = horn_schunck(I1, I2, alpha=alpha, iters=iters, preblur_ksize=blur)

        save_img01(f"out/debug/Ix_alpha{alpha}.png", normalize_for_debug(Ix))
        save_img01(f"out/debug/Iy_alpha{alpha}.png", normalize_for_debug(Iy))
        save_img01(f"out/debug/It_alpha{alpha}.png", normalize_for_debug(It))

        hsv = flow_to_hsv(u, v)
        cv2.imwrite(f"out/flow_hsv_alpha{alpha}.png", hsv)

        I1_warp = warp_image(I1, u, v)
        save_img01(f"out/warp_I1_alpha{alpha}.png", I1_warp)

        diff = np.abs(I2 - I1_warp)
        save_img01(f"out/diff_afterwarp_alpha{alpha}.png", normalize_for_debug(diff))

        print(f"Saved outputs for alpha={alpha} in out/")

        mag = np.sqrt(u*u + v*v)
        print(
            f"alpha={alpha}: "
            f"mean={mag.mean():.6f}, "
            f"max={mag.max():.3f}, "
            f"p95={np.percentile(mag,95):.3f}"
        )
        print(f"Saved outputs for alpha={alpha} in out/")
        
    alpha_show = alphas[1]
    u, v, *_ = horn_schunck(I1, I2, alpha=alpha_show, iters=iters, preblur_ksize=blur)
    quiver_plot(I1, u, v, step=24, title=f"Horn–Schunck (alpha={alpha_show}, stencil={'4N'})")
