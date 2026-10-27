import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from scipy.optimize import minimize

theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2*np.pi, 100)
T, P = np.meshgrid(theta, phi)

ax = np.sin(T) * np.cos(P)
ay = np.sin(T) * np.sin(P)
az = np.cos(T)

def plot_surf(W, title, ax_plot):
    R = (W - np.min(W)) * 3.0 + 1.0
    X = R * ax
    Y = R * ay
    Z = R * az

    norm = plt.Normalize(W.min(), W.max())
    colors = cm.viridis(norm(W))

    ax_plot.plot_surface(X, Y, Z, facecolors=colors, rstride=2, cstride=2, alpha=0.9)
    ax_plot.set_title(title, fontsize=11)
    ax_plot.set_box_aspect((1, 1, 1))
    ax_plot.axis('off')

w_uni_pos = ax**2 + ay**2                                     #одноосная K1>0
w_uni_neg = -(ax**2 + ay**2)                                  #одноосная K1<0
w_cub_pos = ax**2 * ay**2 + ay**2 * az**2 + az**2 * ax**2     #кубическая K1c>0
w_cub_neg = -(ax**2 * ay**2 + ay**2 * az**2 + az**2 * ax**2)  #кубическая K1c<0
w_comb = w_cub_pos + 1.0 * (ax**2 + ay**2)                    #комбинированная(форма вдоль Z)

#2.1
fig1 = plt.figure(figsize=(16, 10))
fig1.suptitle("Задание 2.1: Базовые типы анизотропии", fontsize=16)

plot_surf(w_uni_pos, "Одноосная K1 > 0\n(2 ямы, ось Z)", fig1.add_subplot(221, projection='3d'))
plot_surf(w_uni_neg, "Одноосная K1 < 0\n(лёгкая плоскость XY)", fig1.add_subplot(222, projection='3d'))
plot_surf(w_cub_pos, "Кубическая K1c > 0\n(6 ям, вдоль осей)", fig1.add_subplot(223, projection='3d'))
plot_surf(w_cub_neg, "Кубическая K1c < 0\n(8 ям, вдоль диагоналей)", fig1.add_subplot(224, projection='3d'))
plt.show()

#2.2
fig2 = plt.figure(figsize=(16, 5))
fig2.suptitle("Задание 2.2: Кубическая (K1c > 0) + Форма вдоль Z", fontsize=14)

k_vals = [0.1, 0.5, 1.0, 2.0]
for i, k in enumerate(k_vals):
    w_shape = k * (1 - az**2)
    w_comb_temp = w_cub_pos + w_shape
    plot_surf(w_comb_temp, f"k = {k}", fig2.add_subplot(1, 4, i+1, projection='3d'))
plt.show()

#2.3
fig3 = plt.figure(figsize=(16, 5))
fig3.suptitle("Задание 2.3: Кубическая + Форма под углом (θ=π/4, φ=π/4)", fontsize=14)

nx = np.sin(np.pi/4) * np.cos(np.pi/4)
ny = np.sin(np.pi/4) * np.sin(np.pi/4)
nz = np.cos(np.pi/4)
an = ax*nx + ay*ny + az*nz

for i, k in enumerate(k_vals):
    w_shape = k * (1 - an**2)
    w_comb_temp = w_cub_pos + w_shape
    plot_surf(w_comb_temp, f"k = {k}", fig3.add_subplot(1, 4, i+1, projection='3d'))
plt.show()

#3.1 и 3.2
h_vals = [0, 0.3, 0.7, 4.0]

fig4 = plt.figure(figsize=(15, 10))
fig4.suptitle("Задание 3.1: Внешнее поле (Одноосная K1 > 0, поле вдоль X)", fontsize=14)
for i, h in enumerate(h_vals):
    plot_surf(w_uni_pos - 2 * h * ax, f"h = {h}", fig4.add_subplot(2, 2, i+1, projection='3d'))
plt.show()

fig5 = plt.figure(figsize=(15, 10))
fig5.suptitle("Задание 3.2: Внешнее поле для Кубической (верх) и Комбинированной (низ)", fontsize=14)
for i, h in enumerate(h_vals):
    plot_surf(w_cub_pos - 2 * h * ax, f"Куб. h = {h}", fig5.add_subplot(2, 4, i+1, projection='3d'))
    plot_surf(w_comb - 2 * h * ax, f"Комб. h = {h}", fig5.add_subplot(2, 4, i+5, projection='3d'))
plt.show()

#3.4
def count_minima_exact(h, type_aniso):
    if h < 1e-5:
        return 2 if type_aniso == 'uni' else 6

    def energy(params):
        th, ph = params
        if type_aniso == 'uni':
            return np.sin(th)**2 - 2*h*np.sin(th)*np.cos(ph)
        else:
            ax_ = np.sin(th)*np.cos(ph)
            ay_ = np.sin(th)*np.sin(ph)
            az_ = np.cos(th)
            return ax_**2*ay_**2 + ay_**2*az_**2 + az_**2*ax_**2 - 2*h*ax_

    minima = []
    for th0 in np.linspace(0.1, np.pi-0.1, 15):
        for ph0 in np.linspace(0, 2*np.pi, 15, endpoint=False):
            res = minimize(energy, x0=[th0, ph0], bounds=[(0, np.pi), (0, 2*np.pi)], method='L-BFGS-B')

            if res.success:
                th, ph = res.x

                eps = 1e-5                        #проверка матрицы Гессе
                f0 = energy([th, ph])
                f_th = energy([th+eps, ph])
                f_ph = energy([th, ph+eps])
                f_thth = energy([th+2*eps, ph])
                f_phph = energy([th, ph+2*eps])
                f_thph = energy([th+eps, ph+eps])

                h11 = (f_thth - 2*f_th + f0) / eps**2
                h22 = (f_phph - 2*f_ph + f0) / eps**2
                h12 = (f_thph - f_th - f_ph + f0) / eps**2
                det = h11 * h22 - h12**2

                if h11 > 1e-3 and det > 1e-3:
                    x = np.sin(th)*np.cos(ph)
                    y = np.sin(th)*np.sin(ph)
                    z = np.cos(th)

                    dup = False
                    for mx, my, mz in minima:
                        if (x-mx)**2 + (y-my)**2 + (z-mz)**2 < 0.05:
                            dup = True; break
                    if not dup:
                        minima.append((x, y, z))
    return len(minima)

#3.4
h_range = np.linspace(0, 1.5, 30)
min_uni = [count_minima_exact(h, 'uni') for h in h_range]
min_cub = [count_minima_exact(h, 'cub') for h in h_range]

plt.figure(figsize=(8, 5))
plt.plot(h_range, min_uni, 'bo-', lw=2, label='Одноосная (K1 > 0)')
plt.plot(h_range, min_cub, 'rs-', lw=2, label='Кубическая (K1c > 0)')
plt.title("Зависимость числа минимумов от поля h")
plt.xlabel("Безразмерное поле h")
plt.ylabel("Число минимумов")
plt.yticks([1, 2, 4, 6])
plt.grid(True)
plt.legend()
plt.show()
