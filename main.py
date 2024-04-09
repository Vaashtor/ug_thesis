import numpy as np
import cv2
import time

cf = float(0.75)  # коммерческий коэффициент
n = int(8)  # кол-во разбиений в строке и столбце
Method = 1  # 1 - CMC, 0 - 2000
l = 2  # DeltaECMC
c = 1  # DeltaECMC
kl = 1  # DeltaE00
kc = 1  # DeltaE00
kh = 1  # DeltaE00
p = 3  # кол-во партий


def DeltaECMC(lab1, lab2):
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    C1 = np.sqrt(np.square(a1) + np.square(b1))
    C2 = np.sqrt(np.square(a2) + np.square(b2))

    H = np.arctan2(a1, b1) * 180 / np.pi
    H1 = np.where(H >= 0, H, H + 360)
    delta_a = a2 - a1
    delta_b = b2 - b1

    delta_L = L2 - L1
    delta_C = C2 - C1
    delta_H = np.sqrt(np.abs(np.square(delta_a) + np.square(delta_b) - np.square(delta_C)))

    T = np.where((164 <= H1) & (H1 <= 345), 0.56 + np.abs(0.2 * np.cos(np.pi / 180 * (H1 + 168))),
                 0.36 + np.abs(0.4 * np.cos(np.pi / 180 * (H1 + 35))))
    F = np.sqrt(np.power(C1, 4) / (np.power(C1, 4) + 1900))

    S_L = np.where(L1 < 16, 0.511, (0.040975 * L1) / (1 + 0.01765 * L1))
    S_C = 0.0638 * C1 / (1 + 0.0131 * C1) + 0.638
    S_H = S_C * (F * T + 1 - F)

    delta_E = np.sqrt(np.square(delta_L / (l * S_L)) + np.square(delta_C / (c * S_C)) + np.square(delta_H / S_H))

    return np.mean(delta_E)


def DeltaE00(lab1, lab2):
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    Lm = (L1 + L2) / 2
    delta_Lj = L2 - L1
    S_L = 1 + (0.015 * np.square(Lm - 50)) / np.sqrt(20 + np.square(Lm - 50))

    C1 = np.sqrt(np.square(a1) + np.square(b1))
    C2 = np.sqrt(np.square(a2) + np.square(b2))
    Cm = (C1 + C2) / 2

    G = np.sqrt(np.power(Cm, 7) / (np.power(Cm, 7) + 25 ** 7))
    a1j = a1 + a1 / 2 * (1 - G)
    a2j = a2 + a2 / 2 * (1 - G)

    C1j = np.sqrt(np.square(a1j) + np.square(b1))
    C2j = np.sqrt(np.square(a2j) + np.square(b2))
    delta_Cj = C2j - C1j
    Cmj = (C1j + C2j) / 2
    S_C = 1 + 0.045 * Cmj

    h1 = np.arctan2(a1j, b1) * 180 / np.pi
    h1j = np.where(h1 >= 0, h1, h1 + 360)
    h2 = np.arctan2(a2j, b2) * 180 / np.pi
    h2j = np.where(h2 >= 0, h2, h2 + 360)
    Hmj = np.where(np.abs(h1j - h2j) <= 180, (h1j + h2j) / 2,
                   np.where(h1j + h2j < 360, (h1j + h2j + 360) / 2, (h1j + h2j - 360) / 2))

    T = 1 - 0.17 * np.cos(np.pi / 180 * (Hmj - 30)) + 0.24 * np.cos(np.pi / 180 * 2 * Hmj) + 0.32 * np.cos(
        np.pi / 180 * (3 * Hmj + 6)) - 0.2 * np.cos(np.pi / 180 * (4 * Hmj - 63))
    S_H = 1 + 0.015 * Cmj * T

    delta_hj = np.where(np.abs(h2j - h1j) <= 180, h2j - h1j, np.where(h2j <= h1j, h2j - h1j + 360, h2j - h1j - 360))
    delta_Hj = 2 * np.sqrt(C1j * C2j) * np.sin(np.pi / 180 * delta_hj / 2)
    delta_t = 30 * np.exp(-np.square((Hmj - 275) / 25))
    R_T = -2 * np.sqrt(np.power(Cmj, 7) / (np.power(Cmj, 7) + 25 ** 7)) * np.sin(np.pi / 180 * 2 * delta_t)

    delta_E = np.sqrt(np.square(delta_Lj / (kl * S_L)) + np.square(delta_Cj / (kc * S_C)) + np.square(
        delta_Hj / (kh * S_H)) + R_T * delta_Cj * delta_Hj / (kc * S_C * kh * S_H))

    return np.mean(delta_E)


def extract_lab_values(image_path):
    img_BGR = cv2.imread(image_path)
    median_filtered = cv2.medianBlur(img_BGR, 3)
    img = cv2.cvtColor(np.float32(median_filtered / 255.0), cv2.COLOR_BGR2LAB)
    H, S, V = cv2.split(cv2.cvtColor(np.float32(median_filtered / 255.0), cv2.COLOR_BGR2HSV))
    dy = img.shape[0] // n
    dx = img.shape[1] // n
    segment = np.zeros((n, n, 3))
    for i in range(0, n, 1):
        for j in range(0, n, 1):
            seg = img[dy * i:dy * (i + 1), dx * j:dx * (j + 1), :]
            segment[i, j, :] = mean_pixel(seg[:, :, 0],
                                      seg[:, :, 1],
                                      seg[:, :, 2])
    L, a, b = cv2.split(segment)
    return L, a, b


def mean_pixel(L, a, b):
    Lm = np.mean(L)
    am = np.mean(a)
    bm = np.mean(b)
    Lmax = np.max(L)
    amax = np.max(a)
    bmax = np.max(b)
    if not check((Lm, am, bm), (Lmax, amax, bmax)): print("Err")
    return Lm, am, bm


def check(lab1, lab2):
    if Method == 1:
        if DeltaECMC(lab1, lab2) < cf:
            return True
        else:
            return False
    else:
        if DeltaE00(lab1, lab2) < cf:
            return True
        else:
            return False


def flatten_pixels(x1, x2, x3, k1, k2, k3):
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    k1 = k1.flatten()
    k2 = k2.flatten()
    k3 = k3.flatten()
    sorted_indices = np.lexsort((k3, k2, k1))
    x1 = x1[sorted_indices]
    x2 = x2[sorted_indices]
    x3 = x3[sorted_indices]
    return x1, x2, x3


address = input("Введите путь к изображению первой плитки\n")
Lr = np.zeros((n, n, p))-100
ar = np.zeros((n, n, p))
br = np.zeros((n, n, p))
Lr[:, :, 0], ar[:, :, 0], br[:, :, 0] = extract_lab_values(address)
while True:
    address = input("Введите путь (0, чтобы закрыть)\n")
    t0 = time.time()
    if address == '0':
        break
    Li, ai, bi = extract_lab_values(address)
    t1 = time.time()
    for i in range(0, p):
        if Method == 1:
            delta_E = DeltaECMC((Lr[:, :, i], ar[:, :, i], br[:, :, i]), (Li, ai, bi))
            print("Среднее Delta E_CMC:", delta_E)
            if delta_E < cf:
                print("Партия ", i)
                break
            else:
                if Lr[0,0,i]<0:
                    Lr[:, :, i], ar[:, :, i], br[:, :, i] = Li, ai, bi
                    print("Новая партия ", i)
                    break
                if n - 1 == i: print("Без партии")
        else:
            delta_E00 = DeltaE00((Lr[:, :, i], ar[:, :, i], br[:, :, i]), (Li, ai, bi))
            print("Среднее Delta E_00:", delta_E00)
            if delta_E00 < cf:
                print("Партия ", i)
                break
            else:
                if Lr[0,0,i]<0:
                    Lr[:, :, i], ar[:, :, i], br[:, :, i] = Li, ai, bi
                    print("Новая партия ", i)
                    break
                if n - 1 == i: print("Без партии")
        print("Получение значений", t1 - t0, "секунд\n", "расчет DE", time.time() - t1, "секунд")
