"""Метрики многокритериальной оптимизации: гиперобъём (hypervolume).

Гиперобъём — объём области, доминируемой множеством решений относительно
опорной точки; чем он больше, тем лучше фронт (ближе к идеалу и шире).
Реализован точный подсчёт для 2D и 3D (плоское сканирование), без
готовых библиотечных реализаций.
"""


def hypervolume_2d(points, ref):
    """Точный гиперобъём для двух минимизируемых критериев.

    points — список (x, y); каждый «точка» доминирует прямоугольник
    [x, ref_x] x [y, ref_y]. Площадность объединения таких прямоугольников
    вычисляется сканированием по y: на высоте y покрытая ширина равна
    ref_x - min{x_p : y_p <= y}.
    """
    rx, ry = ref
    pts = sorted((p[0], p[1]) for p in points
                 if p[0] < rx and p[1] < ry)
    if not pts:
        return 0.0
    pts.sort(key=lambda p: p[1])
    area = 0.0
    min_x = float("inf")
    prev_y = None
    i = 0
    n = len(pts)
    while i < n:
        y = pts[i][1]
        if prev_y is not None and y > prev_y:
            area += max(0.0, rx - min_x) * (y - prev_y)
        while i < n and pts[i][1] == y:
            min_x = min(min_x, pts[i][0])
            i += 1
        prev_y = y
    area += max(0.0, rx - min_x) * (ry - prev_y)
    return area


def hypervolume_3d(points, ref):
    """Точный гиперобъём для трёх минимизируемых критериев.

    Сканирование по z: объём = интеграл от площади 2D-объединения
    проекций точек с z_p <= z. Площадь на каждом шаге вычисляется
    функцией hypervolume_2d.
    """
    rz = ref[2]
    pts = [p for p in points
           if p[0] < ref[0] and p[1] < ref[1] and p[2] < rz]
    if not pts:
        return 0.0
    pts.sort(key=lambda p: p[2])
    volume = 0.0
    prev_z = None
    active = []
    i = 0
    n = len(pts)
    while i < n:
        z = pts[i][2]
        if prev_z is not None and z > prev_z:
            volume += hypervolume_2d(active, (ref[0], ref[1])) * (z - prev_z)
        while i < n and pts[i][2] == z:
            active.append((pts[i][0], pts[i][1]))
            i += 1
        prev_z = z
    volume += hypervolume_2d(active, (ref[0], ref[1])) * (rz - prev_z)
    return volume


def hypervolume(points, ref):
    """Гиперобъём для 2D или 3D в зависимости от размерности."""
    points = list(points)
    if not points:
        return 0.0
    dim = len(points[0])
    if dim == 2:
        return hypervolume_2d(points, ref)
    if dim == 3:
        return hypervolume_3d(points, ref)
    raise ValueError(f"Неподдерживаемая размерность: {dim}")


def filter_nondominated(points):
    """Вернуть индексы недоминируемых точек (минимизация)."""
    n = len(points)
    keep = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if j == i:
                continue
            if all(points[j][k] <= points[i][k] for k in range(len(points[i]))) \
                    and any(points[j][k] < points[i][k]
                            for k in range(len(points[i]))):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def is_supported(point, others, weights_grid):
    """Проверить, является ли точка оптимальной для некоторой взвешенной суммы.

    point и others — нормализованные векторы критериев (f1n, f2n, f3n),
    где f1 максимизируется, f2 минимизируется, f3 максимизируется.
    Скаляризация:
    s(p) = w1 * f1n - w2 * f2n + w3 * f3n (максимизация).
    Возвращает True, если найдётся весовой вектор из weights_grid, при
    котором точка не хуже всех остальных. Используется для выделения
    «поддерживаемых» решений, которые в принципе может найти метод
    взвешенной суммы.
    """
    def scalar(p, w):
        return w[0] * p[0] - w[1] * p[1] + w[2] * p[2]
    for w in weights_grid:
        s = scalar(point, w)
        if all(s >= scalar(q, w) - 1e-12 for q in others):
            return True
    return False
