"""Задача размещения датчиков на графе (трек A, вариант 9).

Индивидуальный вариант: V = 9 (N = 15, Q = 1, Y = 26), трек A.

Постановка:
    Дан неориентированный граф района из n вершин (потенциальные места
    установки). Датчик в вершине покрывает саму вершину и всех её соседей.
    Каждый датчик имеет стоимость установки. Требуется выбрать подмножество
    вершин (бинарный вектор x), достигая трёх конфликтующих целей:
      f1(x) = покрытие — число покрытых вершин  -> максимум;
      f2(x) = стоимость — суммарная стоимость датчиков  -> минимум;
      f3(x) = устойчивость — число вершин, покрытых не менее чем двумя
              датчиками (сохраняют покрытие при отказе любого одного
              датчика)  -> максимум.

Критерии противоречивы: каждый новый датчик растит покрытие, но и
стоимость; устойчивость требует избыточности (перекрытия зон), то есть
ещё больших затрат. Жёстких ограничений нет: любое подмножество
допустимо, поэтому интерес представляет не одно оптимальное решение,
а фронт Парето — множество недоминируемых компромиссов.

Данные синтетические; процедура генерации и seed зафиксированы
(см. generate_instance, data/instance.json).
"""

import csv
import json
import math
import os
import random


class Instance:
    """Экземпляр задачи размещения датчиков."""

    def __init__(self, n, coords, edges, costs, seed, gen_params=None):
        self.n = n
        self.coords = [tuple(c) for c in coords]
        self.costs = list(costs)
        self.seed = seed
        self.gen_params = gen_params or {}
        # Списки смежности (с покрытием самой вершины)
        self.cover = [set([i]) for i in range(n)]
        for i, j in edges:
            self.cover[i].add(j)
            self.cover[j].add(i)
        self.edges = [(int(i), int(j)) for i, j in edges]
        # Масштабы для нормализации критериев в [0, 1]
        self.total_cost = float(sum(self.costs))

    def objectives(self, chrom):
        """Вернуть тройку критериев (покрытие, стоимость, устойчивость)."""
        sensors = [i for i, b in enumerate(chrom) if b]
        counts = [0] * self.n
        for s in sensors:
            for v in self.cover[s]:
                counts[v] += 1
        coverage = sum(1 for c in counts if c >= 1)
        robust = sum(1 for c in counts if c >= 2)
        cost = sum(self.costs[i] for i in sensors)
        return (coverage, cost, robust)

    def normalize(self, obj):
        """Нормализовать критерии в [0, 1]: (f1n, f2n, f3n)."""
        return (obj[0] / self.n, obj[1] / self.total_cost,
                obj[2] / self.n)

    def to_minimization(self, obj):
        """Перевести задачу к минимизации: z = (1 - f1n, f2n, 1 - f3n).

        z1 = 1 - f1n: максимум покрытия -> минимум «непокрытости»;
        z3 = 1 - f3n: максимум устойчивости -> минимум «хрупкости».
        """
        f1n, f2n, f3n = self.normalize(obj)
        return (1.0 - f1n, f2n, 1.0 - f3n)

    def n_selected(self, chrom):
        return sum(chrom)

    def summary(self):
        return {
            "n": self.n,
            "n_edges": len(self.edges),
            "total_cost": sum(self.costs),
            "seed": self.seed,
            "gen_params": self.gen_params,
        }


def _connected(n, edges):
    seen = {0}
    stack = [0]
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    while stack:
        v = stack.pop()
        for u in adj[v]:
            if u not in seen:
                seen.add(u)
                stack.append(u)
    return len(seen) == n


def generate_instance(n=30, seed=11, radius=30.0, c_lo=5, c_hi=50):
    """Сгенерировать синтетический экземпляр задачи.

    Процедура генерации (фиксированный seed):
      * координаты вершин — равномерно в квадрате [0, 100]^2;
      * ребро между вершинами, если евклидово расстояние < radius
        (геометрический случайный граф; радиус увеличивается, пока
        граф не станет связным);
      * стоимости датчиков — целые из [c_lo, c_hi] (равномерно).
    Масштаб: n = 30 вершин.
    """
    rng = random.Random(seed)
    coords = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)]
    costs = [rng.randint(c_lo, c_hi) for _ in range(n)]
    r = radius
    while True:
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = coords[i][0] - coords[j][0]
                dy = coords[i][1] - coords[j][1]
                if math.hypot(dx, dy) < r:
                    edges.append((i, j))
        if _connected(n, edges) or r > 200:
            break
        r += 5.0
    gen_params = {"n": n, "radius": r, "c_lo": c_lo, "c_hi": c_hi}
    return Instance(n, coords, edges, costs, seed, gen_params)


def save_instance(inst, data_dir):
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "nodes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "x", "y", "стоимость_датчика", "степень"])
        for i in range(inst.n):
            w.writerow([i, f"{inst.coords[i][0]:.3f}",
                        f"{inst.coords[i][1]:.3f}", inst.costs[i],
                        len(inst.cover[i]) - 1])
    with open(os.path.join(data_dir, "instance.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "summary": inst.summary(),
            "coords": inst.coords,
            "edges": inst.edges,
            "costs": inst.costs,
        }, f, ensure_ascii=False, indent=2)


def load_instance(data_dir):
    with open(os.path.join(data_dir, "instance.json"), encoding="utf-8") as f:
        data = json.load(f)
    return Instance(data["summary"]["n"], data["coords"], data["edges"],
                    data["costs"], data["summary"]["seed"],
                    data["summary"]["gen_params"])


def random_binary_solution(inst, rng):
    return [rng.randint(0, 1) for _ in range(inst.n)]
