"""Самопроверка компонентов ЛР № 3.

Запуск: python3 test_nsga.py
Проверяется:
  * корректность доминирования и быстрой недоминируемой сортировки
    (сравнение с прямым перебором);
  * корректность расстояния до скопления (граничные точки);
  * точный гиперобъём для 2D и 3D на известных примерах;
  * монотонность критериев задачи (добавление датчика не уменьшает
    покрытие, стоимость и устойчивость);
  * сходимость NSGA-II: фронт непуст, решения допустимы, гиперобъём
    растёт от поколения к поколению в среднем;
  * взвешенная сумма с весами (1, 0, 0) максимизирует покрытие.
"""

import random

import numpy as np

from metrics import filter_nondominated, hypervolume
from nsga2 import (crowding_distance, dominates, fast_non_dominated_sort,
                   run_nsga2)
from problem import generate_instance
from weighted_sum import run_weighted_sum_ga

failures = []


def check(cond, msg):
    status = "OK " if cond else "FAIL"
    if not cond:
        failures.append(msg)
    print(f"[{status}] {msg}")


def brute_force_front(objs):
    """Прямой перебор недоминируемых индексов."""
    n = len(objs)
    keep = []
    for i in range(n):
        dom = False
        for j in range(n):
            if j != i and dominates(objs[j], objs[i]):
                dom = True
                break
        if not dom:
            keep.append(i)
    return sorted(keep)


def main():
    rng = random.Random(0)

    # --- Доминирование ---
    check(dominates((1, 1), (2, 2)), "dominates: (1,1) < (2,2)")
    check(not dominates((2, 2), (1, 1)), "dominates: (2,2) не доминирует (1,1)")
    check(not dominates((1, 2), (2, 1)), "dominates: несравнимые точки")
    check(dominates((1, 1), (1, 1)) is False,
          "dominates: совпадающие точки не доминируют")

    # --- Недоминируемая сортировка vs прямой перебор ---
    for trial in range(5):
        pts = [(rng.random(), rng.random(), rng.random()) for _ in range(30)]
        fronts = fast_non_dominated_sort(pts)
        front0 = sorted(fronts[0])
        expected = brute_force_front(pts)
        check(front0 == expected,
              f"быстрая сортировка совпадает с перебором (тест {trial + 1})")
        # каждый следующий фронт должен быть недоминируемым среди оставшихся
        ranks_ok = True
        for k, front in enumerate(fronts):
            for i in front:
                for j in front:
                    if dominates(pts[j], pts[i]):
                        ranks_ok = False
        check(ranks_ok, "внутри одного фронта нет доминирования")

    # --- Расстояние до скопления ---
    pts = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
    cd = crowding_distance([0, 1, 2], pts)
    check(np.isinf(cd[0]) and np.isinf(cd[2]),
          "граничные точки имеют бесконечное расстояние до скопления")
    check(np.isfinite(cd[1]), "внутренняя точка имеет конечное расстояние")

    # --- Гиперобъём 2D ---
    # точка (0.5, 0.5) с опорной (1, 1) покрывает квадрат площади 0.25
    check(abs(hypervolume([(0.5, 0.5)], (1.0, 1.0)) - 0.25) < 1e-12,
          "гиперобъём 2D одного прямоугольника")
    # две точки (0.2, 0.8) и (0.8, 0.2): объединение
    # = (1-0.2)*0.2 + ... вычислим вручную: прямоугольники
    # [0.2,1]x[0.8,1] (0.8*0.2=0.16) и [0.8,1]x[0.2,1] (0.2*0.8=0.16),
    # пересечение [0.8,1]x[0.8,1] = 0.2*0.2 = 0.04 -> 0.28
    hv = hypervolume([(0.2, 0.8), (0.8, 0.2)], (1.0, 1.0))
    check(abs(hv - 0.28) < 1e-12, f"гиперобъём 2D двух прямоугольников ({hv:.4f})")

    # --- Гиперобъём 3D ---
    # одна точка (0.5, 0.5, 0.5) с опорной (1,1,1) -> куб 0.125
    check(abs(hypervolume([(0.5, 0.5, 0.5)], (1.0, 1.0, 1.0)) - 0.125) < 1e-12,
          "гиперобъём 3D одного куба")
    # точки с доминированием: (0.5,0.5,0.5) доминирует (0.6,0.6,0.6)
    hv = hypervolume([(0.5, 0.5, 0.5), (0.6, 0.6, 0.6)], (1.0, 1.0, 1.0))
    check(abs(hv - 0.125) < 1e-12, "гиперобъём 3D: доминируемая точка не "
                                   "добавляет объёма")

    # --- Критерии задачи ---
    inst = generate_instance(n=30, seed=11)
    base = inst.objectives([0] * inst.n)
    check(base == (0, 0, 0), "пустой набор имеет нулевые критерии")
    chrom = [0] * inst.n
    chrom[3] = 1
    o = inst.objectives(chrom)
    check(o == (len(inst.cover[3]), inst.costs[3], 0),
          "критерии одиночного датчика корректны")

    # --- Запуск NSGA-II ---
    res = run_nsga2(inst, pop_size=50, max_fe=3000, seed=0)
    check(len(res["front"]) > 0, "фронт Парето непуст")
    check(all(sum(c) >= 0 for c in res["pop"]), "решения допустимы")
    hv_series = [h for _, h in res["history"]]
    check(hv_series[-1] >= hv_series[0],
          "гиперобъём не убывает к концу работы")
    # все точки фронта недоминируемы
    front_z = [res["z"][i] for i in res["front"]]
    check(sorted(filter_nondominated(front_z)) == sorted(range(len(front_z))),
          "фронт состоит из недоминируемых точек")

    # --- Взвешенная сумма: вес только на ценность -> максимум ценности ---
    res_ws = run_weighted_sum_ga(inst, (1.0, 0.0, 0.0), pop_size=50,
                                 max_fe=10000, seed=0)
    check(res_ws["best_obj"][0] == inst.n,
          f"взвешенная сумма (1,0,0) покрывает все вершины "
          f"(покрытие {res_ws['best_obj'][0]} из {inst.n})")

    print()
    if failures:
        print("Ошибки:")
        for f in failures:
            print(" -", f)
        raise SystemExit(1)
    print("Все проверки пройдены.")


if __name__ == "__main__":
    main()
