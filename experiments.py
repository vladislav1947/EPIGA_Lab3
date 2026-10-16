"""Обязательные эксперименты лабораторной работы № 3 (трек A, вариант 9).

Эксперименты:
  1. Серия из >= 20 независимых запусков NSGA-II; метрика — гиперобъём
     фронта Парето.
  2. Визуализация фронта Парето: три 2D-проекции, 3D-диаграмма и карта
     покрытия для характерных решений.
  3. Три характерных решения из фронта («максимум покрытия»,
     «экономное», «сбалансированное») с предметной трактовкой.
  4. Сравнение NSGA-II с подходом взвешенной суммы критериев при двух
     режимах бюджета, а также со случайным поиском с архивом
     недоминируемых решений.
  5. Анализ «поддерживаемых» и «неподдерживаемых» решений: каких точек
     фронта в принципе не может найти метод взвешенной суммы.
"""

import csv
import os
import time

import numpy as np

from baseline import run_random_search
from metrics import (filter_nondominated, hypervolume, is_supported)
from nsga2 import front_solutions, run_nsga2
import problem
import runner
from weighted_sum import run_weighted_sum_ga, weight_vectors

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")

INSTANCE_PARAMS = dict(n=30, seed=11, radius=30.0, c_lo=5, c_hi=50)
RUNS = 21              # >= 20 независимых запусков
MAX_FE = 20000         # бюджет вычислений функции
POP_SIZE = 100
REF = (1.0, 1.0, 1.0)  # опорная точка гиперобъёма в нормированном пространстве

BASE = dict(pop_size=POP_SIZE, max_fe=MAX_FE, pc=0.9, pm=0.05,
            crossover="uniform", ref=REF)

WS_STEP = 0.25         # шаг сетки весов во взвешенной сумме (15 векторов)


def nsga2_run_record(inst, seed, **kwargs):
    t0 = time.perf_counter()
    res = run_nsga2(inst, seed=seed, **kwargs)
    front_z = [res["z"][i] for i in res["front"]]
    return {
        "result": res,
        "front": front_solutions(res),
        "hv": hypervolume(front_z, REF),
        "front_size": len(res["front"]),
        "history": res["history"],
        "fe": res["fe"],
        "generations": res["generations"],
        "wall_time": time.perf_counter() - t0,
        "seed": seed,
    }


def random_search_record(inst, seed, max_fe):
    t0 = time.perf_counter()
    res = run_random_search(inst, max_fe=max_fe, seed=seed, ref=REF)
    z = [inst.to_minimization(o) for o in res["archive_objs"]]
    return {
        "result": res,
        "front": list(zip(res["archive_objs"], res["archive"])),
        "hv": hypervolume(z, REF),
        "front_size": len(res["archive_objs"]),
        "history": res["history"],
        "fe": res["fe"],
        "generations": 0,
        "wall_time": time.perf_counter() - t0,
        "seed": seed,
    }


def weighted_sum_series(inst, weights, runs, max_fe_per_weight, base_seed=0):
    """Серия «запусков» подхода взвешенной суммы.

    Один «запуск» = по одному оптимуму на каждый весовой вектор; объединение
    этих оптимумов даёт фронт взвешенной суммы. Возвращает список записей
    с фронтом и гиперобъёмом.
    """
    records = []
    for r in range(runs):
        t0 = time.perf_counter()
        solutions = []
        for w in weights:
            res = run_weighted_sum_ga(inst, w, pop_size=POP_SIZE,
                                      max_fe=max_fe_per_weight,
                                      pc=0.9, pm=0.05, crossover="uniform",
                                      seed=base_seed + r)
            solutions.append((res["best_obj"], res["best_x"]))
        nd = filter_nondominated([inst.to_minimization(o) for o, _ in solutions])
        front = [solutions[i] for i in nd]
        front_z = [inst.to_minimization(o) for o, _ in front]
        records.append({
            "front": front,
            "hv": hypervolume(front_z, REF),
            "front_size": len(front),
            "history": [(max_fe_per_weight * len(weights),
                         hypervolume(front_z, REF))],
            "fe": max_fe_per_weight * len(weights),
            "generations": 0,
            "wall_time": time.perf_counter() - t0,
            "seed": base_seed + r,
        })
    return records


def write_representative_solutions_csv(inst, front, path):
    """Три характерных решения из фронта.

    1) «максимум покрытия» — наибольшее покрытие (при равных — дешевле);
    2) «экономное» — минимальная стоимость среди решений, сохраняющих
       не менее половины максимального покрытия фронта;
    3) «сбалансированное» — ближайшее к идеальной точке (макс. покрытие,
       мин. стоимость, макс. устойчивость) в нормированном пространстве.
    """
    objs = [o for o, _ in front]
    i_max_cov = max(range(len(front)),
                    key=lambda i: (objs[i][0], -objs[i][1]))

    cov_threshold = 0.5 * objs[i_max_cov][0]
    economical = [i for i in range(len(front)) if objs[i][0] >= cov_threshold]
    i_economical = min(economical, key=lambda i: (objs[i][1], -objs[i][2]))

    def dist_to_ideal(o):
        f1n, f2n, f3n = inst.normalize(o)
        return (f1n - 1.0) ** 2 + f2n ** 2 + (f3n - 1.0) ** 2
    i_balanced = min(range(len(front)), key=lambda i: dist_to_ideal(objs[i]))

    rows = [
        ("максимум покрытия", front[i_max_cov]),
        ("экономное (>=50% покрытия)", front[i_economical]),
        ("сбалансированное", front[i_balanced]),
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["решение", "покрытие", "стоимость", "устойчивость",
                    "число_датчиков", "покрытие_на_единицу_стоимости",
                    "датчики"])
        for name, (obj, chrom) in rows:
            w.writerow([name, obj[0], obj[1], obj[2], sum(chrom),
                        f"{obj[0] / max(obj[1], 1e-9):.3f}",
                        ";".join(str(i) for i, b in enumerate(chrom) if b)])
    return rows


def write_supported_analysis_csv(inst, front, path, step=0.05):
    """Какие точки фронта может найти метод взвешенной суммы."""
    ns = int(round(1.0 / step))
    grid = []
    for a in range(ns + 1):
        for b in range(ns + 1 - a):
            grid.append((a / ns, b / ns, (ns - a - b) / ns))
    # дедупликация одинаковых целевых векторов
    seen = {}
    for obj, chrom in front:
        key = (obj[0], obj[1], round(obj[2], 6))
        if key not in seen:
            seen[key] = (obj, chrom)
    dedup_front = list(seen.values())
    objs_norm = [inst.normalize(o) for o, _ in dedup_front]
    rows = []
    n_supported = 0
    for (obj, chrom), on in zip(dedup_front, objs_norm):
        supported = is_supported(on, objs_norm, grid)
        n_supported += supported
        rows.append((obj, supported))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["покрытие", "стоимость", "устойчивость",
                    "поддерживается_взвешенной_суммой"])
        for obj, supported in rows:
            w.writerow([obj[0], obj[1], obj[2],
                        "да" if supported else "нет"])
    return n_supported, len(dedup_front)


def write_params_csv(inst, n_weights, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["параметр", "значение"])
        w.writerow(["задача", "размещение датчиков на графе "
                               "(трек A, вариант 9)"])
        w.writerow(["критерии", "максимум покрытия, минимум стоимости, "
                                "максимум устойчивости к отказу"])
        w.writerow(["тип данных", "синтетические, seed=11"])
        w.writerow(["число вершин n", inst.n])
        w.writerow(["граф", "геометрический случайный, связный"])
        w.writerow(["стоимости датчиков", "целые из [5, 50] (равномерно)"])
        w.writerow(["представление особи", "бинарная строка длины n"])
        w.writerow(["алгоритм", "NSGA-II (собственная реализация)"])
        w.writerow(["недоминируемая сортировка", "быстрая, векторизованная"])
        w.writerow(["поддержание разнообразия", "расстояние до скопления"])
        w.writerow(["размер популяции", POP_SIZE])
        w.writerow(["бюджет вычислений функции", MAX_FE])
        w.writerow(["вероятность кроссинговера", 0.9])
        w.writerow(["вероятность мутации на бит", 0.05])
        w.writerow(["кроссовер", "равномерный"])
        w.writerow(["элитизм", "замещение по рангу и расстоянию до скопления"])
        w.writerow(["критерий остановки", f"лимит вычислений функции ({MAX_FE})"])
        w.writerow(["число запусков в серии", RUNS])
        w.writerow(["метрика качества фронта", "гиперобъём (точный, 3D)"])
        w.writerow(["опорная точка гиперобъёма", "(1, 1, 1) в нормированном "
                                                  "пространстве минимизации"])
        w.writerow(["шаг сетки весов (взвешенная сумма)", WS_STEP])
        w.writerow(["число весовых векторов", n_weights])
        w.writerow(["базовый метод", "случайный поиск с архивом "
                                     "недоминируемых решений"])


def print_stats(name, stats):
    print(f"{name}:")
    print(f"  HV: best={stats['best']:.4f} mean={stats['mean']:.4f} "
          f"median={stats['median']:.4f}\n  stdev={stats['stdev']:.4f} "
          f"worst={stats['worst']:.4f} | размер фронта: "
          f"{stats['mean_front_size']:.1f} | время={stats['mean_time_s']:.2f}s")


def main():
    inst = problem.generate_instance(**INSTANCE_PARAMS)
    problem.save_instance(inst, DATA_DIR)
    print(f"Экземпляр: n={inst.n}, рёбер={len(inst.edges)}, "
          f"суммарная стоимость={inst.total_cost:.0f}")
    print(f"Запусков в серии: {RUNS}, бюджет: {MAX_FE}\n")

    compare_csv = os.path.join(RESULTS_DIR, "methods_compare.csv")
    if os.path.exists(compare_csv):
        os.remove(compare_csv)

    # --- Эксперимент 1: серия запусков NSGA-II ---
    nsga_records = [nsga2_run_record(inst, seed=s, **BASE) for s in range(RUNS)]
    for r in nsga_records:
        r.pop("result", None)
    nsga_stats = runner.summarize(nsga_records)
    runner.write_runs_csv(nsga_records, os.path.join(RESULTS_DIR, "runs_nsga2.csv"))
    runner.write_history_csv(nsga_records,
                             os.path.join(RESULTS_DIR, "history_nsga2.csv"))
    runner.append_summary_csv(nsga_stats, "NSGA-II", compare_csv)
    print_stats("NSGA-II", nsga_stats)

    # Фронты всех запусков в один CSV
    front_path = os.path.join(RESULTS_DIR, "nsga2_fronts.csv")
    if os.path.exists(front_path):
        os.remove(front_path)
    for r in nsga_records:
        runner.write_front_csv(inst, r["front"], front_path, seed=r["seed"],
                               append=(r["seed"] != 0))
    print(f"Фронты записаны: {front_path}\n")

    # --- Эксперимент 2: визуализация фронта (репрезентативный запуск) ---
    rep = nsga_records[0]
    rep_front = rep["front"]

    # --- Эксперимент 3: характерные решения ---
    rows = write_representative_solutions_csv(
        inst, rep_front, os.path.join(RESULTS_DIR, "representative_solutions.csv"))
    for name, (obj, chrom) in rows:
        print(f"  {name}: покрытие={obj[0]} стоимость={obj[1]} "
              f"устойчивость={obj[2]} датчиков={sum(chrom)}")

    # --- Эксперимент 4: сравнение методов ---
    weights = weight_vectors(WS_STEP)
    print(f"\nВесовых векторов: {len(weights)}")
    # 4a) взвешенная сумма, каждому весу — полный бюджет
    ws_full = weighted_sum_series(inst, weights, runs=RUNS,
                                  max_fe_per_weight=MAX_FE)
    ws_full_stats = runner.summarize(ws_full)
    runner.append_summary_csv(ws_full_stats,
                              f"weighted sum ({len(weights)}x{MAX_FE} FE)",
                              compare_csv)
    print_stats(f"Взвешенная сумма (каждому весу {MAX_FE} FE)", ws_full_stats)
    # 4b) взвешенная сумма при том же суммарном бюджете, что у NSGA-II
    ws_equal = weighted_sum_series(
        inst, weights, runs=RUNS, max_fe_per_weight=MAX_FE // len(weights))
    ws_equal_stats = runner.summarize(ws_equal)
    runner.append_summary_csv(
        ws_equal_stats,
        f"weighted sum equal budget "
        f"({len(weights)}x{MAX_FE // len(weights)} FE)",
        compare_csv)
    print_stats(f"Взвешенная сумма (равный бюджет: {len(weights)}x"
                f"{MAX_FE // len(weights)} FE)", ws_equal_stats)

    # 4c) случайный поиск с архивом
    rs_records = [random_search_record(inst, seed=s, max_fe=MAX_FE)
                  for s in range(RUNS)]
    rs_stats = runner.summarize(rs_records)
    runner.write_runs_csv(rs_records,
                          os.path.join(RESULTS_DIR, "runs_random_search.csv"))
    runner.write_history_csv(rs_records,
                             os.path.join(RESULTS_DIR,
                                          "history_random_search.csv"))
    runner.append_summary_csv(rs_stats, "random search", compare_csv)
    print_stats("Случайный поиск с архивом", rs_stats)

    # --- Эксперимент 5: поддерживаемые vs неподдерживаемые решения ---
    n_supported, n_total = write_supported_analysis_csv(
        inst, rep_front, os.path.join(RESULTS_DIR, "supported_analysis.csv"))
    print(f"\nПоддерживается взвешенной суммой: {n_supported} из {n_total} "
          f"точек фронта")

    # --- Графики ---
    import plots
    ws_front = ws_equal[0]["front"]   # фронт одного «запуска» ВС
    rs_front = rs_records[0]["front"]
    plots.plot_projections(
        {"NSGA-II": rep_front, "взвешенная сумма": ws_front,
         "случайный поиск": rs_front},
        os.path.join(RESULTS_DIR, "pareto_projections.png"))
    plots.plot_3d({"NSGA-II": rep_front},
                  os.path.join(RESULTS_DIR, "pareto_3d.png"))
    plots.plot_coverage_map(
        inst, {name: sol for name, sol in rows},
        os.path.join(RESULTS_DIR, "coverage_map.png"))
    plots.plot_hv_convergence(nsga_records,
                              os.path.join(RESULTS_DIR, "hv_convergence.png"))
    plots.plot_hv_comparison(
        {"NSGA-II": nsga_stats,
         f"взв. сумма\n({len(weights)}x{MAX_FE} FE)": ws_full_stats,
         f"взв. сумма\n(равный бюджет)": ws_equal_stats,
         "случайный поиск": rs_stats},
        os.path.join(RESULTS_DIR, "hv_comparison.png"))
    plots.plot_flowchart(os.path.join(RESULTS_DIR, "flowchart.png"))

    # --- Параметры ---
    write_params_csv(inst, len(weights),
                     os.path.join(RESULTS_DIR, "params.csv"))
    print(f"\nРезультаты сохранены в {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
