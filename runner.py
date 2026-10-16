"""Проведение серий независимых запусков и запись результатов в CSV (ЛР № 3)."""

import csv
import os
import statistics
import time

from metrics import hypervolume


def run_series(runner, runs=21, base_seed=0, **kwargs):
    results = []
    for i in range(runs):
        seed = base_seed + i
        t0 = time.perf_counter()
        res = runner(seed=seed, **kwargs)
        res["seed"] = seed
        res["wall_time"] = time.perf_counter() - t0
        results.append(res)
    return results, summarize(results)


def summarize(results):
    """Сводная статистика по гиперобъёму фронта и размеру фронта."""
    hvs = [r["hv"] for r in results]
    sizes = [r["front_size"] for r in results]
    times = [r.get("wall_time", 0.0) for r in results]
    return {
        "runs": len(results),
        "best": max(hvs),
        "worst": min(hvs),
        "mean": statistics.fmean(hvs),
        "median": statistics.median(hvs),
        "stdev": statistics.stdev(hvs) if len(hvs) > 1 else 0.0,
        "mean_front_size": statistics.fmean(sizes),
        "mean_time_s": statistics.fmean(times),
        "hvs": hvs,
    }


def curve_stats(results, n_points=200):
    """Среднее/мин/макс гиперобъёма по серии запусков на сетке fe."""
    max_fe = max(r["history"][-1][0] for r in results)
    grid = [int(round(max_fe * i / (n_points - 1))) for i in range(n_points)]

    def interp(hist, g):
        xs = [h[0] for h in hist]
        ys = [h[1] for h in hist]
        out = []
        for target in g:
            if target <= xs[0]:
                out.append(ys[0])
                continue
            if target >= xs[-1]:
                out.append(ys[-1])
                continue
            j = 0
            while xs[j + 1] < target:
                j += 1
            x0, y0 = xs[j], ys[j]
            x1, y1 = xs[j + 1], ys[j + 1]
            out.append(y0 + (y1 - y0) * (target - x0) / (x1 - x0)
                       if x1 != x0 else y0)
        return out

    curves = [interp(r["history"], grid) for r in results]
    mins = [min(c[i] for c in curves) for i in range(len(grid))]
    means = [statistics.fmean([c[i] for c in curves]) for i in range(len(grid))]
    maxs = [max(c[i] for c in curves) for i in range(len(grid))]
    return grid, mins, means, maxs


def write_runs_csv(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "hv", "front_size", "fe", "generations",
                    "wall_time_s"])
        for r in results:
            w.writerow([r["seed"], f"{r['hv']:.6f}", r["front_size"],
                        r.get("fe", 0), r.get("generations", 0),
                        f"{r.get('wall_time', 0.0):.4f}"])


def write_history_csv(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "fe", "hv"])
        for r in results:
            for fe, hv in r["history"]:
                w.writerow([r["seed"], fe, f"{hv:.6f}"])


def write_front_csv(inst, solutions, path, seed=None, append=False):
    """Записать фронт Парето: критерии, число датчиков, список датчиков."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, newline="") as f:
        w = csv.writer(f)
        header = ["покрытие", "стоимость", "устойчивость",
                  "покрытие_норм", "стоимость_норм", "устойчивость_норм",
                  "число_датчиков", "датчики"]
        if seed is not None:
            header = ["seed"] + header
        if not append:
            w.writerow(header)
        for obj, chrom in solutions:
            f1n, f2n, f3n = inst.normalize(obj)
            row = [obj[0], obj[1], round(obj[2], 4),
                   f"{f1n:.4f}", f"{f2n:.4f}", f"{f3n:.4f}",
                   sum(chrom),
                   ";".join(str(i) for i, b in enumerate(chrom) if b)]
            if seed is not None:
                row = [seed] + row
            w.writerow(row)


def append_summary_csv(stats, name, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = ["method", "runs", "hv_best", "hv_mean", "hv_median", "hv_stdev",
              "hv_worst", "mean_front_size", "mean_time_s"]
    row = [name, stats["runs"], f"{stats['best']:.6f}", f"{stats['mean']:.6f}",
           f"{stats['median']:.6f}", f"{stats['stdev']:.6f}",
           f"{stats['worst']:.6f}", f"{stats['mean_front_size']:.1f}",
           f"{stats['mean_time_s']:.4f}"]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)
