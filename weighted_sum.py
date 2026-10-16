"""Подход взвешенной суммы критериев для сравнения с NSGA-II.

Скаляризация: s(x) = w1 * f1n(x) - w2 * f2n(x) + w3 * f3n(x) -> максимум,
где f1n, f2n, f3n — нормированные критерии (покрытие максимизируется,
стоимость минимизируется, устойчивость максимизируется),
а веса w >= 0 в сумме дают 1.

Для построения фронта Парето подход запускается с разными весовыми
векторами; каждое решение оптимизирует только одну скалярную функцию,
поэтому метод способен находить лишь «поддерживаемые» решения.
"""

import random

from nsga2 import CROSSOVERS, flip_mutation, tournament_selection


def run_weighted_sum_ga(inst, weights, pop_size=100, max_fe=20000,
                        pc=0.9, pm=0.05, crossover="uniform", seed=0):
    """Один запуск ГА на взвешенной сумме критериев.

    Возвращает словарь с лучшим решением, его критериями и значением
    скалярной функции.
    """
    rng = random.Random(seed)
    cross = CROSSOVERS[crossover]

    def scalar(obj):
        f1n, f2n, f3n = inst.normalize(obj)
        return weights[0] * f1n - weights[1] * f2n + weights[2] * f3n

    def random_solution():
        return [rng.randint(0, 1) for _ in range(inst.n)]

    pop = [random_solution() for _ in range(pop_size)]
    objs = [inst.objectives(c) for c in pop]
    fit = [scalar(o) for o in objs]
    fe = pop_size

    best_i = max(range(pop_size), key=lambda i: fit[i])
    best_x = list(pop[best_i])
    best_obj = objs[best_i]
    best_fit = fit[best_i]
    history = [(fe, best_fit)]

    generation = 0
    while fe < max_fe:
        generation += 1
        order = sorted(range(pop_size), key=lambda i: -fit[i])
        new_pop = [list(pop[order[0]])]           # элитизм: 1 лучшая особь
        new_objs = [objs[order[0]]]
        new_fit = [fit[order[0]]]
        while len(new_pop) < pop_size and fe < max_fe:
            # турнир размера 2 по приспособленности (максимизация)
            i1 = max([rng.randrange(pop_size) for _ in range(2)],
                     key=lambda i: fit[i])
            i2 = max([rng.randrange(pop_size) for _ in range(2)],
                     key=lambda i: fit[i])
            if rng.random() < pc:
                c1, c2 = cross(pop[i1], pop[i2], rng)
            else:
                c1, c2 = list(pop[i1]), list(pop[i2])
            for c in (c1, c2):
                if len(new_pop) >= pop_size or fe >= max_fe:
                    break
                c = flip_mutation(c, rng, pm)
                new_pop.append(c)
                o = inst.objectives(c)
                new_objs.append(o)
                new_fit.append(scalar(o))
                fe += 1
                if new_fit[-1] > best_fit:
                    best_fit = new_fit[-1]
                    best_x = list(c)
                    best_obj = o
        pop, objs, fit = new_pop, new_objs, new_fit
        history.append((fe, best_fit))

    return {
        "best_x": best_x,
        "best_obj": best_obj,          # (покрытие, стоимость, устойчивость)
        "best_fit": best_fit,
        "history": history,
        "fe": fe,
        "generations": generation,
    }


def weight_vectors(step=0.25):
    """Все весовые векторы w >= 0 с шагом step, сумма компонент равна 1."""
    n_steps = int(round(1.0 / step))
    vectors = []
    for a in range(n_steps + 1):
        for b in range(n_steps + 1 - a):
            c = n_steps - a - b
            vectors.append((a / n_steps, b / n_steps, c / n_steps))
    return vectors
