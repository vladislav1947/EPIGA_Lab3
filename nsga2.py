"""NSGA-II для трёхкритериальной задачи выбора функций (ЛР № 3, трек A).

Реализованы самостоятельно:
  * векторизованная проверка доминирования и быстрая недоминируемая
    сортировка (fast non-dominated sorting);
  * расстояние до скопления (crowding distance);
  * турнирная селекция по (ранг, расстояние до скопления);
  * бинарные операторы вариации (кроссинговер и битовая мутация);
  * элитарное замещение родителей потомками по рангу и скоплению.

Готовые реализации NSGA-II не используются.
"""

import random

import numpy as np

from metrics import hypervolume
import problem


# --------------------------------------------------------------------------
# Доминирование и недоминируемая сортировка
# --------------------------------------------------------------------------

def dominates(a, b):
    """a доминирует b в задаче минимизации: все <= и хотя бы одно <."""
    return (all(x <= y for x, y in zip(a, b))
            and any(x < y for x, y in zip(a, b)))


def _dominates_matrix(Z):
    """Матрица D: D[p, q] = True, если точка p доминирует точку q."""
    le = np.all(Z[:, None, :] <= Z[None, :, :], axis=2)
    lt = np.any(Z[:, None, :] < Z[None, :, :], axis=2)
    return le & lt


def fast_non_dominated_sort(Z):
    """Быстрая недоминируемая сортировка.

    Возвращает список фронтов; fronts[k] — список индексов точек ранга k.
    Точки ранга 0 недоминируемы (фронт Парето).
    """
    Z = np.asarray(Z, dtype=float)
    D = _dominates_matrix(Z)
    nd = D.sum(axis=0).astype(int)      # сколько точек доминируют q
    fronts = []
    remaining = set(range(len(Z)))
    while remaining:
        current = [i for i in sorted(remaining) if nd[i] == 0]
        if not current:
            # защита от вырожденных случаев
            current = sorted(remaining)
        fronts.append(current)
        remaining -= set(current)
        # точки текущего фронта перестают доминировать оставшихся
        for i in current:
            nd[np.where(D[i])[0]] -= 1
    return fronts


def crowding_distance(front_idx, Z):
    """Расстояние до скопления для точек одного фронта (минимизация).

    Граничные точки по каждому критерию получают бесконечность.
    """
    if len(front_idx) <= 2:
        return np.full(len(front_idx), np.inf)
    Zf = np.asarray(Z, dtype=float)[front_idx]
    dist = np.zeros(len(front_idx))
    for k in range(Zf.shape[1]):
        order = np.argsort(Zf[:, k], kind="stable")
        zs = Zf[order, k]
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        span = zs[-1] - zs[0]
        if span == 0:
            continue
        dist[order[1:-1]] += (zs[2:] - zs[:-2]) / span
    return dist


# --------------------------------------------------------------------------
# Операторы вариации (бинарное кодирование)
# --------------------------------------------------------------------------

def one_point_crossover(p1, p2, rng):
    point = rng.randrange(len(p1))
    return p1[:point] + p2[point:], p2[:point] + p1[point:]


def two_point_crossover(p1, p2, rng):
    a, b = sorted(rng.sample(range(len(p1) + 1), 2))
    return p1[:a] + p2[a:b] + p1[b:], p2[:a] + p1[a:b] + p2[b:]


def uniform_crossover(p1, p2, rng):
    c1 = [p1[i] if rng.random() < 0.5 else p2[i] for i in range(len(p1))]
    c2 = [p2[i] if rng.random() < 0.5 else p1[i] for i in range(len(p1))]
    return c1, c2


CROSSOVERS = {"one_point": one_point_crossover,
              "two_point": two_point_crossover,
              "uniform": uniform_crossover}


def flip_mutation(chrom, rng, pm):
    return [1 - chrom[i] if rng.random() < pm else chrom[i]
            for i in range(len(chrom))]


# --------------------------------------------------------------------------
# Селекция
# --------------------------------------------------------------------------

def tournament_selection(rank, crowd, rng, k=2):
    """Турнир по (ранг, -расстояние до скопления): меньше ранг — лучше,
    при равенстве — больше расстояние (поддержание разнообразия)."""
    n = len(rank)
    best = rng.randrange(n)
    for _ in range(k - 1):
        cand = rng.randrange(n)
        if (rank[cand], -crowd[cand]) < (rank[best], -crowd[best]):
            best = cand
    return best


# --------------------------------------------------------------------------
# Основной цикл NSGA-II
# --------------------------------------------------------------------------

def run_nsga2(inst, pop_size=100, max_fe=20000, pc=0.9, pm=0.05,
              crossover="uniform", seed=0, ref=None, hv_every=1):
    """Запустить NSGA-II на экземпляре задачи.

    Параметры:
      pop_size  — размер популяции;
      max_fe    — лимит вычислений функции;
      pc, pm    — вероятности кроссинговера и мутации (на бит);
      ref       — опорная точка для гиперобъёма в нормированном
                  пространстве минимизации (по умолчанию (1, 1, 1)).

    Возвращает словарь:
      pop, objs        — финальная популяция и её критерии (ценность, срок, риск);
      z                — нормированные критерии в пространстве минимизации;
      fronts           — список фронтов (индексы в pop);
      front            — индексы недоминируемых решений (фронт Парето);
      history          — список пар (fe, гиперобъём фронта);
      fe, generations  — затраченные вычисления функции и число поколений.
    """
    rng = random.Random(seed)
    cross = CROSSOVERS[crossover]
    if ref is None:
        ref = (1.0, 1.0, 1.0)

    def eval_pop(pops):
        objs = [inst.objectives(c) for c in pops]
        z = [inst.to_minimization(o) for o in objs]
        return objs, z

    # --- Инициализация ---
    pop = [problem.random_binary_solution(inst, rng) for _ in range(pop_size)]
    objs, z = eval_pop(pop)
    fe = pop_size

    def sort_and_crowd(zz):
        fronts = fast_non_dominated_sort(zz)
        rank = np.zeros(len(zz), dtype=int)
        crowd = np.zeros(len(zz))
        for k, front in enumerate(fronts):
            for i in front:
                rank[i] = k
            cd = crowding_distance(front, zz)
            for pos, i in enumerate(front):
                crowd[i] = cd[pos]
        return fronts, rank, crowd

    fronts, rank, crowd = sort_and_crowd(z)
    front_objs = [z[i] for i in fronts[0]]
    history = [(fe, hypervolume(front_objs, ref))]

    generation = 0
    while fe < max_fe:
        generation += 1
        # --- Скрещивание и мутация ---
        offspring = []
        offspring_z = []
        offspring_objs = []
        while len(offspring) < pop_size and fe < max_fe:
            i1 = tournament_selection(rank, crowd, rng)
            i2 = tournament_selection(rank, crowd, rng)
            if rng.random() < pc:
                c1, c2 = cross(pop[i1], pop[i2], rng)
            else:
                c1, c2 = list(pop[i1]), list(pop[i2])
            for c in (c1, c2):
                if len(offspring) >= pop_size or fe >= max_fe:
                    break
                c = flip_mutation(c, rng, pm)
                offspring.append(c)
                o = inst.objectives(c)
                offspring_objs.append(o)
                offspring_z.append(inst.to_minimization(o))
                fe += 1

        # --- Элитарное замещение: P U Q сортируется, отбор по рангу и
        #     расстоянию до скопления ---
        combined = pop + offspring
        combined_objs = objs + offspring_objs
        combined_z = z + offspring_z
        all_fronts, all_rank, all_crowd = sort_and_crowd(combined_z)

        new_pop, new_objs, new_z = [], [], []
        for front in all_fronts:
            if len(new_pop) + len(front) <= pop_size:
                for i in front:
                    new_pop.append(combined[i])
                    new_objs.append(combined_objs[i])
                    new_z.append(combined_z[i])
            else:
                # фронт не помещается целиком — берём точки с наибольшим
                # расстоянием до скопления
                order = sorted(front, key=lambda i: -all_crowd[i])
                for i in order[:pop_size - len(new_pop)]:
                    new_pop.append(combined[i])
                    new_objs.append(combined_objs[i])
                    new_z.append(combined_z[i])
            if len(new_pop) >= pop_size:
                break

        pop, objs, z = new_pop, new_objs, new_z
        fronts, rank, crowd = sort_and_crowd(z)

        if generation % hv_every == 0 or fe >= max_fe:
            front_objs = [z[i] for i in fronts[0]]
            history.append((fe, hypervolume(front_objs, ref)))

    return {
        "pop": pop,
        "objs": objs,
        "z": z,
        "fronts": fronts,
        "front": fronts[0],
        "history": history,
        "fe": fe,
        "generations": generation,
    }


def front_solutions(result):
    """Извлечь недоминируемые решения: (ценность, срок, риск) и хромосомы."""
    return [(result["objs"][i], result["pop"][i]) for i in result["front"]]
