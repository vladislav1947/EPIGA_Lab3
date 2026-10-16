"""Базовый метод для ЛР № 3: случайный поиск с архивом недоминируемых решений.

Каждое случайное бинарное решение оценивается по трём критериям и
помещается во внешний архив, из которого удаляются доминируемые решения.
Даёт представление о качестве фронта, достигаемого без какой-либо
адаптации поиска, при том же бюджете вычислений функции.
"""

import random

import numpy as np

from metrics import hypervolume

MAX_ARCHIVE = 2000   # верхняя граница размера архива


def run_random_search(inst, max_fe, seed=0, ref=None):
    """Случайный поиск с внешним архивом недоминируемых решений.

    Возвращает словарь:
      archive_objs — критерии недоминируемых решений
      (покрытие, стоимость, устойчивость);
      archive      — соответствующие хромосомы;
      history      — (fe, гиперобъём архива);
      fe           — число вычислений функции.
    """
    rng = random.Random(seed)
    if ref is None:
        ref = (1.0, 1.0, 1.0)

    archive = []        # хромосомы
    archive_objs = []   # (покрытие, стоимость, устойчивость)
    archive_z = np.zeros((0, 3))   # нормированные критерии (минимизация)

    def add(chrom, obj):
        nonlocal archive_z
        z = np.array(inst.to_minimization(obj))
        if archive_z.shape[0] == 0:
            archive.append(chrom)
            archive_objs.append(obj)
            archive_z = z.reshape(1, 3)
            return
        # если новое решение доминируется архивом — оно не нужно
        if np.any(np.all(archive_z >= z, axis=1)
                  & np.any(archive_z > z, axis=1)):
            return
        # удаляем решения, доминируемые новым
        keep = ~(np.all(archive_z <= z, axis=1)
                 & np.any(archive_z < z, axis=1))
        archive[:] = [c for c, k in zip(archive, keep) if k]
        archive_objs[:] = [o for o, k in zip(archive_objs, keep) if k]
        archive.append(chrom)
        archive_objs.append(obj)
        if len(archive) > MAX_ARCHIVE:
            # архив слишком велик — оставляем случайное подмножество
            idx = rng.sample(range(len(archive)), MAX_ARCHIVE)
            archive[:] = [archive[i] for i in idx]
            archive_objs[:] = [archive_objs[i] for i in idx]
        archive_z = np.array([inst.to_minimization(o) for o in archive_objs])

    history = []
    for i in range(max_fe):
        chrom = [rng.randint(0, 1) for _ in range(inst.n)]
        obj = inst.objectives(chrom)
        add(chrom, obj)
        if i % 100 == 0 or i == max_fe - 1:
            history.append((i + 1, hypervolume([tuple(z) for z in archive_z],
                                               ref)))

    return {
        "archive": list(archive),
        "archive_objs": list(archive_objs),
        "history": history,
        "fe": max_fe,
        "generations": 0,
    }
