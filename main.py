"""Клиент командной строки лабораторной работы № 3 (трек A, вариант 9).

Примеры запуска:

  # Полный набор обязательных экспериментов
  python3 experiments.py

  # Один запуск NSGA-II с выводом размера фронта и гиперобъёма
  python3 main.py run --seed 0

  # Запуск с другими параметрами
  python3 main.py run --seed 42 --pop-size 100 --pm 0.1 --crossover two_point

  # Один запуск подхода взвешенной суммы с заданными весами
  python3 main.py weighted-sum --w1 0.5 --w2 0.25 --w3 0.25

  # Случайный поиск с архивом недоминируемых решений
  python3 main.py random-search --max-fe 20000
"""

import argparse
import os

from baseline import run_random_search
from metrics import hypervolume
from nsga2 import front_solutions, run_nsga2
import problem
from weighted_sum import run_weighted_sum_ga

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
INSTANCE_PARAMS = dict(n=30, seed=11, radius=30.0, c_lo=5, c_hi=50)
REF = (1.0, 1.0, 1.0)


def build_parser():
    p = argparse.ArgumentParser(
        description="NSGA-II для размещения датчиков (ЛР № 3, трек A)")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--n", type=int, default=30)
    common.add_argument("--inst-seed", type=int, default=11)
    common.add_argument("--pop-size", type=int, default=100)
    common.add_argument("--max-fe", type=int, default=20000)
    common.add_argument("--pc", type=float, default=0.9)
    common.add_argument("--pm", type=float, default=0.05)
    common.add_argument("--crossover", default="uniform",
                        choices=["one_point", "two_point", "uniform"])
    common.add_argument("--runs", type=int, default=21)

    sub.add_parser("run", parents=[common], help="одиночный запуск NSGA-II")
    sub.add_parser("experiments", parents=[common],
                   help="все обязательные эксперименты")
    p_ws = sub.add_parser("weighted-sum", parents=[common],
                          help="один запуск взвешенной суммы")
    p_ws.add_argument("--w1", type=float, default=0.34,
                      help="вес покрытия")
    p_ws.add_argument("--w2", type=float, default=0.33,
                      help="вес стоимости")
    p_ws.add_argument("--w3", type=float, default=0.33,
                      help="вес устойчивости")
    sub.add_parser("random-search", parents=[common],
                   help="случайный поиск с архивом")
    return p


def get_instance(args):
    params = dict(INSTANCE_PARAMS)
    params["n"] = args.n
    params["seed"] = args.inst_seed
    return problem.generate_instance(**params)


def print_front(inst, front, n_show=None):
    print(f"Размер фронта Парето: {len(front)}")
    z = [inst.to_minimization(o) for o, _ in front]
    print(f"Гиперобъём фронта: {hypervolume(z, REF):.4f}")
    print("Точки фронта (покрытие, стоимость, устойчивость, датчиков):")
    for obj, chrom in front[:n_show or len(front)]:
        print(f"  ({obj[0]:6.0f}, {obj[1]:5.0f}, {obj[2]:6.0f})  n={sum(chrom)}")


def cmd_run(args):
    inst = get_instance(args)
    res = run_nsga2(inst, seed=args.seed, pop_size=args.pop_size,
                    max_fe=args.max_fe, pc=args.pc, pm=args.pm,
                    crossover=args.crossover, ref=REF)
    print_front(inst, front_solutions(res))


def cmd_weighted_sum(args):
    inst = get_instance(args)
    res = run_weighted_sum_ga(inst, (args.w1, args.w2, args.w3),
                              pop_size=args.pop_size, max_fe=args.max_fe,
                              pc=args.pc, pm=args.pm, crossover=args.crossover,
                              seed=args.seed)
    obj = res["best_obj"]
    print(f"Веса: покрытие={args.w1}, стоимость={args.w2}, устойчивость={args.w3}")
    print(f"Лучшее решение: покрытие={obj[0]}, стоимость={obj[1]}, "
          f"устойчивость={obj[2]}, датчиков={sum(res['best_x'])}")


def cmd_random_search(args):
    inst = get_instance(args)
    res = run_random_search(inst, args.max_fe, seed=args.seed, ref=REF)
    print(f"Размер архива недоминируемых решений: {len(res['archive_objs'])}")
    z = [inst.to_minimization(o) for o in res["archive_objs"]]
    print(f"Гиперобъём архива: {hypervolume(z, REF):.4f}")


def cmd_experiments(args):
    import experiments
    experiments.INSTANCE_PARAMS["n"] = args.n
    experiments.INSTANCE_PARAMS["seed"] = args.inst_seed
    experiments.RUNS = args.runs
    experiments.MAX_FE = args.max_fe
    experiments.BASE.update(pop_size=args.pop_size, max_fe=args.max_fe,
                            pc=args.pc, pm=args.pm, crossover=args.crossover)
    experiments.main()


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "run":
        cmd_run(args)
    elif args.command == "experiments":
        cmd_experiments(args)
    elif args.command == "weighted-sum":
        cmd_weighted_sum(args)
    elif args.command == "random-search":
        cmd_random_search(args)


if __name__ == "__main__":
    main()
