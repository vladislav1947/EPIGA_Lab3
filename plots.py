"""Построение графиков для ЛР № 3 (требуется matplotlib)."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from runner import curve_stats  # noqa: E402

OBJ_LABELS = ["покрытие (макс.)", "стоимость (мин.)", "устойчивость (макс.)"]


def plot_projections(series_by_name, path,
                     title="Фронт Парето: проекции на плоскости критериев"):
    """Три 2D-проекции фронта Парето (покрытие-стоимость и др.).

    series_by_name: имя -> список решений [(obj, chrom), ...] или
    имя -> список кортежей критериев.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    pairs = [(0, 1), (0, 2), (1, 2)]
    for ax, (i, j) in zip(axes, pairs):
        for name, items in series_by_name.items():
            pts = [it[0] if isinstance(it, tuple) and isinstance(it[0], tuple)
                   else it for it in items]
            xs = [p[i] for p in pts]
            ys = [p[j] for p in pts]
            ax.scatter(xs, ys, s=18, alpha=0.7, label=name)
        ax.set_xlabel(OBJ_LABELS[i])
        ax.set_ylabel(OBJ_LABELS[j])
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_3d(series_by_name, path, title="Фронт Парето в пространстве критериев"):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for name, items in series_by_name.items():
        pts = [it[0] if isinstance(it, tuple) and isinstance(it[0], tuple)
               else it for it in items]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]
        ax.scatter(xs, ys, zs, s=18, alpha=0.7, label=name)
    ax.set_xlabel(OBJ_LABELS[0])
    ax.set_ylabel(OBJ_LABELS[1])
    ax.set_zlabel(OBJ_LABELS[2])
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_hv_convergence(results, path,
                        title="Гиперобъём фронта в процессе работы"):
    grid, mins, means, maxs = curve_stats(results, n_points=120)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid, means, color="#1f77b4", lw=2, label="среднее")
    ax.fill_between(grid, mins, maxs, color="#1f77b4", alpha=0.25,
                    label="разброс min–max")
    ax.set_xlabel("Число вычислений функции")
    ax.set_ylabel("Гиперобъём фронта")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_hv_comparison(stats_by_name, path,
                       title="Сравнение методов по гиперобъёму"):
    names = list(stats_by_name)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([stats_by_name[n]["hvs"] for n in names], tick_labels=names)
    ax.set_ylabel("Гиперобъём фронта Парето")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_flowchart(path):
    """Блок-схема NSGA-II для трёхкритериальной задачи размещения датчиков."""
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(6.5, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 24)
    ax.axis("off")

    def box(y, text, color="#dbe9f6", w=6.4, x=1.8, h=1.1):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.15",
            fc=color, ec="#333333", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=9)

    def arrow(y_from, y_to, label=None, x=5.0):
        ax.annotate("", xy=(x, y_to + 0.05), xytext=(x, y_from - 0.05),
                    arrowprops=dict(arrowstyle="-|>", lw=1.4))
        if label:
            ax.text(x + 0.2, (y_from + y_to) / 2, label, fontsize=8)

    box(22.5, "Начало: экземпляр задачи (покрытие,\nстоимость, устойчивость), seed", "#e8f6e3")
    box(20.9, "Инициализация популяции:\nслучайные бинарные строки")
    box(19.5, "Оценка трёх критериев\nи перевод к минимизации z")
    box(18.1, "Критерий остановки:\nлимит вычислений функции?", "#fff2cc")
    ax.annotate("", xy=(8.6, 18.15), xytext=(8.2, 18.65),
                arrowprops=dict(arrowstyle="-|>", lw=1.4))
    ax.text(8.3, 18.75, "да", fontsize=8)
    box(17.3, "Фронт Парето и\nгиперобъём", "#f6d6d6", w=1.6, x=8.2, h=1.1)
    ax.annotate("", xy=(9.0, 22.5), xytext=(9.0, 18.45),
                arrowprops=dict(arrowstyle="-|>", lw=1.4))
    ax.text(9.2, 21.0, "конец", fontsize=8, rotation=90)
    arrow(18.1, 16.9, "нет")
    box(15.7, "Недоминируемая сортировка\nи расстояние до скопления")
    box(14.1, "Турнирная селекция родителей\n(ранг, расстояние до скопления)")
    box(12.5, "Равномерный кроссинговер (pc=0.9)\nи битовая мутация (pm=0.05)")
    box(10.9, "Оценка потомков\n(критерии и z)")
    box(9.3, "Элитарное замещение:\nобъединение P+Q, отбор по рангу\nи расстоянию до скопления")
    ax.annotate("", xy=(1.8, 18.65), xytext=(1.8, 9.3),
                arrowprops=dict(arrowstyle="-|>", lw=1.4))
    ax.text(0.05, 14.0, "следующее поколение", fontsize=8, rotation=90)

    box(5.9, "Метрика качества: гиперобъём фронта\nотносительно точки (1, 1, 1)\n"
             "в нормированном пространстве минимизации",
        "#fde9d9", w=8.4, x=0.8, h=2.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_coverage_map(inst, solutions_by_name, path,
                      title="Покрытие района датчиками"):
    """Карта покрытия для характерных решений.

    solutions_by_name: имя -> (obj, chrom). Вершины красятся по кратности
    покрытия (0 — непокрыта, 1 — покрыта, 2+ — с запасом), датчики
    обведены.
    """
    names = list(solutions_by_name)
    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 5))
    if len(names) == 1:
        axes = [axes]
    xs = [c[0] for c in inst.coords]
    ys = [c[1] for c in inst.coords]
    for ax, name in zip(axes, names):
        obj, chrom = solutions_by_name[name]
        sensors = {i for i, b in enumerate(chrom) if b}
        counts = [0] * inst.n
        for s in sensors:
            for v in inst.cover[s]:
                counts[v] += 1
        for i, j in inst.edges:
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color="#bbbbbb",
                    lw=0.7, zorder=1)
        colors = ["#d62728" if c == 0 else ("#ff7f0e" if c == 1 else "#2ca02c")
                  for c in counts]
        ax.scatter(xs, ys, c=colors, s=60, zorder=3,
                   edgecolors="black", linewidths=0.7)
        ax.scatter([xs[i] for i in sensors], [ys[i] for i in sensors],
                   s=160, facecolors="none", edgecolors="black",
                   linewidths=1.6, zorder=4)
        ax.set_aspect("equal")
        ax.set_title(f"{name}\nпокрытие {obj[0]}/{inst.n}, "
                     f"датчиков {sum(chrom)}, цена {obj[1]}", fontsize=9)
        ax.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
