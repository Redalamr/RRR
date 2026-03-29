import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "Output")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "Output", "_comparison_plots")


# regex pour parser les checkpoints dans les logs (iteration + regret)
RE_CHECKPOINT = re.compile(
    r"(?:iteration|it[eé]ration en cours\s*:)\s*(\d+)"
    r".*?"
    r"(?:cumulated regrets|regret cumul[eé])\s*:?\s*([\d]+(?:\.[\d]+)?)",
    re.IGNORECASE
)

RE_FINAL = re.compile(
    r"(?:cumulated regrets|regret cumul[eé] final)\s*:?\s*([\d]+(?:\.[\d]+)?)",
    re.IGNORECASE
)
RE_END_EN = re.compile(r"simulation correctly ended", re.IGNORECASE)
RE_END_FR = re.compile(r"simulation termin[eé]e?", re.IGNORECASE)


def parse_log(log_path):

    runs = []
    current_iters = []
    current_regs = []

    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            m = RE_CHECKPOINT.search(line)
            if m:
                current_iters.append(int(m.group(1)))
                current_regs.append(float(m.group(2)))
                continue

            # quand on detecte la fin d un run, on sauvegarde la courbe
            if RE_END_EN.search(line) or RE_END_FR.search(line):

                if current_iters:
                    runs.append({
                        "iterations": list(current_iters),
                        "regrets": list(current_regs)
                    })
                current_iters = []
                current_regs = []

    if current_iters:
        runs.append({
            "iterations": list(current_iters),
            "regrets": list(current_regs)
        })

    return runs


def detect_mode(config_path, folder_name):

    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as fh:
            for line in fh:
                m = re.search(r"Mode:\s*(\w+)", line, re.IGNORECASE)
                if m:
                    return m.group(1).capitalize()

    parts = folder_name.split("_")
    if len(parts) >= 3:
        day = parts[1]
        if day == "28":
            return "Volatile"
        if day == "29":
            return "Baseline"

    return "Unknown"


def detect_algorithm(config_path):
    if not os.path.isfile(config_path):
        return None
    with open(config_path, encoding="utf-8") as fh:
        for line in fh:
            m = re.search(r"algorithm:\s*(\w+)", line, re.IGNORECASE)
            if m:
                return m.group(1)
    return None


def collect_all_results():

    results = {}

    if not os.path.isdir(OUTPUT_DIR):
        print(f"Dossier introuvable : {OUTPUT_DIR}")
        return results

    # on scan tous les sous-dossiers de output pour recuperer les resultats
    for folder in sorted(os.listdir(OUTPUT_DIR)):
        if folder.startswith("_"):
            continue  # skip our own output folder
        folder_path = os.path.join(OUTPUT_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        config_path = os.path.join(folder_path, "config", "config.txt")
        log_path = os.path.join(folder_path, "logs", "logs.txt")

        algo = detect_algorithm(config_path)
        mode = detect_mode(config_path, folder)

        if algo is None or not os.path.isfile(log_path):
            continue

        runs = parse_log(log_path)
        if not runs:
            continue

        if algo not in results:
            results[algo] = {}
        if mode not in results[algo]:
            results[algo][mode] = []
        results[algo][mode].extend(runs)

    return results



def average_runs(run_list):

    all_iters = [tuple(r["iterations"]) for r in run_list]
    from collections import Counter
    common_iters = Counter(all_iters).most_common(1)[0][0]
    common_iters = list(common_iters)

    regret_matrix = []
    for run in run_list:
        iter_to_reg = dict(zip(run["iterations"], run["regrets"]))
        row = [iter_to_reg.get(it, np.nan) for it in common_iters]
        regret_matrix.append(row)

    # on aligne tous les runs sur les memes iterations et on fait moyenne + ecart type
    regret_matrix = np.array(regret_matrix, dtype=float)
    mean = np.nanmean(regret_matrix, axis=0)
    std = np.nanstd(regret_matrix, axis=0)

    return np.array(common_iters), mean, std, len(run_list)


COLORS = {
    "Baseline": "#2196F3",
    "Volatile": "#F44336",
    "Unknown": "#9E9E9E",
    "VolatileUCB": "#4CAF50",
}

ALGO_COLORS = {
    "Random": "#9E9E9E",
    "EGreedy": "#FF9800",
    "UCB1": "#2196F3",
    "TS": "#9C27B0",
    "LinUCB": "#00BCD4",
    "LinTS": "#3F51B5",
    "VolatileUCB": "#4CAF50",
}


# un graphe par algo : on compare baseline vs volatile
def plot_per_algorithm(results):

    os.makedirs(PLOTS_DIR, exist_ok=True)

    for algo, modes in sorted(results.items()):
        fig, ax = plt.subplots(figsize=(12, 6))
        plotted = False

        for mode in ["Baseline", "Volatile", "Unknown"]:
            if mode not in modes:
                continue
            run_list = modes[mode]
            if not run_list:
                continue

            iters, mean, std, n = average_runs(run_list)
            color = COLORS.get(mode, "#333333")
            ax.plot(
                iters, mean, linewidth=2, color=color,
                label=f"{algo} — {mode} (n={n})"
            )
            ax.fill_between(
                iters, mean - std, mean + std,
                alpha=0.20, color=color
            )
            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        ax.set_xlabel("Itérations", fontsize=12)
        ax.set_ylabel("Regret cumulé", fontsize=12)
        ax.set_title(
            f"Regret cumulé — {algo} : Baseline vs Volatile",
            fontsize=14
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out = os.path.join(PLOTS_DIR, f"algo_{algo}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  [OK] {out}")


# comparaison de tous les algos ensemble en mode volatile
def plot_global_volatile(results):

    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7))
    plotted = False

    for algo in ["Random", "EGreedy", "UCB1", "TS", "LinUCB", "LinTS",
                 "VolatileUCB"]:
        if algo not in results:
            continue
        run_list = results[algo].get("Volatile", [])
        if not run_list:
            continue

        iters, mean, std, n = average_runs(run_list)
        color = ALGO_COLORS.get(algo, "#333333")
        linestyle = "--" if algo == "VolatileUCB" else "-"
        linewidth = 2.5 if algo == "VolatileUCB" else 1.8
        ax.plot(
            iters, mean, linewidth=linewidth, linestyle=linestyle,
            color=color, label=f"{algo} (n={n})"
        )
        plotted = True

    if plotted:
        ax.set_xlabel("Itérations", fontsize=13)
        ax.set_ylabel("Regret cumulé moyen", fontsize=13)
        ax.set_title(
            "Comparaison globale — tous algorithmes — setting Volatile",
            fontsize=15
        )
        ax.legend(fontsize=11, loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = os.path.join(PLOTS_DIR, "global_volatile.png")
        fig.savefig(out, dpi=150)
        print(f"  [OK] {out}")
    plt.close(fig)


def plot_global_baseline(results):

    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7))
    plotted = False

    for algo in ["Random", "EGreedy", "UCB1", "TS", "LinUCB", "LinTS"]:
        if algo not in results:
            continue
        run_list = results[algo].get("Baseline", [])
        if not run_list:
            continue

        iters, mean, std, n = average_runs(run_list)
        color = ALGO_COLORS.get(algo, "#333333")
        ax.plot(
            iters, mean, linewidth=1.8,
            color=color, label=f"{algo} (n={n})"
        )
        plotted = True

    if plotted:
        ax.set_xlabel("Itérations", fontsize=13)
        ax.set_ylabel("Regret cumulé moyen", fontsize=13)
        ax.set_title(
            "Comparaison globale — tous algorithmes — setting Baseline",
            fontsize=15
        )
        ax.legend(fontsize=11, loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = os.path.join(PLOTS_DIR, "global_baseline.png")
        fig.savefig(out, dpi=150)
        print(f"  [OK] {out}")
    plt.close(fig)


# graphe dedie ucb1 vs volatileucb pour montrer l apport de notre algo
def plot_ucb1_vs_volatile_ucb(results):

    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    plotted = False

    for algo, color, label in [
        ("UCB1", "#2196F3", "UCB1 classique (Volatile)"),
        ("VolatileUCB", "#4CAF50", "VolatileUCB — adapté (Volatile)"),
    ]:
        if algo not in results:
            continue
        run_list = results[algo].get("Volatile", [])
        if not run_list:
            continue

        iters, mean, std, n = average_runs(run_list)
        linestyle = "--" if algo == "VolatileUCB" else "-"
        ax.plot(
            iters, mean, linewidth=2.2, linestyle=linestyle,
            color=color, label=f"{label} (n={n})"
        )
        ax.fill_between(iters, mean - std, mean + std, alpha=0.15, color=color)
        plotted = True

    if plotted:
        ax.set_xlabel("Itérations", fontsize=12)
        ax.set_ylabel("Regret cumulé moyen", fontsize=12)
        ax.set_title(
            "UCB1 vs VolatileUCB dans le setting Volatile Arms",
            fontsize=14
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = os.path.join(PLOTS_DIR, "ucb1_vs_volatileucb.png")
        fig.savefig(out, dpi=150)
        print(f"  [OK] {out}")
    plt.close(fig)


def print_summary_table(results):

    print("\n" + "=" * 65)
    print(f"  {'Algorithme':<15} {'Mode':<12} {'Regret final moyen':>20}  {'n runs':>7}")
    print("=" * 65)

    for algo in sorted(results.keys()):
        for mode in ["Baseline", "Volatile", "Unknown"]:
            if mode not in results[algo]:
                continue
            run_list = results[algo][mode]
            _, mean, _, n = average_runs(run_list)
            final = mean[-1]
            print(f"  {algo:<15} {mode:<12} {final:>20.1f}  {n:>7}")

    print("=" * 65 + "\n")



if __name__ == "__main__":
    print(f"\nScan du dossier : {OUTPUT_DIR}\n")
    results = collect_all_results()

    if not results:
        print("aucun résultat trouvé")
    else:
        print(f"{len(results)} algorithme(s) détecté(s) : "
              f"{', '.join(sorted(results.keys()))}\n")

        print_summary_table(results)

        # generation de tous les plots de comparaison
        print(f"Génération des graphiques dans : {PLOTS_DIR}\n")
        plot_per_algorithm(results)
        plot_global_volatile(results)
        plot_global_baseline(results)
        plot_ucb1_vs_volatile_ucb(results)

        print("\nTerminé")