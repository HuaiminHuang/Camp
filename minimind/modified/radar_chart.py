import matplotlib.pyplot as plt
import numpy as np
import os

class BenchmarkRadarChart:
    def __init__(self, labels, r_lim=None):
        """
        labels: list[str] — 各指标名称（如 ['ceval-valid', 'cmmlu', 'aclue', 'tmmluplus']）
        r_lim: tuple(float, float) — 可选，径向范围 (r_min, r_max)
        """
        self.labels = np.array(labels)
        self.num_vars = len(labels)
        self.angles = np.linspace(0, 2 * np.pi, self.num_vars, endpoint=False).tolist()
        self.angles += self.angles[:1]
        self.r_lim = r_lim
        self.models = {}

    def add_model(self, name, stats, color=None):
        """
        name: 模型名称（如 'Qwen2.5-0.5B'）
        stats: 各指标数值（list 或 np.array）
        color: 可选，线条颜色
        """
        stats = np.array(stats)
        if len(stats) != self.num_vars:
            raise ValueError("指标数量与标签数量不匹配！")
        stats = np.concatenate((stats, [stats[0]]))
        self.models[name] = {
            'stats': stats,
            'color': color
        }

    def plot(self, title='Benchmark Radar Chart', save_path=None):
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.set_thetagrids(np.degrees(self.angles[:-1]), self.labels)
        ax.set_rlabel_position(-22.5)

        if self.r_lim:
            ax.set_rlim(*self.r_lim)

        for name, info in self.models.items():
            color = info['color']
            stats = info['stats']
            ax.plot(self.angles, stats, linewidth=2, linestyle='solid', label=name, color=color)
            ax.fill(self.angles, stats, color=color, alpha=0.1)

        plt.title(title, size=16, color='blue', y=1.08)
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Radar chart saved as {os.getcwd()}\\{save_path}")
        else:
            plt.show()

if __name__ == "__main__": 

    labels = ['ceval-valid', 'cmmlu', 'aclue', 'tmmluplus']

    chart = BenchmarkRadarChart(labels, r_lim=(0.2, 0.55))
    chart.add_model("MiniMind-0.1B", [0.2563, 0.2523, 0.2360, 0.2515], color='green')
    # chart.add_model("Qwen2.5-7B-Instaruct", [0.2950, 0.3100, 0.2800, 0.3000], color='red')
    chart.add_model("Qwen2.5-0.5B-Instaruct", [0.5297, 0.5082, 0.3377, 0.3224], color='blue')

    chart.plot(title="Benchmark Comparison Across Models", save_path="1.png")
