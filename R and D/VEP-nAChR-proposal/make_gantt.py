import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Work packages: (label, start_month, duration_months)
tasks = [
    ("WP1  Literature study & curation protocol", 0.0, 1.2),
    ("WP2  Dataset assembly & cross-species alignment", 0.4, 2.2),
    ("WP3  Subunit-aware & structural extension", 3.4, 1.6),
    ("WP4  SOTA comparison & ablation", 4.6, 1.2),
    ("WP5  Project report", 0.0, 6.0),
]

# Milestones: (label, month, task_row_index)
milestones = [
    ("M1", 1.2, 0),
    ("M2", 2.2, 1),
    ("M3", 3.6, 2),
    ("M4", 5.0, 2),
    ("M5", 5.5, 3),
    ("M6", 6.0, 4),
]

months = ["Oct 2026", "Nov", "Dec", "Jan 2027", "Feb", "Mar 2027"]

fig, ax = plt.subplots(figsize=(10, 4.0), dpi=160)

n = len(tasks)
bar_color = "#2c6fbb"
bar_edge = "#1f4e79"
milestone_color = "#c0392b"

for i, (label, start, dur) in enumerate(tasks):
    y = n - 1 - i  # WP1 at top
    ax.barh(y, dur, left=start, height=0.52, color=bar_color,
            edgecolor=bar_edge, linewidth=0.6, zorder=3)
    ax.text(start + dur + 0.06, y, f"{dur:.1f}", va="center", ha="left",
            fontsize=7.5, color="#333333")

# milestone diamonds
for label, m, row in milestones:
    y = n - 1 - row
    ax.plot(m, y, marker="D", markersize=7, color=milestone_color,
            markeredgecolor="white", markeredgewidth=0.8, zorder=5)

ax.set_yticks([n - 1 - i for i in range(n)])
ax.set_yticklabels([t[0] for t in tasks], fontsize=8.5)
ax.set_xlim(0, 6.5)
ax.set_xticks(range(0, 7))
ax.set_xticklabels(months + [""], fontsize=8.5)
ax.set_xlabel("Project month", fontsize=9)

ax.xaxis.grid(True, which="major", color="#dddddd", linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#999999")

legend_elements = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor=bar_color,
           markersize=10, label="Work package"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor=milestone_color,
           markeredgecolor="white", markersize=8, label="Milestone (M1–M6)"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=8, frameon=False)

plt.tight_layout()
plt.savefig("images/gantt.png", bbox_inches="tight")
print("saved images/gantt.png")
