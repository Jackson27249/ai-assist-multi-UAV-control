from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def plot_episode(csv_path: Path, output: Path, safe_distance: float = 0.55) -> dict:
    frame = pd.read_csv(csv_path)
    output.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem
    fig = plt.figure(figsize=(12.0, 8.0))
    ax3d = fig.add_subplot(221, projection="3d")
    axxy = fig.add_subplot(222)
    axtrack = fig.add_subplot(223)
    axsafe = fig.add_subplot(224)
    for agent, group in frame.groupby("agent"):
        ax3d.plot(group.x, group.y, group.z, label=f"UAV {agent}")
        axxy.plot(group.x, group.y, label=f"UAV {agent}")
        speed = np.sqrt(group.vx**2 + group.vy**2 + group.vz**2)
        command = np.sqrt(group.cmd_vx**2 + group.cmd_vy**2 + group.cmd_vz**2)
        axtrack.plot(group.time, speed, label=f"actual {agent}")
        axtrack.plot(group.time, command, linestyle="--", alpha=0.7)
    per_time = frame.groupby("time", as_index=False).agg(
        min_pair_distance=("min_pair_distance", "min"),
        send_latency_ms=("send_latency_ms", "max"),
        telemetry_age_ms=("telemetry_age_ms", "max"),
    )
    axsafe.plot(per_time.time, per_time.min_pair_distance, label="minimum separation")
    axsafe.axhline(safe_distance, color="#c62828", linestyle="--", label="collision threshold")
    latency_axis = axsafe.twinx()
    latency_axis.plot(per_time.time, per_time.send_latency_ms, color="#6a1b9a", alpha=0.55, label="command latency")
    ax3d.set(xlabel="East (m)", ylabel="North (m)", zlabel="Up (m)", title="PX4 SIH trajectories")
    axxy.set(xlabel="East (m)", ylabel="North (m)", title="XY projection")
    axtrack.set(xlabel="Time (s)", ylabel="Speed (m/s)", title="Command tracking")
    axsafe.set(xlabel="Time (s)", ylabel="Separation (m)", title="Safety and latency")
    latency_axis.set_ylabel("Latency (ms)")
    ax3d.legend(fontsize=7)
    axxy.legend(fontsize=7)
    axtrack.legend(fontsize=6, ncol=2)
    axsafe.legend(fontsize=7, loc="upper left")
    fig.suptitle(stem)
    fig.tight_layout()
    png = output / f"{stem}.png"
    pdf = output / f"{stem}.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return {
        "telemetry_file": csv_path.name,
        "png": png.name,
        "pdf": pdf.name,
        "width": Image.open(png).width,
        "height": Image.open(png).height,
        "rows": len(frame),
        "curves": int(frame.agent.nunique()),
        "min_pair_distance": float(frame.min_pair_distance.min()),
        "max_send_latency_ms": float(frame.send_latency_ms.max()),
    }


def contact_sheet(images: list[Path], output: Path) -> None:
    if not images:
        canvas = Image.new("RGB", (1000, 220), "white")
        ImageDraw.Draw(canvas).text((30, 90), "No successful PX4 telemetry image was produced; see episode_summary.jsonl.", fill="black")
        canvas.save(output)
        return
    thumbs = []
    for path in images[:40]:
        image = Image.open(path).convert("RGB")
        image.thumbnail((360, 240))
        thumbs.append((path.name, image.copy()))
    columns = 4
    rows = int(np.ceil(len(thumbs) / columns))
    canvas = Image.new("RGB", (columns * 380, rows * 275), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (name, image) in enumerate(thumbs):
        x = (index % columns) * 380 + 10
        y = (index // columns) * 275 + 10
        canvas.paste(image, (x, y))
        draw.text((x, y + 242), name[:54], fill="black")
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--px4-root", type=Path, required=True)
    args = parser.parse_args()
    telemetry = args.px4_root / "telemetry"
    per_episode = args.px4_root / "per_episode_figures"
    summary_figures = args.px4_root / "summary_figures"
    summary_figures.mkdir(parents=True, exist_ok=True)
    qa = [plot_episode(path, per_episode) for path in sorted(telemetry.glob("*.csv"))]
    contact_sheet([per_episode / item["png"] for item in qa], summary_figures / "px4_contact_sheet.png")

    summary_path = args.px4_root / "episode_summary.jsonl"
    episodes = [json.loads(line) for line in summary_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    frame = pd.DataFrame(episodes)
    if len(frame):
        group = frame.groupby(["policy", "num_agents"], as_index=False).agg(
            success_rate=("success", "mean"),
            collision_rate=("collision", "mean"),
            episodes=("success", "size"),
        )
        group.to_csv(args.px4_root / "summary.csv", index=False)
        plot = group.melt(id_vars=["policy", "num_agents", "episodes"], value_vars=["success_rate", "collision_rate"], var_name="metric", value_name="rate")
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        labels = [f"{row.policy} N={row.num_agents}\n{row.metric}" for row in plot.itertuples()]
        colors = ["#2e7d32" if metric == "success_rate" else "#c62828" for metric in plot.metric]
        ax.bar(np.arange(len(plot)), plot.rate, color=colors)
        ax.set_xticks(np.arange(len(plot)), labels, rotation=30, ha="right")
        ax.set(ylabel="Rate", ylim=(0, 1.05), title="PX4 SIH outcome summary")
        fig.tight_layout()
        fig.savefig(summary_figures / "px4_outcomes.png", dpi=220)
        fig.savefig(summary_figures / "px4_outcomes.pdf")
        plt.close(fig)
    report = {
        "episodes": len(episodes),
        "successful_telemetry_files": len(qa),
        "failed_episodes": sum(not item.get("success", False) for item in episodes),
        "image_qa": qa,
        "claim_boundary": "PX4 SIH validates flight-stack command tracking and multi-instance timing; obstacle-safety evidence remains simulator-based.",
    }
    (args.px4_root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PX4_PLOT_RESULT=PASS images={len(qa)} output={args.px4_root}")


if __name__ == "__main__":
    main()

