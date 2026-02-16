"""Generate training curve plots from logged epoch data."""

from pathlib import Path

import matplotlib.pyplot as plt

ASSETS_DIR = Path(__file__).resolve().parent.parent / 'assets'

# Epoch data from training logs (epochs we have data for)
RUNS = {
    'Run 2 (7.4M, bs=512)': {
        'epochs': [0, 1, 2, 5, 10, 15, 19],
        'cell_acc': [48.2, 84.9, 90.3, 94.2, 96.2, 97.0, 97.2],
        'puzzle_acc': [0.0, 14.8, 34.4, 54.8, 67.3, 73.9, 74.7],
        'color': '#2196F3',
        'marker': 'o',
    },
    'Run 3 (7.4M, bs=2048)': {
        'epochs': [0, 1, 2, 5, 10, 15, 19],
        'cell_acc': [43.8, 73.4, 87.6, 95.3, 97.5, 98.2, 98.3],
        'puzzle_acc': [0.0, 0.7, 19.1, 61.1, 76.6, 82.8, 83.8],
        'color': '#FF9800',
        'marker': 's',
    },
    'Run 4 (7.4M, bs=2048, fixes)': {
        'epochs': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        'cell_acc': [
            64.8,
            80.8,
            88.2,
            92.0,
            94.2,
            95.0,
            95.7,
            96.1,
            96.4,
            96.6,
            96.7,
            97.0,
            97.2,
            97.3,
            97.4,
            97.5,
            97.6,
            97.6,
            97.6,
            97.6,
        ],
        'puzzle_acc': [
            0.2,
            7.7,
            26.4,
            45.1,
            58.5,
            63.1,
            68.6,
            71.4,
            73.2,
            74.7,
            75.9,
            77.6,
            78.9,
            80.0,
            80.8,
            81.5,
            82.0,
            82.3,
            82.5,
            82.5,
        ],
        'color': '#4CAF50',
        'marker': 'D',
    },
    'Run 5 (36.5M, bs=2048)': {
        'epochs': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        'cell_acc': [
            76.3,
            94.7,
            96.9,
            97.3,
            97.6,
            98.0,
            98.3,
            98.5,
            98.7,
            98.8,
            98.9,
            99.0,
            99.1,
            99.1,
            99.2,
            99.2,
            99.2,
            99.3,
        ],
        'puzzle_acc': [
            2.8,
            65.5,
            79.6,
            82.7,
            85.3,
            87.8,
            90.0,
            91.3,
            92.1,
            92.9,
            93.6,
            94.0,
            94.4,
            94.8,
            95.1,
            95.3,
            95.5,
            95.6,
        ],
        'color': '#9C27B0',
        'marker': '^',
    },
}


def plot_comparison() -> None:
    """Generate a 2-panel comparison plot of all training runs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, data in RUNS.items():
        ax1.plot(
            data['epochs'],
            data['cell_acc'],
            color=data['color'],
            marker=data['marker'],
            markersize=5,
            linewidth=2,
            label=name,
        )
        ax2.plot(
            data['epochs'],
            data['puzzle_acc'],
            color=data['color'],
            marker=data['marker'],
            markersize=5,
            linewidth=2,
            label=name,
        )

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cell Accuracy (%)')
    ax1.set_title('Cell Accuracy')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(40, 100)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Puzzle Accuracy (%)')
    ax2.set_title('Puzzle Accuracy')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    ax2.axhline(y=96.2, color='red', linestyle='--', alpha=0.5, label='Kona 1.0 (96.2%)')
    ax2.legend(fontsize=9)

    fig.suptitle('Enso Training Runs — Forward Pass Accuracy', fontsize=14, fontweight='bold')
    fig.tight_layout()

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    out = ASSETS_DIR / 'training_runs.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


if __name__ == '__main__':
    plot_comparison()
