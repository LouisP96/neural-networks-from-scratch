import matplotlib.pyplot as plt
import numpy as np


def display_examples(examples, xs, ys):
    for example in examples:
        print(f"Y Label: {ys[example]}")

        plt.imshow(xs[:, :, example].T, cmap="gray")
        plt.axis("off")
        plt.show()


def flatten_2d_images(nparray):
    return np.reshape(nparray, (nparray.shape[0] * nparray.shape[1], nparray.shape[2]))


def plot_metrics(
    train_metric,
    dev_metric,
    title="Model Performance",
    ylabel="Metric",
    xlabel="Epoch",
    legend_labels=["Training", "Validation"],
    save_fig=False,
    filename="plot.png",
    line_colors=("k", "r"),
    line_style="-",
):
    plt.figure()
    plt.plot(
        train_metric, color=line_colors[0], linestyle=line_style, label=legend_labels[0]
    )
    plt.plot(
        dev_metric, color=line_colors[1], linestyle=line_style, label=legend_labels[1]
    )
    plt.xlabel(xlabel, color="k")
    plt.ylabel(ylabel, color="k")
    plt.title(title, color="k")
    plt.legend(loc="best")
    plt.tick_params(colors="k")
    if save_fig:
        plt.savefig(filename, dpi=300)
    plt.show()
