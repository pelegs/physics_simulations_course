import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable


def get_distribution(runs, dice, sides):
    vals = np.random.randint(low=1, high=sides + 1, size=(dice, runs))
    sum_vals = np.sum(vals, axis=0)
    bins = np.arange(dice, sides * dice + 2, 1)
    return np.histogram(sum_vals, bins, density=True)


def pretty_print(hist, edges):
    table = PrettyTable()
    table.field_names = ["Sum", "Frequency"]
    table.align = "l"
    for sum, freq in zip(edges, hist):
        table.add_row([f"{sum}", f"{freq:0.3f}"])
    print(table)


def create_figure(hist, edges):
    _, ax = plt.subplots()
    ax.set_title("Multiple dice value distribution", size=30)
    ax.set_xlabel("Sum of sides", size=25)
    ax.set_ylabel("Frequency", size=25)
    ax.tick_params(labelsize=20)
    ax.bar(edges[:-1], hist)
    plt.show()


if __name__ == "__main__":
    from sys import argv

    hist, edges = get_distribution(int(argv[1]), int(argv[2]), int(argv[3]))
    # pretty_print(hist, edges)
    create_figure(hist, edges)
