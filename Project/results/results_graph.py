import numpy as np
import matplotlib.pyplot as plt


def main():
    """Plot main results graphs. Uses results from optimal_results.txt"""

    models = ["LR_PCA", "MLP_PCA", "GBN", "GNB_PCA", "Per.", "RF", "SVM"]

    # different results, acc+-2sigma, f1+-2sigma, acc test, f1 test
    LR_PCA_results = [(0.987, 0.022), (0.983, 0.029), 0.980, 0.978]
    MLP_PCA_results = [(0.985, 0.014), (0.980, 0.019), 0.980, 0.978]
    GNB_results = [(0.500, 0.078), (0.422, 0.091), 0.566, 0.533]
    GNB_PCA_results = [(0.958, 0.046), (0.946, 0.058), 0.960, 0.957]
    Per_results = [(0.885, 0.095), (0.851, 0.120), 0.899, 0.887]
    RF_results = [(0.982, 0.032), (0.976, 0.042), 0.995, 0.995]
    SVM_results = [(0.990, 0.014), (0.987, 0.019), 0.985, 0.984]

    results = [
        LR_PCA_results, MLP_PCA_results, GNB_results, GNB_PCA_results,
        Per_results, RF_results, SVM_results
        ]

    v_acc = [result[0][0] for result in results]
    v_acc_std = [result[0][1] for result in results]
    v_f1 = [result[1][0] for result in results]
    v_f1_std = [result[1][1] for result in results]
    t_acc = [result[2] for result in results]
    t_f1 = [result[3] for result in results]

    x_pos = np.arange(len(models))

    plt.errorbar(x=x_pos, y=v_acc, yerr=v_acc_std, fmt="bo", ms=4, label="Validation")
    plt.scatter(x=x_pos, y=t_acc, c="r", marker='o', label="Test")
    plt.xticks(x_pos, models)
    plt.axhline(1, 0, 1, ls="--")
    plt.xlabel("Models")
    plt.ylabel(r"Accuracy$\pm2\sigma$")
    plt.title("Accuracy for optimized models")
    plt.legend()
    plt.savefig("optimized_acc.png")

    plt.clf()

    plt.errorbar(x=x_pos, y=v_f1, yerr=v_f1_std, fmt="bo", ms=4, label="Validation")
    plt.scatter(x=x_pos, y=t_f1, c="r", marker='o', label="Test")
    plt.xticks(x_pos, models)
    plt.axhline(1, 0, 1, ls="--")
    plt.xlabel("Models")
    plt.ylabel(r"F1-score$\pm2\sigma$")
    plt.title("F1-score for optimized models")
    plt.legend()
    plt.savefig("optimized_f1.png")


if __name__ == "__main__":
    main()
