from scipy.stats import wilcoxon


if __name__ == "__main__":
    pretrained_gin = [84.5, 68.7, 79.9, 72.6, 81.3, 62.7, 78.1, 65.7]
    d_mpnn = [80.9, 71, 77.1, 90.6, 78.6, 57, 75.9, 65.5]
    gem = [85.6, 72.4, 80.6, 90.1, 81.7, 67.2, 78.1, 69.2]
    moltop = [82.9, 68.9, 80.8, 73.6, 66.7, 66, 76.3, 64.4]

    test_results = wilcoxon(moltop, pretrained_gin)
    print(f"MOLTOP vs pretrained GIN, p-value: {test_results.pvalue:.3f}")

    test_results = wilcoxon(moltop, d_mpnn)
    print(f"MOLTOP vs D-MPNN, p-value: {test_results.pvalue:.3f}")

    test_results = wilcoxon(moltop, gem)
    print(f"MOLTOP vs GEM, p-value: {test_results.pvalue:.3f}")
