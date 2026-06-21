/* ─────────────────────────────────────────────────────────────────────────────
 * analytics.c — the diagnostics, implemented. Pure reporting: it reads
 * prediction arrays and prints tables. Nothing here affects training.
 * ───────────────────────────────────────────────────────────────────────────*/
#include "analytics.h"
#include "common.h"

#include <stdio.h>

const char *FASHION_NAMES[10] = {
    "t-shirt", "trouser", "pullover", "dress", "coat",
    "sandal", "shirt", "sneaker", "bag", "boot"
};

void ana_per_class(const char *who, const int *pred, const uint8_t *y, int n) {
    int tp[NCLASS] = {0}, fp[NCLASS] = {0}, support[NCLASS] = {0};
    for (int s = 0; s < n; s++) {
        support[y[s]]++;
        if (pred[s] == y[s]) tp[y[s]]++;
        else                 fp[pred[s]]++;
    }
    printf("  per-class report  [%s]\n", who);
    printf("    %-9s %8s %8s %8s %8s\n", "class", "recall", "precis.", "f1", "support");
    for (int c = 0; c < NCLASS; c++) {
        double rec  = support[c] ? (double)tp[c] / support[c] : 0.0;
        double prec = (tp[c] + fp[c]) ? (double)tp[c] / (tp[c] + fp[c]) : 0.0;
        double f1   = (prec + rec) > 0 ? 2 * prec * rec / (prec + rec) : 0.0;
        printf("    %-9s %7.1f%% %7.1f%% %8.3f %8d\n",
               FASHION_NAMES[c], rec * 100, prec * 100, f1, support[c]);
    }
    printf("\n");
}

void ana_confusion(const char *who, const int *pred, const uint8_t *y, int n) {
    int M[NCLASS][NCLASS] = {{0}};
    for (int s = 0; s < n; s++) M[y[s]][pred[s]]++;
    printf("  confusion matrix  [%s]   (row = truth, col = predicted)\n", who);
    printf("    %-9s", "");
    for (int c = 0; c < NCLASS; c++) printf("%5.4s", FASHION_NAMES[c]);
    printf("\n");
    for (int t = 0; t < NCLASS; t++) {
        printf("    %-9s", FASHION_NAMES[t]);
        for (int c = 0; c < NCLASS; c++) {
            if (t == c) printf("\x1b[1m%5d\x1b[0m", M[t][c]);   /* bold the diagonal */
            else        printf("%5d", M[t][c]);
        }
        printf("\n");
    }
    printf("\n");
}

void ana_agreement(const int *a_pred, const int *b_pred, const uint8_t *y, int n) {
    int agree = 0, both_right = 0, both_wrong = 0, a_only = 0, b_only = 0, disagree = 0;
    for (int s = 0; s < n; s++) {
        int ar = (a_pred[s] == y[s]), br = (b_pred[s] == y[s]);
        if (a_pred[s] == b_pred[s]) {
            agree++;
            if (ar) both_right++; else both_wrong++;
        } else {
            disagree++;
            if (ar && !br) a_only++;
            else if (br && !ar) b_only++;
        }
    }
    printf("  LLM vs Brain agreement\n");
    printf("    agree            %6.2f%%  (both right %.2f%%, both wrong %.2f%%)\n",
           100.0 * agree / n, 100.0 * both_right / n, 100.0 * both_wrong / n);
    printf("    disagree         %6.2f%%  (LLM-only right %.2f%%, Brain-only right %.2f%%)\n",
           100.0 * disagree / n, 100.0 * a_only / n, 100.0 * b_only / n);
    printf("    -> on the %d images they split on, Brain wins %d, LLM wins %d.\n\n",
           disagree, b_only, a_only);
}

void ana_committee(const double *solo_acc, const int *expert_pred, int E,
                   const uint8_t *y, int n) {
    (void)y;                                /* labels not needed for diversity */
    printf("  committee internals\n");
    for (int e = 0; e < E; e++)
        printf("    expert %d solo accuracy: %.2f%%\n", e + 1, solo_acc[e] * 100);

    /* average pairwise agreement — low means the experts are diverse, which is
     * exactly what makes a vote help */
    double tot = 0.0; int pairs = 0;
    for (int a = 0; a < E; a++)
        for (int b = a + 1; b < E; b++) {
            int same = 0;
            for (int s = 0; s < n; s++)
                if (expert_pred[(size_t)a * n + s] == expert_pred[(size_t)b * n + s]) same++;
            tot += (double)same / n; pairs++;
        }
    if (pairs) printf("    avg pairwise agreement: %.2f%%  (lower = more diverse = better vote)\n", 100.0 * tot / pairs);
    printf("\n");
}
