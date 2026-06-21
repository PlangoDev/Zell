/* ─────────────────────────────────────────────────────────────────────────────
 * corpus.cpp — the tokenizer + vocabulary builder.
 *
 * This is the one place we lean on C++ (std::string / unordered_map / sort): it
 * makes counting and ranking words trivial. Everything downstream is plain C
 * arrays of ints and floats, exactly like test_011, so the hot loops vectorize.
 * ───────────────────────────────────────────────────────────────────────────*/
#include "corpus.h"
#include "config.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>

/* Read the whole file into a std::string, or die with a helpful message. */
static std::string slurp(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open %s\n", path);
        fprintf(stderr, "       put a UTF-8 text file there (a corpus.txt is shipped in data/).\n");
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string s;
    s.resize((size_t)n);
    size_t got = fread(&s[0], 1, (size_t)n, f);
    fclose(f);
    s.resize(got);
    return s;
}

/* Split into lowercase words. A "word" is a maximal run of letters; everything
 * else (spaces, punctuation, digits, newlines) is just a separator. Simple and
 * good enough — the point of Test 12 is the learning, not perfect tokenizing. */
static std::vector<std::string> tokenize(const std::string &s) {
    std::vector<std::string> toks;
    toks.reserve(s.size() / 5 + 16);
    std::string cur;
    for (unsigned char ch : s) {
        if (isalpha(ch)) {
            cur.push_back((char)tolower(ch));
        } else if (!cur.empty()) {
            toks.push_back(cur);
            cur.clear();
        }
    }
    if (!cur.empty()) toks.push_back(cur);
    return toks;
}

Corpus corpus_load(const char *path) {
    Corpus c;
    memset(&c, 0, sizeof(c));

    std::string text = slurp(path);
    std::vector<std::string> toks = tokenize(text);
    c.total_tok = (long)toks.size();

    /* count word frequencies */
    std::unordered_map<std::string, long> freq;
    freq.reserve(toks.size() / 2 + 16);
    for (const std::string &w : toks) freq[w]++;

    /* rank by frequency, keep the top (VOCAB-1); id 0 is reserved for <unk> */
    std::vector<std::pair<std::string, long>> ranked(freq.begin(), freq.end());
    std::sort(ranked.begin(), ranked.end(),
              [](const std::pair<std::string, long> &a, const std::pair<std::string, long> &b) {
                  if (a.second != b.second) return a.second > b.second;
                  return a.first < b.first;            /* stable, deterministic */
              });

    int keep = (int)ranked.size();
    if (keep > VOCAB - 1) keep = VOCAB - 1;
    c.vocab = keep + 1;                                /* + the <unk> slot       */

    c.word = (char **)malloc((size_t)c.vocab * sizeof(char *));
    c.word[0] = strdup("<unk>");
    std::unordered_map<std::string, int> id;
    id.reserve((size_t)keep * 2 + 16);
    for (int i = 0; i < keep; i++) {
        id[ranked[i].first] = i + 1;
        c.word[i + 1] = strdup(ranked[i].first.c_str());
    }

    /* map the whole stream to ids, counting how many fall through to <unk> */
    int *ids = (int *)malloc(toks.size() * sizeof(int));
    long unk = 0;
    for (size_t i = 0; i < toks.size(); i++) {
        auto it = id.find(toks[i]);
        if (it == id.end()) { ids[i] = 0; unk++; }
        else                  ids[i] = it->second;
    }
    c.unk_frac = toks.empty() ? 0.0 : (double)unk / (double)toks.size();

    /* 90/10 contiguous split: train on the first 90%, test on the last 10% */
    int n = (int)toks.size();
    c.ntrain = (int)((long)n * 9 / 10);
    c.ntest  = n - c.ntrain;
    c.train = (int *)malloc((size_t)c.ntrain * sizeof(int));
    c.test  = (int *)malloc((size_t)c.ntest  * sizeof(int));
    memcpy(c.train, ids,            (size_t)c.ntrain * sizeof(int));
    memcpy(c.test,  ids + c.ntrain, (size_t)c.ntest  * sizeof(int));
    free(ids);

    return c;
}

void corpus_free(Corpus *c) {
    if (c->word) {
        for (int i = 0; i < c->vocab; i++) free(c->word[i]);
        free(c->word);
    }
    free(c->train);
    free(c->test);
    memset(c, 0, sizeof(*c));
}
