/* ─────────────────────────────────────────────────────────────────────────────
 * corpus.h — turns a plain-text file into WORDS the models can learn from.
 *
 * test_011 loaded ready-made binary pixels; here the "pixels" are words, so the
 * loader does the tokenizing itself (no Python step): read the file, split into
 * words, keep the (VOCAB-1) most common ones as the vocabulary, map everything
 * to integer ids, and slice the stream 90/10 into train and test.
 *
 * A "context" example is the previous CTX word-ids; the label is the next word.
 * ───────────────────────────────────────────────────────────────────────────*/
#ifndef CORPUS_H
#define CORPUS_H

#include "config.h"

typedef struct {
    int   *train;       /* train token-id stream                                */
    int    ntrain;      /* number of train tokens                               */
    int   *test;        /* test token-id stream                                 */
    int    ntest;       /* number of test tokens                                */
    char **word;        /* id -> word string (word[0] == "<unk>")               */
    int    vocab;       /* actual vocabulary size (<= VOCAB)                     */
    long   total_tok;   /* total tokens seen in the file                        */
    double unk_frac;    /* fraction of tokens that fell outside the vocabulary  */
} Corpus;

/* Read + tokenize a text file. Exits with a message if the file is missing. */
Corpus corpus_load(const char *path);
void   corpus_free(Corpus *c);

/* number of next-word examples in a stream of `ntok` tokens */
static inline int corpus_examples(int ntok) { return ntok - CTX; }

#endif /* CORPUS_H */
