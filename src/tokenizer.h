#ifndef NEMO_TOKENIZER_H
#define NEMO_TOKENIZER_H
#include "json.h"
typedef struct {
    char **pieces;
    int count;
} vocab_t;
int vocab_from_json(const json_value *arr, vocab_t *v);
void vocab_free(vocab_t *v);
int tok_is_lang_tag(const char *piece);   /* <xx-YY> */
int tok_is_special(const char *piece);    /* <unk> <pad> <s> </s> or lang tag */
/* Text for one token, with the SentencePiece word marker turned into a space.
   Returns malloc'd string ("" for specials / lang tags). */
char *tok_text(const vocab_t *v, int id);
#endif
