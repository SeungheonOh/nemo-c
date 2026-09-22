#include "tokenizer.h"
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

int vocab_from_json(const json_value *arr, vocab_t *v) {
    if (!arr || arr->type != JSON_ARRAY) return -1;
    v->count = (int)arr->count;
    v->pieces = calloc((size_t)v->count, sizeof(char *));
    for (int i = 0; i < v->count; ++i) v->pieces[i] = strdup(json_str(json_at(arr, (size_t)i), ""));
    return 0;
}
void vocab_free(vocab_t *v) {
    for (int i = 0; i < v->count; ++i) free(v->pieces[i]);
    free(v->pieces);
    v->pieces = NULL;
    v->count = 0;
}
int tok_is_lang_tag(const char *p) {
    /* ^<[a-z]{2,3}-[A-Za-z]{2,4}>$ */
    size_t n = strlen(p);
    if (n < 7 || p[0] != '<' || p[n - 1] != '>') return 0;
    const char *dash = strchr(p, '-');
    if (!dash) return 0;
    size_t a = (size_t)(dash - p) - 1, b = n - 2 - a - 1;
    if (a < 2 || a > 3 || b < 2 || b > 4) return 0;
    for (size_t i = 1; i < 1 + a; ++i) if (!islower((unsigned char)p[i])) return 0;
    for (size_t i = (size_t)(dash - p) + 1; i < n - 1; ++i) if (!isalpha((unsigned char)p[i])) return 0;
    return 1;
}
int tok_is_special(const char *p) {
    return !strcmp(p, "<unk>") || !strcmp(p, "<pad>") || !strcmp(p, "<s>") || !strcmp(p, "</s>") || tok_is_lang_tag(p);
}
char *tok_text(const vocab_t *v, int id) {
    if (id < 0 || id >= v->count) return strdup("");
    const char *p = v->pieces[id];
    if (tok_is_special(p)) return strdup("");
    size_t n = strlen(p);
    char *out = malloc(n + 1);
    size_t o = 0;
    for (size_t i = 0; i < n;) {
        /* U+2581 LOWER ONE EIGHTH BLOCK = E2 96 81 */
        if (i + 2 < n && (unsigned char)p[i] == 0xE2 && (unsigned char)p[i + 1] == 0x96 && (unsigned char)p[i + 2] == 0x81) {
            out[o++] = ' ';
            i += 3;
        } else out[o++] = p[i++];
    }
    out[o] = 0;
    return out;
}
