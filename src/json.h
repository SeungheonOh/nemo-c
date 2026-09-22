/* Minimal JSON DOM parser: enough for config.json and safetensors headers. */
#ifndef NEMO_JSON_H
#define NEMO_JSON_H
#include <stddef.h>

typedef enum { JSON_NULL, JSON_BOOL, JSON_NUMBER, JSON_STRING, JSON_ARRAY, JSON_OBJECT } json_type;

typedef struct json_value {
    json_type type;
    int boolean;
    double number;
    char *string;
    struct json_value **items; /* array items or object values */
    char **keys;               /* object keys */
    size_t count;
} json_value;

json_value *json_parse(const char *text, size_t len, char *err, size_t errlen);
void json_free(json_value *v);

const json_value *json_get(const json_value *obj, const char *key);   /* NULL if missing */
double json_num(const json_value *v, double dflt);
const char *json_str(const json_value *v, const char *dflt);
size_t json_len(const json_value *v);                                  /* array/object length */
const json_value *json_at(const json_value *arr, size_t i);

#endif
