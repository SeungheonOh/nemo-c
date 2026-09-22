#include "json.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { const char *s, *start, *end; char *err; size_t errlen; } parser;

static json_value *parse_value(parser *p);

static void fail(parser *p, const char *msg) {
    if (p->err && p->errlen) snprintf(p->err, p->errlen, "json: %s at offset %ld", msg, (long)(p->s - p->start));
}
static void skipws(parser *p) {
    while (p->s < p->end && (*p->s == ' ' || *p->s == '\n' || *p->s == '\r' || *p->s == '\t')) p->s++;
}
static json_value *newv(json_type t) {
    json_value *v = calloc(1, sizeof *v);
    v->type = t;
    return v;
}
static int utf8_put(char *out, unsigned cp) {
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    if (cp < 0x800) { out[0] = (char)(0xC0 | (cp >> 6)); out[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) { out[0] = (char)(0xE0 | (cp >> 12)); out[1] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[2] = (char)(0x80 | (cp & 0x3F)); return 3; }
    out[0] = (char)(0xF0 | (cp >> 18)); out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}
static int hex4(const char *s, unsigned *out) {
    unsigned v = 0;
    for (int i = 0; i < 4; ++i) {
        char c = s[i]; v <<= 4;
        if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
        else return 0;
    }
    *out = v;
    return 1;
}
static char *parse_string(parser *p) {
    if (p->s >= p->end || *p->s != '"') { fail(p, "expected string"); return NULL; }
    p->s++;
    size_t cap = 32, len = 0;
    char *buf = malloc(cap);
    while (p->s < p->end) {
        if (len + 8 > cap) { cap *= 2; buf = realloc(buf, cap); }
        char c = *p->s++;
        if (c == '"') { buf[len] = 0; return buf; }
        if (c != '\\') { buf[len++] = c; continue; }
        if (p->s >= p->end) break;
        char e = *p->s++;
        switch (e) {
            case '"': buf[len++] = '"'; break;
            case '\\': buf[len++] = '\\'; break;
            case '/': buf[len++] = '/'; break;
            case 'b': buf[len++] = '\b'; break;
            case 'f': buf[len++] = '\f'; break;
            case 'n': buf[len++] = '\n'; break;
            case 'r': buf[len++] = '\r'; break;
            case 't': buf[len++] = '\t'; break;
            case 'u': {
                unsigned cp;
                if (p->end - p->s < 4 || !hex4(p->s, &cp)) { fail(p, "bad \\u escape"); free(buf); return NULL; }
                p->s += 4;
                if (cp >= 0xD800 && cp <= 0xDBFF && p->end - p->s >= 6 && p->s[0] == '\\' && p->s[1] == 'u') {
                    unsigned lo;
                    if (hex4(p->s + 2, &lo) && lo >= 0xDC00 && lo <= 0xDFFF) {
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        p->s += 6;
                    }
                }
                len += (size_t)utf8_put(buf + len, cp);
                break;
            }
            default: fail(p, "bad escape"); free(buf); return NULL;
        }
    }
    fail(p, "unterminated string");
    free(buf);
    return NULL;
}
static json_value *parse_array(parser *p) {
    json_value *v = newv(JSON_ARRAY);
    p->s++; /* [ */
    skipws(p);
    if (p->s < p->end && *p->s == ']') { p->s++; return v; }
    size_t cap = 8;
    v->items = malloc(cap * sizeof *v->items);
    for (;;) {
        skipws(p);
        json_value *item = parse_value(p);
        if (!item) { json_free(v); return NULL; }
        if (v->count == cap) { cap *= 2; v->items = realloc(v->items, cap * sizeof *v->items); }
        v->items[v->count++] = item;
        skipws(p);
        if (p->s >= p->end) { fail(p, "unterminated array"); json_free(v); return NULL; }
        if (*p->s == ',') { p->s++; continue; }
        if (*p->s == ']') { p->s++; return v; }
        fail(p, "expected , or ]"); json_free(v); return NULL;
    }
}
static json_value *parse_object(parser *p) {
    json_value *v = newv(JSON_OBJECT);
    p->s++; /* { */
    skipws(p);
    if (p->s < p->end && *p->s == '}') { p->s++; return v; }
    size_t cap = 8;
    v->items = malloc(cap * sizeof *v->items);
    v->keys = malloc(cap * sizeof *v->keys);
    for (;;) {
        skipws(p);
        char *key = parse_string(p);
        if (!key) { json_free(v); return NULL; }
        skipws(p);
        if (p->s >= p->end || *p->s != ':') { fail(p, "expected :"); free(key); json_free(v); return NULL; }
        p->s++;
        skipws(p);
        json_value *val = parse_value(p);
        if (!val) { free(key); json_free(v); return NULL; }
        if (v->count == cap) {
            cap *= 2;
            v->items = realloc(v->items, cap * sizeof *v->items);
            v->keys = realloc(v->keys, cap * sizeof *v->keys);
        }
        v->keys[v->count] = key;
        v->items[v->count++] = val;
        skipws(p);
        if (p->s >= p->end) { fail(p, "unterminated object"); json_free(v); return NULL; }
        if (*p->s == ',') { p->s++; continue; }
        if (*p->s == '}') { p->s++; return v; }
        fail(p, "expected , or }"); json_free(v); return NULL;
    }
}
static json_value *parse_value(parser *p) {
    skipws(p);
    if (p->s >= p->end) { fail(p, "unexpected end"); return NULL; }
    char c = *p->s;
    if (c == '{') return parse_object(p);
    if (c == '[') return parse_array(p);
    if (c == '"') {
        char *s = parse_string(p);
        if (!s) return NULL;
        json_value *v = newv(JSON_STRING);
        v->string = s;
        return v;
    }
    if (!strncmp(p->s, "true", 4) && p->end - p->s >= 4) { p->s += 4; json_value *v = newv(JSON_BOOL); v->boolean = 1; return v; }
    if (!strncmp(p->s, "false", 5) && p->end - p->s >= 5) { p->s += 5; return newv(JSON_BOOL); }
    if (!strncmp(p->s, "null", 4) && p->end - p->s >= 4) { p->s += 4; return newv(JSON_NULL); }
    if (c == '-' || (c >= '0' && c <= '9')) {
        char *endp;
        double d = strtod(p->s, &endp);
        if (endp == p->s) { fail(p, "bad number"); return NULL; }
        p->s = endp;
        json_value *v = newv(JSON_NUMBER);
        v->number = d;
        return v;
    }
    fail(p, "unexpected character");
    return NULL;
}

json_value *json_parse(const char *text, size_t len, char *err, size_t errlen) {
    parser p = { text, text, text + len, err, errlen };
    if (err && errlen) err[0] = 0;
    json_value *v = parse_value(&p);
    return v;
}
void json_free(json_value *v) {
    if (!v) return;
    for (size_t i = 0; i < v->count; ++i) {
        json_free(v->items[i]);
        if (v->keys) free(v->keys[i]);
    }
    free(v->items);
    free(v->keys);
    free(v->string);
    free(v);
}
const json_value *json_get(const json_value *obj, const char *key) {
    if (!obj || obj->type != JSON_OBJECT) return NULL;
    for (size_t i = 0; i < obj->count; ++i)
        if (!strcmp(obj->keys[i], key)) return obj->items[i];
    return NULL;
}
double json_num(const json_value *v, double dflt) { return (v && v->type == JSON_NUMBER) ? v->number : dflt; }
const char *json_str(const json_value *v, const char *dflt) { return (v && v->type == JSON_STRING) ? v->string : dflt; }
size_t json_len(const json_value *v) { return v ? v->count : 0; }
const json_value *json_at(const json_value *arr, size_t i) { return (arr && i < arr->count) ? arr->items[i] : NULL; }
