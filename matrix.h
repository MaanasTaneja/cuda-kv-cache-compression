#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

typedef struct{
    int rows;
    int columns;
    float* data;
} FP32Matrix;

typedef struct {
    int rows;
    int columns;
    int8_t* data;
} INT8Matrix;

#define IDX(cols, i, j) ((i) * (cols) + (j))

FP32Matrix* createFP32Matrix(int rows, int columns);
INT8Matrix* createINT8Matrix(int rows, int columns);

void freeFP32Matrix(FP32Matrix* matrix);
void freeINT8Matrix(INT8Matrix* matrix);

void random_fill_FP32Matrix(FP32Matrix* matrix, float lower, float upper);
void random_fill_query_vector(float* v, int length, float lower, float upper);

float l2_error(FP32Matrix* A, FP32Matrix* B); //error between orignal and quantised - dequantised matrix
float max_abs_error(FP32Matrix* A, FP32Matrix* B);

float attention_score(const float* query, int query_length, const FP32Matrix* M, int row);

void print_FP32Matrix(FP32Matrix* matrix);
void print_INT8Matrix(INT8Matrix* matrix);

//this function calcuates the error between the attention surrogate between orignal k and recon k
//we need to minimze this bascially.. (ideally should be low)
float attention_dot_product_error(const float* Q, const FP32Matrix* K, const FP32Matrix* K_recon);

//KV matrix -> stores keyts of previous tokens, and their values (what they represent)
//and ur current token must be compared to all previous keys to get attention scores, and then weighted sum of values.
//this biases and moves our current token representation to be more in line with previous tokens (aka so we know what this token means
//in context of previous tokens)

//key is like a hash for the token
//so for: how are you doing?
//and we are at doing, then how and are -> hashed to keys and their values in the kv matrix
//although this isnt fully accurate each key is a learned vecotr.

// K -> The key stores what this token offers as a clue for future tokens.
// Q -> The query stores what the current token is looking for.
// V -> The value stores the actual information of the token.

//and these are genrated by learned weight matrices during training of the model. for each token in the sequence
//if you can learn how to map tokens to key value and query vectors well, the model can learn context and relationships
//and thus generate coherent text.

#ifdef __cplusplus
}
#endif

