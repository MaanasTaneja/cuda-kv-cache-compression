#include "matrix.h"

FP32Matrix* createFP32Matrix(int rows, int columns) {
    FP32Matrix* matrix = (FP32Matrix*) malloc(sizeof(FP32Matrix));
    matrix->rows = rows;
    matrix->columns = columns;
    matrix->data = (float*) malloc(sizeof(float) * rows * columns);
    return matrix;
}

INT8Matrix* createINT8Matrix(int rows, int columns) {
    INT8Matrix* matrix = (INT8Matrix*) malloc(sizeof(INT8Matrix));
    matrix->rows = rows;
    matrix->columns = columns;
    matrix->data = (int8_t*) malloc(sizeof(int8_t) * rows * columns);
    return matrix;
}

void freeFP32Matrix(FP32Matrix* matrix) {
    free(matrix->data);
    free(matrix);
}

void freeINT8Matrix(INT8Matrix* matrix) {
    free(matrix->data);
    free(matrix);
}

void random_fill_FP32Matrix(FP32Matrix* matrix, float lower, float upper) {
    //no need to do this 2d matrix access style since ill have to do IDX at the end anyway
    int total_elements = matrix->rows * matrix->columns;
    for(int i  = 0; i < total_elements; i++){
        float r = (float) rand() / (float) RAND_MAX;   //between 0 and 1 
        matrix->data[i] = lower + r * (upper - lower); //scale to [lower, upper]
    }
}

void random_fill_query_vector(float* v, int length, float lower, float upper){
    for(int i = 0; i<length; i++){
        float r = (float) rand() / (float) RAND_MAX;   //between 0 and 1
        v[i] = lower + r * (upper - lower); //scale to [lower, upper]
    }
}

float l2_error(FP32Matrix* A, FP32Matrix* B) {
    if(!A || !B){
        fprintf(stderr, "Error: One or both matrices are NULL in l2_error.\n");
        return -1.0f; // Indicate error
    }

    if(A->rows != B->rows || A->columns != B->columns){
        fprintf(stderr, "Error: Matrix dimensions do not match in l2_error.\n");
        return -1.0f; // Indicate error
    }

    float sum_sq_diff = 0.0f;
    //now we need ot use idx
    for(int i = 0; i < A->rows; i++){
        for(int j = 0; j < A->columns; j++){
            float diff = A->data[IDX(A->columns, i , j)] - B->data[IDX(B->columns, i , j)];
            sum_sq_diff += diff*diff;
        }
    }

    return sqrtf(sum_sq_diff);
}

float max_abs_error(FP32Matrix* A, FP32Matrix* B){
    if(!A || !B){
        fprintf(stderr, "Error: One or both matrices are NULL in max_abs_error.\n");
        return -1.0f; // Indicate error
    }

    if(A->rows != B->rows || A->columns != B->columns){
        fprintf(stderr, "Error: Matrix dimensions do not match in max_abs_error.\n");
        return -1.0f; // Indicate error
    }

    float max_error = 0.0f;
    for(int i = 0; i < A->rows; i++){
        for(int j = 0; j < A->columns; j++){
            float abs_diff = fabsf(A->data[IDX(A->columns, i , j)] - B->data[IDX(B->columns, i , j)]);
            if(abs_diff > max_error){
                max_error = abs_diff;
            }
        }
    }

    return max_error;
}

float attention_score(const float* query, int query_length, const FP32Matrix* M, int row){
    if(query_length != M->columns){
        fprintf(stderr, "Error: Query length does not match matrix columns in attention_score.\n");
        return NAN; // Indicate error
    }

    float score = 0.0f;

    for(int i = 0; i < query_length; i++){
        //now must multiply each elemt of query to the corresponding element in the row of M
        score += query[i] * M->data[IDX(M->columns, row, i)];
    }

    return score;
}
/*
and when calcuating the error (or attention score of our query with one token) take a particular row from kv matrix 
and dot prod with query vector,(which represents our one current token) 

for the error we are not cacluaing attention.. we are just seeing if on average all attention socres with this particular 
query token match the orignal and reconstrutrd matrix thats it

this is NOT ATTENTION. since attention is softmax over all scores. we are just calcuating dot products here
*/
float attention_dot_product_error(const float* Q, const FP32Matrix* K, const FP32Matrix* K_recon){
    if(K->rows != K_recon->rows || K->columns != K_recon->columns){
        fprintf(stderr, "Error: Matrix dimensions do not match in attention_dot_product_error.\n");
        return -1.0f; // Indicate error
    }

    //now q vector (even if ur string input was small)
    //and once we vectorize it the q vector is off same length as the entire k matrix columns, and whole space of tokens.

    float total_error = 0.0f;
    //siulating attention is, we must dot product q with each row of k
    //essentially k is storing the keys for all previoud tokens 
    //and q represents the current token, and we must check this query with all keys.

    int rows = K->rows;

    for(int row = 0; row < K->rows; row++){
        float orig_score = attention_score(Q, K->columns, K, row);
        float recon_score = attention_score(Q, K->columns, K_recon, row);
        float diff = fabsf(orig_score - recon_score);
        total_error += diff ; //squared error
    }

    return total_error/rows; //return mean error
    
}

void print_FP32Matrix(FP32Matrix* matrix) {
    if (!matrix || !matrix->data) {
        fprintf(stderr, "Error: NULL matrix in print_FP32Matrix.\n");
        return;
    }

    printf("\nFP32 Matrix (%d x %d):\n", matrix->rows, matrix->columns);
    printf("----------------------------------------\n");

    for (int i = 0; i < matrix->rows; i++) {
        printf("[ ");
        for (int j = 0; j < matrix->columns; j++) {
            printf("%8.4f ", matrix->data[IDX(matrix->columns, i, j)]);
        }
        printf("]\n");
    }

    printf("----------------------------------------\n\n");
}

void print_INT8Matrix(INT8Matrix* matrix) {
    if (!matrix || !matrix->data) {
        fprintf(stderr, "Error: NULL matrix in print_INT8Matrix.\n");
        return;
    }

    printf("\nINT8 Matrix (%d x %d):\n", matrix->rows, matrix->columns);
    printf("----------------------------------------\n");

    for (int i = 0; i < matrix->rows; i++) {
        printf("[ ");
        for (int j = 0; j < matrix->columns; j++) {
            printf("%4d ", matrix->data[IDX(matrix->columns, i, j)]);
        }
        printf("]\n");
    }

    printf("----------------------------------------\n\n");
}
