#include "quant_cpu.h"

//we will utilize openmp to parrallezie compute scales operation
//not quantize since we are surveying serial vs cuda parallelized

//make sure scale array length is at least matrix->columns, else enjoy a seg fault.
void compute_scales(const FP32Matrix* matrix, float* scales){

    omp_set_num_threads(SCALE_COMPUTATION_NUM_THREADS);
    omp_set_dynamic(0); //disable dynamic threading

    //need to fit our float 32 values into int8 range -128 to 127
    #pragma omp parallel for collapse(1) schedule(static)
    for(int cols = 0 ; cols < matrix->columns; cols++){
        float max_abs = 0.0f;
        for(int rows = 0 ; rows < matrix->rows; rows++){
            float value = fabsf(matrix->data[IDX(matrix->columns, rows, cols)]);
            //for this entire column go through all elemnts in this col..
            if(value > max_abs){
                max_abs = value;
            }
        }

        //now calcucate the scale value for this column.
        if (max_abs < 1e-8f) {
            scales[cols] = 1.0f / 127.0f;  //avouid divide by 0
        } else {
            scales[cols] = max_abs / 127.0f;
        }
    }
}


void quantize_matrix(const FP32Matrix* src, INT8Matrix* dest, const float* scales){
    for(int cols = 0; cols < src->columns; cols++){
        float scale = scales[cols];
        //now go throygh column and scale values.
        for(int rows = 0; rows < src->rows; rows++){
            float value = src->data[IDX(src->columns, rows, cols)];
            //quantize
            int8_t q_value = (int8_t) roundf(value / scale);
            //clamp to int8 range
            if(q_value > 127) q_value = 127;
            if(q_value < -128) q_value = -128;
            dest->data[IDX(dest->columns, rows, cols)] = q_value;
        }
    }

}

void dequantize_matrix(const INT8Matrix* src, FP32Matrix* dest, const float* scales){ 
    for(int cols = 0; cols < src->columns; cols++){
        float scale = scales[cols];
        //now go throygh column and scale values.
        for(int rows = 0; rows < src->rows; rows++){
            int8_t q_value = src->data[IDX(src->columns, rows, cols)];
            //dequantize
            float value = ((float) q_value) * scale;
            dest->data[IDX(dest->columns, rows, cols)] = value;
        }
    }
}