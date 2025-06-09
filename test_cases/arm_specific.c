
        #include <stdio.h>
        #include <stdlib.h>
        
        // ARM-specific intrinsics test
        #ifdef __aarch64__
        #include <arm_neon.h>
        
        float neon_dot_product(float *a, float *b, int size) {
            float result = 0.0f;
            
            int i = 0;
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            
            // Process 4 elements at a time using NEON
            for (; i + 3 < size; i += 4) {
                float32x4_t va = vld1q_f32(&a[i]);
                float32x4_t vb = vld1q_f32(&b[i]);
                sum_vec = vmlaq_f32(sum_vec, va, vb);
            }
            
            // Horizontal sum
            float sum[4];
            vst1q_f32(sum, sum_vec);
            result = sum[0] + sum[1] + sum[2] + sum[3];
            
            // Process remaining elements
            for (; i < size; i++) {
                result += a[i] * b[i];
            }
            
            return result;
        }
        #else
        // Dummy function for non-ARM platforms
        float neon_dot_product(float *a, float *b, int size) {
            float result = 0.0f;
            for (int i = 0; i < size; i++) {
                result += a[i] * b[i];
            }
            return result;
        }
        #endif
        
        int main() {
            float a[1000], b[1000];
            for (int i = 0; i < 1000; i++) {
                a[i] = (float)i / 10.0f;
                b[i] = (float)(1000-i) / 10.0f;
            }
            
            float result = neon_dot_product(a, b, 1000);
            printf("Result: %f\n", result);
            
            return 0;
        }
        