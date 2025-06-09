
        #include <stdio.h>
        #include <stdlib.h>
        
        // x86-specific intrinsics test
        #ifdef __x86_64__
        #include <immintrin.h>
        
        float avx_dot_product(float *a, float *b, int size) {
            float result = 0.0f;
            
            // Check for AVX support
            #ifdef __AVX__
            int i = 0;
            
            // Process 8 elements at a time using AVX
            for (; i + 7 < size; i += 8) {
                __m256 va = _mm256_loadu_ps(&a[i]);
                __m256 vb = _mm256_loadu_ps(&b[i]);
                __m256 mult = _mm256_mul_ps(va, vb);
                
                // Horizontal sum
                __m256 sum = _mm256_hadd_ps(mult, mult);
                sum = _mm256_hadd_ps(sum, sum);
                
                // Extract result
                float temp[8];
                _mm256_storeu_ps(temp, sum);
                result += temp[0] + temp[4];
            }
            
            // Process remaining elements
            for (; i < size; i++) {
                result += a[i] * b[i];
            }
            #else
            // Fallback for non-AVX
            for (int i = 0; i < size; i++) {
                result += a[i] * b[i];
            }
            #endif
            
            return result;
        }
        #else
        // Dummy function for non-x86 platforms
        float avx_dot_product(float *a, float *b, int size) {
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
            
            float result = avx_dot_product(a, b, 1000);
            printf("Result: %f\n", result);
            
            return 0;
        }
        