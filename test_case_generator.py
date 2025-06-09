#!/usr/bin/env python3
"""
Test Case Generator for Compiler-Specific Features

This script generates more specialized test cases that target compiler-specific
optimizations and features for ARM vs x86 architectures.
"""

import os
import argparse
from pathlib import Path

# Advanced test cases focusing on architectural differences
ADVANCED_TEST_CASES = {
    'register_pressure': '''
        // Test register allocation under pressure
        int register_pressure_test(int *data, int size) {
            int a = 0, b = 1, c = 2, d = 3, e = 4, f = 5, g = 6, h = 7;
            int i = 8, j = 9, k = 10, l = 11, m = 12, n = 13, o = 14, p = 15;
            
            for (int idx = 0; idx < size; idx++) {
                a += data[idx] * b;
                c += data[idx] * d;
                e += data[idx] * f;
                g += data[idx] * h;
                i += data[idx] * j;
                k += data[idx] * l;
                m += data[idx] * n;
                o += data[idx] * p;
            }
            
            return a + c + e + g + i + k + m + o;
        }
    ''',
    
    'vector_alignment': '''
        // Test alignment requirements for vectorization
        void vector_alignment_test(float *src, float *dst, int size) {
            // Force misalignment by using offset
            for (int i = 0; i < size; i++) {
                dst[i] = src[i] * 2.0f;
            }
        }
    ''',
    
    'branch_prediction_pattern': '''
        // Test branch prediction patterns
        int branch_pattern_test(int *data, int size, int threshold) {
            int count = 0;
            
            // Pattern 1: Likely to be taken (assuming most values > 0)
            for (int i = 0; i < size; i++) {
                if (data[i] > 0) {
                    count++;
                }
            }
            
            // Pattern 2: Unlikely to be taken
            for (int i = 0; i < size; i++) {
                if (data[i] > threshold && threshold > 1000) {
                    count++;
                }
            }
            
            // Pattern 3: Unpredictable branches
            for (int i = 0; i < size; i++) {
                if ((data[i] % 17) == 0) {
                    count++;
                }
            }
            
            return count;
        }
    ''',
    
    'memory_access_patterns': '''
        // Test different memory access patterns
        void memory_pattern_test(int *data, int size, int *result) {
            // Sequential access
            for (int i = 0; i < size; i++) {
                result[0] += data[i];
            }
            
            // Strided access
            for (int i = 0; i < size; i += 4) {
                result[1] += data[i];
            }
            
            // Random-like access using a simple hash
            for (int i = 0; i < size; i++) {
                int idx = (i * 7919) % size; // Prime number hash
                result[2] += data[idx];
            }
            
            // Indirect access
            int indices[1000];
            for (int i = 0; i < 1000 && i < size; i++) {
                indices[i] = (i * 17) % size;
            }
            
            for (int i = 0; i < 1000 && i < size; i++) {
                result[3] += data[indices[i]];
            }
        }
    ''',
    
    'function_inlining': '''
        // Test compiler's function inlining decisions
        __attribute__((always_inline)) inline int simple_inline(int a, int b) {
            return a + b;
        }
        
        int __attribute__((noinline)) never_inline(int a, int b) {
            return a * b;
        }
        
        // This one lets the compiler decide
        int maybe_inline(int a, int b) {
            return a - b;
        }
        
        int inlining_test(int *data, int size) {
            int sum = 0;
            
            for (int i = 0; i < size; i++) {
                sum += simple_inline(data[i], i);
                sum += never_inline(data[i], i);
                sum += maybe_inline(data[i], i);
            }
            
            return sum;
        }
    ''',
    
    'loop_transformations': '''
        // Test loop transformations (unrolling, fusion, etc.)
        void loop_transform_test(int *data, int size, int *result) {
            // Loop likely to be unrolled
            for (int i = 0; i < 8; i++) {
                result[0] += data[i];
            }
            
            // Loop unlikely to be unrolled due to size
            for (int i = 0; i < size; i++) {
                result[1] += data[i];
            }
            
            // Candidate for loop fusion
            for (int i = 0; i < size; i++) {
                result[2] += data[i];
            }
            for (int i = 0; i < size; i++) {
                result[3] += data[i] * 2;
            }
            
            // Nested loops (candidates for different optimizations)
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < 4; j++) {
                    result[4] += data[i] * j;
                }
            }
        }
    ''',
    
    'conditionals_to_predication': '''
        // Test conversion of conditionals to predicated instructions
        int predication_candidate(int *data, int size) {
            int result = 0;
            
            for (int i = 0; i < size; i++) {
                // Simple condition that might be converted to predicated instructions,
                // especially on ARM (e.g., using CMOV on x86 or predicated adds on ARM)
                if (data[i] > 0) {
                    result += data[i];
                } else {
                    result -= data[i];
                }
            }
            
            return result;
        }
    ''',
    
    'vectorization_opportunities': '''
        // Test different vectorization patterns
        void vectorization_test(float *a, float *b, float *c, int size) {
            // Simple vectorization
            for (int i = 0; i < size; i++) {
                c[i] = a[i] + b[i];
            }
            
            // Harder to vectorize due to data dependencies
            for (int i = 1; i < size; i++) {
                c[i] = a[i] + c[i-1];
            }
            
            // Reduction - may use different strategies
            float sum = 0.0f;
            for (int i = 0; i < size; i++) {
                sum += a[i];
            }
            c[0] = sum;
            
            // Mixed types - may require type conversion in vectors
            int *a_int = (int*)a;
            for (int i = 0; i < size; i++) {
                c[i] = (float)a_int[i] + b[i];
            }
        }
    ''',
    
    'tail_call_optimization': '''
        // Test tail call optimization
        int factorial_tail(int n, int acc) {
            if (n <= 1) return acc;
            return factorial_tail(n - 1, n * acc);  // Tail call
        }
        
        int factorial(int n) {
            return factorial_tail(n, 1);
        }
        
        // Non-tail recursion for comparison
        int fibonacci(int n) {
            if (n <= 1) return n;
            return fibonacci(n-1) + fibonacci(n-2);  // Not a tail call
        }
    ''',
    
    'memory_alignment': '''
        // Test memory alignment handling
        void alignment_test(void *ptr, int size) {
            // Force different alignments
            char *p1 = (char*)ptr;
            char *p2 = p1 + 1;  // Intentionally misaligned
            char *p4 = p1 + 4;  // Might be aligned for int on some architectures
            char *p8 = p1 + 8;  // Might be aligned for double
            
            int *i1 = (int*)p1;
            int *i2 = (int*)p2;
            int *i4 = (int*)p4;
            int *i8 = (int*)p8;
            
            // Access with different alignments
            volatile int sum = 0;
            for (int i = 0; i < size/4; i++) {
                sum += i1[i];
                sum += i2[i];
                sum += i4[i];
                sum += i8[i];
            }
        }
    '''
}

def generate_test_files(output_dir):
    """Generate C files for advanced test cases."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for test_name, code in ADVANCED_TEST_CASES.items():
        with open(output_dir / f"{test_name}.c", 'w') as f:
            f.write(f"""
            #include <stdio.h>
            #include <stdlib.h>
            
            {code}
            
            int main() {{
                // Simple driver for testing
                int data[1000];
                float fdata[1000];
                int result[10] = {{0}};
                
                // Initialize data
                for (int i = 0; i < 1000; i++) {{
                    data[i] = i % 100;
                    fdata[i] = (float)i / 10.0f;
                }}
                
                // Call the test function (adjust based on test type)
                #ifdef TEST_{test_name.upper()}
                    // Add appropriate call here
                #endif
                
                return 0;
            }}
            """)
    
    print(f"Generated {len(ADVANCED_TEST_CASES)} advanced test cases in {output_dir}")

def generate_architecture_specific_tests(output_dir):
    """Generate architecture-specific test cases."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # x86-specific tests
    with open(output_dir / "x86_specific.c", 'w') as f:
        f.write("""
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
            printf("Result: %f\\n", result);
            
            return 0;
        }
        """)
    
    # ARM-specific tests
    with open(output_dir / "arm_specific.c", 'w') as f:
        f.write("""
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
            printf("Result: %f\\n", result);
            
            return 0;
        }
        """)
    
    print(f"Generated architecture-specific test cases in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Generate compiler test cases for ARM vs x86 comparison')
    parser.add_argument('--output-dir', default='specialized_test_cases', help='Output directory for test cases')
    args = parser.parse_args()
    
    generate_test_files(args.output_dir)
    generate_architecture_specific_tests(args.output_dir)

if __name__ == "__main__":
    main()
 