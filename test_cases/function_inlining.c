
            #include <stdio.h>
            #include <stdlib.h>
            
            
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
    
            
            int main() {
                // Simple driver for testing
                int data[1000];
                float fdata[1000];
                int result[10] = {0};
                
                // Initialize data
                for (int i = 0; i < 1000; i++) {
                    data[i] = i % 100;
                    fdata[i] = (float)i / 10.0f;
                }
                
                // Call the test function (adjust based on test type)
                #ifdef TEST_FUNCTION_INLINING
                    // Add appropriate call here
                #endif
                
                return 0;
            }
            