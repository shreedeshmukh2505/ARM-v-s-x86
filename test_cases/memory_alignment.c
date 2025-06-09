
            #include <stdio.h>
            #include <stdlib.h>
            
            
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
                #ifdef TEST_MEMORY_ALIGNMENT
                    // Add appropriate call here
                #endif
                
                return 0;
            }
            