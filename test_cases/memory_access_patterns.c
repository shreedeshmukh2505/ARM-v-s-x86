
            #include <stdio.h>
            #include <stdlib.h>
            
            
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
                #ifdef TEST_MEMORY_ACCESS_PATTERNS
                    // Add appropriate call here
                #endif
                
                return 0;
            }
            