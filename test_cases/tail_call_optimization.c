
            #include <stdio.h>
            #include <stdlib.h>
            
            
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
                #ifdef TEST_TAIL_CALL_OPTIMIZATION
                    // Add appropriate call here
                #endif
                
                return 0;
            }
            