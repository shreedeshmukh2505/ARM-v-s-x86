#!/usr/bin/env python3
"""
ARM vs x86 Compiler Analysis Framework

This script analyzes compiler behavior across ARM and x86 architectures by:
1. Generating test cases
2. Compiling with different optimization levels
3. Analyzing generated assembly
4. Collecting metrics on code generation differences
5. Visualizing results

Requirements:
- Python 3.6+
- gcc/clang compilers (native and cross-compilers if on single architecture)
- binutils (for objdump)
- matplotlib, pandas for analysis
"""

import os
import sys
import subprocess
import re
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration
TEST_CASES = {
    'loop_optimization': '''
        int loop_sum(int n) {
            int sum = 0;
            for (int i = 0; i < n; i++) {
                sum += i;
            }
            return sum;
        }
    ''',
    'recursion': '''
        int fibonacci(int n) {
            if (n <= 1) return n;
            return fibonacci(n-1) + fibonacci(n-2);
        }
    ''',
    'memory_access': '''
        void matrix_multiply(int n, float a[n][n], float b[n][n], float c[n][n]) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    c[i][j] = 0;
                    for (int k = 0; k < n; k++) {
                        c[i][j] += a[i][k] * b[k][j];
                    }
                }
            }
        }
    ''',
    'branch_prediction': '''
        int count_values_greater_than(int arr[], int size, int threshold) {
            int count = 0;
            for (int i = 0; i < size; i++) {
                if (arr[i] > threshold) {
                    count++;
                }
            }
            return count;
        }
    ''',
    'vectorization': '''
        void vector_add(float *a, float *b, float *c, int n) {
            for (int i = 0; i < n; i++) {
                c[i] = a[i] + b[i];
            }
        }
    ''',
    'function_calls': '''
        int add(int a, int b) { return a + b; }
        int subtract(int a, int b) { return a - b; }
        int multiply(int a, int b) { return a * b; }
        
        int compute(int x, int y, int z) {
            int temp1 = add(x, y);
            int temp2 = multiply(temp1, z);
            return subtract(temp2, x);
        }
    '''
}

OPTIMIZATION_LEVELS = ['O0', 'O1', 'O2', 'O3', 'Os', 'Ofast']
COMPILERS = ['gcc', 'clang']
ARCHITECTURES = {
    'x86_64': {
        'name': 'x86_64',
        'gcc': 'gcc',
        'clang': 'clang',
        'objdump': 'objdump',
        'arch_flags': '-march=x86-64',
    },
    'aarch64': {
        'name': 'aarch64',
        'gcc': 'aarch64-linux-gnu-gcc',
        'clang': 'clang --target=aarch64-linux-gnu',
        'objdump': 'aarch64-linux-gnu-objdump',
        'arch_flags': '-march=armv8-a',
    }
}

class CompilerAnalysis:
    def __init__(self, output_dir: str = 'compiler_analysis'):
        """Initialize the analysis framework."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
    def generate_test_files(self) -> None:
        """Generate C files for each test case."""
        test_dir = self.output_dir / 'test_cases'
        test_dir.mkdir(exist_ok=True)
        
        for test_name, code in TEST_CASES.items():
            with open(test_dir / f"{test_name}.c", 'w') as f:
                f.write(f"""
                #include <stdio.h>
                #include <stdlib.h>
                
                {code}
                
                int main() {{
                    // Simple driver to prevent dead code elimination
                    volatile int result = 0;
                    
                    #ifdef LOOP_TEST
                    result = loop_sum(1000);
                    #endif
                    
                    #ifdef RECURSION_TEST
                    result = fibonacci(20);
                    #endif
                    
                    // Other test harnesses as needed
                    
                    return 0;
                }}
                """)
        
        print(f"Generated {len(TEST_CASES)} test cases in {test_dir}")
        
    def compile_test_cases(self):
        """Compile each test case with different optimization levels for each architecture."""
        test_dir = self.test_dir
        asm_dir = self.output_dir / 'assembly'
        asm_dir.mkdir(exist_ok=True)
        
        compilation_data = []
        
        # Find all .c files in the test directory
        test_files = list(test_dir.glob("*.c"))
        print(f"Found {len(test_files)} test files for compilation")
        
        if not test_files:
            print(f"WARNING: No .c files found in {test_dir}")
            return
        
        for arch_name, arch_config in ARCHITECTURES.items():
            arch_dir = asm_dir / arch_name
            arch_dir.mkdir(exist_ok=True)
            
            for compiler in COMPILERS:
                compiler_cmd = arch_config[compiler]
                compiler_dir = arch_dir / compiler
                compiler_dir.mkdir(exist_ok=True)
                
                for test_file in test_files:
                    test_name = test_file.stem  # Get filename without extension
                    
                    for opt_level in OPTIMIZATION_LEVELS:
                        output_asm = compiler_dir / f"{test_name}_{opt_level}.s"
                        
                        # Start timing
                        start_time = time.time()
                        
                        # Compile to assembly
                        cmd = f"{compiler_cmd} -S -{opt_level} {arch_config['arch_flags']} {test_file} -o {output_asm}"
                        try:
                            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                            compile_time = time.time() - start_time
                            
                            # Also compile to object file for binary analysis
                            obj_file = compiler_dir / f"{test_name}_{opt_level}.o"
                            obj_cmd = f"{compiler_cmd} -c -{opt_level} {arch_config['arch_flags']} {test_file} -o {obj_file}"
                            subprocess.run(obj_cmd, shell=True, check=True)
                            
                            # Collect data
                            compilation_data.append({
                                'architecture': arch_name,
                                'compiler': compiler,
                                'test_name': test_name,
                                'optimization': opt_level,
                                'compile_time': compile_time,
                                'assembly_file': str(output_asm),
                                'object_file': str(obj_file),
                                'success': True,
                                'error': None
                            })
                            
                        except subprocess.CalledProcessError as e:
                            print(f"Compilation error: {e.stderr}")
                            compilation_data.append({
                                'architecture': arch_name,
                                'compiler': compiler,
                                'test_name': test_name,
                                'optimization': opt_level,
                                'compile_time': None,
                                'assembly_file': None,
                                'object_file': None,
                                'success': False,
                                'error': e.stderr
                            })
        
        # Save compilation metadata
        compilation_df = pd.DataFrame(compilation_data)
        compilation_df.to_csv(self.output_dir / 'compilation_data.csv', index=False)
        
        # Count successful compilations
        successful = compilation_df[compilation_df['success'] == True]
        print(f"Compiled test cases with {len(successful)} successful compilations out of {len(compilation_data)} attempts")
            
    def analyze_assembly(self) -> Dict:
        """Analyze the generated assembly files for various metrics."""
        try:
            compilation_df = pd.read_csv(self.output_dir / 'compilation_data.csv')
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print("No compilation data found. Cannot analyze assembly.")
            return {}
            
        successful_compilations = compilation_df[compilation_df['success'] == True]
        if successful_compilations.empty:
            print("No successful compilations found to analyze.")
            return {}
        
        analysis_results = []
        processed_files = 0
        
        for _, row in successful_compilations.iterrows():
            if not os.path.exists(row['assembly_file']):
                continue
                
            with open(row['assembly_file'], 'r') as f:
                assembly = f.read()
            
            # Basic metrics
            instruction_count = len(re.findall(r'^\s+\w+\s', assembly, re.MULTILINE))
            code_size = os.path.getsize(row['object_file'])
            
            # Instruction mix analysis
            if row['architecture'] == 'x86_64':
                # x86 specific metrics
                memory_instructions = len(re.findall(r'\bmov[a-z]*\b', assembly, re.IGNORECASE))
                arithmetic_instructions = len(re.findall(r'\b(add|sub|mul|div|inc|dec)[a-z]*\b', assembly, re.IGNORECASE))
                branch_instructions = len(re.findall(r'\b(j[a-z]+|call|ret)\b', assembly, re.IGNORECASE))
                vector_instructions = len(re.findall(r'\b(mm[a-z0-9]+|xmm|ymm|zmm|vex|avx)\b', assembly, re.IGNORECASE))
            else:
                # ARM specific metrics
                memory_instructions = len(re.findall(r'\b(ldr|str)[a-z]*\b', assembly, re.IGNORECASE))
                arithmetic_instructions = len(re.findall(r'\b(add|sub|mul|div)[a-z]*\b', assembly, re.IGNORECASE))
                branch_instructions = len(re.findall(r'\b(b[a-z]*|bl[a-z]*|ret)\b', assembly, re.IGNORECASE))
                vector_instructions = len(re.findall(r'\b(neon|simd|v[a-z0-9]+)\b', assembly, re.IGNORECASE))
            
            # Register usage analysis (simplified)
            if row['architecture'] == 'x86_64':
                register_pattern = r'\b(rax|rbx|rcx|rdx|rsi|rdi|rbp|rsp|r[8-9]|r1[0-5])\b'
            else:
                register_pattern = r'\b(x[0-9]|x[1-2][0-9]|x30|x31|w[0-9]|w[1-2][0-9]|w30|w31)\b'
                
            register_mentions = re.findall(register_pattern, assembly, re.IGNORECASE)
            unique_registers = len(set(register_mentions))
            
            # Store metrics
            analysis_results.append({
                'architecture': row['architecture'],
                'compiler': row['compiler'],
                'test_name': row['test_name'],
                'optimization': row['optimization'],
                'instruction_count': instruction_count,
                'code_size_bytes': code_size,
                'memory_instructions': memory_instructions,
                'arithmetic_instructions': arithmetic_instructions,
                'branch_instructions': branch_instructions,
                'vector_instructions': vector_instructions,
                'unique_registers': unique_registers,
                'register_mentions': len(register_mentions),
            })
        
        # Save analysis results
        analysis_df = pd.DataFrame(analysis_results)
        analysis_df.to_csv(self.output_dir / 'assembly_analysis.csv', index=False)
        print(f"Analyzed {len(analysis_results)} assembly files")
        
        return analysis_df.to_dict('records')
    
    def analyze_function_calling_conventions(self) -> Dict:
        """Analyze function calling conventions specifically."""
        compilation_df = pd.read_csv(self.output_dir / 'compilation_data.csv')
        
        # Filter for the function_calls test case
        function_calls_df = compilation_df[
            (compilation_df['test_name'] == 'function_calls') & 
            (compilation_df['success'] == True)
        ]
        
        function_analysis = []
        
        for _, row in function_calls_df.iterrows():
            if not os.path.exists(row['assembly_file']):
                continue
                
            with open(row['assembly_file'], 'r') as f:
                assembly = f.read()
            
            # Extract the 'compute' function assembly
            if row['architecture'] == 'x86_64':
                function_pattern = r'compute[:\n].*?(?=\.size|\.cfi_endproc|\.LFE)'
            else:
                function_pattern = r'compute[:\n].*?(?=\.size|\.cfi_endproc|\.LFE)'
            
            function_match = re.search(function_pattern, assembly, re.DOTALL)
            if not function_match:
                continue
                
            function_asm = function_match.group(0)
            
            # Analyze parameter passing
            if row['architecture'] == 'x86_64':
                reg_params = len(re.findall(r'\b(rdi|rsi|rdx|rcx|r8|r9)\b', function_asm))
                stack_params = len(re.findall(r'\[rbp\+[0-9a-fx]+\]', function_asm))
            else:
                reg_params = len(re.findall(r'\b(x0|x1|x2|x3|x4|x5|x6|x7)\b', function_asm))
                stack_params = len(re.findall(r'\[sp, [0-9a-fx]+\]', function_asm))
            
            # Analyze prologue/epilogue
            if row['architecture'] == 'x86_64':
                prologue_size = len(re.findall(r'push[a-z]*\s+\%r[a-z0-9]+', function_asm))
                epilogue_size = len(re.findall(r'pop[a-z]*\s+\%r[a-z0-9]+', function_asm))
            else:
                prologue_size = len(re.findall(r'stp\s+[a-z][0-9]+, [a-z][0-9]+, \[sp', function_asm))
                epilogue_size = len(re.findall(r'ldp\s+[a-z][0-9]+, [a-z][0-9]+, \[sp', function_asm))
            
            function_analysis.append({
                'architecture': row['architecture'],
                'compiler': row['compiler'],
                'optimization': row['optimization'],
                'reg_params': reg_params,
                'stack_params': stack_params,
                'prologue_size': prologue_size,
                'epilogue_size': epilogue_size,
                'function_size': len(function_asm.split('\n')),
            })
        
        # Save function calling analysis
        function_df = pd.DataFrame(function_analysis)
        function_df.to_csv(self.output_dir / 'function_calling_analysis.csv', index=False)
        print(f"Analyzed {len(function_analysis)} function implementation variations")
        
        return function_df.to_dict('records')
    
    def visualize_results(self):
        """Create visualizations of the analysis results."""
        try:
            analysis_df = pd.read_csv(self.output_dir / 'assembly_analysis.csv')
            if analysis_df.empty:
                print("No assembly analysis data available for visualization.")
                return
            function_df = pd.read_csv(self.output_dir / 'function_calling_analysis.csv')
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print("Analysis data not found or empty. Cannot create visualizations.")
            return
    
    # Rest of the visualization code...
        
        vis_dir = self.output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        # 1. Code size comparison by architecture and optimization level
        plt.figure(figsize=(12, 8))
        for test_name, group in analysis_df.groupby('test_name'):
            pivot = group.pivot_table(
                index='optimization', 
                columns='architecture', 
                values='code_size_bytes', 
                aggfunc='mean'
            )
            
            plt.figure(figsize=(10, 6))
            pivot.plot(kind='bar', ax=plt.gca())
            plt.title(f'Code Size Comparison for {test_name}')
            plt.ylabel('Bytes')
            plt.xlabel('Optimization Level')
            plt.tight_layout()
            plt.savefig(vis_dir / f'code_size_{test_name}.png')
            plt.close()
        
        # 2. Instruction mix comparison
        for arch in analysis_df['architecture'].unique():
            arch_data = analysis_df[analysis_df['architecture'] == arch]
            
            for opt in ['O0', 'O3']:  # Compare unoptimized vs highly optimized
                opt_data = arch_data[arch_data['optimization'] == opt]
                
                plt.figure(figsize=(12, 8))
                
                # Prepare data for stacked bar chart
                test_names = opt_data['test_name'].unique()
                memory_instr = opt_data.groupby('test_name')['memory_instructions'].mean()
                arithmetic_instr = opt_data.groupby('test_name')['arithmetic_instructions'].mean()
                branch_instr = opt_data.groupby('test_name')['branch_instructions'].mean()
                vector_instr = opt_data.groupby('test_name')['vector_instructions'].mean()
                
                x = np.arange(len(test_names))
                width = 0.6
                
                plt.bar(x, memory_instr, width, label='Memory')
                plt.bar(x, arithmetic_instr, width, bottom=memory_instr, label='Arithmetic')
                plt.bar(x, branch_instr, width, bottom=memory_instr+arithmetic_instr, label='Branch')
                plt.bar(x, vector_instr, width, bottom=memory_instr+arithmetic_instr+branch_instr, label='Vector')
                
                plt.xlabel('Test Case')
                plt.ylabel('Instruction Count')
                plt.title(f'Instruction Mix for {arch} with {opt} Optimization')
                plt.xticks(x, test_names, rotation=45)
                plt.legend()
                plt.tight_layout()
                plt.savefig(vis_dir / f'instruction_mix_{arch}_{opt}.png')
                plt.close()
        
        # 3. Function calling convention visualization
        plt.figure(figsize=(10, 6))
        function_pivot = function_df.pivot_table(
            index='architecture', 
            columns='optimization', 
            values=['reg_params', 'stack_params', 'prologue_size', 'epilogue_size']
        )
        
        function_pivot.plot(kind='bar', ax=plt.gca())
        plt.title('Function Calling Convention Analysis')
        plt.ylabel('Count')
        plt.xlabel('Architecture')
        plt.tight_layout()
        plt.savefig(vis_dir / 'function_calling_conventions.png')
        plt.close()
        
        # 4. Optimization impact on code size
        plt.figure(figsize=(12, 8))
        size_by_opt = analysis_df.groupby(['architecture', 'optimization'])['code_size_bytes'].mean().unstack()
        size_by_opt.plot(kind='line', marker='o')
        plt.title('Impact of Optimization Level on Code Size')
        plt.ylabel('Average Code Size (bytes)')
        plt.xlabel('Architecture')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(vis_dir / 'optimization_impact.png')
        plt.close()
        
        print(f"Generated visualizations in {vis_dir}")

    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting compiler analysis framework...")
        
        # If test_dir hasn't been explicitly set, generate the default test files
        if not hasattr(self, 'test_dir'):
            self.generate_test_files()
            self.test_dir = self.output_dir / 'test_cases'
        
        self.compile_test_cases()
        self.analyze_assembly()
        self.analyze_function_calling_conventions()
        self.visualize_results()
        print(f"Analysis complete. Results in {self.output_dir}")
        
        # Generate summary report
        self.generate_report()
    
    def generate_report(self) -> None:
        """Generate a summary report of findings."""
        report_path = self.output_dir / 'report.md'
        
        try:
            analysis_df = pd.read_csv(self.output_dir / 'assembly_analysis.csv')
            function_df = pd.read_csv(self.output_dir / 'function_calling_analysis.csv')
            compilation_df = pd.read_csv(self.output_dir / 'compilation_data.csv')
        except FileNotFoundError:
            print("Data files not found. Cannot generate report.")
            return
            
        with open(report_path, 'w') as f:
            f.write("# ARM vs x86 Compiler Analysis Report\n\n")
            f.write("## Overview\n\n")
            f.write(f"This report summarizes the analysis of compiler behavior across ARM and x86 architectures.\n")
            f.write(f"- Test cases analyzed: {len(TEST_CASES)}\n")
            f.write(f"- Compilers tested: {', '.join(COMPILERS)}\n")
            f.write(f"- Optimization levels: {', '.join(OPTIMIZATION_LEVELS)}\n\n")
            
            # Code size summary
            f.write("## Code Size Analysis\n\n")
            size_by_arch = analysis_df.groupby('architecture')['code_size_bytes'].mean()
            f.write("Average code size by architecture:\n\n")
            for arch, size in size_by_arch.items():
                f.write(f"- {arch}: {size:.2f} bytes\n")
            f.write("\n")
            
            # Optimization impact summary
            f.write("## Optimization Impact\n\n")
            opt_impact = analysis_df.pivot_table(
                index='optimization', 
                columns='architecture', 
                values='code_size_bytes', 
                aggfunc='mean'
            )
            f.write("Average code size by optimization level (bytes):\n\n")
            f.write(opt_impact.to_markdown() + "\n\n")
            
            # Instruction mix summary
            f.write("## Instruction Mix Analysis\n\n")
            instr_mix = analysis_df.groupby('architecture')[
                ['memory_instructions', 'arithmetic_instructions', 'branch_instructions', 'vector_instructions']
            ].mean()
            f.write("Average instruction distribution by architecture:\n\n")
            f.write(instr_mix.to_markdown() + "\n\n")
            
            # Function calling conventions
            f.write("## Function Calling Convention Analysis\n\n")
            calling_conv = function_df.groupby('architecture')[
                ['reg_params', 'stack_params', 'prologue_size', 'epilogue_size']
            ].mean()
            f.write("Function calling conventions by architecture:\n\n")
            f.write(calling_conv.to_markdown() + "\n\n")
            
            # Key findings
            f.write("## Key Findings\n\n")
            
            # Find the test with biggest arch difference
            test_diff = analysis_df.groupby(['test_name', 'architecture'])['code_size_bytes'].mean().unstack()
            test_diff['difference_pct'] = (test_diff['x86_64'] / test_diff['aarch64'] - 1) * 100
            most_diff_test = test_diff['difference_pct'].abs().idxmax()
            
            f.write(f"1. The test case with the largest architectural difference is '{most_diff_test}', ")
            if test_diff.loc[most_diff_test, 'difference_pct'] > 0:
                f.write(f"where x86_64 code is {test_diff.loc[most_diff_test, 'difference_pct']:.1f}% larger than ARM.\n")
            else:
                f.write(f"where ARM code is {-test_diff.loc[most_diff_test, 'difference_pct']:.1f}% larger than x86_64.\n")
            
            # Register usage difference
            reg_usage = analysis_df.groupby('architecture')['unique_registers'].mean()
            f.write(f"2. ARM code uses {reg_usage['aarch64'] / reg_usage['x86_64']:.2f}x as many unique registers as x86_64 on average.\n")
            
            # Vectorization difference
            vec_usage = analysis_df.groupby(['architecture', 'optimization'])['vector_instructions'].mean().unstack()
            if 'O3' in vec_usage.columns:
                vec_diff = vec_usage['O3']['aarch64'] / vec_usage['O3']['x86_64'] if vec_usage['O3']['x86_64'] > 0 else float('inf')
                f.write(f"3. With high optimization (O3), ARM uses {vec_diff:.2f}x as many vector instructions as x86_64.\n")
            
            # Optimization effectiveness
            opt_effect_x86 = analysis_df[analysis_df['architecture'] == 'x86_64'].groupby('optimization')['code_size_bytes'].mean()
            opt_effect_arm = analysis_df[analysis_df['architecture'] == 'aarch64'].groupby('optimization')['code_size_bytes'].mean()
            
            if 'O0' in opt_effect_x86 and 'O3' in opt_effect_x86:
                x86_reduction = (1 - opt_effect_x86['O3'] / opt_effect_x86['O0']) * 100
                arm_reduction = (1 - opt_effect_arm['O3'] / opt_effect_arm['O0']) * 100
                f.write(f"4. Optimization from O0 to O3 reduces code size by {x86_reduction:.1f}% on x86_64 and {arm_reduction:.1f}% on ARM.\n")
            
            f.write("\n## Conclusion\n\n")
            f.write("This analysis demonstrates significant differences in how compilers generate code for ARM and x86 architectures, ")
            f.write("with implications for performance, code size, and optimization strategies.")
            
        print(f"Generated report at {report_path}")


def main():
    parser = argparse.ArgumentParser(description='ARM vs x86 Compiler Analysis Framework')
    parser.add_argument('--output-dir', default='compiler_analysis', help='Output directory for analysis results')
    parser.add_argument('--test-cases-dir', help='Directory containing test cases to analyze')
    args = parser.parse_args()
    
    analyzer = CompilerAnalysis(output_dir=args.output_dir)
    
    # Use custom test cases directory if provided
    if args.test_cases_dir:
        analyzer.generate_test_files = lambda: None  # Skip generating default test cases
        analyzer.test_dir = Path(args.test_cases_dir)
    
    analyzer.run_complete_analysis()
if __name__ == "__main__":
    main()