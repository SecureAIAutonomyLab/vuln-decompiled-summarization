# Enhancing Reverse Engineering: Investigating and Benchmarking Large Language Models for Vulnerability Analysis in Decompiled Binaries

This repository contains the DeBinVul dataset, source code, and experimental results of the paper Enhancing Reverse Engineering: Investigating and Benchmarking Large Language Models for Vulnerability Analysis in Decompiled Binaries.

## DeBinVul Dataset

The dataset is a compressed CSV file stored as `DeBinVul.zip`. `DeBinVul` has 12 columns:

1. **`source_code`**: The complete source code function definition.
2. **`comments`**: Extracted comments from the function's source code.
3. **`label`**: The assigned CWE label (or `none` if benign).
4. **`file`**: The file containing the function.
5. **`function`**: The name of the function.
6. **`decompiled_code`**: Ghidra decompiled code of the function.
7. **`compiler_options`**: Compiler command used for compiling the function.
8. **`stripped_decompiled_code`**: Ghidra decompiled code with symbols stripped.
9. **`description`**: Description of the vulnerability (if applicable) and the function’s behavior.
10. **`prompt`**: The complete prompt given to the model for the task.
11. **`instruction`**: Specific instructions given to the model for the task.
12. **`task`**: The nature of the task, such as identification, classification, description, or function name recovery.

## Other Repository Contents

All code related to training, inference, and evaluation can be found in the `src` subdirectory. The experimental results can be found under the `Experimental_results` subdirectory. Miscellaneous scripts associated with the dataset are located in the `scripts` directory.