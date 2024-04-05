# Writes decompiled function code to an output file
# Usage: ghidra ~/ghidra_projects SomeProject -import <path_to_bin> \
#   -scriptPath ~/dataset_scripts/ghidra -noanalysis -overwrite \
#   -postScript write_decompiled_function.py <func_name> <output_file>

from ghidra.app.decompiler import DecompInterface
from ghidra.util.task import ConsoleTaskMonitor

program = getCurrentProgram()
ifc = DecompInterface()
ifc.openProgram(program)

func, out_path = getScriptArgs()
function = getGlobalFunctions(func)[0]

# decompile the function and print the pseudo C
results = ifc.decompileFunction(function, 0, ConsoleTaskMonitor())
# Write to output file
with open(out_path, 'w') as out_file:
    out_file.write(results.getDecompiledFunction().getC())