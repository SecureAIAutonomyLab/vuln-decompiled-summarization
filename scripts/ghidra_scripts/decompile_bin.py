# Decompiles a binary file and pickles function as a dict, mapping function name to decompiled code
# Usage: ghidra ~/ghidra_projects SomeProject -import <path_to_bin> \
#   -scriptPath ~/dataset_scripts/ghidra -noanalysis -overwrite \
#   -postScript decompile_bin.py <output_file>

import pickle

from ghidra.app.decompiler import DecompInterface
from ghidra.util.task import ConsoleTaskMonitor

(out_path,) = getScriptArgs()
program = getCurrentProgram()
ifc = DecompInterface()
ifc.openProgram(program)

fm = currentProgram.getFunctionManager()
funcs = fm.getFunctions(True)  # True means 'forward'
func_dict = {}
for func in funcs:
    func_dict[func.getName()] = \
        ifc.decompileFunction(func, 0, ConsoleTaskMonitor()).getDecompiledFunction().getC()

# Pickle function mapping and write to output file
with open(out_path, 'wb') as out_file:
    pickle.dump(func_dict, out_file)
