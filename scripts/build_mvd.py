import itertools
import logging
import os
import pandas as pd
import subprocess
import re

from concurrent.futures import ProcessPoolExecutor
from datasets import DownloadManager
from pathlib import Path
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn

from tempfile import NamedTemporaryFile, TemporaryDirectory
from tree_sitter import Language, Parser
from typer import Argument, BadParameter, Option, Typer
from typing import Annotated, Any, Final, Generator, Literal, Optional

MVD_GITHUB_URL: Final[str] = 'https://github.com/muVulDeePecker/muVulDeePecker/archive/refs/heads/master.zip'
TREE_SITTER_C_GITHUB_URL: Final[str] = 'https://github.com/tree-sitter/tree-sitter-c/archive/refs/heads/master.zip'
TREE_SITTER_CPP_GITHUB_URL: Final[str] = 'https://github.com/tree-sitter/tree-sitter-cpp/archive/refs/heads/master.zip'
GHIDRA_HEADLESS_ENVVAR: Final[str] = 'GHIDRA_HEADLESS'
GHIDRA_SCRIPT_DIR: Final[Path] = Path.home() / 'ghidra_scripts'
DECOMPILE_FUNCTION_SCRIPT: Final[Path] = GHIDRA_SCRIPT_DIR / \
    'write_decompiled_functions.py'
CHECKPOINT_FILE: Final[Path] = Path(__file__).parent / '.checkpoint'

app = Typer()
logging.basicConfig(
    level=logging.INFO, format='%(message)s', datefmt='[%X]',
    handlers=[RichHandler()]
)
logger = logging.getLogger('rich')


def parse_command(command: str) -> str:
    # Run command and check for errors
    cmd = command.split()
    compiler, *_ = cmd
    try:
        out = subprocess.run(cmd, text=True,
                             capture_output=True).stderr
    except FileNotFoundError as e:
        raise BadParameter(f'Unsupported compiler: {compiler}') from e
    match compiler:
        case 'gcc' | 'g++':
            errs = '\n'.join((l for l in out.splitlines()
                             if not any(s in l for s in ['fatal error: no input files',
                                                         'compilation terminated.']))).strip()
        case _:
            raise NotImplementedError(f'Unknown compiler: "{compiler}"')
    if errs:
        errs = '\n\t'.join(errs.splitlines())
        raise BadParameter(f'Errors in {compiler} command:\n\t{errs}')
    return command


def extract_data(path: Path, language: Language) -> dict[Literal['vulnerable', 'patched'],
                                                         dict[str, Any]]:
    data: dict[Literal['vulnerable', 'patched'],
               dict[str, Any]] = {'vulnerable': {}, 'patched': {}}
    # Parse source code AST
    parser = Parser()
    parser.set_language(language)
    ast = parser.parse(path.read_bytes())
    # Find CWE functions
    cwe_func_nodes = (n for n, _ in language.query('(function_definition) @func-decl').captures(ast.root_node)
                      for d, _ in language.query('(function_declarator declarator: (identifier) @func_name)').captures(n)
                      if ['good', 'bad'] in d.text.decode())
    for cwe_func_node in cwe_func_nodes:
        # Get function name and comments
        func_name, = [n.text.decode() for n, _ in language.query(
            '(function_declarator declarator: (identifier) @func_name)').captures(cwe_func_node)]
        if 'good' in func_name:
            key = 'patched'
        elif 'bad' in func_name:
            key = 'vulnerable'
        else:
            raise ValueError(f'Unexpected CWE function: {func_name}')
        data[key][func_name] = {}
        comments = [n.text.decode() for n, _ in language.query(
            '(comment) @comment').captures(cwe_func_node)]
        # Extract metadata
        data[key][func_name]['source_code'] = cwe_func_node.text.decode()
        data[key][func_name]['comments'] = comments
        data[key][func_name]['label'] = re.findall(r'CWE\d+', func_name) \
            if key == 'vulnerable' else None
    return data


def get_decompiled_function(function: str, path: Path, ghidra: Path, ghidra_dir: Path) -> str:
    with NamedTemporaryFile(suffix='.c', delete_on_close=False) as decompiled_file:
        decompiled_file = Path(decompiled_file.name)
        try:
            subprocess.run([ghidra, ghidra_dir, f'MVDProject_PID_{os.getpid()}',
                            '-import', path, '-scriptPath', GHIDRA_SCRIPT_DIR,
                            '-noanalysis', '-overwrite', '-postScript', DECOMPILE_FUNCTION_SCRIPT,
                            function, decompiled_file], check=True, capture_output=True)
            if decompiled_code := decompiled_file.read_text():
                return decompiled_code
            raise ValueError('Decompiled code was not written')
        except (subprocess.CalledProcessError, ValueError) as e:
            raise ValueError('Could not get decompiled function: ' +
                             f'"{function}"') from e


def generate_sample(cwe_file: Path, languages_file: Path, commands: list[str],
                    ghidra: Path, ghidra_dir: Path) -> tuple[list[dict[str, Any]], int]:
    samples: list[dict[str, Any]] = []
    errors = 0
    # Get compile commands and grammar for C or C++ sample
    if cwe_file.suffix == '.c':
        language = Language(str(languages_file), 'c')
        current_commands = [c for c in commands if 'gcc' in c]
    else:
        language = Language(str(languages_file), 'cpp')
        current_commands = [c for c in commands if 'g++' in c]
    try:
        # Extract source code comments, and other data
        cwe_file_data = extract_data(cwe_file, language)
    except Exception as e:
        logger.error(f'Failed to extract {cwe_file.name}')
        errors += len(current_commands) * 2
    else:
        # Get other source files and headers
        other_source = [s for s in cwe_file.parent.glob('*.c')
                        if s != cwe_file]
        headers = [h for h in cwe_file.parent.glob('*.h')]
        # Run through all compile commands
        for command in current_commands:
            compiler, *options = command.split()
            # Compile vulnerable/non-vulnerable
            for omit_def in ['OMITGOOD', 'OMITBAD']:
                with NamedTemporaryFile(suffix='.exe', delete_on_close=False) as out_file:
                    out_file.close()
                    # Compile sample
                    try:
                        subprocess.run([compiler, '-D', 'INCLUDEMAIN', '-D', omit_def, *options,
                                        *itertools.chain.from_iterable([['-I', h] for h in headers]),
                                        '-o', out_file.name, cwe_file, *other_source],
                                       capture_output=True, check=True)
                    except subprocess.CalledProcessError as e:
                        logger.error('Failed to compile ' +
                                     f'{omit_def} {cwe_file.name}')
                        errors += len(current_commands)
                    else:
                        key = 'vulnerable' if omit_def == 'OMITGOOD' else 'patched'
                        for func in cwe_file_data[key]:
                            try:
                                decompiled_code = get_decompiled_function(func, Path(out_file.name),
                                                                          ghidra,
                                                                          ghidra_dir)
                            except ValueError as e:
                                logger.error('Could not get decompiled ' +
                                             f'"{func}" function')
                                errors += 1
                            else:
                                samples.append({**cwe_file_data[key][func],
                                                'function': func,
                                                'decompiled_code': decompiled_code,
                                                'compiler_options': [compiler, *options]})
    return samples, errors


@app.command()
def command(path: Annotated[Path, Argument(help='CSV file to store the dataset.', dir_okay=False)] =
            Path('/workspace') / 'storage' / 'mvd_decomp.csv',
            commands: Annotated[Optional[list[str]], Argument(help='Compiler commands to use ' +
                                                              'for each sample (e.g. "gcc -g -O0")',
                                                              parser=parse_command)] = None,
            ghidra: Annotated[Optional[Path], Option(envvar=GHIDRA_HEADLESS_ENVVAR, dir_okay=False,
                                                     help="Path to Ghidra's analyzeHeadless command.")] = None,
            workers: Annotated[Optional[int], Option(min=1,
                                                     help='Number of subprocesses to build the dataset.')] = None,
            checkpoint: Annotated[int, Option(min=1, help='How many samples until checkpoint.')] = 10) -> None:
    '''
    Compiles the dataset, associating source code summaries with decompiled snippets.
    '''
    if not ghidra:
        # Get Ghidra path
        if ghidra_path := os.environ.get(GHIDRA_HEADLESS_ENVVAR):
            ghidra = Path(ghidra_path)
        else:
            raise BadParameter(f'{GHIDRA_HEADLESS_ENVVAR} is not set.',
                               param_hint='--ghidra')
    path.parent.mkdir(parents=True, exist_ok=True)
    if commands is None:
        commands = []
    # Add any default compiler commands
    for compiler in ['gcc', 'g++']:
        if not any(compiler in c for c in commands):
            commands.append(compiler)

    # Download MVD
    logger.info('Preparing MVD...')
    mvd_path = Path(DownloadManager(dataset_name='MVD')
                    .download_and_extract(MVD_GITHUB_URL))  # type: ignore
    cwe_files = (mvd_path / 'muVulDeePecker-master' /
                 'source files' / 'upload_source_1').rglob('*CWE*.c')
    cwe_files = sorted(cwe_files)
    if CHECKPOINT_FILE.exists():
        checkpoint = int(CHECKPOINT_FILE.read_text()) + 1
        # Load from checkpoint
        logger.warning(
            f'Loading from checkpoint. Starting at sample {checkpoint}')
        cwe_files = cwe_files[checkpoint:]
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame()

    # Download tree-sitter grammar implementations for C and C++
    logger.info('Preparing C and C++ tree-sitter grammars...')
    c_repo_path, cpp_repo_path = [Path(p) for p in DownloadManager(dataset_name='tree-sitter-langs')
                                  .download_and_extract([TREE_SITTER_C_GITHUB_URL,
                                                         TREE_SITTER_CPP_GITHUB_URL])]
    # Build the language library for MVD
    languages_file = Path.cwd() / 'mvd_languages.so'
    Language.build_library(str(languages_file), [str(c_repo_path / 'tree-sitter-c-master'),
                                                 str(cpp_repo_path / 'tree-sitter-cpp-master')])

    # Create a temporary directory for Ghidra
    with TemporaryDirectory() as ghidra_dir:
        ghidra_dir = Path(ghidra_dir)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Create dataset of decompiled code and associated comments
            progress = Progress(TextColumn("[progress.description]{task.description}"), BarColumn(),
                                TaskProgressColumn(), TextColumn('Time Elapsed:'), TimeElapsedColumn(),
                                TextColumn('ETA:'), TimeRemainingColumn(),
                                TextColumn('Entries: {task.completed}'),
                                TextColumn('Errors: {task.fields[errors]}'))
            task = progress.add_task(
                'Building dataset...', total=len(cwe_files))
            progress.tasks[task].fields['errors'] = 0
            with progress:
                for idx, results in enumerate(executor.map(generate_sample, cwe_files, itertools.repeat(languages_file),
                                                           itertools.repeat(
                                                               commands), itertools.repeat(ghidra),
                                                           itertools.repeat(ghidra_dir))):
                    samples, errors = results
                    progress.tasks[task].fields['errors'] += errors
                    for sample in samples:
                        df = pd.concat(
                            [df, pd.Series(sample).to_frame().T], ignore_index=True)
                        progress.advance(task)
                    if idx % checkpoint == 0:
                        # Create a checkpoint
                        logger.info('Checkpoint saved')
                        CHECKPOINT_FILE.write_text(str(idx))
                        df.to_csv(path)
    # Save dataset
    CHECKPOINT_FILE.unlink()
    df.to_csv(path)


app()
