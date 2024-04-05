import itertools
import json
import os
import pandas as pd
import re
import subprocess

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from rich import print
from rich.progress import track
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typer import Argument, Option, Typer
from typing import Annotated, Final, Literal, Optional

GHIDRA_DIR: Final[Path] = Path.home() / 'Ghidra' / \
    'ghidra_11.0.1_PUBLIC' / 'support' / 'analyzeHeadless'
GHIDRA_SCRIPT_DIR: Final[Path] = Path().home() / 'ghidra_scripts'
DECOMPILE_FUNCTION_SCRIPT: Final[Path] = GHIDRA_SCRIPT_DIR / \
    'write_decompiled_function.py'

app = Typer()


def get_decompiled_function(function: str, path: Path, project_dir: Path) -> str:
    with NamedTemporaryFile(suffix='.c') as decompiled_file:
        decompiled_file = Path(decompiled_file.name)
        try:
            subprocess.run([GHIDRA_DIR, project_dir, f'VulnProject_PID_{os.getpid()}',
                            '-import', path, '-scriptPath', GHIDRA_SCRIPT_DIR,
                            '-noanalysis', '-overwrite', '-postScript', DECOMPILE_FUNCTION_SCRIPT,
                            function, decompiled_file], check=True, capture_output=True)
            if decompiled_code := decompiled_file.read_text():
                return decompiled_code
            raise ValueError('Decompiled code was not written')
        except (subprocess.CalledProcessError, ValueError) as e:
            raise ValueError('Could not get decompiled function: ' +
                             f'"{function}"') from e


def process_entry(project_dir: Path, repo: str, benign_bin: Path, vuln_bin: Path,
                  vuln_src_file: Path, func_name: str,
                  code_dict: dict[Literal['original', 'vulnerable'], str]) -> dict[Literal['repo', 'benign_src_code', 'benign_decompiled_code',
                                                                                           'vuln_src_file', 'vuln_src_code', 'vuln_decompiled_code',
                                                                                           'cwes'], list[str | Path | list[str]]]:
    decomp_code = get_decompiled_function(func_name, vuln_bin, project_dir)
    return {'repo': [repo], 'benign_src_code': [code_dict['original']],
            'benign_decompiled_code': [get_decompiled_function(func_name, benign_bin, project_dir)],
            'vuln_src_file': [vuln_src_file], 'vuln_src_code': [code_dict['vulnerable']],
            'vuln_decompiled_code': [decomp_code],
            'cwes': [re.findall(r'//\s*VULNERABILITY:.+', code_dict['vulnerable'])]}


@app.command()
def command(repo: Annotated[str, Argument(help='Name of the repository.')],
            report: Annotated[Path, Argument(dir_okay=False, exists=True,
                                             help='Path to JSON report produced from create_vulns.')],
            benign_bin: Annotated[Path, Argument(dir_okay=False, exists=True,
                                                 help='Path to benign binary.')],
            vuln_bin: Annotated[Path, Argument(dir_okay=False, exists=True,
                                               help='Path to vulnerable binary')],
            dataset: Annotated[Path, Argument(dir_okay=False,
                                              help='Path to store the CSV dataset file.')],
            checkpoint: Annotated[int, Option(min=1,
                                              help='Number of steps to save the report file.')] = 10) -> None:
    assert benign_bin != vuln_bin
    report_json: dict[str, dict[str, dict[Literal['original', 'vulnerable'], str]]] = \
        json.loads(report.read_text())
    df = pd.DataFrame(columns=['repo', 'benign_src_code', 'benign_decompiled_code',
                               'vuln_src_file', 'vuln_src_code', 'vuln_decompiled_code',
                               'cwes'])
    step = 0
    with TemporaryDirectory() as ghidra_projects_dir:
        for src_file, func_name, code_dict in track(list((outer_key, inner_key, value)
                                                         for outer_key, inner_dict in report_json.items()
                                                         for inner_key, value in inner_dict.items()),
                                                    description='Building dataset...'):
            try:
                entry = process_entry(Path(ghidra_projects_dir), repo, benign_bin, vuln_bin,
                                      Path(src_file), func_name, code_dict)
            except ValueError as e:
                print(f'[yellow]Failed processing an entry: {e}')
            else:
                df = pd.concat([df, pd.DataFrame(entry)])
                step += 1
                if step % checkpoint == 0:
                    # Save partial report
                    print('[green]Partial dataset saved.')
                    df.to_csv(dataset)
    df.to_csv(dataset)
    print(f'[green]Successfully saved {len(df)} entries to {dataset}')
    dedup_df = df[df['benign_decompiled_code'] != df['vuln_decompiled_code']]
    dedup_dataset = dataset.with_stem(f'{dataset.stem}_dedup')
    dedup_df.to_csv(dedup_dataset)
    print(f'[green]Successfully saved {len(dedup_df)} (de-duplicated) entries to {dedup_dataset}')

app()
