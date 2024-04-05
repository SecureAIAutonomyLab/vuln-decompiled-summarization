import itertools
import json
import random
import requests
import shutil
import warnings

from pathlib import Path
from rich import print
from rich.progress import track
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from tree_sitter import Language, Parser
from typer import Argument, Option, Typer
from typing import Annotated, Final, Generator, Iterator, Literal, Optional
from zipfile import ZipFile

warnings.filterwarnings('ignore')

C_GRAMMAR_GITHUB_URL: Final[str] = 'https://github.com/tree-sitter/tree-sitter-c/archive/refs/heads/master.zip'
CPP_GRAMMAR_GITHUB_URL: Final[str] = 'https://github.com/tree-sitter/tree-sitter-cpp/archive/refs/heads/master.zip'
TREE_SITTER_DIR: Final[Path] = Path.home() / '.tree-sitter'
LANGUAGES_FILE = TREE_SITTER_DIR / 'languages.so'

app = Typer()


def get_parser() -> Parser:
    TREE_SITTER_DIR.mkdir(exist_ok=True, parents=True)
    c_grammar_path = TREE_SITTER_DIR / 'tree-sitter-c-master'
    cpp_grammar_path = TREE_SITTER_DIR / 'tree-sitter-cpp-master'
    unbuilt_grammar_paths = [g for g in [c_grammar_path, cpp_grammar_path]
                             if not g.exists()]
    for grammar_path in unbuilt_grammar_paths:
        # Download grammar and extract
        grammar_zip_file = grammar_path.with_suffix('.zip')
        grammar_zip_file.write_bytes(requests.get(C_GRAMMAR_GITHUB_URL if grammar_path is c_grammar_path
                                                  else CPP_GRAMMAR_GITHUB_URL).content)
        with ZipFile(grammar_zip_file, 'r') as zip_file:
            zip_file.extractall(TREE_SITTER_DIR)
        # Remove archive
        grammar_zip_file.unlink()
    if any(unbuilt_grammar_paths):
        # Build C/C++ grammar library
        Language.build_library(str(LANGUAGES_FILE),
                               [str(c_grammar_path), str(cpp_grammar_path)])
    return Parser()


def get_functions(src_file: Path, repo_dir: Path,
                  parser: Parser) -> Generator[tuple[str, str], None, None]:
    language = Language(str(LANGUAGES_FILE), 'c') if src_file.suffix == '.c' \
        else Language(str(LANGUAGES_FILE), 'cpp')
    parser.set_language(language)
    nodes_and_funcs = [(n, n.text.decode()) for n, _ in language.query('(function_definition) @func-def')
                       .captures(parser.parse(src_file.read_bytes()).root_node)]
    for node, func in nodes_and_funcs:
        try:
            func_name, = [n.text.decode() for n, _ in language.query('''
            (function_definition
                declarator: (function_declarator
                    declarator: (identifier) @func-name
                )
            )
            ''').captures(node)]
        except ValueError:
            print('[yellow]Could not extract function name from ' +
                  f'{repo_dir.name / src_file.relative_to(repo_dir)}')
        else:
            yield func_name, func


def generate_vulnerable_code(code: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                             additional_tokens: int = 50) -> str | None:
    messages = [
        {"role": "user", "content": 'Can you help me rewrite this code to contain a simple vulnerability?'},
        {"role": "assistant", "content": 'Of course, I am an expert at coding C and I am also a red team cybersecurity expert.'},
        {"role": "user",
            "content": 'Great! Please only generate the vulnerable code and no other text. On the same line where you inserted the vulnerability, ' +
            'make a comment that says "//VULNERABILITY: CWE-xx", where "CWE-xx" is the CWE of the vulnerability. Do not include ' +
            'any imports. Be mindful to not produce potential dead-code. Do not modify function parameters or add/remove parameters. ' +
            f'Here is the code:\n```c{code}\n```'}
    ]
    token_length = sum(len(m['content'].split())
                       for m in messages) + len(code.split()) + additional_tokens
    model_inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt").to("cuda")  # type: ignore
    generated_ids = model.generate(
        model_inputs, max_new_tokens=token_length, do_sample=True)
    try:
        return tokenizer.batch_decode(generated_ids)[0].split('[/INST]')[-1].split('```')[1].removeprefix('c')
    except IndexError:
        return None


@app.command()
def command(repo: Annotated[Path, Argument(file_okay=False,  # exists=True,
                                           help='Path to C/C++ repository.')],
            vuln_repo: Annotated[Path, Argument(file_okay=False, exists=False,
                                                help='Path to store the vulnerable repository.')],
            report: Annotated[Path, Argument(dir_okay=False,
                                             help='Path to JSON file detailing what was changed.')],
            samples: Annotated[int, Argument(min=1,
                                             help='Requested number of functions to make vulnerable.')] = 200,
            overwrite: Annotated[bool,
                                 Option(help='Overwrite vulnerable repository.')] = False,
            subdir: Annotated[Optional[list[Path]],
                              Option(help='Subdirectories/files to only search in.')] = None,
            max_funcs: Annotated[Optional[int],
                                 Option(min=1,
                                        help='Maximum number of functions per file.')] = None,
            additional_tokens: Annotated[int, Option(min=1,
                                                     help='Number of additional tokens to generate during vulnerability generation.')] = 50,
            checkpoint: Annotated[int, Option(min=1,
                                              help='Number of steps to save the report file.')] = 50) -> None:
    print('Loading Mistral-7b...')
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2")
    if overwrite:
        shutil.rmtree(vuln_repo, ignore_errors=True)
    # Copy repository
    for _ in track([None], description='Copying repository...'):
        shutil.copytree(repo, vuln_repo)
    # Setup parser get source files
    parser = get_parser()
    if not subdir:
        src_files = list(itertools.chain.from_iterable([vuln_repo.rglob('*.c'),
                                                        vuln_repo.rglob('*.cpp')]))
    else:
        subdirs = set(s if s.is_absolute() else vuln_repo / s for s in subdir)
        src_files = list(itertools.chain.from_iterable(itertools.chain
                                                       .from_iterable([p.rglob('*.c'),
                                                                       p.rglob('*.cpp')]) for p in subdirs))
    # Randomize entries
    random.shuffle(src_files)
    funcs: dict[Path, dict[str, str]] = {}
    file_it = iter(src_files)
    func_it: Iterator[tuple[str, str]] = iter([])
    src_file: Path | None = None
    for idx in track(range(samples), description='Locating functions...'):
        # Get next function
        while not (func_info := next(func_it, None)):
            try:
                # Get next source file
                src_file = next(file_it)
            except StopIteration:
                break
            funcs[src_file] = {}
            try:
                func_it = iter(get_functions(src_file, vuln_repo, parser))
            except ValueError:
                print('[yellow]Could not parse ' +
                      f'{vuln_repo.name / src_file.relative_to(vuln_repo)}')
        if not src_file or not func_info:
            print(f'[red]Could only produce {idx + 1} out of ' +
                  f'{samples} vulnerable samples')
            break
        else:
            # Add file-function mapping to dictionary
            func_name, func_code = func_info
            funcs[src_file][func_name] = func_code
            if max_funcs and len(funcs[src_file]) >= max_funcs:
                # Maximum functions per file reached, go to next file
                func_it = iter([])
    print('[green]Successfully located ' +
          f'{sum(len(f) for f in funcs.values())} functions')
    # Write vulnerabilities in code using Mistral-7B
    report_json: dict[str, dict[str,
                                dict[Literal['original', 'vulnerable'], str]]] = {}
    step = 0
    for src_file, func_name, func_code in track(list((outer_key, inner_key, value)
                                                     for outer_key, inner_dict in funcs.items()
                                                     for inner_key, value in inner_dict.items()),
                                                description='Writing vulnerabilities...'):
        if vuln_func_code := generate_vulnerable_code(func_code, model, tokenizer,
                                                      additional_tokens=additional_tokens):
            # Ensure the code is parsable
            try:
                parser.parse(vuln_func_code.encode())
            except ValueError:
                print('[yellow]Could not parse vulnerable code for ' +
                      f'{vuln_repo.name / src_file.relative_to(vuln_repo)}:{func_name}')
            else:
                # Replace benign code with vulnerable code
                src_file.write_text(src_file.read_text().replace(func_code, vuln_func_code))
                func_dict = report_json.setdefault(str(src_file), {})
                func_dict[func_name] = {'original': func_code,
                                        'vulnerable': vuln_func_code}
                step += 1
                if step % checkpoint == 0:
                    # Save partial report
                    print('[green]Partial report saved.')
                    report.write_text(json.dumps(report_json))
        else:
            print('[yellow]Failed to generate vulnerable code for ' +
                  f'{vuln_repo.name / src_file.relative_to(vuln_repo)}:{func_name}')
    report.write_text(json.dumps(report_json))
    print(f'[green]Successfully wrote vulnerable functions to {vuln_repo}. Report ' +
          f'is stored at {report}')


app()
