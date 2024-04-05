import json

from pathlib import Path
from rich import print
from rich.progress import track
from typer import Argument, Typer
from typing import Annotated, Literal

app = Typer()


@app.command()
def command(report: Annotated[Path, Argument(help='Path to the report file.')],
            funcs: Annotated[list[str], Argument(help='Vulnerable functions to revert to benign functions.')]) -> None:
    report_json: dict[str, dict[str, dict[Literal['original', 'vulnerable'], str]]] = \
        json.loads(report.read_text())
    for src_file, func_name, code_dict in track(list((outer_key, inner_key, value)
                                                     for outer_key, inner_dict in report_json.items()
                                                     for inner_key, value in inner_dict.items()
                                                     if inner_key in funcs),
                                                description='Fixing compile errors...'):
        src_file_path = Path(src_file)
        # Fix error in source code
        src_file_path.write_text(src_file_path.read_text().replace(code_dict['vulnerable'],
                                                                   code_dict['original']))
        # Remove from report
        del report_json[src_file][func_name]
    # Write new report
    report.write_text(json.dumps(report_json))
    print(f'[green]Successfully removed {len(funcs)} compile errors. ' +
          f'Report size is now {sum(len(d) for d in report_json.values())} entries')


app()
