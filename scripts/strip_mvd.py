import pandas as pd
import sys
import uuid

from pathlib import Path
from tree_sitter import Language, Parser
from typing import Final

TREE_SITTER_DIR: Final[Path] = Path.home() / '.tree-sitter'
LANGUAGES_FILE = TREE_SITTER_DIR / 'languages.so'
STRIP_QUERY: Final[str] = '''
(function_definition
  declarator: (function_declarator
    declarator: (identifier) @func-def
  )
)
(call_expression
  function: (identifier) @call-expr
)
'''

parser = Parser()
language = Language(str(LANGUAGES_FILE), 'cpp')
parser.set_language(language)


def strip_func(func_def: str) -> str:
    func_names = set(n.text.decode() for n, t in language.query(STRIP_QUERY)
                     .captures(parser.parse(func_def.encode()).root_node))
    stripped_mappings = {n: f'func_{uuid.uuid4().hex[:16]}'
                         for n in func_names}
    stripped_def = func_def
    for orig_name, strip_name in stripped_mappings.items():
        stripped_def = stripped_def.replace(orig_name, strip_name)
    return stripped_def


df = pd.read_csv(sys.argv[1])
test: str = df.iloc[0]['decompiled_code']

df['stripped_decompiled_code'] = df['decompiled_code'].apply(strip_func)
print(df)
df.to_csv(sys.argv[1])
