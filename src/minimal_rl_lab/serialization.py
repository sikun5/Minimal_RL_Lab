"""Portable JSON metadata; filesystem paths are never persisted as absolute paths."""
import json as _json
import os
import re
from pathlib import Path

class JsonCodec:

    def __init__(self, workspace):
        self.workspace = Path(workspace)
        self.absolute_root = str(self.workspace.absolute())
        self.relative_root = str(self.workspace)

    def portable(self, value):
        if isinstance(value, dict):
            return {str(k): self.portable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.portable(v) for v in value]
        if isinstance(value, Path):
            return Path(os.path.relpath(value, self.workspace)).as_posix()
        if isinstance(value, str):
            if os.path.isabs(value):
                return Path(os.path.relpath(value, self.workspace)).as_posix()
            for root in (self.absolute_root, self.relative_root):
                if root not in ('', '.'):
                    value = value.replace(root + os.sep, '')
            value = re.sub(
                '(?<![:/\\w])(/[A-Za-z0-9_.~-]+(?:/[A-Za-z0-9_.~+%-]+)+)',
                lambda m: Path(os.path.relpath(m.group(1), self.workspace)).as_posix(),
                value,
            )
        return value

    def dump(self, value, stream, **kwargs):
        return _json.dump(self.portable(value), stream, **kwargs)

    def dumps(self, value, **kwargs):
        return _json.dumps(value, **kwargs)
    load = staticmethod(_json.load)
    loads = staticmethod(_json.loads)

def sanitize_saved_metadata(folder, workspace):
    """Normalize JSON configuration files written by Transformers/PEFT, not weights."""
    codec = JsonCodec(workspace)
    for path in Path(folder).rglob('*.json'):
        with path.open(encoding='utf-8') as stream:
            original = _json.load(stream)
        portable = codec.portable(original)
        if portable != original:
            with path.open('w', encoding='utf-8') as stream:
                _json.dump(portable, stream, indent=2, ensure_ascii=False)
