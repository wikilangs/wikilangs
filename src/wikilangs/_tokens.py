"""Read token sequences from current and legacy model parquet cells."""
import json


def parse_tokens(raw):
    """Read a JSON array or sequence. Reject malformed cells one row at a time."""
    try:
        tokens = json.loads(raw) if isinstance(raw, str) else list(raw)
    except (TypeError, json.JSONDecodeError) as err:
        raise ValueError('Invalid token sequence') from err
    if not isinstance(tokens, (list, tuple)) or not all(isinstance(t, str) for t in tokens):
        raise ValueError('Expected an array of token strings')
    return tuple(tokens)
