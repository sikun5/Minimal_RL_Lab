"""Numeric readout: R1 primary and #### secondary."""
import re
_HASH_RE = re.compile('####\\s*([^\\n]+)')

def _normalize_number(s: str):
    """Parse a numeric answer string with thousands separators / sign / decimals.
    Returns float or None if not parseable as a single number."""
    t = s.strip()
    t = t.rstrip('.').strip()
    m = re.match('^([+-]?[0-9][0-9,]*(?:\\.[0-9]+)?|[+-]?\\.[0-9]+)', t)
    if not m:
        return None
    tok = m.group(1).replace(',', '')
    try:
        return float(tok)
    except ValueError:
        return None

def extract_answer(text: str):
    """Extract the value after the LAST '####' marker.
    Returns (parsed_float_or_None, format_valid_bool)."""
    matches = _HASH_RE.findall(text or '')
    if not matches:
        return (None, False)
    val = _normalize_number(matches[-1])
    return (val, val is not None)

def score_response(response_text: str, gold_answer: str, truncated: bool) -> dict:
    """Score one model response against the gold numeric answer.

    correct       = 1 iff extracted value parses and equals gold exactly (float equality)
    format_valid  = True iff a final '#### <parseable number>' exists
    truncated     = caller-reported finish reason == length (recorded honestly)
    A truncated response with no final answer scores 0; a truncated response that
    still contains a valid final answer keeps its score but keeps truncated=True.
    """
    pred, fmt_ok = extract_answer(response_text)
    if '####' in (gold_answer or ''):
        gold, _ = extract_answer(gold_answer)
    else:
        gold = _normalize_number(gold_answer)
    correct = 1 if pred is not None and gold is not None and (pred == gold) else 0
    return {
        'correct': correct,
        'extracted': pred,
        'gold': gold,
        'format_valid': bool(fmt_ok),
        'truncated': bool(truncated),
    }
_TAIL_WINDOW = 400

def _clean_number_text(s: str):
    """Strip currency symbols and trailing units/punctuation from a value string;
    return normalized float or None if not a single clean number."""
    t = s.strip()
    t = re.sub('^[€£¥$]+', '', t).strip()
    t = t.rstrip('.,;:!').strip()
    m = re.match(
        '^([+-]?[0-9][0-9,]*(?:\\.[0-9]+)?|[+-]?\\.[0-9]+)(?:\\s+[A-Za-z][A-Za-z ]{0,24})?[.!,;:]?$',
        t,
    )
    if not m:
        return None
    tok = m.group(1).replace(',', '')
    try:
        return float(tok)
    except ValueError:
        return None

def _read_declared_value(seg: str):
    """From the text segment right after an explicit marker, read the declared
    final value. Rules:
      - markdown emphasis / math fencing stripped
      - if the segment asserts a result ('... = N'), read N from after the last '='
        (rhs must be a single clean value, optional units)
      - otherwise the segment must contain EXACTLY ONE number (a bare declared
        value, optional units); multiple numbers without '=' = expression or
        multi-value fragment -> reject (never guess)
    """
    seg = seg.strip()
    seg = re.sub('[*_`~]+', '', seg)
    seg = re.sub('\\$+', '', seg)
    seg = seg.strip()
    if not seg:
        return None
    if '=' in seg:
        rhs = seg.rsplit('=', 1)[1].strip()
        return _clean_number_text(rhs)
    nums = list(re.finditer('[+-]?[0-9][0-9,]*(?:\\.[0-9]+)?', seg))
    if len(nums) != 1:
        return None
    m = nums[0]
    return _clean_number_text(m.group(0) + seg[m.end():])
_DECLARED_PATTERNS = [re.compile('final\\s*answer\\s*[*_`~\\s]*(?::|is)\\s*[*_`~\\s]*([^\\n]+)', re.IGNORECASE), re.compile('the\\s+answer\\s+is\\s*:?\\s*([^.\\n]{0,60})', re.IGNORECASE), re.compile('(?<![A-Za-z])answer\\s*:\\s*([^.\\n]{0,60})', re.IGNORECASE)]
_BOXED_RE = re.compile('\\\\boxed\\{([^{}]+)\\}')

def readout_answer_r1(text: str):
    """R1 readout. Returns (value_or_None, source_str_or_None).
    Sources: '####' (strict line), 'boxed', 'declared:<pattern>'. Conflicting
    distinct values among tail declarations -> (None, 'ambiguous')."""
    if not text:
        return (None, None)
    tail = text[-_TAIL_WINDOW:]
    values = []
    v, fmt = extract_answer(text)
    if fmt:
        values.append((v, '####'))
    for m in _BOXED_RE.finditer(tail):
        vv = _clean_number_text(m.group(1))
        if vv is not None:
            values.append((vv, 'boxed'))
    for pat in _DECLARED_PATTERNS:
        for m in pat.finditer(tail):
            vv = _read_declared_value(m.group(1))
            if vv is not None:
                values.append((vv, 'declared'))
    if not values:
        return (None, None)
    distinct = {round(v, 6) for v, _ in values}
    if len(distinct) > 1:
        return (None, 'ambiguous')
    return (values[0][0], values[0][1])

def score_response_r1(response_text: str, gold_answer: str, truncated: bool) -> dict:
    """Primary scoring under protocol answer_readout_r1.

    correct        = R1 numeric correctness (explicit marked final answer only)
    correct_strict = #### correctness (secondary metric)
    format_valid   = strict #### presence (never overridden by R1)
    truncated      = caller-reported finish == length cap
    readout_source = how the R1 value was read (####/boxed/declared/ambiguous/none)
    """
    strict_val, strict_fmt = extract_answer(response_text)
    gold_val, _ = extract_answer(gold_answer) if '####' in (gold_answer or '') else (_normalize_number(gold_answer), None)
    r1_val, source = readout_answer_r1(response_text)
    correct = 1 if r1_val is not None and gold_val is not None and (r1_val == gold_val) else 0
    correct_strict = 1 if strict_val is not None and gold_val is not None and (strict_val == gold_val) else 0
    return {
        'correct': correct,
        'correct_strict': correct_strict,
        'extracted': r1_val,
        'extracted_strict': strict_val,
        'gold': gold_val,
        'format_valid': bool(strict_fmt),
        'truncated': bool(truncated),
        'readout_source': source,
    }
