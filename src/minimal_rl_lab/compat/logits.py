"""Consume real rollout logits in order without retaining a full vocabulary batch."""

class FreeingLogits:
    """Sequence of per-chunk real logits tensors; serves the exact
    slice pattern train() uses (logitss[i:i+lrbs]); releases each chunk
    reference when served; raises EXPLICIT errors on double consumption,
    out-of-order slices, or unsupported access.."""

    def __init__(self, chunks, lrbs):
        self._chunks = list(chunks)
        self._lrbs = lrbs
        self._nrows = sum((c.shape[0] for c in chunks))
        self._served = 0

    def _expected_rows(self, k):
        return min(self._lrbs, self._nrows - k * self._lrbs)

    def __getitem__(self, idx):
        if not isinstance(idx, slice):
            raise TypeError(f'FreeingLogits: only slice access is supported, got {type(idx)}')
        start, stop, step = idx.indices(self._nrows)
        k, rem = divmod(start, self._lrbs)
        exp = self._expected_rows(k)
        if step != 1 or rem != 0 or stop - start != exp or (start >= self._nrows):
            raise IndexError(f'FreeingLogits: unsupported slice {idx} (expect [k*lrbs : k*lrbs+rows({k})={exp}])')
        if k != self._served:
            raise IndexError(
                f'FreeingLogits: out-of-order/double consumption: requested chunk {k}, served counter at {self._served}',
            )
        self._served += 1
        c = self._chunks[k]
        self._chunks[k] = None
        return c

    @property
    def live(self):
        return sum((1 for c in self._chunks if c is not None))
