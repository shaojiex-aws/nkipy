"""Code emitter: indentation, name generation, import collection, output.

The :class:`Emitter` is a pure text builder — it knows nothing about MLIR or
the kernel_builder API. The IR-walking layer (``__init__``/``emit_*``) feeds it
already-formatted statements and import requests; the Emitter is responsible
only for turning those into a correctly-indented, deduplicated Python source
string.

Keeping all formatting concerns here (and all API-string concerns in
``api.py``) means the IR walkers read as a sequence of intent — "open a loop",
"emit this assignment" — rather than string plumbing.
"""

from __future__ import annotations

import keyword
import re
from contextlib import contextmanager


_IDENT_RE = re.compile(r"[^0-9a-zA-Z_]")


def _sanitize_identifier(name: str) -> str:
    """Turn an arbitrary string into a valid, non-keyword Python identifier."""
    cleaned = _IDENT_RE.sub("_", name)
    if not cleaned or cleaned[0].isdigit():
        cleaned = "_" + cleaned
    if keyword.iskeyword(cleaned):
        cleaned = cleaned + "_"
    return cleaned


class Emitter:
    """Accumulates indented Python source lines plus the imports they need.

    Usage::

        em = Emitter()
        em.add_import("import nki.compiler.kernel_builder as nb")
        em.line("def kernel(in_0, out_0):")
        with em.indent():
            em.line("tile = nb.compiler.alloc(...)")
        source = em.getvalue()
    """

    def __init__(self, indent_unit: str = "    ") -> None:
        self._indent_unit = indent_unit
        self._level = 0
        self._lines: list[str] = []
        # Imports are kept as a set for dedup but emitted in sorted order so
        # output is deterministic regardless of walk order.
        self._imports: set[str] = set()
        # Names already handed out, to guarantee uniqueness.
        self._used_names: set[str] = set()
        # Stable per-prefix counters for generated names (par_0, par_1, ...).
        self._name_counters: dict[str, int] = {}

    # -- imports -----------------------------------------------------------

    def add_import(self, statement: str) -> None:
        """Register an import line (deduplicated). E.g. ``import numpy as np``."""
        self._imports.add(statement.rstrip())

    # -- indentation -------------------------------------------------------

    @contextmanager
    def indent(self):
        """Context manager that indents every :meth:`line` emitted within it."""
        self._level += 1
        try:
            yield
        finally:
            self._level -= 1

    def push_indent(self) -> None:
        self._level += 1

    def pop_indent(self) -> None:
        if self._level == 0:
            raise RuntimeError("pop_indent() below zero")
        self._level -= 1

    # -- line output -------------------------------------------------------

    def line(self, text: str = "") -> None:
        """Emit one line at the current indent. Empty string -> blank line."""
        if text == "":
            self._lines.append("")
        else:
            self._lines.append(self._indent_unit * self._level + text)

    def comment(self, text: str) -> None:
        """Emit a ``# ...`` comment at the current indent."""
        self.line(f"# {text}")

    def blank(self) -> None:
        self._lines.append("")

    # -- name generation ---------------------------------------------------

    def fresh_name(self, hint: str) -> str:
        """Return a unique, valid Python identifier derived from ``hint``.

        ``hint`` is sanitized; if it (or a numbered variant) is already taken,
        a numeric suffix is appended. The chosen name is recorded as used.
        """
        base = _sanitize_identifier(hint)
        if base not in self._used_names:
            self._used_names.add(base)
            return base
        # Disambiguate with a per-base counter.
        n = self._name_counters.get(base, 0)
        while True:
            candidate = f"{base}_{n}"
            n += 1
            if candidate not in self._used_names:
                self._name_counters[base] = n
                self._used_names.add(candidate)
                return candidate

    def reserve_name(self, name: str) -> str:
        """Record an exact name as used (e.g. a function arg). Returns it.

        If the sanitized name collides with an existing one, falls back to
        :meth:`fresh_name`.
        """
        sanitized = _sanitize_identifier(name)
        if sanitized in self._used_names:
            return self.fresh_name(sanitized)
        self._used_names.add(sanitized)
        return sanitized

    # -- assembly ----------------------------------------------------------

    def getvalue(self) -> str:
        """Assemble the final source: imports, blank line, then body.

        Imports are ordered conventionally: plain ``import X`` statements first
        (sorted), then ``from X import Y`` statements (sorted), so output is
        deterministic and reads naturally regardless of registration order.
        """
        parts: list[str] = []
        if self._imports:
            plain = sorted(i for i in self._imports if i.startswith("import "))
            froms = sorted(i for i in self._imports if not i.startswith("import "))
            parts.extend(plain)
            parts.extend(froms)
            parts.append("")
            parts.append("")
        parts.extend(self._lines)
        # Single trailing newline, no trailing blank lines.
        text = "\n".join(parts).rstrip("\n")
        return text + "\n"
