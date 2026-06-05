"""Unit tests for the kernelbuilder Emitter.

The Emitter is a pure text builder (no MLIR / no NKI wheel), so these tests are
fast and dependency-free.
"""

import pytest

from nkigen.codegen.kernelbuilder.emitter import Emitter, _sanitize_identifier


# ---------------------------------------------------------------------------
# Indentation & line output
# ---------------------------------------------------------------------------

def test_flat_lines():
    em = Emitter()
    em.line("a = 1")
    em.line("b = 2")
    assert em.getvalue() == "a = 1\nb = 2\n"


def test_indent_context_manager():
    em = Emitter()
    em.line("def f():")
    with em.indent():
        em.line("x = 1")
        with em.indent():
            em.line("y = 2")
        em.line("z = 3")
    em.line("done")
    assert em.getvalue() == (
        "def f():\n"
        "    x = 1\n"
        "        y = 2\n"
        "    z = 3\n"
        "done\n"
    )


def test_push_pop_indent():
    em = Emitter()
    em.line("a")
    em.push_indent()
    em.line("b")
    em.pop_indent()
    em.line("c")
    assert em.getvalue() == "a\n    b\nc\n"


def test_pop_indent_below_zero_raises():
    em = Emitter()
    with pytest.raises(RuntimeError):
        em.pop_indent()


def test_blank_and_empty_line():
    em = Emitter()
    em.line("a")
    em.blank()
    em.line("b")
    # Trailing blanks are stripped, interior preserved.
    assert em.getvalue() == "a\n\nb\n"


def test_comment():
    em = Emitter()
    em.line("def f():")
    with em.indent():
        em.comment("a note")
        em.line("pass")
    assert "    # a note\n" in em.getvalue()


def test_custom_indent_unit():
    em = Emitter(indent_unit="  ")
    em.line("def f():")
    with em.indent():
        em.line("x")
    assert em.getvalue() == "def f():\n  x\n"


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

def test_imports_dedup_and_order():
    em = Emitter()
    em.add_import("from x import y")
    em.add_import("import a")
    em.add_import("import a")  # duplicate
    em.line("body")
    out = em.getvalue()
    # plain imports first (sorted), then from-imports, blank line, body.
    assert out == "import a\nfrom x import y\n\n\nbody\n"


def test_no_imports_no_leading_blanks():
    em = Emitter()
    em.line("body")
    assert em.getvalue() == "body\n"


# ---------------------------------------------------------------------------
# Name generation
# ---------------------------------------------------------------------------

def test_fresh_name_unique():
    em = Emitter()
    a = em.fresh_name("tile")
    b = em.fresh_name("tile")
    c = em.fresh_name("tile")
    assert a == "tile"
    assert len({a, b, c}) == 3
    assert b != a and c != a and b != c


def test_reserve_name_then_fresh_collides():
    em = Emitter()
    assert em.reserve_name("input_0") == "input_0"
    # A later fresh_name asking for the same base must not reuse it.
    other = em.fresh_name("input_0")
    assert other != "input_0"


def test_reserve_name_collision_falls_back():
    em = Emitter()
    first = em.reserve_name("x")
    second = em.reserve_name("x")
    assert first == "x"
    assert second != "x"


# ---------------------------------------------------------------------------
# Identifier sanitization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("simple", "simple"),
    ("with space", "with_space"),
    ("dots.and-dashes", "dots_and_dashes"),
    ("%arg0", "_arg0"),
    ("123abc", "_123abc"),
])
def test_sanitize_identifier(raw, expected):
    assert _sanitize_identifier(raw) == expected


def test_sanitize_keyword():
    # Python keywords get a trailing underscore.
    assert _sanitize_identifier("class") == "class_"
    assert _sanitize_identifier("for") == "for_"


def test_sanitize_empty():
    assert _sanitize_identifier("") == "_"


def test_fresh_name_sanitizes():
    em = Emitter()
    name = em.fresh_name("%tile.0")
    # Valid identifier, no illegal chars.
    assert name.isidentifier()
