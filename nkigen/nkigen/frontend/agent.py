"""Kernel agents for ``knob(...).use(agent)``.

An *agent* rewrites the kernel_builder **source** nkigen's kernelbuilder backend
emits for a region, and returns kernel_builder source to use instead. The source
is a complete, self-contained representation of the region — so the agent's job
is essentially source-in, source-out. What it does in between (rewrite, tune,
call an LLM) is up to the agent.

The agent receives an :class:`AgentContext`, not a bare string, so it also gets a
**workspace** — a per-site folder it can read and write freely (scratch, logs,
persisted "memory", candidate kernels). Under ``prog.tune(db=...)`` that folder
lives under the tuning DB and persists across runs; otherwise it is an ephemeral
temp dir. Keeping the input a context (rather than a bare ``str``) means future
fields — region key, shapes, target, a search deadline — can be added without
changing every agent's signature.

    class MyAgent:
        def transform(self, ctx: AgentContext) -> str:
            (ctx.workspace / "notes.md").write_text(...)   # persist memory
            return rewrite(ctx.source)                     # return kb source

    knob(x, y).use(MyAgent())

:class:`EchoAgent` is the identity starter example — it returns the source
unchanged, so the region compiles to exactly the kernel nkigen would have
generated. A useful baseline and a round-trip test of the whole
emit → agent → re-materialize → splice loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class AgentContext:
    """What an agent is given for one ``.use(agent)`` site."""

    source: str
    """The region's kernel_builder source (the agent's input IR)."""

    workspace: Path
    """A per-site folder the agent may read/write (memory, scratch, artifacts)."""

    key: str
    """Human-readable site label (the workspace folder name)."""


@runtime_checkable
class KernelAgent(Protocol):
    """Transforms a region's kernel_builder source: ``AgentContext -> str``."""

    def transform(self, ctx: AgentContext) -> str:
        ...


class EchoAgent:
    """Starter agent: returns the kernel_builder source unchanged.

    Exercises the full emit → agent → re-materialize → splice loop as a no-op
    transformation (the region compiles to the kernel nkigen emits for it).
    """

    def transform(self, ctx: AgentContext) -> str:
        return ctx.source
