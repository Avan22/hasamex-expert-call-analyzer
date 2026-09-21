"""Core pipeline: parse -> retrieve -> generate -> verify -> (themes).

Importable and runnable without the UI, e.g.::

    from core.pipeline import Pipeline
    p = Pipeline()
    ans = p.ask("How long do purchase decisions take?")
"""
