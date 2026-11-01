"""Static extraction (declared_tools + schemas) from agent source code. Parses, never imports."""

from agenticreq.ast.extractor import ProjectExtractor, extract

__all__ = ["ProjectExtractor", "extract"]
