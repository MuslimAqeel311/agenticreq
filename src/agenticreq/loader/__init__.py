"""Spec loader: YAML with line tracking, ``extends`` deep-merge, normalisation, builder API."""

from agenticreq.loader.builder import SpecBuilder, spec_to_yaml, write_spec
from agenticreq.loader.yaml_loader import LoadedSpec, load_spec, load_spec_text

__all__ = ["LoadedSpec", "SpecBuilder", "load_spec", "load_spec_text", "spec_to_yaml", "write_spec"]
