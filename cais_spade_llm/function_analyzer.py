#!/usr/bin/env python3
"""
Analysis of Python functions and entire classes using introspection
for creating descriptions usable with the OpenAI API
"""
from __future__ import annotations

import inspect
import json
import typing
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Union, Iterable

import yaml                                 # ← NEW
from collections import defaultdict

@dataclass
class VariableDescription:
    name: str
    type: str
    description: str

    def to_dict(self) -> dict:
        return {self.name: {"type": self.type, "description": self.description}}


class FunctionAnalyzer:
    openai_types = {
        float: "number",
        int  : "number",
        str  : "string",
        bool : "boolean",
    }

    # ------------------------------------------------------------------ #
    # NEW ─ convert typing-annotation → JSON-Schema fragment
    # ------------------------------------------------------------------ #
    @staticmethod
    def _json_schema(py_type) -> dict:
        origin = typing.get_origin(py_type)

        # list[...] / List[...]
        if origin in (list, typing.List):
            elem = typing.get_args(py_type)[0] if typing.get_args(py_type) else str
            return {
                "type": "array",
                "items": {"type": FunctionAnalyzer.openai_types.get(elem, "string")},
            }

        # tuple / set  → treat like list of strings
        if origin in (tuple, set) or py_type in (list, tuple, set):
            return {"type": "array", "items": {"type": "string"}}

        # dict / Dict[...] → opaque object
        if origin in (dict, typing.Dict) or py_type is dict:
            return {"type": "object"}

        # primitive
        return {"type": FunctionAnalyzer.openai_types.get(py_type, "string")}



    def analyze_function(self, fn) -> dict:
        """
        Analyzes a python function and returns a description compatible with the OpenAI API.
        
        Assumptions:
        * Docstring includes a function description and (optionally) parameter descriptions separated by 2 linebreaks.
        * Parameter descriptions are indicated by `:param x:`.
        * Handles functions with zero or more parameters.
        """
        name = fn.__name__

        # 1. type hints -------------------------------------------------
        hints = typing.get_type_hints(fn)
        hints.pop("return", None)

        required = [
            p for p, t in hints.items()
            if not (typing.get_origin(t) is Union and type(None) in typing.get_args(t))
        ]

        # 2. doc-string -------------------------------------------------
        doc = inspect.getdoc(fn) or ""
        block, *_ = doc.split("\n\n") if doc else [""]
        function_description = block.strip()

        # collect ":param x:" lines
        param_descriptions = {}
        for line in doc.split(":param ")[1:]:
            key, _, text = line.partition(": ")
            param_descriptions[key.strip()] = text.strip()

        # 3. build schema ----------------------------------------------
        properties = {}
        for p, hint in hints.items():
            if p not in param_descriptions:
                raise ValueError(f"Missing `:param {p}:` in docstring of `{name}`")
            schema = self._json_schema(hint)
            schema["description"] = param_descriptions[p]
            properties[p] = schema

        return {
            "name": name,
            "description": function_description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
    
    def analyze_class(self, class_: object) -> list:
        """
        Analyzes a python class and returns a description of all its non-private functions
            compatible with the OpenAI API
        """
        functions = [
            self.analyze_function(getattr(class_, func))
            for func in dir(class_)
            if callable(getattr(class_, func)) and not func.startswith("_")
        ]
        return functions

    # ────────────────────────────────────────────────────────────────
    # ADD-ON: build_tools_catalogue
    # ----------------------------------------------------------------
    @staticmethod
    def _extract_yaml_frontmatter(fn) -> dict:
        """
        Return the YAML dict from the first '--- … ---' block of a function's
        doc-string, or {} if no such block exists.
        """
        doc = inspect.getdoc(fn) or ""
        if doc.lstrip().startswith("---"):
            try:
                _, rest = doc.split("---", 1)
                yaml_block, _ = rest.split("---", 1)
                return yaml.safe_load(yaml_block) or {}
            except ValueError:
                pass  # malformed front-matter → ignore
        return {}

    @staticmethod
    def build_tools_catalogue(
        agents: Iterable[object],
        allowed: dict[str, set[str]] | None = None,
        outfile: Path = Path("tools.json"),
    ) -> None:

        rows = []
        for agent in agents:
            owner = getattr(agent, "name", agent.__class__.__name__)
            fn_whitelist = allowed.get(owner, set()) if allowed else set(dir(agent))

            for fn_name in fn_whitelist:
                if not hasattr(agent, fn_name):
                    continue
                fn = getattr(agent, fn_name)
                if not callable(fn):
                    continue

                meta = FunctionAnalyzer._extract_yaml_frontmatter(fn)
                rows.append({"function": fn_name, "function_owner_agent": owner, **meta})

        outfile.write_text(json.dumps(rows, indent=2))
        print(f"wrote {len(rows)} capability rows to {outfile}")