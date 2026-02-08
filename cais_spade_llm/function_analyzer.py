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
        """Convert a Python type annotation into a minimal JSON Schema fragment."""
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
        frontmatter, doc_body = FunctionAnalyzer._frontmatter_and_body(doc)
        block, *_ = doc_body.split("\n\n") if doc_body else [""]
        function_description = block.strip()

        # collect ":param x:" lines
        param_descriptions = {}
        for line in doc_body.split(":param ")[1:]:
            key, _, text = line.partition(": ")
            param_descriptions[key.strip()] = text.strip()

        param_meta = (
            frontmatter.get("params") if isinstance(frontmatter, dict) else None
        )
        if not isinstance(param_meta, dict):
            param_meta = {}

        # 3. build schema ----------------------------------------------
        properties = {}
        for p, hint in hints.items():
            schema = self._json_schema(hint)
            detail = param_meta.get(p)
            if isinstance(detail, dict):
                if detail.get("type"):
                    schema["type"] = detail["type"]
                desc = detail.get("description")
            elif isinstance(detail, str):
                desc = detail
            else:
                desc = None

            if not desc:
                desc = param_descriptions.get(p)

            if desc:
                schema["description"] = desc
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
    def _frontmatter_and_body(doc: str) -> tuple[dict, str]:
        """
        Given a docstring, return (frontmatter_dict, body_without_frontmatter).
        Frontmatter is expected to be enclosed in leading --- blocks.
        """
        stripped = doc.lstrip()
        if stripped.startswith("---"):
            try:
                _, rest = stripped.split("---", 1)
                yaml_block, remainder = rest.split("---", 1)
                meta = yaml.safe_load(yaml_block) or {}
                return meta, remainder.lstrip()
            except ValueError:
                pass
        return {}, doc

    @staticmethod
    def _extract_yaml_frontmatter(fn) -> dict:
        """
        Return the YAML dict from the first '--- … ---' block of a function's
        doc-string, or {} if no such block exists.
        """
        doc = inspect.getdoc(fn) or ""
        meta, _ = FunctionAnalyzer._frontmatter_and_body(doc)
        return meta

    @staticmethod
    def _schema_type_to_str(schema: dict) -> str:
        """Return a compact representation for JSON schema fragments."""
        type_name = schema.get("type") or ""
        if type_name == "array":
            inner = schema.get("items", {}).get("type", "string")
            return f"array[{inner}]"
        if not type_name and schema.get("items"):
            return f"array[{schema['items'].get('type', 'string')}]"
        return type_name or "string"

    @staticmethod
    def build_tools_catalogue(
        agents: Iterable[object],
        allowed: dict[str, set[str]] | None = None,
        outfile: Path | str = Path("tools.json"),
    ) -> None:

        analyzer = FunctionAnalyzer()
        rows = []
        for agent in agents:
            # prefer agent.agent_name; fallback to .name; else class name
            owner = getattr(agent, "agent_name", getattr(agent, "name", agent.__class__.__name__))
            fn_whitelist = allowed.get(owner, set()) if isinstance(allowed, dict) else {
                n for n in dir(agent) if not n.startswith("_") and callable(getattr(agent, n, None))
            }

            for fn_name in fn_whitelist:
                fn = getattr(agent, fn_name, None)
                if not callable(fn):
                    continue
                meta = FunctionAnalyzer._extract_yaml_frontmatter(fn)
                row = {"function": fn_name, "function_owner_agent": owner, **(meta or {})}

                try:
                    analyzed = analyzer.analyze_function(fn)
                except Exception:
                    rows.append(row)
                    continue

                # Prefer docstring description if YAML block omitted it
                if analyzed.get("description"):
                    row.setdefault("description", analyzed["description"])

                params_schema = (
                    analyzed.get("parameters", {}).get("properties", {}) or {}
                )
                if params_schema:
                    params_payload = {}
                    for param_name, schema in params_schema.items():
                        entry = {
                            "type": FunctionAnalyzer._schema_type_to_str(schema),
                        }
                        desc = schema.get("description")
                        if desc:
                            entry["description"] = desc
                        params_payload[param_name] = entry
                    row["params"] = params_payload

                rows.append(row)

        out_path = Path(outfile)                     # ← coerce to Path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {len(rows)} capability rows to {out_path}")
