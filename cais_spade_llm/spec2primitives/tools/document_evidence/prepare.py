"""Prepare cached generic overviews for approved Spec2Primitives PDFs."""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path

from cais_spade_llm.spec2primitives.config import load_model_runtime_config
from cais_spade_llm.spec2primitives.tools.document_evidence.interpreter import (
    OpenAIDocumentVisionRuntime,
    prepare_document_overview,
)
from cais_spade_llm.spec2primitives.tools.exact_ref_resolver import (
    approved_document_refs,
    resolve_context_ref,
)

LOGGER = logging.getLogger(__name__)
_DEFAULT_CACHE_ROOT = Path(__file__).resolve().parents[2] / "contexts/source_cache"


async def prepare_documents(
    context_refs: Sequence[str],
    *,
    cache_root: Path = _DEFAULT_CACHE_ROOT,
) -> int:
    """Validate approved PDFs and prepare their generic overview caches.

    Args:
        context_refs: Exact approved document refs to prepare in order.
        cache_root: Root for generated content-addressed cache records.

    Returns:
        Zero after all requested documents have valid overview records.

    Raises:
        RuntimeError: If an approved ref cannot be served.
    """
    config = load_model_runtime_config().document_vlm
    vision_runtime = OpenAIDocumentVisionRuntime(config)
    for context_ref in context_refs:
        resolved = resolve_context_ref({"context_ref": context_ref})
        served_context = resolved.get("served_context")
        if not isinstance(served_context, dict):
            rejection = resolved.get("rejection")
            raise RuntimeError(
                f"Approved document could not be served: {context_ref}: {rejection}"
            )
        result = await prepare_document_overview(
            served_context=served_context,
            cache_root=cache_root,
            config=config,
            vision_runtime=vision_runtime,
        )
        LOGGER.info(
            "Document overview %s: %s (%s)",
            result.cache_status,
            context_ref,
            result.record_path,
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the preparation command and run the async cache workflow."""
    parser = argparse.ArgumentParser(
        description="Prepare ontology-neutral overviews for approved PDFs.",
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--all",
        action="store_true",
        help="prepare every PDF registered in approved_sources.json",
    )
    selection.add_argument(
        "--context-ref",
        help="prepare one exact approved PDF context_ref",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=_DEFAULT_CACHE_ROOT,
        help="override the generated overview cache root",
    )
    args = parser.parse_args(argv)
    context_refs = approved_document_refs() if args.all else (args.context_ref,)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return asyncio.run(prepare_documents(context_refs, cache_root=args.cache_root))


if __name__ == "__main__":
    raise SystemExit(main())
