"""Batch operations extension for AsyncMemoryKernel.

This module provides advanced batch operations for concurrent memory
management, including batch retrieval, updates, and deletions.

Example:
    >>> import asyncio
    >>> from memograph.core.kernel_batch import BatchMemoryKernel
    >>>
    >>> async def main():
    ...     kernel = BatchMemoryKernel(vault_path="./vault")
    ...     await kernel.ingest_async()
    ...
    ...     # Batch retrieval
    ...     queries = ["python", "docker", "kubernetes"]
    ...     results = await kernel.retrieve_batch_async(queries)
    ...
    ...     # Batch update
    ...     updates = [
    ...         {"id": "mem1", "tags": ["updated", "python"]},
    ...         {"id": "mem2", "salience": 0.9}
    ...     ]
    ...     await kernel.update_batch_async(updates)
    >>>
    >>> asyncio.run(main())
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from memograph.core.kernel_async import AsyncMemoryKernel
from memograph.core.node import MemoryNode
from memograph.core.validation import ValidationError, validate_memory_id

logger = logging.getLogger("memograph.batch")


class BatchMemoryKernel(AsyncMemoryKernel):
    """Extended async kernel with advanced batch operations.

    This kernel extends AsyncMemoryKernel with batch retrieval, update,
    and delete operations, enabling efficient bulk memory management.

    Features:
    - Batch retrieval with query aggregation
    - Batch updates with validation
    - Batch deletions with safety checks
    - Progress tracking for all batch operations
    - Automatic result deduplication

    Example:
        >>> import asyncio
        >>>
        >>> async def example():
        ...     kernel = BatchMemoryKernel(vault_path="./vault")
        ...     await kernel.ingest_async()
        ...
        ...     # Retrieve multiple queries at once
        ...     queries = ["python tips", "docker setup", "git workflow"]
        ...     results = await kernel.retrieve_batch_async(
        ...         queries,
        ...         deduplicate=True
        ...     )
        ...
        ...     # Update multiple memories
        ...     updates = [
        ...         {"id": "mem1", "tags": ["updated"]},
        ...         {"id": "mem2", "salience": 0.9}
        ...     ]
        ...     updated = await kernel.update_batch_async(updates)
        >>>
        >>> asyncio.run(example())
    """

    async def retrieve_batch_async(
        self,
        queries: list[str],
        tags: list[str] | None = None,
        depth: int = 2,
        top_k: int = 8,
        deduplicate: bool = True,
        show_progress: bool = True,
        **kwargs,
    ) -> dict[str, list[MemoryNode]]:
        """Retrieve results for multiple queries concurrently.

        This method processes multiple queries in parallel and returns
        a dictionary mapping each query to its results. Optionally
        deduplicates nodes that appear in multiple result sets.

        Args:
            queries: List of search queries
            tags: Filter by tags (applied to all queries)
            depth: Graph traversal depth
            top_k: Number of results per query
            deduplicate: Remove duplicate nodes across queries
            show_progress: Show progress indicator
            **kwargs: Additional arguments for retrieve_nodes_async

        Returns:
            Dictionary mapping queries to their result lists

        Raises:
            ValidationError: If any query is invalid

        Example:
            >>> async def search_multiple():
            ...     kernel = BatchMemoryKernel(vault_path="./vault")
            ...     await kernel.ingest_async()
            ...
            ...     queries = [
            ...         "python best practices",
            ...         "docker deployment",
            ...         "kubernetes scaling"
            ...     ]
            ...
            ...     results = await kernel.retrieve_batch_async(
            ...         queries,
            ...         top_k=5,
            ...         deduplicate=True
            ...     )
            ...
            ...     for query, nodes in results.items():
            ...         print(f"{query}: {len(nodes)} results")
        """
        # Validate all queries first
        for query in queries:
            if not query or not isinstance(query, str):
                raise ValidationError(f"Invalid query: {query}")

        if show_progress:
            try:
                from rich.progress import Progress

                with Progress() as progress:
                    task = progress.add_task("[cyan]Retrieving...", total=len(queries))

                    async def retrieve_with_progress(
                        q: str,
                    ) -> tuple[str, list[MemoryNode]]:
                        nodes = await self.retrieve_nodes_async(
                            q, tags=tags, depth=depth, top_k=top_k, **kwargs
                        )
                        progress.update(task, advance=1)
                        return (q, nodes)

                    results_list = await asyncio.gather(
                        *[retrieve_with_progress(q) for q in queries]
                    )
            except ImportError:
                # Fallback without progress
                results_list = await asyncio.gather(
                    *[
                        self._retrieve_with_query(q, tags, depth, top_k, **kwargs)
                        for q in queries
                    ]
                )
        else:
            results_list = await asyncio.gather(
                *[
                    self._retrieve_with_query(q, tags, depth, top_k, **kwargs)
                    for q in queries
                ]
            )

        # Convert to dictionary
        results = dict(results_list)

        # Deduplicate if requested
        if deduplicate:
            results = self._deduplicate_results(results)

        logger.info(f"Retrieved batch results for {len(queries)} queries")
        return results

    async def _retrieve_with_query(
        self, query: str, tags: list[str] | None, depth: int, top_k: int, **kwargs
    ) -> tuple[str, list[MemoryNode]]:
        """Helper to retrieve with query tuple."""
        nodes = await self.retrieve_nodes_async(
            query, tags=tags, depth=depth, top_k=top_k, **kwargs
        )
        return (query, nodes)

    def _deduplicate_results(
        self, results: dict[str, list[MemoryNode]]
    ) -> dict[str, list[MemoryNode]]:
        """Remove duplicate nodes across result sets.

        Keeps the first occurrence of each node based on query order.
        """
        seen_ids = set()
        deduplicated = {}

        for query, nodes in results.items():
            unique_nodes = []
            for node in nodes:
                if node.id not in seen_ids:
                    unique_nodes.append(node)
                    seen_ids.add(node.id)
            deduplicated[query] = unique_nodes

        return deduplicated

    async def update_batch_async(
        self, updates: list[dict[str, Any]], show_progress: bool = True
    ) -> list[str]:
        """Update multiple memories concurrently.

        Each update dict must contain an 'id' field and any fields to update
        (e.g., 'tags', 'salience', 'content').

        Args:
            updates: List of update dicts with 'id' and fields to update
            show_progress: Show progress indicator

        Returns:
            List of updated memory IDs

        Raises:
            ValidationError: If any update is invalid
            FileNotFoundError: If memory file doesn't exist

        Example:
            >>> async def update_memories():
            ...     kernel = BatchMemoryKernel(vault_path="./vault")
            ...
            ...     updates = [
            ...         {
            ...             "id": "memory-1",
            ...             "tags": ["updated", "important"],
            ...             "salience": 0.9
            ...         },
            ...         {
            ...             "id": "memory-2",
            ...             "content": "Updated content"
            ...         }
            ...     ]
            ...
            ...     updated_ids = await kernel.update_batch_async(updates)
            ...     print(f"Updated {len(updated_ids)} memories")
        """
        # Validate all updates first
        for update in updates:
            if "id" not in update:
                raise ValidationError("Each update must contain 'id' field")
            validate_memory_id(update["id"])

        if show_progress:
            try:
                from rich.progress import Progress

                with Progress() as progress:
                    task = progress.add_task("[cyan]Updating...", total=len(updates))

                    async def update_with_progress(upd: dict[str, Any]) -> str:
                        memory_id = await self._update_single_async(upd)
                        progress.update(task, advance=1)
                        return memory_id

                    updated_ids = await asyncio.gather(
                        *[update_with_progress(u) for u in updates]
                    )
            except ImportError:
                # Fallback without progress
                updated_ids = await asyncio.gather(
                    *[self._update_single_async(u) for u in updates]
                )
        else:
            updated_ids = await asyncio.gather(
                *[self._update_single_async(u) for u in updates]
            )

        # Re-ingest after updates
        await self.ingest_async(force=True, show_progress=False)

        logger.info(f"Updated {len(updated_ids)} memories in batch")
        return updated_ids

    async def _update_single_async(self, update: dict[str, Any]) -> str:
        """Update a single memory asynchronously."""
        async with self._semaphore:
            memory_id = update["id"]

            # Get current node
            node = self.graph.get(memory_id)
            if not node:
                raise FileNotFoundError(f"Memory not found: {memory_id}")

            # Build updated content
            file_path = Path(node.source_path)

            # Read current content
            content = await asyncio.to_thread(file_path.read_text, encoding="utf-8")

            # Parse frontmatter
            import yaml

            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = yaml.safe_load(parts[1])
                    body = parts[2].strip()
                else:
                    frontmatter = {}
                    body = content
            else:
                frontmatter = {}
                body = content

            # Apply updates
            if "tags" in update:
                frontmatter["tags"] = update["tags"]
            if "salience" in update:
                frontmatter["salience"] = update["salience"]
            if "content" in update:
                body = update["content"]

            # Write updated content
            new_content = (
                f"---\n{yaml.dump(frontmatter, default_flow_style=False)}---\n{body}"
            )
            await asyncio.to_thread(file_path.write_text, new_content, encoding="utf-8")

            logger.debug(f"Updated memory: {memory_id}")
            return memory_id

    async def delete_batch_async(
        self, memory_ids: list[str], show_progress: bool = True
    ) -> list[str]:
        """Delete multiple memories concurrently.

        Args:
            memory_ids: List of memory IDs to delete
            show_progress: Show progress indicator

        Returns:
            List of deleted memory IDs

        Raises:
            ValidationError: If any ID is invalid
            FileNotFoundError: If memory file doesn't exist

        Example:
            >>> async def cleanup():
            ...     kernel = BatchMemoryKernel(vault_path="./vault")
            ...
            ...     # Delete old memories
            ...     old_ids = ["memory-1", "memory-2", "memory-3"]
            ...     deleted = await kernel.delete_batch_async(old_ids)
            ...     print(f"Deleted {len(deleted)} memories")
        """
        # Validate all IDs first
        for memory_id in memory_ids:
            validate_memory_id(memory_id)

        if show_progress:
            try:
                from rich.progress import Progress

                with Progress() as progress:
                    task = progress.add_task("[cyan]Deleting...", total=len(memory_ids))

                    async def delete_with_progress(mid: str) -> str:
                        deleted_id = await self._delete_single_async(mid)
                        progress.update(task, advance=1)
                        return deleted_id

                    deleted_ids = await asyncio.gather(
                        *[delete_with_progress(mid) for mid in memory_ids]
                    )
            except ImportError:
                # Fallback without progress
                deleted_ids = await asyncio.gather(
                    *[self._delete_single_async(mid) for mid in memory_ids]
                )
        else:
            deleted_ids = await asyncio.gather(
                *[self._delete_single_async(mid) for mid in memory_ids]
            )

        # Re-ingest after deletions
        await self.ingest_async(force=True, show_progress=False)

        logger.info(f"Deleted {len(deleted_ids)} memories in batch")
        return deleted_ids

    async def _delete_single_async(self, memory_id: str) -> str:
        """Delete a single memory asynchronously."""
        async with self._semaphore:
            # Get node
            node = self.graph.get(memory_id)
            if not node:
                raise FileNotFoundError(f"Memory not found: {memory_id}")

            # Delete file
            file_path = Path(node.source_path)
            await asyncio.to_thread(file_path.unlink)

            logger.debug(f"Deleted memory: {memory_id}")
            return memory_id

    async def aggregate_results_async(
        self, queries: list[str], aggregation: str = "union", **kwargs
    ) -> list[MemoryNode]:
        """Aggregate results from multiple queries.

        Args:
            queries: List of search queries
            aggregation: Aggregation method ('union', 'intersection')
            **kwargs: Additional arguments for retrieve_batch_async

        Returns:
            Aggregated list of memory nodes

        Example:
            >>> async def find_common():
            ...     kernel = BatchMemoryKernel(vault_path="./vault")
            ...     await kernel.ingest_async()
            ...
            ...     # Find memories matching all queries
            ...     common = await kernel.aggregate_results_async(
            ...         ["python", "testing", "best practices"],
            ...         aggregation="intersection"
            ...     )
            ...     print(f"Found {len(common)} common results")
        """
        results = await self.retrieve_batch_async(queries, deduplicate=False, **kwargs)

        if aggregation == "union":
            # Union: all unique nodes
            seen = set()
            union_nodes = []
            for nodes in results.values():
                for node in nodes:
                    if node.id not in seen:
                        union_nodes.append(node)
                        seen.add(node.id)
            return union_nodes

        elif aggregation == "intersection":
            # Intersection: nodes in all result sets
            if not results:
                return []

            # Start with first result set
            first_query = list(results.keys())[0]
            intersection_ids = {node.id for node in results[first_query]}

            # Intersect with remaining sets
            for _query, nodes in list(results.items())[1:]:
                intersection_ids &= {node.id for node in nodes}

            # Return nodes in intersection
            all_nodes = {node.id: node for nodes in results.values() for node in nodes}
            return [all_nodes[nid] for nid in intersection_ids]

        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")


# Convenience function
async def create_batch_kernel(
    vault_path: str, enable_cache: bool = True, max_concurrent: int = 10, **kwargs
) -> BatchMemoryKernel:
    """Create and initialize a batch memory kernel.

    Args:
        vault_path: Path to vault directory
        enable_cache: Whether to enable caching
        max_concurrent: Maximum concurrent operations
        **kwargs: Additional arguments for BatchMemoryKernel

    Returns:
        Initialized BatchMemoryKernel instance

    Example:
        >>> async def setup():
        ...     kernel = await create_batch_kernel("./vault")
        ...     return kernel
    """
    kernel = BatchMemoryKernel(
        vault_path=vault_path,
        enable_cache=enable_cache,
        max_concurrent=max_concurrent,
        **kwargs,
    )
    await kernel.ingest_async()
    return kernel
