"""Async GAM (Graph Attention Memory) integration for MemoryKernel.

This module provides async support for GAM-based retrieval, enabling
attention-based scoring with concurrent operations.

Example:
    >>> import asyncio
    >>> from memograph.core.kernel_gam_async import GAMAsyncKernel
    >>>
    >>> async def main():
    ...     kernel = GAMAsyncKernel(
    ...         vault_path="./vault",
    ...         enable_gam=True
    ...     )
    ...     await kernel.ingest_async()
    ...
    ...     # GAM-enhanced retrieval
    ...     results = await kernel.retrieve_nodes_async(
    ...         "python tips",
    ...         use_gam=True
    ...     )
    >>>
    >>> asyncio.run(main())
"""

import asyncio
import logging
from typing import Any

from memograph.core.access_tracker import AccessTracker
from memograph.core.gam_retriever import GAMRetriever
from memograph.core.gam_scorer import GAMScorer
from memograph.core.kernel_batch import BatchMemoryKernel
from memograph.core.node import MemoryNode

logger = logging.getLogger("memograph.gam_async")


class GAMAsyncKernel(BatchMemoryKernel):
    """Async kernel with GAM (Graph Attention Memory) support.

    This kernel extends BatchMemoryKernel with GAM-based retrieval,
    providing attention-based scoring that considers relationship
    strength, co-access patterns, and temporal decay.

    Features:
    - Async GAM-enhanced retrieval
    - Attention-based node scoring
    - Access pattern tracking
    - Temporal decay modeling
    - Concurrent GAM operations

    Example:
        >>> import asyncio
        >>>
        >>> async def example():
        ...     kernel = GAMAsyncKernel(
        ...         vault_path="./vault",
        ...         enable_gam=True,
        ...         gam_config={
        ...             'attention_weight': 0.4,
        ...             'recency_weight': 0.3,
        ...             'frequency_weight': 0.3
        ...         }
        ...     )
        ...     await kernel.ingest_async()
        ...
        ...     # GAM-enhanced retrieval
        ...     results = await kernel.retrieve_nodes_async(
        ...         "machine learning",
        ...         use_gam=True,
        ...         top_k=10
        ...     )
        ...
        ...     for node in results:
        ...         print(f"{node.title}: {node.salience:.3f}")
        >>>
        >>> asyncio.run(example())
    """

    def __init__(
        self,
        vault_path: str,
        enable_gam: bool = False,
        gam_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Initialize GAM async kernel.

        Args:
            vault_path: Path to vault directory
            enable_gam: Whether to enable GAM retrieval
            gam_config: GAM configuration dict with weights
            **kwargs: Additional arguments for BatchMemoryKernel
        """
        super().__init__(vault_path=vault_path, **kwargs)

        self.enable_gam = enable_gam
        self.gam_config = gam_config or {}

        # Initialize GAM components if enabled
        if self.enable_gam:
            from memograph.core.gam_scorer import GAMConfig

            self.access_tracker = AccessTracker()

            # Create GAM config from provided config dict
            if self.gam_config:
                gam_config_obj = GAMConfig(**self.gam_config)
            else:
                gam_config_obj = GAMConfig()

            self.gam_scorer = GAMScorer(config=gam_config_obj)
            self.gam_retriever = GAMRetriever(
                graph=self.graph,
                embedding_adapter=self.embedding_adapter
                if hasattr(self, "embedding_adapter")
                else None,
                use_gam=True,
                gam_config=gam_config_obj,
                access_tracker=self.access_tracker,
            )
            logger.info("GAM components initialized")
        else:
            self.access_tracker = None
            self.gam_scorer = None
            self.gam_retriever = None

    async def retrieve_nodes_async(
        self,
        query: str,
        tags: list[str] | None = None,
        depth: int = 2,
        top_k: int = 8,
        use_cache: bool = True,
        use_gam: bool = None,
        **kwargs,
    ) -> list[MemoryNode]:
        """Retrieve memory nodes with optional GAM enhancement.

        This method can use either standard retrieval or GAM-enhanced
        retrieval based on the use_gam parameter.

        Args:
            query: Search query
            tags: Filter by tags
            depth: Graph traversal depth
            top_k: Number of results to return
            use_cache: Whether to use query cache
            use_gam: Use GAM retrieval (defaults to self.enable_gam)
            **kwargs: Additional arguments

        Returns:
            List of relevant memory nodes

        Example:
            >>> async def search():
            ...     kernel = GAMAsyncKernel(
            ...         vault_path="./vault",
            ...         enable_gam=True
            ...     )
            ...     await kernel.ingest_async()
            ...
            ...     # Standard retrieval
            ...     standard = await kernel.retrieve_nodes_async(
            ...         "python",
            ...         use_gam=False
            ...     )
            ...
            ...     # GAM-enhanced retrieval
            ...     gam_results = await kernel.retrieve_nodes_async(
            ...         "python",
            ...         use_gam=True
            ...     )
        """
        # Determine whether to use GAM
        should_use_gam = use_gam if use_gam is not None else self.enable_gam

        if should_use_gam and self.gam_retriever:
            # Use GAM retrieval
            async with self._semaphore:
                # GAM retriever needs seed_ids as empty list if None
                seed_ids_for_gam = []

                results = await asyncio.to_thread(
                    self.gam_retriever.retrieve,
                    query,
                    seed_ids=seed_ids_for_gam,
                    tags=tags,
                    memory_type=None,
                    depth=depth,
                    top_k=top_k,
                    min_salience=0.0,
                )

                # Track access for GAM
                if self.access_tracker and results:
                    self.access_tracker.record_access(query, results)

                logger.debug(f"GAM retrieved {len(results)} nodes for: {query}")
                return results
        else:
            # Use standard retrieval
            return await super().retrieve_nodes_async(
                query, tags=tags, depth=depth, top_k=top_k, use_cache=use_cache, **kwargs
            )

    async def retrieve_batch_async(
        self,
        queries: list[str],
        tags: list[str] | None = None,
        depth: int = 2,
        top_k: int = 8,
        deduplicate: bool = True,
        show_progress: bool = True,
        use_gam: bool = None,
        **kwargs,
    ) -> dict[str, list[MemoryNode]]:
        """Batch retrieval with optional GAM enhancement.

        Args:
            queries: List of search queries
            tags: Filter by tags
            depth: Graph traversal depth
            top_k: Number of results per query
            deduplicate: Remove duplicate nodes
            show_progress: Show progress indicator
            use_gam: Use GAM retrieval (defaults to self.enable_gam)
            **kwargs: Additional arguments

        Returns:
            Dictionary mapping queries to result lists

        Example:
            >>> async def batch_search():
            ...     kernel = GAMAsyncKernel(
            ...         vault_path="./vault",
            ...         enable_gam=True
            ...     )
            ...     await kernel.ingest_async()
            ...
            ...     queries = ["python", "docker", "kubernetes"]
            ...     results = await kernel.retrieve_batch_async(
            ...         queries,
            ...         use_gam=True
            ...     )
        """
        # Determine whether to use GAM
        should_use_gam = use_gam if use_gam is not None else self.enable_gam

        if should_use_gam and self.gam_retriever:
            # Use GAM for each query
            if show_progress:
                try:
                    from rich.progress import Progress

                    with Progress() as progress:
                        task = progress.add_task("[cyan]GAM Retrieving...", total=len(queries))

                        async def retrieve_with_progress(q: str):
                            nodes = await self.retrieve_nodes_async(
                                q, tags=tags, depth=depth, top_k=top_k, use_gam=True
                            )
                            progress.update(task, advance=1)
                            return (q, nodes)

                        results_list = await asyncio.gather(
                            *[retrieve_with_progress(q) for q in queries]
                        )
                except ImportError:
                    results_list = await asyncio.gather(
                        *[self._retrieve_with_query_gam(q, tags, depth, top_k) for q in queries]
                    )
            else:
                results_list = await asyncio.gather(
                    *[self._retrieve_with_query_gam(q, tags, depth, top_k) for q in queries]
                )

            results = dict(results_list)

            if deduplicate:
                results = self._deduplicate_results(results)

            logger.info(f"GAM batch retrieved for {len(queries)} queries")
            return results
        else:
            # Use standard batch retrieval
            return await super().retrieve_batch_async(
                queries,
                tags=tags,
                depth=depth,
                top_k=top_k,
                deduplicate=deduplicate,
                show_progress=show_progress,
                **kwargs,
            )

    async def _retrieve_with_query_gam(
        self, query: str, tags: list[str] | None, depth: int, top_k: int
    ) -> tuple:
        """Helper for GAM retrieval with query tuple."""
        nodes = await self.retrieve_nodes_async(
            query, tags=tags, depth=depth, top_k=top_k, use_gam=True
        )
        return (query, nodes)

    async def remember_batch_async(
        self,
        memories: list[dict[str, Any]],
        show_progress: bool = False,
        batch_size: int = 10,
    ) -> list[str]:
        """Create multiple memories asynchronously with proper indexing.

        Args:
            memories: List of memory dictionaries
            show_progress: Whether to show progress bar
            batch_size: Number of concurrent operations per batch

        Returns:
            List of file paths for created memories
        """

        async def create_memory(memory: dict[str, Any]) -> str:
            """Helper to create a single memory."""
            # Build kwargs, only include memory_type if provided
            kwargs = {}
            if "memory_type" in memory and memory["memory_type"] is not None:
                kwargs["memory_type"] = memory["memory_type"]

            return await self.remember_async(
                memory.get("title", ""),
                memory.get("content", ""),
                tags=memory.get("tags"),
                **kwargs,
            )

        if show_progress:
            try:
                from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TextColumn("{task.completed}/{task.total}"),
                ) as progress:
                    task = progress.add_task("Creating memories...", total=len(memories))

                    results = []
                    for i in range(0, len(memories), batch_size):
                        batch = memories[i : i + batch_size]
                        batch_results = await asyncio.gather(*[create_memory(m) for m in batch])
                        results.extend(batch_results)
                        progress.update(task, advance=len(batch))

                    # Ensure filesystem sync and refresh index
                    await asyncio.sleep(0.1)
                    await self.ingest_async(force=True)
                    return results

            except ImportError:
                logger.warning("rich not installed, falling back to basic batch")

        # Fallback: process in batches
        results = []
        for i in range(0, len(memories), batch_size):
            batch = memories[i : i + batch_size]
            batch_results = await asyncio.gather(*[create_memory(m) for m in batch])
            results.extend(batch_results)

        # Ensure filesystem sync and refresh index
        await asyncio.sleep(0.1)
        await self.ingest_async(force=True)
        return results

    async def get_gam_stats_async(self) -> dict[str, Any]:
        """Get GAM statistics asynchronously.

        Returns:
            Dictionary with GAM statistics

        Example:
            >>> async def check_stats():
            ...     kernel = GAMAsyncKernel(
            ...         vault_path="./vault",
            ...         enable_gam=True
            ...     )
            ...     stats = await kernel.get_gam_stats_async()
            ...     print(f"Total accesses: {stats['total_accesses']}")
        """
        if not self.enable_gam or not self.access_tracker:
            return {"enabled": False}

        return await asyncio.to_thread(
            lambda: {
                "enabled": True,
                "total_accesses": len(self.access_tracker.access_history),
                "unique_nodes": len(self.access_tracker.node_access_counts),
                "config": self.gam_config,
            }
        )

    async def reset_gam_stats_async(self):
        """Reset GAM access statistics asynchronously.

        Example:
            >>> async def reset():
            ...     kernel = GAMAsyncKernel(
            ...         vault_path="./vault",
            ...         enable_gam=True
            ...     )
            ...     await kernel.reset_gam_stats_async()
        """
        if self.access_tracker:
            await asyncio.to_thread(self.access_tracker.reset)
            logger.info("GAM stats reset")


# Convenience function
async def create_gam_async_kernel(
    vault_path: str,
    enable_cache: bool = True,
    enable_gam: bool = True,
    max_concurrent: int = 10,
    gam_config: dict[str, Any] | None = None,
    **kwargs,
) -> GAMAsyncKernel:
    """Create and initialize a GAM async kernel.

    Args:
        vault_path: Path to vault directory
        enable_cache: Whether to enable caching
        enable_gam: Whether to enable GAM retrieval
        max_concurrent: Maximum concurrent operations
        gam_config: GAM configuration dict
        **kwargs: Additional arguments

    Returns:
        Initialized GAMAsyncKernel instance

    Example:
        >>> async def setup():
        ...     kernel = await create_gam_async_kernel(
        ...         "./vault",
        ...         gam_config={
        ...             'attention_weight': 0.4,
        ...             'recency_weight': 0.3,
        ...             'frequency_weight': 0.3
        ...         }
        ...     )
        ...     return kernel
    """
    kernel = GAMAsyncKernel(
        vault_path=vault_path,
        enable_cache=enable_cache,
        enable_gam=enable_gam,
        max_concurrent=max_concurrent,
        gam_config=gam_config,
        **kwargs,
    )
    await kernel.ingest_async()
    return kernel
