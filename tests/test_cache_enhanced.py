"""Tests for enhanced caching system.

This test suite validates:
- LRU cache functionality
- Disk cache persistence
- Multi-level cache coordination
- Query result caching
- Cache statistics tracking
- Performance improvements
"""

import time

import pytest

from memograph.storage.cache_enhanced import (
    DiskCache,
    LRUCache,
    MultiLevelCache,
    QueryResultCache,
)


class TestLRUCache:
    """Test in-memory LRU cache."""

    def test_basic_operations(self):
        """Test basic get/put operations."""
        cache = LRUCache(max_size=3)

        # Put items
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")

        # Get items
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") is None

    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = LRUCache(max_size=2)

        cache.put("key1", "value1")
        cache.put("key2", "value2")

        # Access key1 to make it most recently used
        cache.get("key1")

        # Add key3, should evict key2 (least recently used)
        cache.put("key3", "value3")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"

    def test_memory_limit(self):
        """Test memory-based eviction."""
        cache = LRUCache(max_size=100, max_memory_mb=1)  # 1MB limit

        # Add large items
        large_value = [0.1] * 10000  # ~80KB

        for i in range(20):
            cache.put(f"key{i}", large_value)

        # Should have evicted some items to stay under memory limit
        assert len(cache._cache) < 20

    def test_statistics(self):
        """Test cache statistics tracking."""
        cache = LRUCache(max_size=2)

        cache.put("key1", "value1")
        cache.put("key2", "value2")

        # Hits
        cache.get("key1")
        cache.get("key1")

        # Misses
        cache.get("key3")
        cache.get("key4")

        assert cache.stats.hits == 2
        assert cache.stats.misses == 2
        assert cache.stats.hit_rate == 0.5

        # Eviction
        cache.put("key3", "value3")
        assert cache.stats.evictions == 1

    def test_clear(self):
        """Test cache clearing."""
        cache = LRUCache(max_size=10)

        for i in range(5):
            cache.put(f"key{i}", f"value{i}")

        assert len(cache._cache) == 5

        cache.clear()

        assert len(cache._cache) == 0
        assert cache.stats.total_size_bytes == 0


class TestDiskCache:
    """Test disk-based persistent cache."""

    def test_basic_operations(self, tmp_path):
        """Test basic get/put operations."""
        cache = DiskCache(tmp_path)

        # Put items
        cache.put("key1", {"data": "value1"})
        cache.put("key2", {"data": "value2"})

        # Get items
        assert cache.get("key1") == {"data": "value1"}
        assert cache.get("key2") == {"data": "value2"}
        assert cache.get("key3") is None

    def test_persistence(self, tmp_path):
        """Test that cache persists across instances."""
        # Create cache and add items
        cache1 = DiskCache(tmp_path)
        cache1.put("key1", {"data": "value1"})
        cache1.put("key2", [1, 2, 3])

        # Create new cache instance
        cache2 = DiskCache(tmp_path)

        # Should be able to retrieve items
        assert cache2.get("key1") == {"data": "value1"}
        assert cache2.get("key2") == [1, 2, 3]

    def test_statistics(self, tmp_path):
        """Test disk cache statistics."""
        cache = DiskCache(tmp_path)

        cache.put("key1", {"data": "value1"})
        cache.put("key2", {"data": "value2"})

        # Hits
        cache.get("key1")
        cache.get("key1")

        # Misses
        cache.get("key3")

        assert cache.stats.hits == 2
        assert cache.stats.misses == 1
        assert cache.stats.disk_reads == 2
        assert cache.stats.disk_writes == 2

    def test_clear(self, tmp_path):
        """Test cache clearing."""
        cache = DiskCache(tmp_path)

        for i in range(5):
            cache.put(f"key{i}", {"data": f"value{i}"})

        # Check files exist
        cache_files = list(tmp_path.glob("*.json"))
        assert len(cache_files) > 0

        cache.clear()

        # Check files removed (except metadata)
        cache_files = [f for f in tmp_path.glob("*.json") if f.name != "_metadata.json"]
        assert len(cache_files) == 0


class TestMultiLevelCache:
    """Test multi-level cache coordination."""

    def test_memory_first(self, tmp_path):
        """Test that memory cache is checked first."""
        cache = MultiLevelCache(tmp_path)

        # Put in cache
        cache.put("key1", [0.1, 0.2, 0.3])

        # Should hit memory cache
        value = cache.get("key1")
        assert value == [0.1, 0.2, 0.3]
        assert cache.memory_cache.stats.hits == 1
        assert cache.disk_cache.stats.hits == 0

    def test_disk_fallback(self, tmp_path):
        """Test fallback to disk cache."""
        cache = MultiLevelCache(tmp_path, memory_max_size=2)

        # Add items
        cache.put("key1", [0.1])
        cache.put("key2", [0.2])
        cache.put("key3", [0.3])  # Evicts key1 from memory

        # Clear memory cache to force disk lookup
        cache.memory_cache.clear()

        # Should hit disk cache and promote to memory
        value = cache.get("key1")
        assert value == [0.1]
        assert cache.disk_cache.stats.hits == 1

        # Second access should hit memory
        value = cache.get("key1")
        assert cache.memory_cache.stats.hits == 1

    def test_combined_stats(self, tmp_path):
        """Test combined statistics."""
        cache = MultiLevelCache(tmp_path)

        cache.put("key1", [0.1])
        cache.put("key2", [0.2])

        # Memory hits
        cache.get("key1")
        cache.get("key2")

        # Clear memory, force disk hits
        cache.memory_cache.clear()
        cache.get("key1")

        # Miss
        cache.get("key3")

        stats = cache.get_stats()
        assert stats["combined"]["hits"] == 3  # 2 memory + 1 disk
        assert (
            stats["combined"]["misses"] == 3
        )  # 2 from memory after clear, 1 actual miss
        assert stats["combined"]["hit_rate"] == 0.5

    def test_clear_all(self, tmp_path):
        """Test clearing all caches."""
        cache = MultiLevelCache(tmp_path)

        for i in range(5):
            cache.put(f"key{i}", [float(i)])

        cache.clear()

        # All caches should be empty
        assert len(cache.memory_cache._cache) == 0
        assert cache.get("key0") is None


class TestQueryResultCache:
    """Test query result caching with TTL."""

    def test_basic_operations(self):
        """Test basic get/put operations."""
        cache = QueryResultCache(ttl_seconds=60)

        # Put results
        cache.put("python tips", ["result1", "result2"])
        cache.put("machine learning", ["result3"])

        # Get results
        assert cache.get("python tips") == ["result1", "result2"]
        assert cache.get("machine learning") == ["result3"]
        assert cache.get("unknown") is None

    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        cache = QueryResultCache(ttl_seconds=1)  # 1 second TTL

        cache.put("query1", ["result1"])

        # Should be available immediately
        assert cache.get("query1") == ["result1"]

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        assert cache.get("query1") is None

    def test_lru_eviction(self):
        """Test LRU eviction when at capacity."""
        cache = QueryResultCache(ttl_seconds=60, max_size=2)

        cache.put("query1", ["result1"])
        cache.put("query2", ["result2"])

        # Access query1 to make it most recently used
        cache.get("query1")

        # Add query3, should evict query2
        cache.put("query3", ["result3"])

        assert cache.get("query1") == ["result1"]
        assert cache.get("query2") is None  # Evicted
        assert cache.get("query3") == ["result3"]

    def test_statistics(self):
        """Test cache statistics."""
        cache = QueryResultCache(ttl_seconds=60)

        cache.put("query1", ["result1"])

        # Hits
        cache.get("query1")
        cache.get("query1")

        # Misses
        cache.get("query2")

        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 2 / 3


class TestCachePerformance:
    """Test cache performance improvements."""

    def test_embedding_cache_speedup(self, tmp_path):
        """Test that caching provides significant speedup."""
        cache = MultiLevelCache(tmp_path)

        # Simulate embedding generation (slow)
        def generate_embedding(text):
            time.sleep(0.01)  # Simulate 10ms generation time
            return [0.1] * 384

        # First access (cache miss)
        start = time.time()
        embedding = generate_embedding("test text")
        cache.put("embed:test", embedding)
        first_time = time.time() - start

        # Second access (cache hit)
        start = time.time()
        cached_embedding = cache.get("embed:test")
        second_time = time.time() - start

        # Cache should be much faster
        assert cached_embedding == embedding
        assert second_time < first_time / 10  # At least 10x faster

    def test_query_cache_speedup(self):
        """Test query cache speedup."""
        cache = QueryResultCache(ttl_seconds=60)

        # Simulate slow query
        def execute_query(query):
            time.sleep(0.01)  # Simulate 10ms query time
            return [f"result_{i}" for i in range(10)]

        # First query (cache miss)
        start = time.time()
        results = execute_query("python tips")
        cache.put("python tips", results)
        first_time = time.time() - start

        # Second query (cache hit)
        start = time.time()
        cached_results = cache.get("python tips")
        second_time = time.time() - start

        # Cache should be much faster
        assert cached_results == results
        assert second_time < first_time / 10  # At least 10x faster


@pytest.fixture
def temp_vault(tmp_path):
    """Create temporary vault for testing."""
    vault = tmp_path / "vault"
    vault.mkdir()
    return vault
