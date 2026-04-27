# MCP Registry Submission Guide

This guide explains how to submit MemoGraph to the official [MCP Registry](https://modelcontextprotocol.io/registry) and the [Smithery Marketplace](https://smithery.ai).

---

## Prerequisites

Before submitting, verify all checks pass:

```bash
python scripts/verify_mcp_registry.py
```

All 5 checks must pass:

| Check | Description |
|-------|-------------|
| Server Metadata | `server.json` valid and correctly configured |
| PyPI Package | `memograph` is published on PyPI |
| MCP Server | `python -m memograph.mcp.run_server --help` exits cleanly |
| GitHub Repository | Repository is public and accessible |
| Required Files | All required files present in the repo |

---

## server.json Configuration

The [`server.json`](../server.json) file at the repo root is the MCP Registry manifest. Key fields:

```json
{
  "$schema": "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json",
  "name": "io.github.Indhar01/memograph",
  "description": "...",
  "repository": {
    "url": "https://github.com/Indhar01/MemoGraph",
    "source": "github"
  },
  "version": "0.3.0",
  "packages": [
    {
      "registryType": "pypi",
      "identifier": "memograph",
      "version": "0.3.0",
      "transport": { "type": "stdio" }
    }
  ]
}
```

**Rules:**
- `name` must follow `io.github.<GitHubUsername>/<package>` exactly (case-sensitive)
- `description` must be ≤ 100 characters
- `$schema` must point to the 2025-12-11 schema URL
- `version` must match `pyproject.toml` version

Version sync is automated by the [`mcp-release.yml`](../.github/workflows/mcp-release.yml) workflow.

---

## Submitting to the MCP Registry

### Step 1 — Publish to PyPI

The package must be on PyPI before registry submission. Automated via the `publish.yml` workflow on GitHub release.

Verify at: https://pypi.org/project/memograph/

### Step 2 — Submit to MCP Registry

1. Visit: https://modelcontextprotocol.io/registry
2. Click **"Publish a Server"** or **"Submit Server"**
3. Authenticate with GitHub (`@Indhar01`)
4. Submit namespace: `io.github.Indhar01/memograph`
5. The registry will pull metadata from `server.json`

### Step 3 — Verify Registration

After submission, your server should appear at:
```
https://modelcontextprotocol.io/registry/io.github.Indhar01/memograph
```

---

## Smithery Marketplace (Optional)

Smithery provides an additional discovery channel for MCP servers.

- **Automated**: The `mcp-release.yml` workflow attempts to publish via `mcp-publisher` if available
- **Manual**: Visit https://smithery.ai and publish directly
- **Config**: `smithery.yaml` at the repo root contains the Smithery manifest

---

## Version Management

When releasing a new version:

1. Update `version` in [`pyproject.toml`](../pyproject.toml)
2. Push to `main` — the `mcp-release.yml` workflow automatically syncs `server.json`
3. Create a GitHub Release — triggers PyPI publish and MCP reminder issue
4. Complete the MCP Registry submission manually (registry requires human auth)

---

## MCP Server Configuration

### Running the Server

```bash
# With vault path
python -m memograph.mcp.run_server --vault ~/my-vault

# With environment variable
export MEMOGRAPH_VAULT=~/my-vault
python -m memograph.mcp.run_server
```

### Claude Desktop Configuration

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "memograph": {
      "command": "python",
      "args": ["-m", "memograph.mcp.run_server"],
      "env": {
        "MEMOGRAPH_VAULT": "/path/to/your/vault"
      }
    }
  }
}
```

### Available Tools

The MCP server exposes these tool categories:

| Category | Tools |
|----------|-------|
| Search | `search_vault`, `query_with_context` |
| Create | `create_memory`, `import_document`, `bulk_create` |
| Read | `get_memory`, `list_memories` |
| Update | `update_memory`, `batch_update` |
| Delete | `delete_memory`, `batch_delete` |
| Graph | `relate_memories`, `search_by_graph`, `find_path` |
| Vault | `get_vault_info`, `get_vault_stats` |
| Backup | `export_vault_tool`, `import_backup_tool`, `create_backup_tool` |
| AI | `suggest_tags`, `suggest_links`, `detect_knowledge_gaps`, `analyze_knowledge_base` |
| Autonomous | `auto_hook_query`, `auto_hook_response`, `configure_autonomous_mode` |

Run `list_available_tools` from any MCP client to see the full list with descriptions.

---

## Troubleshooting

### `server.json` description too long

The description field has a **100 character limit**. Check with:

```python
import json
data = json.load(open("server.json"))
print(len(data["description"]), "chars")
```

### MCP server import error on Linux

If `python -m memograph.mcp.run_server --help` fails:

```bash
# Install all dependencies
pip install -e ".[dev,all]"

# Check for import errors
python -c "from memograph.mcp.run_server import main; print('OK')"
```

### Namespace format

The `name` field in `server.json` must match exactly — it is **case-sensitive**:

```
io.github.Indhar01/memograph   ✅ correct
io.github.indhar01/memograph   ❌ wrong (lowercase)
```

---

## Related Files

| File | Purpose |
|------|---------|
| [`server.json`](../server.json) | MCP Registry manifest |
| [`smithery.yaml`](../smithery.yaml) | Smithery marketplace config |
| [`scripts/verify_mcp_registry.py`](../scripts/verify_mcp_registry.py) | Pre-submission verification script |
| [`.github/workflows/mcp-release.yml`](../.github/workflows/mcp-release.yml) | Release automation |
| [`memograph/mcp/server.py`](../memograph/mcp/server.py) | MCP server implementation |
| [`memograph/mcp/run_server.py`](../memograph/mcp/run_server.py) | Server entry point |
