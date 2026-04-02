#!/usr/bin/env python3
"""Verification script for MCP Registry submission.

This script validates that MemoGraph is ready for submission to the
official MCP Registry at https://modelcontextprotocol.io/registry
"""

import json
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError


def print_header(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_status(check: str, status: bool, message: str = "") -> None:
    """Print a check status."""
    icon = "✅" if status else "❌"
    print(f"{icon} {check}")
    if message:
        print(f"   → {message}")


def check_server_json() -> bool:
    """Verify server.json exists and is valid."""
    print_header("Checking server.json")

    server_json_path = Path("server.json")

    # Check if file exists
    if not server_json_path.exists():
        print_status("server.json exists", False, "File not found")
        return False
    print_status("server.json exists", True)

    # Check if valid JSON
    try:
        with open(server_json_path) as f:
            data = json.load(f)
        print_status("Valid JSON format", True)
    except json.JSONDecodeError as e:
        print_status("Valid JSON format", False, f"JSON error: {e}")
        return False

    # Check required fields
    required_fields = ["name", "displayName", "description", "location", "execution"]
    missing_fields = [field for field in required_fields if field not in data]

    if missing_fields:
        print_status(
            "Required fields present", False, f"Missing: {', '.join(missing_fields)}"
        )
        return False
    print_status("Required fields present", True)

    # Verify namespace format
    expected_namespace = "io.github.indhar01/memograph"
    actual_namespace = data.get("name", "")

    if actual_namespace != expected_namespace:
        print_status(
            "Correct namespace",
            False,
            f"Expected: {expected_namespace}, Got: {actual_namespace}",
        )
        return False
    print_status(f"Correct namespace: {expected_namespace}", True)

    # Verify location
    if data.get("location", {}).get("type") != "pypi":
        print_status("Location type", False, "Expected 'pypi'")
        return False
    print_status("Location type: pypi", True)

    if data.get("location", {}).get("package") != "memograph":
        print_status("Package name", False, "Expected 'memograph'")
        return False
    print_status("Package name: memograph", True)

    # Verify execution
    expected_command = "python"
    actual_command = data.get("execution", {}).get("command", "")

    if actual_command != expected_command:
        print_status(
            "Execution command",
            False,
            f"Expected: {expected_command}, Got: {actual_command}",
        )
        return False
    print_status("Execution command: python", True)

    expected_args = ["-m", "memograph.mcp.run_server"]
    actual_args = data.get("execution", {}).get("args", [])

    if actual_args != expected_args:
        print_status(
            "Execution args", False, f"Expected: {expected_args}, Got: {actual_args}"
        )
        return False
    print_status("Execution args correct", True)

    return True


def check_pypi_package() -> bool:
    """Verify package is published on PyPI."""
    print_header("Checking PyPI Package")

    try:
        url = "https://pypi.org/pypi/memograph/json"
        with urlopen(url, timeout=10) as response:
            data = json.loads(response.read())

        version = data["info"]["version"]
        print_status("Package accessible on PyPI", True, f"Version: {version}")
        return True

    except URLError as e:
        print_status("Package accessible on PyPI", False, f"Error: {e}")
        return False
    except Exception as e:
        print_status("Package accessible on PyPI", False, f"Unexpected error: {e}")
        return False


def check_mcp_server() -> bool:
    """Verify MCP server can be executed."""
    print_header("Checking MCP Server")

    try:
        # Test help command
        result = subprocess.run(
            ["python", "-m", "memograph.mcp.run_server", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            print_status("MCP server executable", True)
            return True
        else:
            print_status(
                "MCP server executable", False, f"Exit code: {result.returncode}"
            )
            if result.stderr:
                print(f"   Error: {result.stderr[:200]}")
            return False

    except subprocess.TimeoutExpired:
        print_status("MCP server executable", False, "Timeout")
        return False
    except FileNotFoundError:
        print_status("MCP server executable", False, "Python or module not found")
        return False
    except Exception as e:
        print_status("MCP server executable", False, f"Error: {e}")
        return False


def check_github_repo() -> bool:
    """Verify GitHub repository is accessible."""
    print_header("Checking GitHub Repository")

    try:
        url = "https://api.github.com/repos/Indhar01/MemoGraph"
        with urlopen(url, timeout=10) as response:
            data = json.loads(response.read())

        is_public = not data.get("private", True)

        if is_public:
            print_status("Repository is public", True)
            print_status(
                "Repository accessible", True, "https://github.com/Indhar01/MemoGraph"
            )
            return True
        else:
            print_status("Repository is public", False, "Repository is private")
            return False

    except URLError as e:
        print_status("Repository accessible", False, f"Error: {e}")
        return False
    except Exception as e:
        print_status("Repository accessible", False, f"Unexpected error: {e}")
        return False


def check_files() -> bool:
    """Verify required files exist."""
    print_header("Checking Required Files")

    required_files = [
        "server.json",
        "README.md",
        "docs/MCP_REGISTRY_GUIDE.md",
        "memograph/mcp/server.py",
        "memograph/mcp/run_server.py",
    ]

    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        exists = path.exists()
        print_status(f"{file_path}", exists)
        if not exists:
            all_exist = False

    return all_exist


def main() -> int:
    """Run all verification checks."""
    print("\n🔍 MemoGraph MCP Registry Verification")
    print("=====================================\n")
    print("This script verifies that MemoGraph is ready for submission to")
    print("the official MCP Registry at https://modelcontextprotocol.io/registry\n")

    checks = [
        ("Server Metadata", check_server_json),
        ("PyPI Package", check_pypi_package),
        ("MCP Server", check_mcp_server),
        ("GitHub Repository", check_github_repo),
        ("Required Files", check_files),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Unexpected error in {name}: {e}")
            results.append((name, False))

    # Summary
    print_header("Summary")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    print(f"\n✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}\n")

    if passed == total:
        print("🎉 All checks passed! You're ready to submit to MCP Registry!")
        print("\nNext steps:")
        print("1. Commit and push all changes to GitHub")
        print("2. Visit: https://modelcontextprotocol.io/registry")
        print("3. Click 'Publish a Server' or 'Submit Server'")
        print("4. Authenticate with GitHub (Indhar01)")
        print("5. Submit namespace: io.github.indhar01/memograph")
        print("\nSee docs/MCP_REGISTRY_GUIDE.md for detailed instructions.")
        return 0
    else:
        print("⚠️  Some checks failed. Please fix the issues above before submitting.")
        print("\nFor help, see:")
        print("- docs/MCP_REGISTRY_GUIDE.md")
        print("- README.md")
        return 1


if __name__ == "__main__":
    sys.exit(main())
