"""
tests/test_diff_application.py — Multi-hunk diff application.

The previous ``CoderAgent._apply_diff`` reset its source cursor to line 0 on
every hunk and re-indexed each hunk by original line numbers after earlier
hunks had changed the buffer length. Single-hunk patches happened to survive
both mistakes, which is why the defect went unnoticed; anything with two or
more hunks was silently corrupted.

Run:
    python -m pytest tests/test_diff_application.py -v
"""

from __future__ import annotations

import pytest

from core.agents.agents import CoderAgent

ORIGINAL = """def alpha():
    return 1


def beta():
    return 2


def gamma():
    return 3
"""


def apply(diff: str) -> str:
    return CoderAgent._apply_diff(ORIGINAL, diff)


def test_single_hunk():
    diff = """```diff
--- a/m.py
+++ b/m.py
@@ -1,2 +1,2 @@
 def alpha():
-    return 1
+    return 100
```"""
    result = apply(diff)
    assert "return 100" in result
    assert "return 2" in result and "return 3" in result


def test_two_hunks_both_land_correctly():
    """The case the old implementation corrupted."""
    diff = """```diff
--- a/m.py
+++ b/m.py
@@ -1,2 +1,2 @@
 def alpha():
-    return 1
+    return 100
@@ -9,2 +9,2 @@
 def gamma():
-    return 3
+    return 300
```"""
    result = apply(diff)
    assert "return 100" in result, "first hunk lost"
    assert "return 300" in result, "second hunk landed at the wrong offset"
    assert "return 2" in result, "untouched function was clobbered"
    assert result.count("def alpha():") == 1
    assert result.count("def beta():") == 1
    assert result.count("def gamma():") == 1


def test_hunk_that_grows_the_file_shifts_later_hunks():
    """An insertion in hunk 1 moves every subsequent line by one."""
    diff = """```diff
--- a/m.py
+++ b/m.py
@@ -1,2 +1,3 @@
 def alpha():
+    # guard
     return 1
@@ -9,2 +10,2 @@
 def gamma():
-    return 3
+    return 300
```"""
    result = apply(diff)
    assert "# guard" in result
    assert "return 300" in result, "later hunk not shifted by the insertion"
    assert "return 1" in result
    assert "return 2" in result


def test_context_lines_come_from_the_hunk_not_the_file_head():
    """
    The old cursor bug copied context from line 0 onward, so a hunk late in
    the file would splice the file's opening lines into the middle.
    """
    diff = """```diff
--- a/m.py
+++ b/m.py
@@ -9,2 +9,2 @@
 def gamma():
-    return 3
+    return 300
```"""
    result = apply(diff)
    assert result.count("def alpha():") == 1, "file head duplicated into a later hunk"
    assert "return 300" in result
    assert result.startswith("def alpha():")


def test_out_of_range_hunk_returns_original_unchanged():
    diff = """```diff
--- a/m.py
+++ b/m.py
@@ -400,2 +400,2 @@
 def nowhere():
-    return 0
+    return 1
```"""
    assert apply(diff) == ORIGINAL


@pytest.mark.parametrize("garbage", ["", "not a diff at all", "```diff\n```"])
def test_unparseable_input_returns_original(garbage):
    assert apply(garbage) == ORIGINAL


def test_unfenced_diff_is_accepted():
    diff = """--- a/m.py
+++ b/m.py
@@ -1,2 +1,2 @@
 def alpha():
-    return 1
+    return 100
"""
    assert "return 100" in apply(diff)
