"""Execution-verified coding recipes.

Each recipe is a small, typed Python function with an independent
example-based contract.  A recipe is selected from a structural request
cue, rendered, and executed against its own examples before the caller
may use it.  A recipe that fails its contract is discarded, so this
module cannot turn a failed check into a verification claim.

The recipes are deliberately composable building blocks (sort, merge,
search, fold, predicate), not a replacement for the operation pipeline:
compound requests keep the normal semantic composer.
"""
from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from typing import Any


_COMPOUND = re.compile(
    r"\b(?:and|then|followed\s+by|after\s+that|as\s+well\s+as)\b|"
    r"(?:하고|한\s+뒤|한\s+후|그다음|각각|동시에|골라서|정렬해서)",
    re.IGNORECASE)
_ORDINAL_SUPERLATIVE = re.compile(
    r"\b(?:second|third|fourth|fifth|nth)\s+"
    r"(?:largest|smallest|highest|lowest|maximum|minimum|biggest|greatest)\b|"
    r"(?:두|세|네)\s*번째\s*(?:큰|작은|높은|낮은)",
    re.IGNORECASE)

# The composition primitives a recipe's meaning covers.  A recipe may
# answer a request only when every operation the semantic composer found
# is inside this set; otherwise the sentence asks for more than the
# recipe does ("the average, highest first" is not average_list()).
_RECIPE_COMPOSITION_OPS = {
    "average_list": {"calculate_average"},
    "sum_list": {"calculate_sum"},
    "find_max": {"find_max"},
    "find_min": {"find_min"},
    "filter_even": {"filter", "is_multiple_of"},
    "sort_descending": {"sort_descending"},
    "remove_duplicates": {"remove_duplicates"},
    "reverse_string": {"reverse_sequence"},
    "reverse_list": {"reverse_sequence"},
    "is_prime": {"is_prime"},
    "fibonacci": {"fibonacci"},
    "factorial": {"factorial"},
    "multiply_two_numbers": {"multiply"},
    "add_two_numbers": {"add"},
    "two_sum": {"add"},
    "binary_search": {"search"},
    "most_frequent_char": {"find_max", "count_frequency"},
}


def _c(expected: Any, *args: Any) -> tuple[tuple[Any, ...], Any]:
    return (tuple(args), expected)


@dataclass(frozen=True)
class CodingRecipe:
    name: str
    title: str
    patterns: tuple[str, ...]
    code: str
    tests: tuple[tuple[tuple[Any, ...], Any], ...]
    ir_type: str = ""

    def matches(self, text: str) -> bool:
        return any(re.search(pattern, text, re.IGNORECASE)
                   for pattern in self.patterns)

    def verify(self) -> dict:
        result = {
            "success": False,
            "behavioral": False,
            "source": "coding_recipe",
            "tests_passed": 0,
            "tests_total": len(self.tests),
            "errors": [],
        }
        try:
            ast.parse(self.code)
        except SyntaxError as exc:
            result["errors"].append(f"SyntaxError: {exc}")
            return result
        namespace: dict[str, Any] = {}
        try:
            exec(self.code, namespace)
        except Exception as exc:
            result["errors"].append(f"{type(exc).__name__}: {exc}")
            return result
        function = namespace.get(self.name)
        if not callable(function):
            result["errors"].append(f"missing function: {self.name}")
            return result
        for args, expected in self.tests:
            try:
                actual = function(*args)
            except Exception as exc:
                result["errors"].append(
                    f"{self.name}{args!r} raised {type(exc).__name__}: {exc}")
                return result
            if not _same(actual, expected):
                result["errors"].append(
                    f"{self.name}{args!r} returned {actual!r}, "
                    f"expected {expected!r}")
                return result
            result["tests_passed"] += 1
        result["success"] = True
        result["behavioral"] = True
        return result


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(actual, float) or isinstance(expected, float):
        try:
            return math.isclose(float(actual), float(expected),
                                rel_tol=1e-9, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same(a, b) for a, b in zip(actual, expected))
    if isinstance(actual, tuple) and isinstance(expected, tuple):
        return len(actual) == len(expected) and all(
            _same(a, b) for a, b in zip(actual, expected))
    return actual == expected


_RECIPES: tuple[CodingRecipe, ...] = (
    CodingRecipe(
        "group_anagrams", "group words into anagram classes",
        (r"\bgroup\s+anagrams?\b", r"\banagram\s+groups?\b",
         r"애너그램.*(?:그룹|묶|분류)"),
        "def group_anagrams(words: list[str]) -> list[list[str]]:\n"
        "    groups: dict[str, list[str]] = {}\n"
        "    order: list[str] = []\n"
        "    for word in words:\n"
        "        key = ''.join(sorted(word))\n"
        "        if key not in groups:\n"
        "            groups[key] = []\n"
        "            order.append(key)\n"
        "        groups[key].append(word)\n"
        "    return [groups[key] for key in order]\n",
        (
            _c([["eat", "tea", "ate"], ["tan", "nat"], ["bat"]],
               ["eat", "tea", "tan", "ate", "nat", "bat"]),
            _c([], []),
            _c([["a"]], ["a"]),
        ),
    ),
    CodingRecipe(
        "is_anagram", "check whether two words are anagrams",
        (r"\banagrams?\b", r"애너그램"),
        "def is_anagram(left: str, right: str) -> bool:\n"
        "    return sorted(left) == sorted(right)\n",
        (
            _c(True, "listen", "silent"),
            _c(False, "abc", "abd"),
            _c(True, "", ""),
        ),
    ),
    CodingRecipe(
        "two_sum", "return the indices of two values that sum to a target",
        (r"\btwo\s+sum\b", r"\bpair\s+sum\b",
         r"(?:two|2)\s+(?:numbers?|values?).*\b(?:add|sum)\b.*\btarget\b",
         r"(?:두|2)\s*(?:수|숫자).*(?:합|더).*(?:타겟|target)",
         r"(?:합이|합을).*(?:되는|만드는).*(?:두|2)"),
        "def two_sum(numbers: list, target: float) -> tuple | None:\n"
        "    seen: dict = {}\n"
        "    for index, value in enumerate(numbers):\n"
        "        need = target - value\n"
        "        if need in seen:\n"
        "            return (seen[need], index)\n"
        "        seen[value] = index\n"
        "    return None\n",
        (
            _c((0, 1), [2, 7, 11, 15], 9),
            _c((1, 2), [3, 2, 4], 6),
            _c((0, 1), [3, 3], 6),
            _c(None, [1, 2], 5),
        ),
    ),
    CodingRecipe(
        "merge_sorted_lists", "merge two sorted lists into one sorted list",
        (r"\bmerge\s+(?:two\s+)?(?:sorted\s+)?(?:lists?|arrays?)\b",
         r"(?:정렬된\s*)?(?:두\s*)?(?:리스트|배열).*(?:병합|합치)"),
        "def merge_sorted_lists(left: list, right: list) -> list:\n"
        "    merged: list = []\n"
        "    i = 0\n"
        "    j = 0\n"
        "    while i < len(left) and j < len(right):\n"
        "        if left[i] <= right[j]:\n"
        "            merged.append(left[i])\n"
        "            i += 1\n"
        "        else:\n"
        "            merged.append(right[j])\n"
        "            j += 1\n"
        "    return merged + left[i:] + right[j:]\n",
        (
            _c([1, 2, 3, 4], [1, 3], [2, 4]),
            _c([1, 2], [], [1, 2]),
            _c([1, 1, 1, 2], [1, 1], [1, 2]),
            _c([1, 2, 3, 5, 9], [1, 5, 9], [2, 3]),
        ),
    ),
    CodingRecipe(
        "flatten_list", "flatten arbitrarily nested lists",
        (r"\bflatten\b", r"평탄화", r"중첩.*(?:풀|펼|평탄)"),
        "def flatten_list(values: list) -> list:\n"
        "    flat: list = []\n"
        "    def visit(item) -> None:\n"
        "        if isinstance(item, (list, tuple)):\n"
        "            for child in item:\n"
        "                visit(child)\n"
        "        else:\n"
        "            flat.append(item)\n"
        "    visit(values)\n"
        "    return flat\n",
        (
            _c([1, 2, 3, 4], [1, [2, [3]], 4]),
            _c([1, 2], [[], [1], [[2]]]),
            _c([], []),
        ),
    ),
    CodingRecipe(
        "longest_common_prefix", "find the longest common prefix",
        (r"\blongest\s+common\s+prefix\b", r"공통.*접두"),
        "def longest_common_prefix(words: list[str]) -> str:\n"
        "    if not words:\n"
        "        return ''\n"
        "    prefix = words[0]\n"
        "    for word in words[1:]:\n"
        "        while not word.startswith(prefix):\n"
        "            prefix = prefix[:-1]\n"
        "            if not prefix:\n"
        "                return ''\n"
        "    return prefix\n",
        (
            _c("fl", ["flower", "flow", "flight"]),
            _c("", ["dog", "racecar", "car"]),
            _c("same", ["same"]),
        ),
    ),
    CodingRecipe(
        "count_vowels", "count vowels in a text",
        (r"\bcount\s+(?:the\s+)?vowels?\b", r"\bvowel\s+count\b",
         r"모음.*(?:개수|세|세기|count|얼마)"),
        "def count_vowels(text: str) -> int:\n"
        "    return sum(1 for char in text.lower() if char in 'aeiou')\n",
        (
            _c(2, "hello"),
            _c(5, "AEIOU"),
            _c(0, "rhythm"),
            _c(0, ""),
        ),
    ),
    CodingRecipe(
        "is_palindrome", "check whether a string reads the same backwards",
        (r"\bpalindrome\b", r"회문", r"팰린드롬"),
        "def is_palindrome(text: str) -> bool:\n"
        "    return text == text[::-1]\n",
        (
            _c(True, "racecar"),
            _c(False, "hello"),
            _c(True, "abba"),
            _c(True, ""),
        ),
    ),
    CodingRecipe(
        "reverse_string", "reverse a string",
        (r"\breverse\s+(?:a\s+)?(?:string|text)\b",
         r"\breverse\s+the\s+characters\b",
         r"문자열.*(?:뒤집|반전|거꾸로)"),
        "def reverse_string(text: str) -> str:\n"
        "    return text[::-1]\n",
        (
            _c("cba", "abc"),
            _c("dlrow olleh", "hello world"),
            _c("", ""),
        ),
    ),
    CodingRecipe(
        "reverse_list", "reverse a list",
        (r"\breverse\s+(?:a\s+)?(?:list|array)\b",
         r"(?:리스트|배열).*(?:뒤집|반전|거꾸로)"),
        "def reverse_list(values: list) -> list:\n"
        "    return list(reversed(values))\n",
        (
            _c([3, 2, 1], [1, 2, 3]),
            _c([], []),
        ),
    ),
    CodingRecipe(
        "binary_search", "search a value in a sorted list",
        (r"\bbinary\s+search\b", r"이진\s*탐색", r"이분\s*탐색"),
        "def binary_search(values: list, target) -> int:\n"
        "    low = 0\n"
        "    high = len(values) - 1\n"
        "    while low <= high:\n"
        "        middle = (low + high) // 2\n"
        "        if values[middle] == target:\n"
        "            return middle\n"
        "        if values[middle] < target:\n"
        "            low = middle + 1\n"
        "        else:\n"
        "            high = middle - 1\n"
        "    return -1\n",
        (
            _c(2, [1, 3, 5, 7], 5),
            _c(0, [1, 3, 5, 7], 1),
            _c(-1, [1, 3, 5, 7], 8),
            _c(-1, [], 1),
        ),
    ),
    CodingRecipe(
        "remove_duplicates", "remove duplicate values while keeping order",
        (r"\bremove\s+duplicat", r"\bdeduplicat", r"\bunique\b",
         r"중복.*(?:제거|없)"),
        "def remove_duplicates(values: list) -> list:\n"
        "    result: list = []\n"
        "    for value in values:\n"
        "        if value not in result:\n"
        "            result.append(value)\n"
        "    return result\n",
        (
            _c([3, 1, 2], [3, 1, 3, 2, 1]),
            _c([], []),
            _c(["a", "b"], ["a", "b", "a"]),
        ),
    ),
    CodingRecipe(
        "filter_even", "keep only even numbers",
        (r"\bfilter\s+even\b", r"\beven\s+numbers?\b", r"짝수"),
        "def filter_even(values: list) -> list:\n"
        "    return [value for value in values if value % 2 == 0]\n",
        (
            _c([2, 4], [1, 2, 3, 4]),
            _c([], []),
            _c([-2, 0], [-2, -1, 0]),
        ),
    ),
    CodingRecipe(
        "sort_descending", "sort values in descending order",
        (r"\bsort\b.*\b(?:descending|desc|reverse\s+order)\b",
         r"내림차순"),
        "def sort_descending(values: list) -> list:\n"
        "    return sorted(values, reverse=True)\n",
        (
            _c([3, 2, 1], [1, 3, 2]),
            _c([], []),
            _c(["b", "a"], ["b", "a"]),
        ),
    ),
    CodingRecipe(
        "sort_list", "sort values in ascending order",
        (r"\bsort\b.*\b(?:list|array|values|numbers)\b",
         r"\bsort\s+(?:a\s+)?(?:list|array)\b", r"정렬"),
        "def copy_list(values: list) -> list:\n"
        "    return list(values)\n"
        "\n"
        "def sort_list(values: list) -> list:\n"
        "    sorted_list = copy_list(values)\n"
        "    for i in range(len(sorted_list)):\n"
        "        for j in range(i + 1, len(sorted_list)):\n"
        "            if sorted_list[j] < sorted_list[i]:\n"
        "                sorted_list[i], sorted_list[j] = (\n"
        "                    sorted_list[j], sorted_list[i])\n"
        "    return sorted_list\n",
        (
            _c([1, 2, 3], [3, 1, 2]),
            _c([], []),
            _c(["a", "b"], ["b", "a"]),
        ),
        ir_type="List<int>",
    ),
    CodingRecipe(
        "average_list", "compute the arithmetic mean",
        (r"\baverage\b", r"\bmean\b", r"평균"),
        "def average_list(values: list) -> float:\n"
        "    if not values:\n"
        "        raise ValueError('empty list')\n"
        "    return sum(values) / len(values)\n",
        (
            _c(2.0, [1, 2, 3]),
            _c(3.0, [2, 4]),
        ),
    ),
    CodingRecipe(
        "sum_list", "sum the values in a list",
        (r"\bsum\s+(?:of\s+)?(?:a\s+)?(?:list|array|numbers|elements)\b",
         r"리스트.*(?:합|합계)", r"배열.*(?:합|합계)"),
        "def sum_list(values: list) -> float:\n"
        "    return sum(values)\n",
        (
            _c(6, [1, 2, 3]),
            _c(0, []),
            _c(4.0, [1.5, 2.5]),
        ),
    ),
    CodingRecipe(
        "find_min", "find the minimum value in a list",
        (r"\b(?:find|get|return|compute)\s+(?:the\s+)?(?:minimum|min|"
         r"smallest)\b",
         r"\b(?:minimum|min|smallest)\s+(?:element|value|number)\b",
         r"최솟값", r"최소값", r"가장\s*작"),
        "def find_min(values: list):\n"
        "    if not values:\n"
        "        raise ValueError('empty list')\n"
        "    result = values[0]\n"
        "    for value in values[1:]:\n"
        "        if value < result:\n"
        "            result = value\n"
        "    return result\n",
        (
            _c(1, [3, 1, 2]),
            _c(-9, [-5, -2, -9]),
        ),
    ),
    CodingRecipe(
        "find_max", "find the maximum value in a list",
        (r"\b(?:find|get|return|compute)\s+(?:the\s+)?(?:maximum|max|"
         r"largest|biggest)\b",
         r"\b(?:maximum|max|largest|biggest)\s+(?:element|value|number)\b",
         r"최댓값", r"최대값", r"가장\s*큰"),
        "def find_max(values: list):\n"
        "    if not values:\n"
        "        raise ValueError('empty list')\n"
        "    result = values[0]\n"
        "    for value in values[1:]:\n"
        "        if value > result:\n"
        "            result = value\n"
        "    return result\n",
        (
            _c(3, [3, 1, 2]),
            _c(-2, [-5, -2, -9]),
        ),
    ),
    CodingRecipe(
        "lcm", "compute the least common multiple",
        (r"\blcm\b", r"\bleast\s+common\s+multiple\b", r"최소공배수"),
        "def lcm(left: int, right: int) -> int:\n"
        "    if left == 0 or right == 0:\n"
        "        return 0\n"
        "    a, b = abs(left), abs(right)\n"
        "    while b:\n"
        "        a, b = b, a % b\n"
        "    return abs(left * right) // a\n",
        (
            _c(12, 4, 6),
            _c(0, 0, 5),
            _c(21, 7, 3),
        ),
    ),
    CodingRecipe(
        "gcd", "compute the greatest common divisor",
        (r"\bgcd\b", r"\bgreatest\s+common\s+divisor\b", r"최대공약수"),
        "def gcd(left: int, right: int) -> int:\n"
        "    a, b = abs(left), abs(right)\n"
        "    while b:\n"
        "        a, b = b, a % b\n"
        "    return a\n",
        (
            _c(6, 12, 18),
            _c(1, 7, 3),
            _c(5, 0, 5),
        ),
    ),
    CodingRecipe(
        "most_frequent_char", "find the most frequent character",
        (r"\bmost\s+frequent\s+char", r"빈도.*문자"),
        "def most_frequent_char(text: str) -> str:\n"
        "    if not text:\n"
        "        return ''\n"
        "    counts: dict[str, int] = {}\n"
        "    for char in text:\n"
        "        counts[char] = counts.get(char, 0) + 1\n"
        "    return max(counts, key=counts.get)\n",
        (
            _c("a", "abaca"),
            _c("a", "abc"),
            _c("", ""),
        ),
    ),
    CodingRecipe(
        "is_prime", "check whether a number is prime",
        (r"\bis[_\s]?prime\b", r"\bprime\s+number\b",
         r"\bcheck\s+(?:if\s+)?(?:a\s+)?number\s+is\s+prime\b",
         r"소수(?:인지|판별|확인|검사|여부)"),
        "def is_prime(n: int) -> bool:\n"
        "    if n < 2:\n"
        "        return False\n"
        "    if n % 2 == 0:\n"
        "        return n == 2\n"
        "    factor = 3\n"
        "    while factor * factor <= n:\n"
        "        if n % factor == 0:\n"
        "            return False\n"
        "        factor += 2\n"
        "    return True\n",
        (
            _c(True, 2),
            _c(True, 3),
            _c(False, 4),
            _c(False, 9),
            _c(True, 97),
            _c(False, 1),
        ),
    ),
    CodingRecipe(
        "fibonacci", "compute the n-th Fibonacci number",
        (r"\bfibonacci\b", r"피보나치"),
        "def fibonacci(n: int) -> int:\n"
        "    if n < 0:\n"
        "        raise ValueError('n must be non-negative')\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n):\n"
        "        a, b = b, a + b\n"
        "    return a\n",
        (
            _c(0, 0),
            _c(1, 1),
            _c(1, 2),
            _c(55, 10),
        ),
    ),
    CodingRecipe(
        "factorial", "compute the factorial",
        (r"\bfactorial\b", r"팩토리얼"),
        "def factorial(n: int) -> int:\n"
        "    if n < 0:\n"
        "        raise ValueError('n must be non-negative')\n"
        "    result = 1\n"
        "    for value in range(2, n + 1):\n"
        "        result *= value\n"
        "    return result\n",
        (
            _c(1, 0),
            _c(1, 1),
            _c(120, 5),
            _c(5040, 7),
        ),
    ),
    CodingRecipe(
        "multiply_two_numbers", "multiply two numbers",
        (r"\bmultiply\b", r"곱하"),
        "def multiply_two_numbers(a: float, b: float) -> float:\n"
        "    return a * b\n",
        (
            _c(12, 4, 3),
            _c(-10, -2, 5),
            _c(2.0, 0.5, 4),
        ),
    ),
    CodingRecipe(
        "add_two_numbers", "add two numbers",
        (r"\badd\s+(?:two|2)\s+(?:numbers?|values?|integers?)\b",
         r"\bsum\s+(?:of\s+)?(?:two|2)\s+(?:numbers?|values?)\b",
         r"두\s*(?:개의?\s*)?(?:숫자|수).*(?:더하|합)"),
        "def add_two_numbers(a: float, b: float) -> float:\n"
        "    return a + b\n",
        (
            _c(5, 2, 3),
            _c(0, -1, 1),
            _c(1.0, 0.5, 0.5),
        ),
    ),
)


def match_recipe(text: str) -> CodingRecipe | None:
    """Return the most specific recipe for a simple request, or None."""
    source = str(text or "")
    if (not source.strip() or _COMPOUND.search(source)
            or _ORDINAL_SUPERLATIVE.search(source)):
        return None
    for recipe in _RECIPES:
        if recipe.matches(source):
            return recipe
    return None


def select_verified_recipe(text: str,
                           composition_ops: list[str] | None = None) -> dict | None:
    """Select a recipe and return it only after its own tests pass.

    ``composition_ops`` is the operation set the semantic composer read
    from the same request.  A recipe whose meaning covers only part of
    that set is not a candidate: its tests would verify the recipe, not
    the sentence.  An empty list keeps the simple-cue path (the request
    carried no composed operation).
    """
    recipe = match_recipe(text)
    if recipe is None:
        return None
    if composition_ops:
        allowed = _RECIPE_COMPOSITION_OPS.get(recipe.name)
        if allowed is not None and not set(composition_ops).issubset(allowed):
            return None
    verification = recipe.verify()
    if not verification["success"]:
        return None
    return {
        "name": recipe.name,
        "title": recipe.title,
        "code": recipe.code,
        "ir_type": recipe.ir_type,
        "verification": verification,
        "test_cases": [
            {"args": list(args), "expected": expected}
            for args, expected in recipe.tests
        ],
    }


def recipe_names() -> list[str]:
    return [recipe.name for recipe in _RECIPES]
