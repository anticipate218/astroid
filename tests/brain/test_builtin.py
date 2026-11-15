# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/pylint-dev/astroid/blob/main/LICENSE
# Copyright (c) https://github.com/pylint-dev/astroid/blob/main/CONTRIBUTORS.txt

"""Unit Tests for the builtins brain module."""

import unittest

import pytest

from astroid import InferenceError, bases, nodes, objects, util
from astroid.builder import _extract_single_node, extract_node


class BuiltinsTest(unittest.TestCase):
    def test_infer_property(self):
        property_assign = _extract_single_node("""
        class Something:
            def getter():
                return 5
            asd = property(getter) #@
        """)
        inferred_property = next(iter(property_assign.value.infer()))
        self.assertTrue(isinstance(inferred_property, objects.Property))
        class_parent = property_assign.scope()
        self.assertIsInstance(class_parent, nodes.ClassDef)
        self.assertFalse(
            any(
                isinstance(def_, objects.Property)
                for def_list in class_parent.locals.values()
                for def_ in def_list
            )
        )
        self.assertTrue(hasattr(inferred_property, "args"))


class TestStringNodes:
    @pytest.mark.parametrize(
        "format_string",
        [
            pytest.param(
                """"My name is {}, I'm {}".format("Daniel", 12)""", id="empty-indexes"
            ),
            pytest.param(
                """"My name is {0}, I'm {1}".format("Daniel", 12)""",
                id="numbered-indexes",
            ),
            pytest.param(
                """"My name is {fname}, I'm {age}".format(fname = "Daniel", age = 12)""",
                id="named-indexes",
            ),
            pytest.param(
                """
        name = "Daniel"
        age = 12
        "My name is {0}, I'm {1}".format(name, age)
        """,
                id="numbered-indexes-from-positional",
            ),
            pytest.param(
                """
        name = "Daniel"
        age = 12
        "My name is {fname}, I'm {age}".format(fname = name, age = age)
        """,
                id="named-indexes-from-keyword",
            ),
            pytest.param(
                """
        name = "Daniel"
        age = 12
        "My name is {0}, I'm {age}".format(name, age = age)
        """,
                id="mixed-indexes-from-mixed",
            ),
            pytest.param(
                """
        string = "My name is {}, I'm {}"
        string.format("Daniel", 12)
        """,
                id="empty-indexes-on-variable",
            ),
        ],
    )
    def test_string_format(self, format_string: str) -> None:
        node: nodes.Call = _extract_single_node(format_string)
        inferred = next(node.infer())
        assert isinstance(inferred, nodes.Const)
        assert inferred.value == "My name is Daniel, I'm 12"

    @pytest.mark.parametrize(
        "format_string",
        [
            """
            from missing import Unknown
            name = Unknown
            age = 12
            "My name is {fname}, I'm {age}".format(fname = name, age = age)
            """,
            """
            from missing import Unknown
            age = 12
            "My name is {fname}, I'm {age}".format(fname = Unknown, age = age)
            """,
            """
            from missing import Unknown
            "My name is {}, I'm {}".format(Unknown, 12)
            """,
            """"I am {}".format()""",
            """
            "My name is {fname}, I'm {age}".format(fsname = "Daniel", age = 12)
            """,
            """
            "My unicode character is {:c}".format(None)
            """,
            """
            "My hex format is {:4x}".format('1')
            """,
            """
            daniel_age = 12
            "My name is {0.name}".format(daniel_age)
            """,
            pytest.param(""""{:>2000000000}".format("x")""", id="oversized-width"),
            pytest.param(""""{:.2000000000f}".format(1.0)""", id="oversized-precision"),
            pytest.param(
                """"{:>{}}".format("x", 2000000000)""", id="oversized-nested-width"
            ),
            pytest.param(""""{:c}".format(-1)""", id="char-negative"),
            pytest.param(""""{:c}".format(0x110000)""", id="char-too-large"),
        ],
    )
    def test_string_format_uninferable(self, format_string: str) -> None:
        node: nodes.Call = _extract_single_node(format_string)
        inferred = next(node.infer())
        assert inferred is util.Uninferable

    def test_string_format_with_specs(self) -> None:
        node: nodes.Call = _extract_single_node(
            """"My name is {}, I'm {:.2f}".format("Daniel", 12)"""
        )
        inferred = next(node.infer())
        assert isinstance(inferred, nodes.Const)
        assert inferred.value == "My name is Daniel, I'm 12.00"

    def test_string_format_in_dataclass_pylint8109(self) -> None:
        """https://github.com/pylint-dev/pylint/issues/8109"""
        function_def = extract_node("""
from dataclasses import dataclass

@dataclass
class Number:
    amount: int | float
    round: int = 2

    def __str__(self): #@
        number_format = "{:,.%sf}" % self.round
        return number_format.format(self.amount).rstrip("0").rstrip(".")
""")
        inferit = function_def.infer_call_result(function_def, context=None)
        assert [a.name for a in inferit] == [util.Uninferable]


@pytest.mark.parametrize(
    "arguments",
    ["*(1, 5, 2)", "*[1, 5, 2]", "1, *(5, 2)", "*(1,), 5, *(2,)", "*(), 1, 5, 2"],
)
def test_slice_with_starred_args(arguments: str) -> None:
    """Regression test for https://github.com/pylint-dev/astroid/issues/186.

    Unpacked arguments must retain the slice bounds, not just its type.
    """
    node = extract_node(f"slice({arguments})")
    inferred = node.inferred()
    assert len(inferred) == 1
    assert isinstance(inferred[0], nodes.Slice)
    assert inferred[0].pytype() == "builtins.slice"
    assert (
        inferred[0].lower.value,
        inferred[0].upper.value,
        inferred[0].step.value,
    ) == (
        1,
        5,
        2,
    )
    sliced = next(extract_node(f"[0, 1, 2, 3, 4, 5][slice({arguments})]").infer())
    assert isinstance(sliced, nodes.List)
    assert [element.value for element in sliced.elts] == [1, 3]


@pytest.mark.parametrize(
    "arguments",
    [
        '*(Target, "value")',
        '*[Target, "value", None]',
        'Target, *("value",)',
        '*(), Target, "value"',
        '*(Target, "missing", 42)',
        'Target, *("missing",), 42',
    ],
)
def test_getattr_with_starred_args(arguments: str) -> None:
    node = extract_node(f"""
        class Target:
            value = 42
        getattr({arguments})
    """)
    inferred = next(node.infer())
    assert isinstance(inferred, nodes.Const)
    assert inferred.value == 42


@pytest.mark.parametrize("attribute, expected", [("upper", True), ("missing", False)])
def test_hasattr_with_starred_args(attribute: str, expected: bool) -> None:
    node = extract_node(f'hasattr(*("a", "{attribute}"))')
    inferred = next(node.infer())
    assert isinstance(inferred, nodes.Const)
    assert inferred.value is expected


@pytest.mark.parametrize("arguments", ["1, *unknown", "1, *42", "*()", "*(1, 2, 3, 4)"])
def test_slice_with_invalid_starred_args(arguments: str) -> None:
    inferred = next(extract_node(f"slice({arguments})").infer())
    assert isinstance(inferred, bases.Instance)
    assert inferred.pytype() == "builtins.slice"


@pytest.mark.parametrize(
    "arguments",
    ['"a", "upper", *unknown', '"a", "upper", *42', "*()", '*("a", "upper", 1, 2)'],
)
def test_getattr_with_invalid_starred_args(arguments: str) -> None:
    with pytest.raises(InferenceError):
        next(extract_node(f"getattr({arguments})").infer())


@pytest.mark.parametrize(
    "arguments",
    ['"a", "upper", *unknown', '"a", "upper", *42', '*("a", "upper", None)'],
)
def test_hasattr_with_invalid_starred_args(arguments: str) -> None:
    assert next(extract_node(f"hasattr({arguments})").infer()) is util.Uninferable


@pytest.mark.parametrize("arguments", ["3", "*(3,)"])
def test_slice_single_argument_is_stop(arguments: str) -> None:
    inferred = next(extract_node(f"slice({arguments})").infer())
    assert isinstance(inferred, nodes.Slice)
    assert next(inferred.igetattr("start")).value is None
    assert next(inferred.igetattr("stop")).value == 3
    assert next(inferred.igetattr("step")).value is None
    sliced = next(extract_node(f"[0, 1, 2, 3, 4][slice({arguments})]").infer())
    assert [element.value for element in sliced.elts] == [0, 1, 2]


def test_slice_with_two_starred_args() -> None:
    inferred = next(extract_node("slice(*(1, 4))").infer())
    assert isinstance(inferred, nodes.Slice)
    assert next(inferred.igetattr("start")).value == 1
    assert next(inferred.igetattr("stop")).value == 4
    assert next(inferred.igetattr("step")).value is None


@pytest.mark.parametrize("assigned", [False, True])
def test_slice_does_not_assume_set_iteration_order(assigned: bool) -> None:
    code = "args = {5, 1, 2}\nslice(*args)" if assigned else "slice(*{5, 1, 2})"
    inferred = next(extract_node(code).infer())
    assert isinstance(inferred, bases.Instance)
    assert inferred.pytype() == "builtins.slice"


@pytest.mark.parametrize("assigned", [False, True])
def test_getattr_does_not_assume_set_iteration_order(assigned: bool) -> None:
    code = (
        'args = {"a", "upper"}\ngetattr(*args)'
        if assigned
        else 'getattr(*{"a", "upper"})'
    )
    with pytest.raises(InferenceError):
        next(extract_node(code).infer())


@pytest.mark.parametrize("assigned", [False, True])
def test_hasattr_does_not_assume_set_iteration_order(assigned: bool) -> None:
    code = (
        'args = {"a", "upper"}\nhasattr(*args)'
        if assigned
        else 'hasattr(*{"a", "upper"})'
    )
    assert next(extract_node(code).infer()) is util.Uninferable


@pytest.mark.parametrize("keywords", ["extra=True", "**unknown", '**{"extra": True}'])
def test_getattr_with_starred_args_rejects_keywords(keywords: str) -> None:
    with pytest.raises(InferenceError):
        next(extract_node(f'getattr(*("a", "upper"), {keywords})').infer())


@pytest.mark.parametrize("keywords", ["extra=True", "**unknown", '**{"extra": True}'])
def test_hasattr_with_starred_args_rejects_keywords(keywords: str) -> None:
    assert (
        next(extract_node(f'hasattr(*("a", "upper"), {keywords})').infer())
        is util.Uninferable
    )


@pytest.mark.parametrize("keywords", ["stop=5", "**unknown", '**{"stop": 5}'])
def test_slice_with_starred_args_rejects_keywords(keywords: str) -> None:
    inferred = next(extract_node(f"slice(*(1, 5, 2), {keywords})").infer())
    assert isinstance(inferred, bases.Instance)
    assert inferred.pytype() == "builtins.slice"


def test_builtin_starred_args_with_empty_keywords() -> None:
    method = next(extract_node('getattr(*("a", "upper"), **{})').infer())
    assert isinstance(method, bases.BoundMethod)
    assert next(extract_node('hasattr(*("a", "upper"), **{})').infer()).value is True
    inferred = next(extract_node("slice(*(1, 5, 2), **{})").infer())
    assert isinstance(inferred, nodes.Slice)
    assert (inferred.lower.value, inferred.upper.value, inferred.step.value) == (
        1,
        5,
        2,
    )
