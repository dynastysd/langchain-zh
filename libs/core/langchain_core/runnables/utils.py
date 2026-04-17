"""`Runnable`（可运行单元）对象的工具代码。"""

from __future__ import annotations

import ast
import asyncio
import inspect
import sys
import textwrap

# 无法移动到 TYPE_CHECKING，因为 Mapping 和 Sequence 在运行时被
# RunnableConfigurableFields 需要
from collections.abc import Mapping, Sequence  # noqa: TC003
from functools import lru_cache
from inspect import signature
from itertools import groupby
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    Protocol,
    TypeGuard,
    TypeVar,
)

from typing_extensions import override

# 为保持向后兼容性，重新导出 create-model
from langchain_core.utils.pydantic import create_model  # noqa: F401

if TYPE_CHECKING:
    from collections.abc import (
        AsyncIterable,
        AsyncIterator,
        Awaitable,
        Callable,
        Coroutine,
        Iterable,
    )
    from contextvars import Context

    from langchain_core.runnables.schema import StreamEvent

Input = TypeVar("Input", contravariant=True)  # noqa: PLC0105
# 输出类型应实现 __concat__，如 str、list、dict
Output = TypeVar("Output", covariant=True)  # noqa: PLC0105


async def gated_coro(semaphore: asyncio.Semaphore, coro: Coroutine) -> Any:
    """使用信号量运行协程。

    参数:
        semaphore: 要使用的信号量。
        coro: 要运行的协程。

    返回:
        协程的结果。
    """
    async with semaphore:
        return await coro


async def gather_with_concurrency(n: int | None, *coros: Coroutine) -> list:
    """限制并发协程数量来收集协程。

    参数:
        n: 同时运行的协程数量。
        *coros: 要运行的协程。

    返回:
        协程的结果列表。
    """
    if n is None:
        return await asyncio.gather(*coros)

    semaphore = asyncio.Semaphore(n)

    return await asyncio.gather(*(gated_coro(semaphore, c) for c in coros))


def accepts_run_manager(callable: Callable[..., Any]) -> bool:  # noqa: A002
    """检查可调用对象是否接受 run_manager 参数。

    参数:
        callable: 要检查的可调用对象。

    返回:
        如果可调用对象接受 run_manager 参数则为 `True`，否则为 `False`。
    """
    try:
        return signature(callable).parameters.get("run_manager") is not None
    except ValueError:
        return False


def accepts_config(callable: Callable[..., Any]) -> bool:  # noqa: A002
    """检查可调用对象是否接受 config 参数。

    参数:
        callable: 要检查的可调用对象。

    返回:
        如果可调用对象接受 config 参数则为 `True`，否则为 `False`。
    """
    try:
        return signature(callable).parameters.get("config") is not None
    except ValueError:
        return False


def accepts_context(callable: Callable[..., Any]) -> bool:  # noqa: A002
    """检查可调用对象是否接受 context 参数。

    参数:
        callable: 要检查的可调用对象。

    返回:
        如果可调用对象接受 context 参数则为 `True`，否则为 `False`。
    """
    try:
        return signature(callable).parameters.get("context") is not None
    except ValueError:
        return False


def asyncio_accepts_context() -> bool:
    """检查 asyncio.create_task 是否接受 `context` 参数。

    返回:
        如果 `asyncio.create_task` 接受 context 参数则为 True，否则为 False。
    """
    return sys.version_info >= (3, 11)


_T = TypeVar("_T")


def coro_with_context(
    coro: Awaitable[_T], context: Context, *, create_task: bool = False
) -> Awaitable[_T]:
    """使用上下文等待协程。

    参数:
        coro: 要等待的协程。
        context: 要使用的上下文。
        create_task: 是否创建任务。

    返回:
        带上下文的协程。
    """
    if asyncio_accepts_context():
        return asyncio.create_task(coro, context=context)  # type: ignore[arg-type,call-arg,unused-ignore]
    if create_task:
        return asyncio.create_task(coro)  # type: ignore[arg-type]
    return coro


class IsLocalDict(ast.NodeVisitor):
    """检查名称是否是局部字典。"""

    def __init__(self, name: str, keys: set[str]) -> None:
        """初始化访问者。

        参数:
            name: 要检查的名称。
            keys: 要填充的键集合。
        """
        self.name = name
        self.keys = keys

    @override
    def visit_Subscript(self, node: ast.Subscript) -> None:
        """访问下标节点。

        参数:
            node: 要访问的节点。
        """
        if (
            isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Name)
            and node.value.id == self.name
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            # 找到了对目标名称的下标访问
            self.keys.add(node.slice.value)

    @override
    def visit_Call(self, node: ast.Call) -> None:
        """访问调用节点。

        参数:
            node: 要访问的节点。
        """
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == self.name
            and node.func.attr == "get"
            and len(node.args) in {1, 2}
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            # 找到了对目标名称的 .get() 调用
            self.keys.add(node.args[0].value)


class IsFunctionArgDict(ast.NodeVisitor):
    """检查函数的第一个参数是否是字典。"""

    def __init__(self) -> None:
        """创建 IsFunctionArgDict 访问者。"""
        self.keys: set[str] = set()

    @override
    def visit_Lambda(self, node: ast.Lambda) -> None:
        """访问 lambda 函数。

        参数:
            node: 要访问的节点。
        """
        if not node.args.args:
            return
        input_arg_name = node.args.args[0].arg
        IsLocalDict(input_arg_name, self.keys).visit(node.body)

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """访问函数定义。

        参数:
            node: 要访问的节点。
        """
        if not node.args.args:
            return
        input_arg_name = node.args.args[0].arg
        IsLocalDict(input_arg_name, self.keys).visit(node)

    @override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """访问异步函数定义。

        参数:
            node: 要访问的节点。
        """
        if not node.args.args:
            return
        input_arg_name = node.args.args[0].arg
        IsLocalDict(input_arg_name, self.keys).visit(node)


class NonLocals(ast.NodeVisitor):
    """获取访问的非局部变量。"""

    def __init__(self) -> None:
        """创建 NonLocals 访问者。"""
        self.loads: set[str] = set()
        self.stores: set[str] = set()

    @override
    def visit_Name(self, node: ast.Name) -> None:
        """访问名称节点。

        参数:
            node: 要访问的节点。
        """
        if isinstance(node.ctx, ast.Load):
            self.loads.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            self.stores.add(node.id)

    @override
    def visit_Attribute(self, node: ast.Attribute) -> None:
        """访问属性节点。

        参数:
            node: 要访问的节点。
        """
        if isinstance(node.ctx, ast.Load):
            parent = node.value
            attr_expr = node.attr
            while isinstance(parent, ast.Attribute):
                attr_expr = parent.attr + "." + attr_expr
                parent = parent.value
            if isinstance(parent, ast.Name):
                self.loads.add(parent.id + "." + attr_expr)
                self.loads.discard(parent.id)
            elif isinstance(parent, ast.Call):
                if isinstance(parent.func, ast.Name):
                    self.loads.add(parent.func.id)
                else:
                    parent = parent.func
                    attr_expr = ""
                    while isinstance(parent, ast.Attribute):
                        if attr_expr:
                            attr_expr = parent.attr + "." + attr_expr
                        else:
                            attr_expr = parent.attr
                        parent = parent.value
                    if isinstance(parent, ast.Name):
                        self.loads.add(parent.id + "." + attr_expr)


class FunctionNonLocals(ast.NodeVisitor):
    """获取函数访问的非局部变量。"""

    def __init__(self) -> None:
        """创建 FunctionNonLocals 访问者。"""
        self.nonlocals: set[str] = set()

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """访问函数定义。

        参数:
            node: 要访问的节点。
        """
        visitor = NonLocals()
        visitor.visit(node)
        self.nonlocals.update(visitor.loads - visitor.stores)

    @override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """访问异步函数定义。

        参数:
            node: 要访问的节点。
        """
        visitor = NonLocals()
        visitor.visit(node)
        self.nonlocals.update(visitor.loads - visitor.stores)

    @override
    def visit_Lambda(self, node: ast.Lambda) -> None:
        """访问 lambda 函数。

        参数:
            node: 要访问的节点。
        """
        visitor = NonLocals()
        visitor.visit(node)
        self.nonlocals.update(visitor.loads - visitor.stores)


class GetLambdaSource(ast.NodeVisitor):
    """获取 lambda 函数的源代码。"""

    def __init__(self) -> None:
        """初始化访问者。"""
        self.source: str | None = None
        self.count = 0

    @override
    def visit_Lambda(self, node: ast.Lambda) -> None:
        """访问 lambda 函数。

        参数:
            node: 要访问的节点。
        """
        self.count += 1
        if hasattr(ast, "unparse"):
            self.source = ast.unparse(node)


def get_function_first_arg_dict_keys(func: Callable) -> list[str] | None:
    """如果函数的第一个参数是字典，则获取其键。

    参数:
        func: 要检查的函数。

    返回:
        如果第一个参数是字典则返回其键，否则返回 None。
    """
    try:
        code = inspect.getsource(func)
        tree = ast.parse(textwrap.dedent(code))
        visitor = IsFunctionArgDict()
        visitor.visit(tree)
        return sorted(visitor.keys) if visitor.keys else None
    except (SyntaxError, TypeError, OSError, SystemError):
        return None


def get_lambda_source(func: Callable) -> str | None:
    """获取 lambda 函数的源代码。

    参数:
        func: 可以是 lambda 函数的可调用对象。

    返回:
        lambda 函数的源代码。
    """
    try:
        name = func.__name__ if func.__name__ != "<lambda>" else None
    except AttributeError:
        name = None
    try:
        code = inspect.getsource(func)
        tree = ast.parse(textwrap.dedent(code))
        visitor = GetLambdaSource()
        visitor.visit(tree)
    except (SyntaxError, TypeError, OSError, SystemError):
        return name
    return visitor.source if visitor.count == 1 else name


@lru_cache(maxsize=256)
def get_function_nonlocals(func: Callable) -> list[Any]:
    """获取函数访问的非局部变量。

    参数:
        func: 要检查的函数。

    返回:
        函数访问的非局部变量列表。
    """
    try:
        code = inspect.getsource(func)
        tree = ast.parse(textwrap.dedent(code))
        visitor = FunctionNonLocals()
        visitor.visit(tree)
        values: list[Any] = []
        closure = (
            inspect.getclosurevars(func.__wrapped__)
            if hasattr(func, "__wrapped__") and callable(func.__wrapped__)
            else inspect.getclosurevars(func)
        )
        candidates = {**closure.globals, **closure.nonlocals}
        for k, v in candidates.items():
            if k in visitor.nonlocals:
                values.append(v)
            for kk in visitor.nonlocals:
                if "." in kk and kk.startswith(k):
                    vv = v
                    for part in kk.split(".")[1:]:
                        if vv is None:
                            break
                        try:
                            vv = getattr(vv, part)
                        except AttributeError:
                            break
                    else:
                        values.append(vv)
    except (SyntaxError, TypeError, OSError, SystemError):
        return []

    return values


def indent_lines_after_first(text: str, prefix: str) -> str:
    """缩进第一行之后的所有文本行。

    参数:
        text: 要缩进的文本。
        prefix: 用于确定缩进空格数。

    返回:
        缩进后的文本。
    """
    n_spaces = len(prefix)
    spaces = " " * n_spaces
    lines = text.splitlines()
    return "\n".join([lines[0]] + [spaces + line for line in lines[1:]])


class AddableDict(dict[str, Any]):
    """可添加到另一个字典的字典（可添加字典）。"""

    def __add__(self, other: AddableDict) -> AddableDict:
        """将另一个字典添加到此字典。

        参数:
            other: 要添加的另一个字典。

        返回:
            两个字典相加的结果字典。
        """
        chunk = AddableDict(self)
        for key in other:
            if key not in chunk or chunk[key] is None:
                chunk[key] = other[key]
            elif other[key] is not None:
                try:
                    added = chunk[key] + other[key]
                except TypeError:
                    added = other[key]
                chunk[key] = added
        return chunk

    def __radd__(self, other: AddableDict) -> AddableDict:
        """将此字典添加到另一个字典。

        参数:
            other: 要被添加到的另一个字典。

        返回:
            两个字典相加的结果字典。
        """
        chunk = AddableDict(other)
        for key in self:
            if key not in chunk or chunk[key] is None:
                chunk[key] = self[key]
            elif self[key] is not None:
                try:
                    added = chunk[key] + self[key]
                except TypeError:
                    added = self[key]
                chunk[key] = added
        return chunk


_T_co = TypeVar("_T_co", covariant=True)
_T_contra = TypeVar("_T_contra", contravariant=True)


class SupportsAdd(Protocol[_T_contra, _T_co]):
    """支持加法的对象的协议。"""

    def __add__(self, x: _T_contra, /) -> _T_co:
        """将对象添加到另一个对象。"""


Addable = TypeVar("Addable", bound=SupportsAdd[Any, Any])


def add(addables: Iterable[Addable]) -> Addable | None:
    """将一系列可添加对象相加。

    参数:
        addables: 要添加的可添加对象。

    返回:
        添加可添加对象的结果。
    """
    final: Addable | None = None
    for chunk in addables:
        final = chunk if final is None else final + chunk
    return final


async def aadd(addables: AsyncIterable[Addable]) -> Addable | None:
    """异步将一系列可添加对象相加。

    参数:
        addables: 要添加的可添加对象。

    返回:
        添加可添加对象的结果。
    """
    final: Addable | None = None
    async for chunk in addables:
        final = chunk if final is None else final + chunk
    return final


class ConfigurableField(NamedTuple):
    """可由用户配置的字段（可配置字段）。"""

    id: str
    """字段的唯一标识符。"""

    name: str | None = None
    """字段的名称。"""

    description: str | None = None
    """字段的描述。"""

    annotation: Any | None = None
    """字段的注解。"""

    is_shared: bool = False
    """字段是否是共享的。"""

    @override
    def __hash__(self) -> int:
        return hash((self.id, self.annotation))


class ConfigurableFieldSingleOption(NamedTuple):
    """可由用户配置且具有默认值的字段（单选项可配置字段）。"""

    id: str
    """字段的唯一标识符。"""

    options: Mapping[str, Any]
    """字段的选项。"""

    default: str
    """字段的默认值。"""

    name: str | None = None
    """字段的名称。"""

    description: str | None = None
    """字段的描述。"""

    is_shared: bool = False
    """字段是否是共享的。"""

    @override
    def __hash__(self) -> int:
        return hash((self.id, tuple(self.options.keys()), self.default))


class ConfigurableFieldMultiOption(NamedTuple):
    """可由用户配置且具有多个默认值的字段（多选项可配置字段）。"""

    id: str
    """字段的唯一标识符。"""

    options: Mapping[str, Any]
    """字段的选项。"""

    default: Sequence[str]
    """字段的默认值列表。"""

    name: str | None = None
    """字段的名称。"""

    description: str | None = None
    """字段的描述。"""

    is_shared: bool = False
    """字段是否是共享的。"""

    @override
    def __hash__(self) -> int:
        return hash((self.id, tuple(self.options.keys()), tuple(self.default)))


AnyConfigurableField = (
    ConfigurableField | ConfigurableFieldSingleOption | ConfigurableFieldMultiOption
)


class ConfigurableFieldSpec(NamedTuple):
    """可由用户配置的字段的规范（可配置字段规范）。"""

    id: str
    """字段的唯一标识符。"""

    annotation: Any
    """字段的注解。"""

    name: str | None = None
    """字段的名称。"""

    description: str | None = None
    """字段的描述。"""

    default: Any = None
    """字段的默认值。"""

    is_shared: bool = False
    """字段是否是共享的。"""

    dependencies: list[str] | None = None
    """字段的依赖项。"""


def get_unique_config_specs(
    specs: Iterable[ConfigurableFieldSpec],
) -> list[ConfigurableFieldSpec]:
    """从配置规范序列中获取唯一的配置规范。

    参数:
        specs: 配置规范。

    返回:
        唯一的配置规范列表。

    异常:
        ValueError: 如果可运行单元序列包含冲突的配置规范。
    """
    grouped = groupby(
        sorted(specs, key=lambda s: (s.id, *(s.dependencies or []))), lambda s: s.id
    )
    unique: list[ConfigurableFieldSpec] = []
    for spec_id, dupes in grouped:
        first = next(dupes)
        others = list(dupes)
        if len(others) == 0 or all(o == first for o in others):
            unique.append(first)
        else:
            msg = (
                "RunnableSequence contains conflicting config specs"
                f"for {spec_id}: {[first, *others]}"
            )
            raise ValueError(msg)
    return unique


class _RootEventFilter:
    def __init__(
        self,
        *,
        include_names: Sequence[str] | None = None,
        include_types: Sequence[str] | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_names: Sequence[str] | None = None,
        exclude_types: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
    ) -> None:
        """在 astream_events 实现中过滤根事件的工具。

        这只是将参数绑定到命名空间，以在 astream_events 实现中节省一些输入。
        """
        self.include_names = include_names
        self.include_types = include_types
        self.include_tags = include_tags
        self.exclude_names = exclude_names
        self.exclude_types = exclude_types
        self.exclude_tags = exclude_tags

    def include_event(self, event: StreamEvent, root_type: str) -> bool:
        """确定是否包含事件。"""
        if (
            self.include_names is None
            and self.include_types is None
            and self.include_tags is None
        ):
            include = True
        else:
            include = False

        event_tags = event.get("tags") or []

        if self.include_names is not None:
            include = include or event["name"] in self.include_names
        if self.include_types is not None:
            include = include or root_type in self.include_types
        if self.include_tags is not None:
            include = include or any(tag in self.include_tags for tag in event_tags)

        if self.exclude_names is not None:
            include = include and event["name"] not in self.exclude_names
        if self.exclude_types is not None:
            include = include and root_type not in self.exclude_types
        if self.exclude_tags is not None:
            include = include and all(
                tag not in self.exclude_tags for tag in event_tags
            )

        return include


def is_async_generator(
    func: Any,
) -> TypeGuard[Callable[..., AsyncIterator]]:
    """检查函数是否是异步生成器。

    参数:
        func: 要检查的函数。

    返回:
        如果函数是异步生成器则为 `True`，否则为 `False`。
    """
    return inspect.isasyncgenfunction(func) or (
        hasattr(func, "__call__")  # noqa: B004
        and inspect.isasyncgenfunction(func.__call__)
    )


def is_async_callable(
    func: Any,
) -> TypeGuard[Callable[..., Awaitable]]:
    """检查函数是否是异步的。

    参数:
        func: 要检查的函数。

    返回:
        如果函数是异步的则为 `True`，否则为 `False`。
    """
    return asyncio.iscoroutinefunction(func) or (
        hasattr(func, "__call__")  # noqa: B004
        and asyncio.iscoroutinefunction(func.__call__)
    )
