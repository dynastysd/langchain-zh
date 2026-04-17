"""Runnable 对象的配置工具。"""

from __future__ import annotations

import asyncio

# 不能将 uuid 移到 TYPE_CHECKING 中，因为 RunnableConfig 在 Pydantic 模型中使用
import uuid  # noqa: TC003
import warnings
from collections.abc import Awaitable, Callable, Generator, Iterable, Iterator, Sequence
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import Context, ContextVar, Token, copy_context
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    ParamSpec,
    TypeVar,
    cast,
)

from typing_extensions import TypedDict

from langchain_core.callbacks.manager import AsyncCallbackManager, CallbackManager
from langchain_core.runnables.utils import (
    Input,
    Output,
    accepts_config,
    accepts_run_manager,
)

if TYPE_CHECKING:
    from langchain_core.callbacks.base import BaseCallbackManager, Callbacks
    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForChainRun,
        CallbackManagerForChainRun,
    )
else:
    # Pydantic 通过类型字典验证，但回调需要前向引用更新
    Callbacks = list | Any | None


class EmptyDict(TypedDict, total=False):
    """空字典类型。"""


class RunnableConfig(TypedDict, total=False):
    """可运行单元的配置。

    !!! note 自定义值

        `TypedDict` 特意设置 `total=False` 以：

        - 允许创建部分配置并通过 `merge_configs` 合并
        - 支持配置从父可运行单元传播到子可运行单元（通过
            `var_child_runnable_config`，这是一个 `ContextVar`，会自动传递
            配置到调用栈下方而无需显式参数传递），配置是合并而非替换

        !!! example

            ```python
            # 父级设置标签
            chain.invoke(input, config={"tags": ["parent"]})
            # 子级自动继承并可添加：
            # ensure_config({"tags": ["child"]}) -> {"tags": ["parent", "child"]}
            ```
    """

    tags: list[str]
    """此次调用及所有子调用的标签（例如链调用 LLM）。

    可用于过滤调用。
    """

    metadata: dict[str, Any]
    """此次调用及所有子调用的元数据（例如链调用 LLM）。

    键应该是字符串，值应该是 JSON 可序列化的。
    """

    callbacks: Callbacks
    """此次调用及所有子调用的回调（例如链调用 LLM）。

    标签会传递给所有回调，元数据会传递给 handle*Start 回调。
    """

    run_name: str
    """此次调用的追踪器运行名称。

    默认为类名。
    """

    max_concurrency: int | None
    """最大并行调用数。

    如果未提供，默认为 `ThreadPoolExecutor` 的默认值。
    """

    recursion_limit: int
    """调用可以递归的最大次数。

    如果未提供，默认为 `25`。
    """

    configurable: dict[str, Any]
    """此前在此 Runnable 或子 Runnable 上通过 `configurable_fields` 或
    `configurable_alternatives` 设置为可配置的属性的运行时值。

    有关已配置属性的描述，请查看 `output_schema`。
    """

    run_id: uuid.UUID | None
    """此次调用的追踪器运行唯一标识符。

    如果未提供，将生成一个新的 UUID。
    """


CONFIG_KEYS = [
    "tags",
    "metadata",
    "callbacks",
    "run_name",
    "max_concurrency",
    "recursion_limit",
    "configurable",
    "run_id",
]

COPIABLE_KEYS = [
    "tags",
    "metadata",
    "callbacks",
    "configurable",
]


# 用户应使用带有上下文对象的 `context` API（不会被追踪）
CONFIGURABLE_TO_TRACING_METADATA_EXCLUDED_KEYS = frozenset(("api_key",))


def _get_langsmith_inheritable_metadata_from_config(
    config: RunnableConfig,
) -> dict[str, Any] | None:
    """从配置中获取仅 LangSmith 可继承的元数据默认值。"""
    configurable = config.get("configurable") or {}
    metadata = {
        key: value
        for key, value in configurable.items()
        if not key.startswith("__")
        and isinstance(value, (str, int, float, bool))
        and key not in config.get("metadata", {})
        and key not in CONFIGURABLE_TO_TRACING_METADATA_EXCLUDED_KEYS
    }
    return metadata or None


DEFAULT_RECURSION_LIMIT = 25


var_child_runnable_config: ContextVar[RunnableConfig | None] = ContextVar(
    "child_runnable_config", default=None
)


# 此模块在 langgraph 中被导入和使用，所以不要破坏它。
def _set_config_context(
    config: RunnableConfig,
) -> tuple[Token[RunnableConfig | None], dict[str, Any] | None]:
    """设置子 Runnable 配置 + 追踪上下文。

    参数:
        config: 要设置的配置。

    返回:
        重置配置的令牌和先前的追踪上下文。
    """
    # 延迟导入以避免在模块级别导入 langsmith（约 132ms）。
    from langsmith.run_helpers import (  # noqa: PLC0415
        _set_tracing_context,
        get_tracing_context,
    )

    from langchain_core.tracers.langchain import LangChainTracer  # noqa: PLC0415

    config_token = var_child_runnable_config.set(config)
    current_context = None
    if (
        (callbacks := config.get("callbacks"))
        and (
            parent_run_id := getattr(callbacks, "parent_run_id", None)
        )  # Is callback manager
        and (
            tracer := next(
                (
                    handler
                    for handler in getattr(callbacks, "handlers", [])
                    if isinstance(handler, LangChainTracer)
                ),
                None,
            )
        )
        and (run := tracer.run_map.get(str(parent_run_id)))
    ):
        current_context = get_tracing_context()
        _set_tracing_context({"parent": run})
    return config_token, current_context


@contextmanager
def set_config_context(config: RunnableConfig) -> Generator[Context, None, None]:
    """设置子 Runnable 配置 + 追踪上下文。

    参数:
        config: 要设置的配置。

    产出:
        配置上下文。
    """
    # 延迟导入以避免在模块级别导入 langsmith（约 132ms）。
    from langsmith.run_helpers import _set_tracing_context  # noqa: PLC0415

    ctx = copy_context()
    config_token, _ = ctx.run(_set_config_context, config)
    try:
        yield ctx
    finally:
        ctx.run(var_child_runnable_config.reset, config_token)
        ctx.run(
            _set_tracing_context,
            {
                "parent": None,
                "project_name": None,
                "tags": None,
                "metadata": None,
                "enabled": None,
                "client": None,
            },
        )


def ensure_config(config: RunnableConfig | None = None) -> RunnableConfig:
    """确保配置是一个包含所有键的字典。

    参数:
        config: 要确保的配置。

    返回:
        确保后的配置。
    """
    empty = RunnableConfig(
        tags=[],
        metadata={},
        callbacks=None,
        recursion_limit=DEFAULT_RECURSION_LIMIT,
        configurable={},
    )
    if var_config := var_child_runnable_config.get():
        empty.update(
            cast(
                "RunnableConfig",
                {
                    k: v.copy() if k in COPIABLE_KEYS else v  # type: ignore[attr-defined]
                    for k, v in var_config.items()
                    if v is not None
                },
            )
        )
    if config is not None:
        empty.update(
            cast(
                "RunnableConfig",
                {
                    k: v.copy() if k in COPIABLE_KEYS else v  # type: ignore[attr-defined]
                    for k, v in config.items()
                    if v is not None and k in CONFIG_KEYS
                },
            )
        )
    if config is not None:
        for k, v in config.items():
            if k not in CONFIG_KEYS and v is not None:
                empty["configurable"][k] = v
    for configurable_key in ("model", "checkpoint_ns"):
        if (
            isinstance(
                configurable_value := empty.get("configurable", {}).get(
                    configurable_key
                ),
                str,
            )
            and configurable_key not in empty["metadata"]
        ):
            empty["metadata"][configurable_key] = configurable_value
    return empty


def get_config_list(
    config: RunnableConfig | Sequence[RunnableConfig] | None, length: int
) -> list[RunnableConfig]:
    """从单个配置或配置列表获取配置列表。

    这对于子类重写 batch() 或 abatch() 很有用。

    参数:
        config: 配置或配置列表。
        length: 列表的长度。

    返回:
        配置列表。

    异常:
        ValueError: 如果列表长度与输入长度不相等。
    """
    if length < 0:
        msg = f"length must be >= 0, but got {length}"
        raise ValueError(msg)
    if isinstance(config, Sequence) and len(config) != length:
        msg = (
            f"config must be a list of the same length as inputs, "
            f"but got {len(config)} configs for {length} inputs"
        )
        raise ValueError(msg)

    if isinstance(config, Sequence):
        return list(map(ensure_config, config))
    if length > 1 and isinstance(config, dict) and config.get("run_id") is not None:
        warnings.warn(
            "Provided run_id be used only for the first element of the batch.",
            category=RuntimeWarning,
            stacklevel=3,
        )
        subsequent = cast(
            "RunnableConfig", {k: v for k, v in config.items() if k != "run_id"}
        )
        return [
            ensure_config(subsequent) if i else ensure_config(config)
            for i in range(length)
        ]
    return [ensure_config(config) for i in range(length)]


def patch_config(
    config: RunnableConfig | None,
    *,
    callbacks: BaseCallbackManager | None = None,
    recursion_limit: int | None = None,
    max_concurrency: int | None = None,
    run_name: str | None = None,
    configurable: dict[str, Any] | None = None,
) -> RunnableConfig:
    """用新值修补配置。

    参数:
        config: 要修补的配置。
        callbacks: 要设置的回调。
        recursion_limit: 要设置的递归限制。
        max_concurrency: 要设置的最大并发数。
        run_name: 要设置的运行名称。
        configurable: 要设置的可配置项。

    返回:
        修补后的配置。
    """
    config = ensure_config(config)
    if callbacks is not None:
        # 如果我们要替换回调，需要取消设置 run_name
        # 因为它应该只适用于与原始回调相同的运行
        config["callbacks"] = callbacks
        if "run_name" in config:
            del config["run_name"]
        if "run_id" in config:
            del config["run_id"]
    if recursion_limit is not None:
        config["recursion_limit"] = recursion_limit
    if max_concurrency is not None:
        config["max_concurrency"] = max_concurrency
    if run_name is not None:
        config["run_name"] = run_name
    if configurable is not None:
        config["configurable"] = {**config.get("configurable", {}), **configurable}
    return config


def merge_configs(*configs: RunnableConfig | None) -> RunnableConfig:
    """将多个配置合并为一个。

    参数:
        *configs: 要合并的配置。

    返回:
        合并后的配置。
    """
    base: RunnableConfig = {}
    # 即使键不是字面量，这也是正确的
    # 因为两个字典是相同类型
    for config in (ensure_config(c) for c in configs if c is not None):
        for key in config:
            if key == "metadata":
                base["metadata"] = {
                    **base.get("metadata", {}),
                    **(config.get("metadata") or {}),
                }
            elif key == "tags":
                base["tags"] = sorted(
                    set(base.get("tags", []) + (config.get("tags") or [])),
                )
            elif key == "configurable":
                base["configurable"] = {
                    **base.get("configurable", {}),
                    **(config.get("configurable") or {}),
                }
            elif key == "callbacks":
                base_callbacks = base.get("callbacks")
                these_callbacks = config["callbacks"]
                # 回调可以是 None、list[handler] 或 manager
                # 所以合并两个回调值有 6 种情况
                if isinstance(these_callbacks, list):
                    if base_callbacks is None:
                        base["callbacks"] = these_callbacks.copy()
                    elif isinstance(base_callbacks, list):
                        base["callbacks"] = base_callbacks + these_callbacks
                    else:
                        # base_callbacks 是一个 manager
                        mngr = base_callbacks.copy()
                        for callback in these_callbacks:
                            mngr.add_handler(callback, inherit=True)
                        base["callbacks"] = mngr
                elif these_callbacks is not None:
                    # these_callbacks 是一个 manager
                    if base_callbacks is None:
                        base["callbacks"] = these_callbacks.copy()
                    elif isinstance(base_callbacks, list):
                        mngr = these_callbacks.copy()
                        for callback in base_callbacks:
                            mngr.add_handler(callback, inherit=True)
                        base["callbacks"] = mngr
                    else:
                        # base_callbacks 也是一个 manager
                        base["callbacks"] = base_callbacks.merge(these_callbacks)
            elif key == "recursion_limit":
                if config["recursion_limit"] != DEFAULT_RECURSION_LIMIT:
                    base["recursion_limit"] = config["recursion_limit"]
            elif key in COPIABLE_KEYS and config[key] is not None:  # type: ignore[literal-required]
                base[key] = config[key].copy()  # type: ignore[literal-required]
            else:
                base[key] = config[key] or base.get(key)  # type: ignore[literal-required]
    return base


def call_func_with_variable_args(
    func: Callable[[Input], Output]
    | Callable[[Input, RunnableConfig], Output]
    | Callable[[Input, CallbackManagerForChainRun], Output]
    | Callable[[Input, CallbackManagerForChainRun, RunnableConfig], Output],
    input: Input,
    config: RunnableConfig,
    run_manager: CallbackManagerForChainRun | None = None,
    **kwargs: Any,
) -> Output:
    """调用可以可选接受 run_manager 和/或 config 的函数。

    参数:
        func: 要调用的函数。
        input: 函数的输入。
        config: 要传递给函数的配置。
        run_manager: 要传递给函数的运行管理器。
        **kwargs: 要传递给函数的关键字参数。

    返回:
        函数的输出。
    """
    if accepts_config(func):
        if run_manager is not None:
            kwargs["config"] = patch_config(config, callbacks=run_manager.get_child())
        else:
            kwargs["config"] = config
    if run_manager is not None and accepts_run_manager(func):
        kwargs["run_manager"] = run_manager
    return func(input, **kwargs)  # type: ignore[call-arg]


def acall_func_with_variable_args(
    func: Callable[[Input], Awaitable[Output]]
    | Callable[[Input, RunnableConfig], Awaitable[Output]]
    | Callable[[Input, AsyncCallbackManagerForChainRun], Awaitable[Output]]
    | Callable[
        [Input, AsyncCallbackManagerForChainRun, RunnableConfig], Awaitable[Output]
    ],
    input: Input,
    config: RunnableConfig,
    run_manager: AsyncCallbackManagerForChainRun | None = None,
    **kwargs: Any,
) -> Awaitable[Output]:
    """异步调用可以可选接受 run_manager 和/或 config 的函数。

    参数:
        func: 要调用的函数。
        input: 函数的输入。
        config: 要传递给函数的配置。
        run_manager: 要传递给函数的运行管理器。
        **kwargs: 要传递给函数的关键字参数。

    返回:
        函数的输出。
    """
    if accepts_config(func):
        if run_manager is not None:
            kwargs["config"] = patch_config(config, callbacks=run_manager.get_child())
        else:
            kwargs["config"] = config
    if run_manager is not None and accepts_run_manager(func):
        kwargs["run_manager"] = run_manager
    return func(input, **kwargs)  # type: ignore[call-arg]


def get_callback_manager_for_config(config: RunnableConfig) -> CallbackManager:
    """获取配置的回调管理器。

    参数:
        config: 配置。

    返回:
        回调管理器。
    """
    return CallbackManager.configure(
        inheritable_callbacks=config.get("callbacks"),
        inheritable_tags=config.get("tags"),
        inheritable_metadata=config.get("metadata"),
        langsmith_inheritable_metadata=_get_langsmith_inheritable_metadata_from_config(
            config
        ),
    )


def get_async_callback_manager_for_config(
    config: RunnableConfig,
) -> AsyncCallbackManager:
    """获取配置的异步回调管理器。

    参数:
        config: 配置。

    返回:
        异步回调管理器。
    """
    return AsyncCallbackManager.configure(
        inheritable_callbacks=config.get("callbacks"),
        inheritable_tags=config.get("tags"),
        inheritable_metadata=config.get("metadata"),
        langsmith_inheritable_metadata=_get_langsmith_inheritable_metadata_from_config(
            config
        ),
    )


P = ParamSpec("P")
T = TypeVar("T")


class ContextThreadPoolExecutor(ThreadPoolExecutor):
    """复制上下文到子线程的 ThreadPoolExecutor。"""

    def submit(  # type: ignore[override]
        self,
        func: Callable[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Future[T]:
        """向执行器提交函数。

        参数:
            func: 要提交的函数。
            *args: 函数的位置参数。
            **kwargs: 函数的关键字参数。

        返回:
            函数的 Future。
        """
        return super().submit(
            cast("Callable[..., T]", partial(copy_context().run, func, *args, **kwargs))
        )

    def map(
        self,
        fn: Callable[..., T],
        *iterables: Iterable[Any],
        **kwargs: Any,
    ) -> Iterator[T]:
        """将函数映射到多个可迭代对象。

        参数:
            fn: 要映射的函数。
            *iterables: 要映射的可迭代对象。
            timeout: 映射的超时时间。
            chunksize: 映射的块大小。

        返回:
            映射函数的迭代器。
        """
        contexts = [copy_context() for _ in range(len(iterables[0]))]  # type: ignore[arg-type]

        def _wrapped_fn(*args: Any) -> T:
            return contexts.pop().run(fn, *args)

        return super().map(
            _wrapped_fn,
            *iterables,
            **kwargs,
        )


@contextmanager
def get_executor_for_config(
    config: RunnableConfig | None,
) -> Generator[Executor, None, None]:
    """获取配置的执行器。

    参数:
        config: 配置。

    产出:
        执行器。
    """
    config = config or {}
    with ContextThreadPoolExecutor(
        max_workers=config.get("max_concurrency")
    ) as executor:
        yield executor


async def run_in_executor(
    executor_or_config: Executor | RunnableConfig | None,
    func: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """在执行器中运行函数。

    参数:
        executor_or_config: 要运行所在的执行器或配置。
        func: 函数。
        *args: 函数的位置参数。
        **kwargs: 函数的关键字参数。

    返回:
        函数的输出。
    """

    def wrapper() -> T:
        try:
            return func(*args, **kwargs)
        except StopIteration as exc:
            # StopIteration 无法设置在 asyncio.Future 上
            # 它会引发 TypeError 并使 Future 保持待处理状态
            # 所以我们需要将其转换为 RuntimeError
            raise RuntimeError from exc

    if executor_or_config is None or isinstance(executor_or_config, dict):
        # 使用默认执行器，上下文中复制自当前上下文
        return await asyncio.get_running_loop().run_in_executor(
            None,
            cast("Callable[..., T]", partial(copy_context().run, wrapper)),
        )

    return await asyncio.get_running_loop().run_in_executor(executor_or_config, wrapper)
