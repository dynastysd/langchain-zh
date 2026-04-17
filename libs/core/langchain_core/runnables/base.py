"""`Runnable` 可运行单元的基础类和工具函数。"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import functools
import inspect
import threading
from abc import ABC, abstractmethod
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Iterator,
    Mapping,
    Sequence,
)
from concurrent.futures import FIRST_COMPLETED, wait
from functools import wraps
from itertools import tee
from operator import itemgetter
from types import GenericAlias
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Protocol,
    TypeVar,
    cast,
    get_args,
    get_type_hints,
    overload,
)

from pydantic import BaseModel, ConfigDict, Field, RootModel
from typing_extensions import override

from langchain_core._api import beta_decorator
from langchain_core.callbacks.manager import AsyncCallbackManager, CallbackManager
from langchain_core.load.serializable import (
    Serializable,
    SerializedConstructor,
    SerializedNotImplemented,
)
from langchain_core.runnables.config import (
    RunnableConfig,
    acall_func_with_variable_args,
    call_func_with_variable_args,
    ensure_config,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
    get_config_list,
    get_executor_for_config,
    merge_configs,
    patch_config,
    run_in_executor,
    set_config_context,
)
from langchain_core.runnables.utils import (
    AddableDict,
    AnyConfigurableField,
    ConfigurableField,
    ConfigurableFieldSpec,
    Input,
    Output,
    accepts_config,
    accepts_run_manager,
    coro_with_context,
    gated_coro,
    gather_with_concurrency,
    get_function_first_arg_dict_keys,
    get_function_nonlocals,
    get_lambda_source,
    get_unique_config_specs,
    indent_lines_after_first,
    is_async_callable,
    is_async_generator,
)
from langchain_core.tracers._streaming import _StreamingCallbackHandler
from langchain_core.tracers.event_stream import (
    _astream_events_implementation_v1,
    _astream_events_implementation_v2,
)
from langchain_core.tracers.log_stream import (
    LogStreamCallbackHandler,
    _astream_log_implementation,
)
from langchain_core.tracers.root_listeners import (
    AsyncRootListenersTracer,
    RootListenersTracer,
)
from langchain_core.utils.aiter import aclosing, atee
from langchain_core.utils.iter import safetee
from langchain_core.utils.pydantic import create_model_v2

if TYPE_CHECKING:
    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForChainRun,
        CallbackManagerForChainRun,
    )
    from langchain_core.prompts.base import BasePromptTemplate
    from langchain_core.runnables.fallbacks import (
        RunnableWithFallbacks as RunnableWithFallbacksT,
    )
    from langchain_core.runnables.graph import Graph
    from langchain_core.runnables.retry import ExponentialJitterParams
    from langchain_core.runnables.schema import StreamEvent
    from langchain_core.tools import BaseTool
    from langchain_core.tracers.log_stream import RunLog, RunLogPatch
    from langchain_core.tracers.root_listeners import AsyncListener
    from langchain_core.tracers.schemas import Run


Other = TypeVar("Other")

_RUNNABLE_GENERIC_NUM_ARGS = 2  # Input and Output


class Runnable(ABC, Generic[Input, Output]):
    """可运行单元（Runnable），可以被调用、批量调用、流式调用、转换和组合。

    核心方法
    ===========

    - `invoke`/`ainvoke`：将单个输入转换为输出。
    - `batch`/`abatch`：高效地将多个输入转换为输出。
    - `stream`/`astream`：从单个输入中流式生成输出。
    - `astream_log`：流式输出以及输入的选定中间结果。

    内置优化：

    - **批量调用（Batch）**：默认情况下，batch 使用线程池并发执行 invoke()。
        可通过重写来优化批量调用性能。

    - **异步**：带有 'a' 前缀的方法是异步的。默认情况下，它们使用 asyncio
        的线程池执行同步对应方法。可通过重写实现原生异步。

    所有方法都接受一个可选的 config 参数，可用于配置执行、添加标签和元数据
    以便追踪和调试等。

    可运行单元通过 `input_schema` 属性、`output_schema` 属性和
    `config_schema` 方法来暴露关于其输入、输出和配置的概要信息。

    组合
    ===========

    可运行单元可以以声明式方式组合在一起创建链。

    以这种方式构建的任何链都会自动支持同步、异步、批量调用和流式调用。

    主要的组合原语是 `RunnableSequence`（顺序运行单元）和
    `RunnableParallel`（并行运行单元）。

    **`RunnableSequence`**（顺序运行单元）按顺序调用一系列可运行单元，
    前一个的输出作为下一个的输入。使用 `|` 运算符或将要组合的可运行单元
    列表传递给 `RunnableSequence` 来构建。

    **`RunnableParallel`**（并行运行单元）并发调用多个可运行单元，
    向每个提供相同的输入。使用序列中的字典字面量或将字典传递给
    `RunnableParallel` 来构建。


    For example,

    ```python
    from langchain_core.runnables import RunnableLambda

    # A RunnableSequence constructed using the `|` operator
    sequence = RunnableLambda(lambda x: x + 1) | RunnableLambda(lambda x: x * 2)
    sequence.invoke(1)  # 4
    sequence.batch([1, 2, 3])  # [4, 6, 8]


    # A sequence that contains a RunnableParallel constructed using a dict literal
    sequence = RunnableLambda(lambda x: x + 1) | {
        "mul_2": RunnableLambda(lambda x: x * 2),
        "mul_5": RunnableLambda(lambda x: x * 5),
    }
    sequence.invoke(1)  # {'mul_2': 4, 'mul_5': 10}
    ```

    Standard Methods
    ================

    All `Runnable`s expose additional methods that can be used to modify their
    behavior (e.g., add a retry policy, add lifecycle listeners, make them
    configurable, etc.).

    These methods will work on any `Runnable`, including `Runnable` chains
    constructed by composing other `Runnable`s.
    See the individual methods for details.

    For example,

    ```python
    from langchain_core.runnables import RunnableLambda

    import random

    def add_one(x: int) -> int:
        return x + 1


    def buggy_double(y: int) -> int:
        \"\"\"Buggy code that will fail 70% of the time\"\"\"
        if random.random() > 0.3:
            print('This code failed, and will probably be retried!')  # noqa: T201
            raise ValueError('Triggered buggy code')
        return y * 2

    sequence = (
        RunnableLambda(add_one) |
        RunnableLambda(buggy_double).with_retry( # Retry on failure
            stop_after_attempt=10,
            wait_exponential_jitter=False
        )
    )

    print(sequence.input_schema.model_json_schema()) # Show inferred input schema
    print(sequence.output_schema.model_json_schema()) # Show inferred output schema
    print(sequence.invoke(2)) # invoke the sequence (note the retry above!!)
    ```

    Debugging and tracing
    =====================

    As the chains get longer, it can be useful to be able to see intermediate results
    to debug and trace the chain.

    You can set the global debug flag to True to enable debug output for all chains:

    ```python
    from langchain_core.globals import set_debug

    set_debug(True)
    ```

    Alternatively, you can pass existing or custom callbacks to any given chain:

    ```python
    from langchain_core.tracers import ConsoleCallbackHandler

    chain.invoke(..., config={"callbacks": [ConsoleCallbackHandler()]})
    ```

    For a UI (and much more) checkout [LangSmith](https://docs.langchain.com/langsmith/home).

    """

    name: str | None
    """可运行单元（Runnable）的名称。用于调试和追踪。"""

    def get_name(self, suffix: str | None = None, *, name: str | None = None) -> str:
        """获取可运行单元（Runnable）的名称。

        Args:
            suffix: 要附加到名称的可选后缀。
            name: 可选的自定义名称，用于替代可运行单元（Runnable）的名称。

        Returns:
            可运行单元（Runnable）的名称。
        """
        if name:
            name_ = name
        elif hasattr(self, "name") and self.name:
            name_ = self.name
        else:
            # 这里处理可运行单元子类同时也是 pydantic 模型的情况。
            cls = self.__class__
            # 然后检查它是否是泛型，如果是则恢复原始名称。
            if (
                hasattr(
                    cls,
                    "__pydantic_generic_metadata__",
                )
                and "origin" in cls.__pydantic_generic_metadata__
                and cls.__pydantic_generic_metadata__["origin"] is not None
            ):
                name_ = cls.__pydantic_generic_metadata__["origin"].__name__
            else:
                name_ = cls.__name__

        if suffix:
            if name_[0].isupper():
                return name_ + suffix.title()
            return name_ + "_" + suffix.lower()
        return name_

    @property
    def InputType(self) -> type[Input]:  # noqa: N802
        """输入类型。

        此可运行单元（Runnable）接受的输入类型，指定为类型注解。

        Raises:
            TypeError: 如果无法推断输入类型。
        """
        # 首先遍历所有父类，如果其中任何一个是 Pydantic 模型，
        # 我们将通过 __pydantic_generic_metadata__ 属性
        # 从该模型中获取泛型参数化信息。
        for base in self.__class__.mro():
            if hasattr(base, "__pydantic_generic_metadata__"):
                metadata = base.__pydantic_generic_metadata__
                if (
                    "args" in metadata
                    and len(metadata["args"]) == _RUNNABLE_GENERIC_NUM_ARGS
                ):
                    return cast("type[Input]", metadata["args"][0])

        # 如果在父类中没有找到 Pydantic 模型，
        # 则遍历 __orig_bases__。这对应于不是 pydantic 模型的可运行单元。
        for cls in self.__class__.__orig_bases__:  # type: ignore[attr-defined]
            type_args = get_args(cls)
            if type_args and len(type_args) == _RUNNABLE_GENERIC_NUM_ARGS:
                return cast("type[Input]", type_args[0])

        msg = (
            f"Runnable {self.get_name()} doesn't have an inferable InputType. "
            "Override the InputType property to specify the input type."
        )
        raise TypeError(msg)

    @property
    def OutputType(self) -> type[Output]:  # noqa: N802
        """输出类型。

        此可运行单元（Runnable）产生的输出类型，指定为类型注解。

        Raises:
            TypeError: 如果无法推断输出类型。
        """
        # 首先遍历基类——这将帮助处理任何 pydantic 模型的泛型。
        for base in self.__class__.mro():
            if hasattr(base, "__pydantic_generic_metadata__"):
                metadata = base.__pydantic_generic_metadata__
                if (
                    "args" in metadata
                    and len(metadata["args"]) == _RUNNABLE_GENERIC_NUM_ARGS
                ):
                    return cast("type[Output]", metadata["args"][1])

        for cls in self.__class__.__orig_bases__:  # type: ignore[attr-defined]
            type_args = get_args(cls)
            if type_args and len(type_args) == _RUNNABLE_GENERIC_NUM_ARGS:
                return cast("type[Output]", type_args[1])

        msg = (
            f"Runnable {self.get_name()} doesn't have an inferable OutputType. "
            "Override the OutputType property to specify the output type."
        )
        raise TypeError(msg)

    @property
    def input_schema(self) -> type[BaseModel]:
        """此可运行单元（Runnable）接受的输入类型，指定为 Pydantic 模型。"""
        return self.get_input_schema()

    def get_input_schema(
        self,
        config: RunnableConfig | None = None,
    ) -> type[BaseModel]:
        """获取一个可用于验证可运行单元（Runnable）输入的 Pydantic 模型。

        使用了 `configurable_fields` 和 `configurable_alternatives` 方法的
        可运行单元（Runnable）对象会有动态输入模式，取决于调用该可运行单元时
        使用的配置。

        此方法允许获取特定配置的输入模式。

        Args:
            config: 生成模式时使用的配置。

        Returns:
            可用于验证输入的 Pydantic 模型。
        """
        _ = config
        root_type = self.InputType

        if (
            inspect.isclass(root_type)
            and not isinstance(root_type, GenericAlias)
            and issubclass(root_type, BaseModel)
        ):
            return root_type

        return create_model_v2(
            self.get_name("Input"),
            root=root_type,
            # 创建模型需要访问适当的类型注解才能构造 Pydantic 模型。
            # 当创建模型时，我们传递有关模型创建所在命名空间的信息，
            # 以便类型注解也能被正确解析。
            # self.__class__.__module__ 处理可运行单元在不同的模块中
            # 被子类化的情况。
            module_name=self.__class__.__module__,
        )

    def get_input_jsonschema(
        self, config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        """获取表示可运行单元（Runnable）输入的 JSON 模式。

        Args:
            config: 生成模式时使用的配置。

        Returns:
            表示可运行单元（Runnable）输入的 JSON 模式。

        Example:
            ```python
            from langchain_core.runnables import RunnableLambda


            def add_one(x: int) -> int:
                return x + 1


            runnable = RunnableLambda(add_one)

            print(runnable.get_input_jsonschema())
            ```

        !!! version-added "Added in `langchain-core` 0.3.0"

        """
        return self.get_input_schema(config).model_json_schema()

    @property
    def output_schema(self) -> type[BaseModel]:
        """输出模式。

        此可运行单元（Runnable）产生的输出类型，指定为 Pydantic 模型。
        """
        return self.get_output_schema()

    def get_output_schema(
        self,
        config: RunnableConfig | None = None,
    ) -> type[BaseModel]:
        """获取一个可用于验证可运行单元（Runnable）输出的 Pydantic 模型。

        使用了 `configurable_fields` 和 `configurable_alternatives` 方法的
        可运行单元（Runnable）对象会有动态输出模式，取决于调用该可运行单元时
        使用的配置。

        此方法允许获取特定配置的输出模式。

        Args:
            config: 生成模式时使用的配置。

        Returns:
            可用于验证输出的 Pydantic 模型。
        """
        _ = config
        root_type = self.OutputType

        if (
            inspect.isclass(root_type)
            and not isinstance(root_type, GenericAlias)
            and issubclass(root_type, BaseModel)
        ):
            return root_type

        return create_model_v2(
            self.get_name("Output"),
            root=root_type,
            # 创建模型需要访问适当的类型注解才能构造 Pydantic 模型。
            # 当创建模型时，我们传递有关模型创建所在命名空间的信息，
            # 以便类型注解也能被正确解析。
            # self.__class__.__module__ 处理可运行单元在不同的模块中
            # 被子类化的情况。
            module_name=self.__class__.__module__,
        )

    def get_output_jsonschema(
        self, config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        """获取表示可运行单元（Runnable）输出的 JSON 模式。

        Args:
            config: 生成模式时使用的配置。

        Returns:
            表示可运行单元（Runnable）输出的 JSON 模式。

        Example:
            ```python
            from langchain_core.runnables import RunnableLambda


            def add_one(x: int) -> int:
                return x + 1


            runnable = RunnableLambda(add_one)

            print(runnable.get_output_jsonschema())
            ```

        !!! version-added "Added in `langchain-core` 0.3.0"

        """
        return self.get_output_schema(config).model_json_schema()

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        """此可运行单元（Runnable）的可配置字段列表。"""
        return []

    def config_schema(self, *, include: Sequence[str] | None = None) -> type[BaseModel]:
        """此可运行单元（Runnable）接受的配置类型，指定为 Pydantic 模型。

        要将字段标记为可配置的，请参阅 `configurable_fields`
        和 `configurable_alternatives` 方法。

        Args:
            include: 要包含在配置模式中的字段列表。

        Returns:
            可用于验证配置的 Pydantic 模型。

        """
        include = include or []
        config_specs = self.config_specs
        configurable = (
            create_model_v2(
                "Configurable",
                field_definitions={
                    spec.id: (
                        spec.annotation,
                        Field(
                            spec.default, title=spec.name, description=spec.description
                        ),
                    )
                    for spec in config_specs
                },
            )
            if config_specs
            else None
        )

        # Many need to create a typed dict instead to implement NotRequired!
        # 需要创建一个类型字典来实现 NotRequired！
        all_fields = {
            **({"configurable": (configurable, None)} if configurable else {}),
            **{
                field_name: (field_type, None)
                for field_name, field_type in get_type_hints(RunnableConfig).items()
                if field_name in [i for i in include if i != "configurable"]
            },
        }
        return create_model_v2(self.get_name("Config"), field_definitions=all_fields)

    def get_config_jsonschema(
        self, *, include: Sequence[str] | None = None
    ) -> dict[str, Any]:
        """获取表示可运行单元（Runnable）配置的 JSON 模式。

        Args:
            include: 要包含在配置模式中的字段列表。

        Returns:
            表示可运行单元（Runnable）配置的 JSON 模式。

        !!! version-added "Added in `langchain-core` 0.3.0"

        """
        return self.config_schema(include=include).model_json_schema()

    def get_graph(self, config: RunnableConfig | None = None) -> Graph:
        """返回此可运行单元（Runnable）的图形表示。"""
        # 在本地导入以避免循环导入
        from langchain_core.runnables.graph import Graph  # noqa: PLC0415

        graph = Graph()
        try:
            input_node = graph.add_node(self.get_input_schema(config))
        except TypeError:
            input_node = graph.add_node(create_model_v2(self.get_name("Input")))
        runnable_node = graph.add_node(
            self, metadata=config.get("metadata") if config else None
        )
        try:
            output_node = graph.add_node(self.get_output_schema(config))
        except TypeError:
            output_node = graph.add_node(create_model_v2(self.get_name("Output")))
        graph.add_edge(input_node, runnable_node)
        graph.add_edge(runnable_node, output_node)
        return graph

    def get_prompts(
        self, config: RunnableConfig | None = None
    ) -> list[BasePromptTemplate]:
        """返回此可运行单元（Runnable）使用的提示模板列表。"""
        # 在本地导入以避免循环导入
        from langchain_core.prompts.base import BasePromptTemplate  # noqa: PLC0415

        return [
            node.data
            for node in self.get_graph(config=config).nodes.values()
            if isinstance(node.data, BasePromptTemplate)
        ]

    def __or__(
        self,
        other: Runnable[Any, Other]
        | Callable[[Iterator[Any]], Iterator[Other]]
        | Callable[[AsyncIterator[Any]], AsyncIterator[Other]]
        | Callable[[Any], Other]
        | Mapping[str, Runnable[Any, Other] | Callable[[Any], Other] | Any],
    ) -> RunnableSerializable[Input, Other]:
        """可运行单元（Runnable）的"或"运算符。

        将此可运行单元（Runnable）与另一个对象组合以创建
        顺序运行单元（RunnableSequence）。

        Args:
            other: 另一个可运行单元（Runnable）或类似可运行单元的对象。

        Returns:
            新的可运行单元（Runnable）。
        """
        return RunnableSequence(self, coerce_to_runnable(other))

    def __ror__(
        self,
        other: Runnable[Other, Any]
        | Callable[[Iterator[Other]], Iterator[Any]]
        | Callable[[AsyncIterator[Other]], AsyncIterator[Any]]
        | Callable[[Other], Any]
        | Mapping[str, Runnable[Other, Any] | Callable[[Other], Any] | Any],
    ) -> RunnableSerializable[Other, Output]:
        """可运行单元（Runnable）的"反向或"运算符。

        将此可运行单元（Runnable）与另一个对象组合以创建
        顺序运行单元（RunnableSequence）。

        Args:
            other: 另一个可运行单元（Runnable）或类似可运行单元的对象。

        Returns:
            新的可运行单元（Runnable）。
        """
        return RunnableSequence(coerce_to_runnable(other), self)

    def pipe(
        self,
        *others: Runnable[Any, Other] | Callable[[Any], Other],
        name: str | None = None,
    ) -> RunnableSerializable[Input, Other]:
        """管道连接（Pipe）可运行单元（Runnable）对象。

        将此可运行单元（Runnable）与类似可运行单元的对象组合以创建
        顺序运行单元（RunnableSequence）。

        等价于 `RunnableSequence(self, *others)` 或 `self | others[0] | ...`

        Example:
            ```python
            from langchain_core.runnables import RunnableLambda


            def add_one(x: int) -> int:
                return x + 1


            def mul_two(x: int) -> int:
                return x * 2


            runnable_1 = RunnableLambda(add_one)
            runnable_2 = RunnableLambda(mul_two)
            sequence = runnable_1.pipe(runnable_2)
            # Or equivalently:
            # sequence = runnable_1 | runnable_2
            # sequence = RunnableSequence(first=runnable_1, last=runnable_2)
            sequence.invoke(1)
            await sequence.ainvoke(1)
            # -> 4

            sequence.batch([1, 2, 3])
            await sequence.abatch([1, 2, 3])
            # -> [4, 6, 8]
            ```

        Args:
            *others: 要组合的其他可运行单元（Runnable）或类似可运行单元的对象。
            name: 生成的可运行单元序列（RunnableSequence）的可选名称。

        Returns:
            新的可运行单元（Runnable）。
        """
        return RunnableSequence(self, *others, name=name)

    def pick(self, keys: str | list[str]) -> RunnableSerializable[Any, Any]:
        """从可运行单元（Runnable）输出字典中选择键。

        !!! example "选择单个键"

            ```python
            import json

            from langchain_core.runnables import RunnableLambda, RunnableMap

            as_str = RunnableLambda(str)
            as_json = RunnableLambda(json.loads)
            chain = RunnableMap(str=as_str, json=as_json)

            chain.invoke("[1, 2, 3]")
            # -> {"str": "[1, 2, 3]", "json": [1, 2, 3]}

            json_only_chain = chain.pick("json")
            json_only_chain.invoke("[1, 2, 3]")
            # -> [1, 2, 3]
            ```

        !!! example "选择键列表"

            ```python
            from typing import Any

            import json

            from langchain_core.runnables import RunnableLambda, RunnableMap

            as_str = RunnableLambda(str)
            as_json = RunnableLambda(json.loads)


            def as_bytes(x: Any) -> bytes:
                return bytes(x, "utf-8")


            chain = RunnableMap(
                str=as_str, json=as_json, bytes=RunnableLambda(as_bytes)
            )

            chain.invoke("[1, 2, 3]")
            # -> {"str": "[1, 2, 3]", "json": [1, 2, 3], "bytes": b"[1, 2, 3]"}

            json_and_bytes_chain = chain.pick(["json", "bytes"])
            json_and_bytes_chain.invoke("[1, 2, 3]")
            # -> {"json": [1, 2, 3], "bytes": b"[1, 2, 3]"}
            ```

        Args:
            keys: 要从输出字典中选择的键或键列表。

        Returns:
            新的可运行单元（Runnable）。

        """
        # 在本地导入以避免循环导入
        from langchain_core.runnables.passthrough import RunnablePick  # noqa: PLC0415

        return self | RunnablePick(keys)

    def assign(
        self,
        **kwargs: Runnable[dict[str, Any], Any]
        | Callable[[dict[str, Any]], Any]
        | Mapping[str, Runnable[dict[str, Any], Any] | Callable[[dict[str, Any]], Any]],
    ) -> RunnableSerializable[Any, Any]:
        """为可运行单元（Runnable）的字典输出分配新字段。

        ```python
        from langchain_core.language_models.fake import FakeStreamingListLLM
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import SystemMessagePromptTemplate
        from langchain_core.runnables import Runnable
        from operator import itemgetter

        prompt = (
            SystemMessagePromptTemplate.from_template("You are a nice assistant.")
            + "{question}"
        )
        model = FakeStreamingListLLM(responses=["foo-lish"])

        chain: Runnable = prompt | model | {"str": StrOutputParser()}

        chain_with_assign = chain.assign(hello=itemgetter("str") | model)

        print(chain_with_assign.input_schema.model_json_schema())
        # {'title': 'PromptInput', 'type': 'object', 'properties':
        {'question': {'title': 'Question', 'type': 'string'}}}
        print(chain_with_assign.output_schema.model_json_schema())
        # {'title': 'RunnableSequenceOutput', 'type': 'object', 'properties':
        {'str': {'title': 'Str',
        'type': 'string'}, 'hello': {'title': 'Hello', 'type': 'string'}}}
        ```

        Args:
            **kwargs: 键到可运行单元（Runnable）或类似可运行单元对象的映射，
                它们将使用此可运行单元（Runnable）的整个输出字典来调用。

        Returns:
            新的可运行单元（Runnable）。

        """
        # 在本地导入以避免循环导入
        from langchain_core.runnables.passthrough import RunnableAssign  # noqa: PLC0415

        return self | RunnableAssign(RunnableParallel[dict[str, Any]](kwargs))

    """ --- 公共 API --- """

    @abstractmethod
    def invoke(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Output:
        """将单个输入转换为输出。

        Args:
            input: 可运行单元（Runnable）的输入。
            config: 调用可运行单元（Runnable）时使用的配置。

                配置支持标准键，如用于追踪目的的 `'tags'`、`'metadata'`，
                用于控制并发工作量的 `'max_concurrency'` 等。

                详情请参阅 `RunnableConfig`。

        Returns:
            可运行单元（Runnable）的输出。
        """

    async def ainvoke(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Output:
        """将单个输入转换为输出。

        Args:
            input: 可运行单元（Runnable）的输入。
            config: 调用可运行单元（Runnable）时使用的配置。

                配置支持标准键，如用于追踪目的的 `'tags'`、`'metadata'`，
                用于控制并发工作量的 `'max_concurrency'` 等。

                详情请参阅 `RunnableConfig`。

        Returns:
            可运行单元（Runnable）的输出。
        """
        return await run_in_executor(config, self.invoke, input, config, **kwargs)

    def batch(
        self,
        inputs: list[Input],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> list[Output]:
        """默认实现使用线程池执行器并发运行 invoke。

        默认的批量调用实现适用于 IO 密集型可运行单元。

        如果子类可以更高效地进行批量处理，则必须重写此方法；
        例如，如果底层可运行单元（Runnable）使用的 API 支持批量模式。

        Args:
            inputs: 可运行单元（Runnable）的输入列表。
            config: 调用可运行单元（Runnable）时使用的配置。配置支持
                标准键，如用于追踪目的的 `'tags'`、`'metadata'`，
                用于控制并发工作量的 `'max_concurrency'` 等。

                详情请参阅 `RunnableConfig`。
            return_exceptions: 是否返回异常而不是抛出异常。
            **kwargs: 传递给可运行单元（Runnable）的其他关键字参数。

        Returns:
            可运行单元（Runnable）的输出列表。
        """
        if not inputs:
            return []

        configs = get_config_list(config, len(inputs))

        def invoke(input_: Input, config: RunnableConfig) -> Output | Exception:
            if return_exceptions:
                try:
                    return self.invoke(input_, config, **kwargs)
                except Exception as e:
                    return e
            else:
                return self.invoke(input_, config, **kwargs)

        # 如果只有一个输入，就不需要使用执行器
        if len(inputs) == 1:
            return cast("list[Output]", [invoke(inputs[0], configs[0])])

        with get_executor_for_config(configs[0]) as executor:
            return cast("list[Output]", list(executor.map(invoke, inputs, configs)))

    @overload
    def batch_as_completed(
        self,
        inputs: Sequence[Input],
        config: RunnableConfig | Sequence[RunnableConfig] | None = None,
        *,
        return_exceptions: Literal[False] = False,
        **kwargs: Any,
    ) -> Iterator[tuple[int, Output]]: ...

    @overload
    def batch_as_completed(
        self,
        inputs: Sequence[Input],
        config: RunnableConfig | Sequence[RunnableConfig] | None = None,
        *,
        return_exceptions: Literal[True],
        **kwargs: Any,
    ) -> Iterator[tuple[int, Output | Exception]]: ...

    def batch_as_completed(
        self,
        inputs: Sequence[Input],
        config: RunnableConfig | Sequence[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> Iterator[tuple[int, Output | Exception]]:
        """在输入列表上并行运行 `invoke`。

        在完成时生成结果。

        Args:
            inputs: 可运行单元（Runnable）的输入列表。
            config: 调用可运行单元（Runnable）时使用的配置。

                配置支持标准键，如用于追踪目的的 `'tags'`、`'metadata'`，
                用于控制并发工作量的 `'max_concurrency'` 等。

                详情请参阅 `RunnableConfig`。
            return_exceptions: 是否返回异常而不是抛出异常。
            **kwargs: 传递给可运行单元（Runnable）的其他关键字参数。

        Yields:
            可运行单元（Runnable）输入的索引和输出的元组。

        """
        if not inputs:
            return

        configs = get_config_list(config, len(inputs))

        def invoke(
            i: int, input_: Input, config: RunnableConfig
        ) -> tuple[int, Output | Exception]:
            if return_exceptions:
                try:
                    out: Output | Exception = self.invoke(input_, config, **kwargs)
                except Exception as e:
                    out = e
            else:
                out = self.invoke(input_, config, **kwargs)

            return (i, out)

        if len(inputs) == 1:
            yield invoke(0, inputs[0], configs[0])
            return

        with get_executor_for_config(configs[0]) as executor:
            futures = {
                executor.submit(invoke, i, input_, config)
                for i, (input_, config) in enumerate(zip(inputs, configs, strict=False))
            }

            try:
                while futures:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                    while done:
                        yield done.pop().result()
            finally:
                for future in futures:
                    future.cancel()

    async def abatch(
        self,
        inputs: list[Input],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> list[Output]:
        """默认实现使用 `asyncio.gather` 并行运行 `ainvoke`。

        默认的批量调用实现适用于 IO 密集型可运行单元。

        如果子类可以更高效地进行批量处理，则必须重写此方法；
        例如，如果底层可运行单元（Runnable）使用的 API 支持批量模式。

        Args:
            inputs: 可运行单元（Runnable）的输入列表。
            config: 调用可运行单元（Runnable）时使用的配置。

                配置支持标准键，如用于追踪目的的 `'tags'`、`'metadata'`，
                用于控制并发工作量的 `'max_concurrency'` 等。

                详情请参阅 `RunnableConfig`。
            return_exceptions: 是否返回异常而不是抛出异常。
            **kwargs: 传递给可运行单元（Runnable）的其他关键字参数。

        Returns:
            可运行单元（Runnable）的输出列表。

        """
        if not inputs:
            return []

        configs = get_config_list(config, len(inputs))

        async def ainvoke(value: Input, config: RunnableConfig) -> Output | Exception:
            if return_exceptions:
                try:
                    return await self.ainvoke(value, config, **kwargs)
                except Exception as e:
                    return e
            else:
                return await self.ainvoke(value, config, **kwargs)

        coros = map(ainvoke, inputs, configs)
        return await gather_with_concurrency(configs[0].get("max_concurrency"), *coros)

    @overload
    def abatch_as_completed(
        self,
        inputs: Sequence[Input],
        config: RunnableConfig | Sequence[RunnableConfig] | None = None,
        *,
        return_exceptions: Literal[False] = False,
        **kwargs: Any | None,
    ) -> AsyncIterator[tuple[int, Output]]: ...

    @overload
    def abatch_as_completed(
        self,
        inputs: Sequence[Input],
        config: RunnableConfig | Sequence[RunnableConfig] | None = None,
        *,
        return_exceptions: Literal[True],
        **kwargs: Any | None,
    ) -> AsyncIterator[tuple[int, Output | Exception]]: ...

    async def abatch_as_completed(
        self,
        inputs: Sequence[Input],
        config: RunnableConfig | Sequence[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> AsyncIterator[tuple[int, Output | Exception]]:
        """在输入列表上并行运行 `ainvoke`。

        在完成时生成结果。

        Args:
            inputs: 可运行单元（Runnable）的输入列表。
            config: 调用可运行单元（Runnable）时使用的配置。

                配置支持标准键，如用于追踪目的的 `'tags'`、`'metadata'`，
                用于控制并发工作量的 `'max_concurrency'` 等。

                详情请参阅 `RunnableConfig`。
            return_exceptions: 是否返回异常而不是抛出异常。
            **kwargs: 传递给可运行单元（Runnable）的其他关键字参数。

        Yields:
            可运行单元（Runnable）输入的索引和输出的元组。

        """
        if not inputs:
            return

        configs = get_config_list(config, len(inputs))
        # 从第一个配置中获取 max_concurrency，默认为 None（无限制）
        max_concurrency = configs[0].get("max_concurrency") if configs else None
        semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

        async def ainvoke_task(
            i: int, input_: Input, config: RunnableConfig
        ) -> tuple[int, Output | Exception]:
            if return_exceptions:
                try:
                    out: Output | Exception = await self.ainvoke(
                        input_, config, **kwargs
                    )
                except Exception as e:
                    out = e
            else:
                out = await self.ainvoke(input_, config, **kwargs)
            return (i, out)

        coros = [
            gated_coro(semaphore, ainvoke_task(i, input_, config))
            if semaphore
            else ainvoke_task(i, input_, config)
            for i, (input_, config) in enumerate(zip(inputs, configs, strict=False))
        ]

        for coro in asyncio.as_completed(coros):
            yield await coro

    def stream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Output]:
        """`stream` 的默认实现，调用 `invoke`。

        如果子类支持流式输出，则必须重写此方法。

        Args:
            input: 可运行单元（Runnable）的输入。
            config: 可运行单元（Runnable）使用的配置。
            **kwargs: 传递给可运行单元（Runnable）的其他关键字参数。

        Yields:
            可运行单元（Runnable）的输出。

        """
        yield self.invoke(input, config, **kwargs)

    async def astream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Output]:
        """`astream` 的默认实现，调用 `ainvoke`。

        如果子类支持流式输出，则必须重写此方法。

        Args:
            input: 可运行单元（Runnable）的输入。
            config: 可运行单元（Runnable）使用的配置。
            **kwargs: 传递给可运行单元（Runnable）的其他关键字参数。

        Yields:
            可运行单元（Runnable）的输出。

        """
        yield await self.ainvoke(input, config, **kwargs)

    @overload
    def astream_log(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        diff: Literal[True] = True,
        with_streamed_output_list: bool = True,
        include_names: Sequence[str] | None = None,
        include_types: Sequence[str] | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_names: Sequence[str] | None = None,
        exclude_types: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[RunLogPatch]: ...

    @overload
    def astream_log(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        diff: Literal[False],
        with_streamed_output_list: bool = True,
        include_names: Sequence[str] | None = None,
        include_types: Sequence[str] | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_names: Sequence[str] | None = None,
        exclude_types: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[RunLog]: ...

    async def astream_log(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        diff: bool = True,
        with_streamed_output_list: bool = True,
        include_names: Sequence[str] | None = None,
        include_types: Sequence[str] | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_names: Sequence[str] | None = None,
        exclude_types: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[RunLogPatch] | AsyncIterator[RunLog]:
        """流式输出可运行单元（Runnable）的所有输出，报告给回调系统。

        这包括 LLM、检索器、工具等的所有内部运行。

        输出作为日志对象流式传输，包括描述每一步运行状态如何变化的
        Jsonpatch 操作列表，以及运行的最终状态。

        可以按顺序应用 Jsonpatch 操作来构建状态。

        Args:
            input: 可运行单元（Runnable）的输入。
            config: 可运行单元（Runnable）使用的配置。
            diff: 是否在每一步或当前状态之间生成差异。
            with_streamed_output_list: 是否生成 `streamed_output` 列表。
            include_names: 仅包含具有这些名称的日志。
            include_types: 仅包含具有这些类型的日志。
            include_tags: 仅包含具有这些标签的日志。
            exclude_names: 排除具有这些名称的日志。
            exclude_types: 排除具有这些类型的日志。
            exclude_tags: 排除具有这些标签的日志。
            **kwargs: 传递给可运行单元（Runnable）的其他关键字参数。

        Yields:
            `RunLogPatch` 或 `RunLog` 对象。

        """
        stream = LogStreamCallbackHandler(
            auto_close=False,
            include_names=include_names,
            include_types=include_types,
            include_tags=include_tags,
            exclude_names=exclude_names,
            exclude_types=exclude_types,
            exclude_tags=exclude_tags,
            _schema_format="original",
        )

        # Mypy 在这里没有正确解析重载
        # 可能是因为 `self` 被传递了
        # 而且它无法将其映射到 Runnable[Input,Output]？
        async for item in _astream_log_implementation(  # type: ignore[call-overload]
            self,
            input,
            config,
            diff=diff,
            stream=stream,
            with_streamed_output_list=with_streamed_output_list,
            **kwargs,
        ):
            yield item

    async def astream_events(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        version: Literal["v1", "v2"] = "v2",
        include_names: Sequence[str] | None = None,
        include_types: Sequence[str] | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_names: Sequence[str] | None = None,
        exclude_types: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """生成事件流。

        用于创建 `StreamEvent` 的迭代器，提供关于可运行单元（Runnable）
        进度的实时信息，包括来自中间结果的 `StreamEvent`。

        `StreamEvent` 是一个具有以下模式的字典：

        - `event`：事件名称格式为：
            `on_[runnable_type]_(start|stream|end)`。
        - `name`：生成事件的 `Runnable` 的名称。
        - `run_id`：与发出事件的 `Runnable` 的给定执行相关联的随机生成的 ID。
            作为父 `Runnable` 执行的一部分而被调用的子 `Runnable` 会被分配自己唯一的 ID。
        - `parent_ids`：生成事件的父可运行单元的 ID。
            根 `Runnable` 将有一个空列表。父 ID 的顺序是从根到直接父级。
            仅在 API 的 v2 版本中可用。API 的 v1 版本将返回空列表。
        - `tags`：生成事件的 `Runnable` 的标签。
        - `metadata`：生成事件的 `Runnable` 的元数据。
        - `data`：与事件关联的数据。此字段的内容取决于事件的类型。
            详见下表。

        以下表格展示了一些可能由各种链发出的事件。为简洁起见，
        表中省略了元数据字段。表格后面包含了链的定义。

        !!! note
            此参考表是针对模式 v2 版本的。

        | event                  | name                 | chunk                               | input                                             | output                                              |
        | ---------------------- | -------------------- | ----------------------------------- | ------------------------------------------------- | --------------------------------------------------- |
        | `on_chat_model_start`  | `'[model name]'`     |                                     | `{"messages": [[SystemMessage, HumanMessage]]}`   |                                                     |
        | `on_chat_model_stream` | `'[model name]'`     | `AIMessageChunk(content="hello")`   |                                                   |                                                     |
        | `on_chat_model_end`    | `'[model name]'`     |                                     | `{"messages": [[SystemMessage, HumanMessage]]}`   | `AIMessageChunk(content="hello world")`             |
        | `on_llm_start`         | `'[model name]'`     |                                     | `{'input': 'hello'}`                              |                                                     |
        | `on_llm_stream`        | `'[model name]'`     | `'Hello' `                          |                                                   |                                                     |
        | `on_llm_end`           | `'[model name]'`     |                                     | `'Hello human!'`                                  |                                                     |
        | `on_chain_start`       | `'format_docs'`      |                                     |                                                   |                                                     |
        | `on_chain_stream`      | `'format_docs'`      | `'hello world!, goodbye world!'`    |                                                   |                                                     |
        | `on_chain_end`         | `'format_docs'`      |                                     | `[Document(...)]`                                 | `'hello world!, goodbye world!'`                    |
        | `on_tool_start`        | `'some_tool'`        |                                     | `{"x": 1, "y": "2"}`                              |                                                     |
        | `on_tool_end`          | `'some_tool'`        |                                     |                                                   | `{"x": 1, "y": "2"}`                                |
        | `on_retriever_start`   | `'[retriever name]'` |                                     | `{"query": "hello"}`                              |                                                     |
        | `on_retriever_end`     | `'[retriever name]'` |                                     | `{"query": "hello"}`                              | `[Document(...), ..]`                               |
        | `on_prompt_start`      | `'[template_name]'`  |                                     | `{"question": "hello"}`                           |                                                     |
        | `on_prompt_end`        | `'[template_name]'`  |                                     | `{"question": "hello"}`                           | `ChatPromptValue(messages: [SystemMessage, ...])`   |

        除了标准事件外，用户还可以调度自定义事件（见下面的示例）。

        自定义事件仅在 API 的 v2 版本中显示！

        自定义事件具有以下格式：

        | Attribute   | Type   | Description                                                                                               |
        | ----------- | ------ | --------------------------------------------------------------------------------------------------------- |
        | `name`      | `str`  | 用户定义的事件名称。                                                                          |
        | `data`      | `Any`  | 与事件关联的数据。可以是任何内容，尽管建议使其可 JSON 序列化。 |

        以下是与上面所示标准事件关联的声明：

        `format_docs`：

        ```python
        def format_docs(docs: list[Document]) -> str:
            '''Format the docs.'''
            return ", ".join([doc.page_content for doc in docs])


        format_docs = RunnableLambda(format_docs)
        ```

        `some_tool`：

        ```python
        @tool
        def some_tool(x: int, y: str) -> dict:
            '''Some_tool.'''
            return {"x": x, "y": y}
        ```

        `prompt`：

        ```python
        template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are Cat Agent 007"),
                ("human", "{question}"),
            ]
        ).with_config({"run_name": "my_template", "tags": ["my_template"]})
        ```

        !!! example

            ```python
            from langchain_core.runnables import RunnableLambda


            async def reverse(s: str) -> str:
                return s[::-1]


            chain = RunnableLambda(func=reverse)

            events = [
                event async for event in chain.astream_events("hello", version="v2")
            ]

            # 将产生以下事件
            # (run_id 和 parent_ids 为简洁起见已省略)：
            [
                {
                    "data": {"input": "hello"},
                    "event": "on_chain_start",
                    "metadata": {},
                    "name": "reverse",
                    "tags": [],
                },
                {
                    "data": {"chunk": "olleh"},
                    "event": "on_chain_stream",
                    "metadata": {},
                    "name": "reverse",
                    "tags": [],
                },
                {
                    "data": {"output": "olleh"},
                    "event": "on_chain_end",
                    "metadata": {},
                    "name": "reverse",
                    "tags": [],
                },
            ]
            ```

        ```python title="调度自定义事件"
        from langchain_core.callbacks.manager import (
            adispatch_custom_event,
        )
        from langchain_core.runnables import RunnableLambda, RunnableConfig
        import asyncio


        async def slow_thing(some_input: str, config: RunnableConfig) -> str:
            \"\"\"Do something that takes a long time.\"\"\"
            await asyncio.sleep(1) # Placeholder for some slow operation
            await adispatch_custom_event(
                "progress_event",
                {"message": "Finished step 1 of 3"},
                config=config # Must be included for python < 3.10
            )
            await asyncio.sleep(1) # Placeholder for some slow operation
            await adispatch_custom_event(
                "progress_event",
                {"message": "Finished step 2 of 3"},
                config=config # Must be included for python < 3.10
            )
            await asyncio.sleep(1) # Placeholder for some slow operation
            return "Done"

        slow_thing = RunnableLambda(slow_thing)

        async for event in slow_thing.astream_events("some_input", version="v2"):
            print(event)
        ```

        Args:
            input: 可运行单元（Runnable）的输入。
            config: 可运行单元（Runnable）使用的配置。
            version: 要使用的模式版本，`'v2'` 或 `'v1'`。

                用户应使用 `'v2'`。

                `'v1'` 是为了向后兼容，将在 `0.4.0` 中弃用。

                在 API 稳定之前不会分配默认值。
                自定义事件仅在 `'v2'` 中显示。
            include_names: 仅包含具有匹配名称的可运行单元（Runnable）的事件。
            include_types: 仅包含具有匹配类型的可运行单元（Runnable）的事件。
            include_tags: Only include events from `Runnable` objects with matching tags.
            exclude_names: Exclude events from `Runnable` objects with matching names.
            exclude_types: Exclude events from `Runnable` objects with matching types.
            exclude_tags: Exclude events from `Runnable` objects with matching tags.
            **kwargs: Additional keyword arguments to pass to the `Runnable`.

                这些将被传递给 `astream_log`，因为 `astream_events`
                的实现是构建在 `astream_log` 之上的。

        Yields:
            `StreamEvent` 的异步流。

        Raises:
            NotImplementedError: 如果版本不是 `'v1'` 或 `'v2'`。

        """  # noqa: E501
        if version == "v2":
            event_stream = _astream_events_implementation_v2(
                self,
                input,
                config=config,
                include_names=include_names,
                include_types=include_types,
                include_tags=include_tags,
                exclude_names=exclude_names,
                exclude_types=exclude_types,
                exclude_tags=exclude_tags,
                **kwargs,
            )
        elif version == "v1":
            # 第一个实现，构建在 astream_log API 之上
            # 此实现将在 0.2.0 中弃用
            event_stream = _astream_events_implementation_v1(
                self,
                input,
                config=config,
                include_names=include_names,
                include_types=include_types,
                include_tags=include_tags,
                exclude_names=exclude_names,
                exclude_types=exclude_types,
                exclude_tags=exclude_tags,
                **kwargs,
            )
        else:
            msg = 'Only versions "v1" and "v2" of the schema is currently supported.'
            raise NotImplementedError(msg)

        async with aclosing(event_stream):
            async for event in event_stream:
                yield event

    def transform(
        self,
        input: Iterator[Input],
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Output]:
        """将输入转换为输出。

        transform 的默认实现，它缓冲输入并调用 `astream`。

        如果子类可以在输入仍在生成时开始产生输出，则必须重写此方法。

        Args:
            input: 可运行单元（Runnable）的输入迭代器。
            config: 可运行单元（Runnable）使用的配置。
            **kwargs: 传递给可运行单元（Runnable）的其他关键字参数。

        Yields:
            可运行单元（Runnable）的输出。

        """
        final: Input
        got_first_val = False

        for ichunk in input:
            # transform 的默认实现是缓冲输入，
            # 然后调用 stream。
            # 它会尝试使用 `+` 运算符将所有输入收集到一个块中。
            # 如果输入不可加，那么我们假定只能操作最后一个块，
            # 我们会迭代直到到达最后一个块。
            if not got_first_val:
                final = ichunk
                got_first_val = True
            else:
                try:
                    final = final + ichunk  # type: ignore[operator]
                except TypeError:
                    final = ichunk

        if got_first_val:
            yield from self.stream(final, config, **kwargs)

    async def atransform(
        self,
        input: AsyncIterator[Input],
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Output]:
        """将输入转换为输出。

        atransform 的默认实现，它缓冲输入并调用 `astream`。

        如果子类可以在输入仍在生成时开始产生输出，则必须重写此方法。

        Args:
            input: 可运行单元（Runnable）输入的异步迭代器。
            config: 可运行单元（Runnable）使用的配置。
            **kwargs: 传递给可运行单元（Runnable）的其他关键字参数。

        Yields:
            可运行单元（Runnable）的输出。

        """
        final: Input
        got_first_val = False

        async for ichunk in input:
            # transform 的默认实现是缓冲输入，
            # 然后调用 stream。
            # 它会尝试使用 `+` 运算符将所有输入收集到一个块中。
            # 如果输入不可加，那么我们假定只能操作最后一个块，
            # 我们会迭代直到到达最后一个块。
            if not got_first_val:
                final = ichunk
                got_first_val = True
            else:
                try:
                    final = final + ichunk  # type: ignore[operator]
                except TypeError:
                    final = ichunk

        if got_first_val:
            async for output in self.astream(final, config, **kwargs):
                yield output

    def bind(self, **kwargs: Any) -> Runnable[Input, Output]:
        """将参数绑定到可运行单元（Runnable），返回一个新的可运行单元（Runnable）。

        当链中的可运行单元（Runnable）需要上一个可运行单元输出中不包含
        或用户输入中未包含的参数时，此方法很有用。

        Args:
            **kwargs: 要绑定到可运行单元（Runnable）的参数。

        Returns:
            绑定参数的新可运行单元（Runnable）。

        Example:
            ```python
            from langchain_ollama import ChatOllama
            from langchain_core.output_parsers import StrOutputParser

            model = ChatOllama(model="llama3.1")

            # Without bind
            chain = model | StrOutputParser()

            chain.invoke("Repeat quoted words exactly: 'One two three four five.'")
            # Output is 'One two three four five.'

            # With bind
            chain = model.bind(stop=["three"]) | StrOutputParser()

            chain.invoke("Repeat quoted words exactly: 'One two three four five.'")
            # Output is 'One two'
            ```
        """
        return RunnableBinding(bound=self, kwargs=kwargs, config={})

    def with_config(
        self,
        config: RunnableConfig | None = None,
        # Sadly Unpack is not well-supported by mypy so this will have to be untyped
        # 不幸的是，Unpack 没有被 mypy 很好地支持，所以这里必须不进行类型检查
        **kwargs: Any,
    ) -> Runnable[Input, Output]:
        """将配置绑定到可运行单元（Runnable），返回一个新的可运行单元（Runnable）。

        Args:
            config: 要绑定到可运行单元（Runnable）的配置。
            **kwargs: 传递给可运行单元（Runnable）的其他关键字参数。

        Returns:
            绑定配置的新可运行单元（Runnable）。

        """
        return RunnableBinding(
            bound=self,
            config=cast(
                "RunnableConfig",
                {**(config or {}), **kwargs},
            ),
            kwargs={},
        )

    def with_listeners(
        self,
        *,
        on_start: Callable[[Run], None]
        | Callable[[Run, RunnableConfig], None]
        | None = None,
        on_end: Callable[[Run], None]
        | Callable[[Run, RunnableConfig], None]
        | None = None,
        on_error: Callable[[Run], None]
        | Callable[[Run, RunnableConfig], None]
        | None = None,
    ) -> Runnable[Input, Output]:
        """将生命周期监听器绑定到可运行单元（Runnable），返回一个新的可运行单元（Runnable）。

        Run 对象包含有关运行的信息，包括其 `id`、`type`、`input`、`output`、
        `error`、`start_time`、`end_time` 以及添加到运行中的任何标签或元数据。

        Args:
            on_start: 在可运行单元（Runnable）开始运行之前调用，传入 `Run` 对象。
            on_end: 在可运行单元（Runnable）运行完成后调用，传入 `Run` 对象。
            on_error: 在可运行单元（Runnable）抛出错误时调用，传入 `Run` 对象。

        Returns:
            绑定监听器的新可运行单元（Runnable）。

        Example:
            ```python
            from langchain_core.runnables import RunnableLambda
            from langchain_core.tracers.schemas import Run

            import time


            def test_runnable(time_to_sleep: int):
                time.sleep(time_to_sleep)


            def fn_start(run_obj: Run):
                print("start_time:", run_obj.start_time)


            def fn_end(run_obj: Run):
                print("end_time:", run_obj.end_time)


            chain = RunnableLambda(test_runnable).with_listeners(
                on_start=fn_start, on_end=fn_end
            )
            chain.invoke(2)
            ```
        """
        return RunnableBinding(
            bound=self,
            config_factories=[
                lambda config: {
                    "callbacks": [
                        RootListenersTracer(
                            config=config,
                            on_start=on_start,
                            on_end=on_end,
                            on_error=on_error,
                        )
                    ],
                }
            ],
        )

    def with_alisteners(
        self,
        *,
        on_start: AsyncListener | None = None,
        on_end: AsyncListener | None = None,
        on_error: AsyncListener | None = None,
    ) -> Runnable[Input, Output]:
        """将异步生命周期监听器绑定到可运行单元（Runnable）。

        返回一个新的可运行单元（Runnable）。

        Run 对象包含有关运行的信息，包括其 `id`、`type`、`input`、`output`、
        `error`、`start_time`、`end_time` 以及添加到运行中的任何标签或元数据。

        Args:
            on_start: 在可运行单元（Runnable）开始运行之前异步调用，
                传入 `Run` 对象。
            on_end: 在可运行单元（Runnable）运行完成后异步调用，
                传入 `Run` 对象。
            on_error: 在可运行单元（Runnable）抛出错误时异步调用，
                传入 `Run` 对象。

        Returns:
            绑定监听器的新可运行单元（Runnable）。

        Example:
            ```python
            from langchain_core.runnables import RunnableLambda, Runnable
            from datetime import datetime, timezone
            import time
            import asyncio


            def format_t(timestamp: float) -> str:
                return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


            async def test_runnable(time_to_sleep: int):
                print(f"Runnable[{time_to_sleep}s]: starts at {format_t(time.time())}")
                await asyncio.sleep(time_to_sleep)
                print(f"Runnable[{time_to_sleep}s]: ends at {format_t(time.time())}")


            async def fn_start(run_obj: Runnable):
                print(f"on start callback starts at {format_t(time.time())}")
                await asyncio.sleep(3)
                print(f"on start callback ends at {format_t(time.time())}")


            async def fn_end(run_obj: Runnable):
                print(f"on end callback starts at {format_t(time.time())}")
                await asyncio.sleep(2)
                print(f"on end callback ends at {format_t(time.time())}")


            runnable = RunnableLambda(test_runnable).with_alisteners(
                on_start=fn_start, on_end=fn_end
            )


            async def concurrent_runs():
                await asyncio.gather(runnable.ainvoke(2), runnable.ainvoke(3))


            asyncio.run(concurrent_runs())
            # Result:
            # on start callback starts at 2025-03-01T07:05:22.875378+00:00
            # on start callback starts at 2025-03-01T07:05:22.875495+00:00
            # on start callback ends at 2025-03-01T07:05:25.878862+00:00
            # on start callback ends at 2025-03-01T07:05:25.878947+00:00
            # Runnable[2s]: starts at 2025-03-01T07:05:25.879392+00:00
            # Runnable[3s]: starts at 2025-03-01T07:05:25.879804+00:00
            # Runnable[2s]: ends at 2025-03-01T07:05:27.881998+00:00
            # on end callback starts at 2025-03-01T07:05:27.882360+00:00
            # Runnable[3s]: ends at 2025-03-01T07:05:28.881737+00:00
            # on end callback starts at 2025-03-01T07:05:28.882428+00:00
            # on end callback ends at 2025-03-01T07:05:29.883893+00:00
            # on end callback ends at 2025-03-01T07:05:30.884831+00:00
            ```
        """
        return RunnableBinding(
            bound=self,
            config_factories=[
                lambda config: {
                    "callbacks": [
                        AsyncRootListenersTracer(
                            config=config,
                            on_start=on_start,
                            on_end=on_end,
                            on_error=on_error,
                        )
                    ],
                }
            ],
        )

    def with_types(
        self,
        *,
        input_type: type[Input] | None = None,
        output_type: type[Output] | None = None,
    ) -> Runnable[Input, Output]:
        """将输入和输出类型绑定到可运行单元（Runnable），返回一个新的可运行单元（Runnable）。

        Args:
            input_type: 要绑定到可运行单元（Runnable）的输入类型。
            output_type: 要绑定到可运行单元（Runnable）的输出类型。

        Returns:
            绑定类型的新可运行单元（Runnable）。
        """
        return RunnableBinding(
            bound=self,
            custom_input_type=input_type,
            custom_output_type=output_type,
            kwargs={},
        )

    def with_retry(
        self,
        *,
        retry_if_exception_type: tuple[type[BaseException], ...] = (Exception,),
        wait_exponential_jitter: bool = True,
        exponential_jitter_params: ExponentialJitterParams | None = None,
        stop_after_attempt: int = 3,
    ) -> Runnable[Input, Output]:
        """创建一个在异常时重试原始可运行单元（Runnable）的新可运行单元（Runnable）。

        Args:
            retry_if_exception_type: 要重试的异常类型元组。
            wait_exponential_jitter: 是否在重试之间添加抖动。
            stop_after_attempt: 放弃前的最大尝试次数。
            exponential_jitter_params: `tenacity.wait_exponential_jitter` 的参数。
                即：`initial`、`max`、`exp_base` 和 `jitter`（均为 `float` 值）。

        Returns:
            在异常时重试原始可运行单元（Runnable）的新可运行单元（Runnable）。

        Example:
            ```python
            from langchain_core.runnables import RunnableLambda

            count = 0


            def _lambda(x: int) -> None:
                global count
                count = count + 1
                if x == 1:
                    raise ValueError("x is 1")
                else:
                    pass


            runnable = RunnableLambda(_lambda)
            try:
                runnable.with_retry(
                    stop_after_attempt=2,
                    retry_if_exception_type=(ValueError,),
                ).invoke(1)
            except ValueError:
                pass

            assert count == 2
            ```
        """
        # 在本地导入以避免循环导入
        from langchain_core.runnables.retry import RunnableRetry  # noqa: PLC0415

        return RunnableRetry(
            bound=self,
            kwargs={},
            config={},
            retry_exception_types=retry_if_exception_type,
            wait_exponential_jitter=wait_exponential_jitter,
            max_attempt_number=stop_after_attempt,
            exponential_jitter_params=exponential_jitter_params,
        )

    def map(self) -> Runnable[list[Input], list[Output]]:
        """返回将输入列表映射到输出列表的新可运行单元（Runnable）。

        对每个输入调用 `invoke`。

        Returns:
            将输入列表映射到输出列表的新可运行单元（Runnable）。

        Example:
            ```python
            from langchain_core.runnables import RunnableLambda


            def _lambda(x: int) -> int:
                return x + 1


            runnable = RunnableLambda(_lambda)
            print(runnable.map().invoke([1, 2, 3]))  # [2, 3, 4]
            ```
        """
        return RunnableEach(bound=self)

    def with_fallbacks(
        self,
        fallbacks: Sequence[Runnable[Input, Output]],
        *,
        exceptions_to_handle: tuple[type[BaseException], ...] = (Exception,),
        exception_key: str | None = None,
    ) -> RunnableWithFallbacksT[Input, Output]:
        """为可运行单元（Runnable）添加降级方案，返回一个新的可运行单元（Runnable）。

        新的可运行单元（Runnable）将在失败时依次尝试原始可运行单元（Runnable），
        然后尝试每个降级方案。

        Args:
            fallbacks: 原始可运行单元（Runnable）失败时要尝试的可运行单元序列。
            exceptions_to_handle: 要处理的异常类型元组。
            exception_key: 如果指定为 `string`，则被处理的异常将作为输入的一部分，
                通过指定的键传递给降级方案。

                如果为 `None`，异常不会传递给降级方案。

                如果使用此参数，基础可运行单元（Runnable）及其降级方案
                必须接受字典作为输入。

        Returns:
            一个新的可运行单元（Runnable），在失败时将依次尝试
            原始可运行单元（Runnable），然后是每个降级方案。

        Example:
            ```python
            from typing import Iterator

            from langchain_core.runnables import RunnableGenerator


            def _generate_immediate_error(input: Iterator) -> Iterator[str]:
                raise ValueError()
                yield ""


            def _generate(input: Iterator) -> Iterator[str]:
                yield from "foo bar"


            runnable = RunnableGenerator(_generate_immediate_error).with_fallbacks(
                [RunnableGenerator(_generate)]
            )
            print("".join(runnable.stream({})))  # foo bar
            ```

        Args:
            fallbacks: 原始可运行单元（Runnable）失败时要尝试的可运行单元序列。
            exceptions_to_handle: 要处理的异常类型元组。
            exception_key: 如果指定为 `string`，则被处理的异常将作为输入的一部分，
                通过指定的键传递给降级方案。

                如果为 `None`，异常不会传递给降级方案。

                如果使用此参数，基础可运行单元（Runnable）及其降级方案
                必须接受字典作为输入。

        Returns:
            一个新的可运行单元（Runnable），在失败时将依次尝试
            原始可运行单元（Runnable），然后是每个降级方案。
        """
        # 在本地导入以避免循环导入
        from langchain_core.runnables.fallbacks import (  # noqa: PLC0415
            RunnableWithFallbacks,
        )

        return RunnableWithFallbacks(
            runnable=self,
            fallbacks=fallbacks,
            exceptions_to_handle=exceptions_to_handle,
            exception_key=exception_key,
        )

    """ --- 子类的辅助方法 --- """

    def _call_with_config(
        self,
        func: Callable[[Input], Output]
        | Callable[[Input, CallbackManagerForChainRun], Output]
        | Callable[[Input, CallbackManagerForChainRun, RunnableConfig], Output],
        input_: Input,
        config: RunnableConfig | None,
        run_type: str | None = None,
        serialized: dict[str, Any] | None = None,
        **kwargs: Any | None,
    ) -> Output:
        """使用配置调用。

        辅助方法，用于在有回调的情况下将 `Input` 值转换为 `Output` 值。

        使用此方法在子类中实现 `invoke`。

        """
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        run_manager = callback_manager.on_chain_start(
            serialized,
            input_,
            run_type=run_type,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        try:
            child_config = patch_config(config, callbacks=run_manager.get_child())
            with set_config_context(child_config) as context:
                output = cast(
                    "Output",
                    context.run(
                        call_func_with_variable_args,  # type: ignore[arg-type]
                        func,
                        input_,
                        config,
                        run_manager,
                        **kwargs,
                    ),
                )
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        else:
            run_manager.on_chain_end(output)
            return output

    async def _acall_with_config(
        self,
        func: Callable[[Input], Awaitable[Output]]
        | Callable[[Input, AsyncCallbackManagerForChainRun], Awaitable[Output]]
        | Callable[
            [Input, AsyncCallbackManagerForChainRun, RunnableConfig], Awaitable[Output]
        ],
        input_: Input,
        config: RunnableConfig | None,
        run_type: str | None = None,
        serialized: dict[str, Any] | None = None,
        **kwargs: Any | None,
    ) -> Output:
        """异步调用（使用配置）。

        辅助方法，用于在有回调的情况下将 `Input` 值转换为 `Output` 值。

        使用此方法在子类中实现 `ainvoke`。
        """
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        run_manager = await callback_manager.on_chain_start(
            serialized,
            input_,
            run_type=run_type,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        try:
            child_config = patch_config(config, callbacks=run_manager.get_child())
            with set_config_context(child_config) as context:
                coro = acall_func_with_variable_args(
                    func, input_, config, run_manager, **kwargs
                )
                output: Output = await coro_with_context(coro, context)
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        else:
            await run_manager.on_chain_end(output)
            return output

    def _batch_with_config(
        self,
        func: Callable[[list[Input]], list[Exception | Output]]
        | Callable[
            [list[Input], list[CallbackManagerForChainRun]], list[Exception | Output]
        ]
        | Callable[
            [list[Input], list[CallbackManagerForChainRun], list[RunnableConfig]],
            list[Exception | Output],
        ],
        inputs: list[Input],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        run_type: str | None = None,
        **kwargs: Any | None,
    ) -> list[Output]:
        """使用回调将输入列表转换为输出列表。

        辅助方法，用于在有回调的情况下将 `Input` 值转换为 `Output` 值。
        使用此方法在子类中实现 `invoke`。

        """
        if not inputs:
            return []

        configs = get_config_list(config, len(inputs))
        callback_managers = [get_callback_manager_for_config(c) for c in configs]
        run_managers = [
            callback_manager.on_chain_start(
                None,
                input_,
                run_type=run_type,
                name=config.get("run_name") or self.get_name(),
                run_id=config.pop("run_id", None),
            )
            for callback_manager, input_, config in zip(
                callback_managers, inputs, configs, strict=False
            )
        ]
        try:
            if accepts_config(func):
                kwargs["config"] = [
                    patch_config(c, callbacks=rm.get_child())
                    for c, rm in zip(configs, run_managers, strict=False)
                ]
            if accepts_run_manager(func):
                kwargs["run_manager"] = run_managers
            output = func(inputs, **kwargs)  # type: ignore[call-arg]
        except BaseException as e:
            for run_manager in run_managers:
                run_manager.on_chain_error(e)
            if return_exceptions:
                return cast("list[Output]", [e for _ in inputs])
            raise
        else:
            first_exception: Exception | None = None
            for run_manager, out in zip(run_managers, output, strict=False):
                if isinstance(out, Exception):
                    first_exception = first_exception or out
                    run_manager.on_chain_error(out)
                else:
                    run_manager.on_chain_end(out)
            if return_exceptions or first_exception is None:
                return cast("list[Output]", output)
            raise first_exception

    async def _abatch_with_config(
        self,
        func: Callable[[list[Input]], Awaitable[list[Exception | Output]]]
        | Callable[
            [list[Input], list[AsyncCallbackManagerForChainRun]],
            Awaitable[list[Exception | Output]],
        ]
        | Callable[
            [list[Input], list[AsyncCallbackManagerForChainRun], list[RunnableConfig]],
            Awaitable[list[Exception | Output]],
        ],
        inputs: list[Input],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        run_type: str | None = None,
        **kwargs: Any | None,
    ) -> list[Output]:
        """使用回调将输入列表转换为输出列表。

        辅助方法，用于在有回调的情况下将 `Input` 值转换为 `Output` 值。

        使用此方法在子类中实现 `invoke`。

        """
        if not inputs:
            return []

        configs = get_config_list(config, len(inputs))
        callback_managers = [get_async_callback_manager_for_config(c) for c in configs]
        run_managers: list[AsyncCallbackManagerForChainRun] = await asyncio.gather(
            *(
                callback_manager.on_chain_start(
                    None,
                    input_,
                    run_type=run_type,
                    name=config.get("run_name") or self.get_name(),
                    run_id=config.pop("run_id", None),
                )
                for callback_manager, input_, config in zip(
                    callback_managers, inputs, configs, strict=False
                )
            )
        )
        try:
            if accepts_config(func):
                kwargs["config"] = [
                    patch_config(c, callbacks=rm.get_child())
                    for c, rm in zip(configs, run_managers, strict=False)
                ]
            if accepts_run_manager(func):
                kwargs["run_manager"] = run_managers
            output = await func(inputs, **kwargs)  # type: ignore[call-arg]
        except BaseException as e:
            await asyncio.gather(
                *(run_manager.on_chain_error(e) for run_manager in run_managers)
            )
            if return_exceptions:
                return cast("list[Output]", [e for _ in inputs])
            raise
        else:
            first_exception: Exception | None = None
            coros: list[Awaitable[None]] = []
            for run_manager, out in zip(run_managers, output, strict=False):
                if isinstance(out, Exception):
                    first_exception = first_exception or out
                    coros.append(run_manager.on_chain_error(out))
                else:
                    coros.append(run_manager.on_chain_end(out))
            await asyncio.gather(*coros)
            if return_exceptions or first_exception is None:
                return cast("list[Output]", output)
            raise first_exception

    def _transform_stream_with_config(
        self,
        inputs: Iterator[Input],
        transformer: Callable[[Iterator[Input]], Iterator[Output]]
        | Callable[[Iterator[Input], CallbackManagerForChainRun], Iterator[Output]]
        | Callable[
            [Iterator[Input], CallbackManagerForChainRun, RunnableConfig],
            Iterator[Output],
        ],
        config: RunnableConfig | None,
        run_type: str | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Output]:
        """使用配置转换流。

        辅助方法，用于在有回调的情况下将 `Input` 值的 `Iterator`
        转换为 `Output` 值的 `Iterator`。

        使用此方法在 `Runnable` 子类中实现 `stream` 或 `transform`。

        """
        # 如果存在，从 kwargs 中提取 defers_inputs
        defers_inputs = kwargs.pop("defers_inputs", False)

        # 对输入进行 tee 以便我们可以迭代两次
        input_for_tracing, input_for_transform = tee(inputs, 2)
        # 启动输入迭代器以确保输入 Runnable 在此之前开始
        final_input: Input | None = next(input_for_tracing, None)
        final_input_supported = True
        final_output: Output | None = None
        final_output_supported = True

        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        run_manager = callback_manager.on_chain_start(
            None,
            {"input": ""},
            run_type=run_type,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
            defers_inputs=defers_inputs,
        )
        try:
            child_config = patch_config(config, callbacks=run_manager.get_child())
            if accepts_config(transformer):
                kwargs["config"] = child_config
            if accepts_run_manager(transformer):
                kwargs["run_manager"] = run_manager
            with set_config_context(child_config) as context:
                iterator = context.run(transformer, input_for_transform, **kwargs)  # type: ignore[arg-type]
                if stream_handler := next(
                    (
                        cast("_StreamingCallbackHandler", h)
                        for h in run_manager.handlers
                        # 实例检查在这里没问题，它是一个 mixin
                        if isinstance(h, _StreamingCallbackHandler)
                    ),
                    None,
                ):
                    # populates streamed_output in astream_log() output if needed
                    iterator = stream_handler.tap_output_iter(
                        run_manager.run_id, iterator
                    )
                try:
                    while True:
                        chunk: Output = context.run(next, iterator)
                        yield chunk
                        if final_output_supported:
                            if final_output is None:
                                final_output = chunk
                            else:
                                try:
                                    final_output = final_output + chunk  # type: ignore[operator]
                                except TypeError:
                                    final_output = chunk
                                    final_output_supported = False
                        else:
                            final_output = chunk
                except (StopIteration, GeneratorExit):
                    pass
                for ichunk in input_for_tracing:
                    if final_input_supported:
                        if final_input is None:
                            final_input = ichunk
                        else:
                            try:
                                final_input = final_input + ichunk  # type: ignore[operator]
                            except TypeError:
                                final_input = ichunk
                                final_input_supported = False
                    else:
                        final_input = ichunk
        except BaseException as e:
            run_manager.on_chain_error(e, inputs=final_input)
            raise
        else:
            run_manager.on_chain_end(final_output, inputs=final_input)

    async def _atransform_stream_with_config(
        self,
        inputs: AsyncIterator[Input],
        transformer: Callable[[AsyncIterator[Input]], AsyncIterator[Output]]
        | Callable[
            [AsyncIterator[Input], AsyncCallbackManagerForChainRun],
            AsyncIterator[Output],
        ]
        | Callable[
            [AsyncIterator[Input], AsyncCallbackManagerForChainRun, RunnableConfig],
            AsyncIterator[Output],
        ],
        config: RunnableConfig | None,
        run_type: str | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Output]:
        """使用配置转换异步流。

        辅助方法，用于在有回调的情况下将 `Input` 值的异步 `Iterator`
        转换为 `Output` 值的异步 `Iterator`。

        使用此方法在 `Runnable` 子类中实现 `astream` 或 `atransform`。

        """
        # 如果存在，从 kwargs 中提取 defers_inputs
        defers_inputs = kwargs.pop("defers_inputs", False)

        # 对输入进行 tee 以便我们可以迭代两次
        input_for_tracing, input_for_transform = atee(inputs, 2)
        # 启动输入迭代器以确保输入 Runnable 在此之前开始
        final_input: Input | None = await anext(input_for_tracing, None)
        final_input_supported = True
        final_output: Output | None = None
        final_output_supported = True

        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        run_manager = await callback_manager.on_chain_start(
            None,
            {"input": ""},
            run_type=run_type,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
            defers_inputs=defers_inputs,
        )
        try:
            child_config = patch_config(config, callbacks=run_manager.get_child())
            if accepts_config(transformer):
                kwargs["config"] = child_config
            if accepts_run_manager(transformer):
                kwargs["run_manager"] = run_manager
            with set_config_context(child_config) as context:
                iterator_ = context.run(transformer, input_for_transform, **kwargs)  # type: ignore[arg-type]

                if stream_handler := next(
                    (
                        cast("_StreamingCallbackHandler", h)
                        for h in run_manager.handlers
                        # 实例检查在这里没问题，它是一个 mixin
                        if isinstance(h, _StreamingCallbackHandler)
                    ),
                    None,
                ):
                    # 如果需要，在 astream_log() 输出中填充 streamed_output
                    iterator = stream_handler.tap_output_aiter(
                        run_manager.run_id, iterator_
                    )
                else:
                    iterator = iterator_
                try:
                    while True:
                        chunk = await coro_with_context(anext(iterator), context)
                        yield chunk
                        if final_output_supported:
                            if final_output is None:
                                final_output = chunk
                            else:
                                try:
                                    final_output = final_output + chunk
                                except TypeError:
                                    final_output = chunk
                                    final_output_supported = False
                        else:
                            final_output = chunk
                except StopAsyncIteration:
                    pass
                async for ichunk in input_for_tracing:
                    if final_input_supported:
                        if final_input is None:
                            final_input = ichunk
                        else:
                            try:
                                final_input = final_input + ichunk  # type: ignore[operator]
                            except TypeError:
                                final_input = ichunk
                                final_input_supported = False
                    else:
                        final_input = ichunk
        except BaseException as e:
            await run_manager.on_chain_error(e, inputs=final_input)
            raise
        else:
            await run_manager.on_chain_end(final_output, inputs=final_input)
        finally:
            if iterator_ is not None and hasattr(iterator_, "aclose"):
                await iterator_.aclose()

    @beta_decorator.beta(message="This API is in beta and may change in the future.")
    def as_tool(
        self,
        args_schema: type[BaseModel] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        arg_types: dict[str, type] | None = None,
    ) -> BaseTool:
        """从可运行单元（Runnable）创建基础工具（BaseTool）。

        `as_tool` 将从可运行单元（Runnable）实例化一个具有名称、描述和
        `args_schema` 的基础工具（BaseTool）。在可能的情况下，
        模式从 `runnable.get_input_schema` 推断。

        或者（例如，如果可运行单元（Runnable）接受字典作为输入且特定
        `dict` 键没有类型化），可以直接通过 `args_schema` 指定模式。

        您也可以传递 `arg_types` 来仅指定所需的参数及其类型。

        Args:
            args_schema: 工具的模式。
            name: 工具的名称。
            description: 工具的描述。
            arg_types: 参数名称到类型的字典。

        Returns:
            `BaseTool` 实例。

        !!! example "`TypedDict` 输入"

            ```python
            from typing_extensions import TypedDict
            from langchain_core.runnables import RunnableLambda


            class Args(TypedDict):
                a: int
                b: list[int]


            def f(x: Args) -> str:
                return str(x["a"] * max(x["b"]))


            runnable = RunnableLambda(f)
            as_tool = runnable.as_tool()
            as_tool.invoke({"a": 3, "b": [1, 2]})
            ```

        !!! example "`dict` 输入，通过 `args_schema` 指定模式"

            ```python
            from typing import Any
            from pydantic import BaseModel, Field
            from langchain_core.runnables import RunnableLambda

            def f(x: dict[str, Any]) -> str:
                return str(x["a"] * max(x["b"]))

            class FSchema(BaseModel):
                \"\"\"Apply a function to an integer and list of integers.\"\"\"

                a: int = Field(..., description="Integer")
                b: list[int] = Field(..., description="List of ints")

            runnable = RunnableLambda(f)
            as_tool = runnable.as_tool(FSchema)
            as_tool.invoke({"a": 3, "b": [1, 2]})
            ```

        !!! example "`dict` 输入，通过 `arg_types` 指定模式"

            ```python
            from typing import Any
            from langchain_core.runnables import RunnableLambda


            def f(x: dict[str, Any]) -> str:
                return str(x["a"] * max(x["b"]))


            runnable = RunnableLambda(f)
            as_tool = runnable.as_tool(arg_types={"a": int, "b": list[int]})
            as_tool.invoke({"a": 3, "b": [1, 2]})
            ```

        !!! example "`str` 输入"

            ```python
            from langchain_core.runnables import RunnableLambda


            def f(x: str) -> str:
                return x + "a"


            def g(x: str) -> str:
                return x + "z"


            runnable = RunnableLambda(f) | g
            as_tool = runnable.as_tool()
            as_tool.invoke("b")
            ```
        """
        # 避免循环导入
        from langchain_core.tools import convert_runnable_to_tool  # noqa: PLC0415

        return convert_runnable_to_tool(
            self,
            args_schema=args_schema,
            name=name,
            description=description,
            arg_types=arg_types,
        )


class RunnableSerializable(Serializable, Runnable[Input, Output]):
    """可序列化到 JSON 的可运行单元（Runnable）。"""

    name: str | None = None
    """可运行单元（Runnable）的名称。

    用于调试和追踪。
    """

    model_config = ConfigDict(
        # 抑制来自 pydantic 受保护命名空间的警告
        # (例如 `model_`)
        protected_namespaces=(),
    )

    @override
    def to_json(self) -> SerializedConstructor | SerializedNotImplemented:
        """将可运行单元（Runnable）序列化为 JSON。

        Returns:
            可运行单元（Runnable）的 JSON 可序列化表示。

        """
        dumped = super().to_json()
        with contextlib.suppress(Exception):
            dumped["name"] = self.get_name()
        return dumped

    def configurable_fields(
        self, **kwargs: AnyConfigurableField
    ) -> RunnableSerializable[Input, Output]:
        """在运行时配置特定的可运行单元（Runnable）字段。

        Args:
            **kwargs: 要配置的可配置字段（`ConfigurableField`）实例字典。

        Raises:
            ValueError: 如果在可运行单元（Runnable）中找不到配置键。

        Returns:
            配置了字段的新可运行单元（Runnable）。

        !!! example

            ```python
            from langchain_core.runnables import ConfigurableField
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(max_tokens=20).configurable_fields(
                max_tokens=ConfigurableField(
                    id="output_token_number",
                    name="Max tokens in the output",
                    description="The maximum number of tokens in the output",
                )
            )

            # max_tokens = 20
            print(
                "max_tokens_20: ", model.invoke("tell me something about chess").content
            )

            # max_tokens = 200
            print(
                "max_tokens_200: ",
                model.with_config(configurable={"output_token_number": 200})
                .invoke("tell me something about chess")
                .content,
            )
            ```
        """
        # 在本地导入以避免循环导入
        from langchain_core.runnables.configurable import (  # noqa: PLC0415
            RunnableConfigurableFields,
        )

        model_fields = type(self).model_fields
        for key in kwargs:
            if key not in model_fields:
                msg = (
                    f"Configuration key {key} not found in {self}: "
                    f"available keys are {model_fields.keys()}"
                )
                raise ValueError(msg)

        return RunnableConfigurableFields(default=self, fields=kwargs)

    def configurable_alternatives(
        self,
        which: ConfigurableField,
        *,
        default_key: str = "default",
        prefix_keys: bool = False,
        **kwargs: Runnable[Input, Output] | Callable[[], Runnable[Input, Output]],
    ) -> RunnableSerializable[Input, Output]:
        """配置可在运行时设置的可运行单元（Runnable）对象的替代方案。

        Args:
            which: 将用于选择替代方案的可配置字段（`ConfigurableField`）实例。
            default_key: 如果没有选择替代方案，则使用默认键。
            prefix_keys: 是否使用可配置字段（`ConfigurableField`）id 作为键的前缀。
            **kwargs: 键到可运行单元（Runnable）实例或返回可运行单元（Runnable）实例的可调用对象的字典。

        Returns:
            配置了替代方案的新可运行单元（Runnable）。

        !!! example

            ```python
            from langchain_anthropic import ChatAnthropic
            from langchain_core.runnables.utils import ConfigurableField
            from langchain_openai import ChatOpenAI

            model = ChatAnthropic(
                model_name="claude-sonnet-4-5-20250929"
            ).configurable_alternatives(
                ConfigurableField(id="llm"),
                default_key="anthropic",
                openai=ChatOpenAI(),
            )

            # uses the default model ChatAnthropic
            print(model.invoke("which organization created you?").content)

            # uses ChatOpenAI
            print(
                model.with_config(configurable={"llm": "openai"})
                .invoke("which organization created you?")
                .content
            )
            ```
        """
        # 在本地导入以避免循环导入
        from langchain_core.runnables.configurable import (  # noqa: PLC0415
            RunnableConfigurableAlternatives,
        )

        return RunnableConfigurableAlternatives(
            which=which,
            default=self,
            alternatives=kwargs,
            default_key=default_key,
            prefix_keys=prefix_keys,
        )


def _seq_input_schema(
    steps: list[Runnable[Any, Any]], config: RunnableConfig | None
) -> type[BaseModel]:
    # 在本地导入以避免循环导入
    from langchain_core.runnables.passthrough import (  # noqa: PLC0415
        RunnableAssign,
        RunnablePick,
    )

    first = steps[0]
    if len(steps) == 1:
        return first.get_input_schema(config)
    if isinstance(first, RunnableAssign):
        next_input_schema = _seq_input_schema(steps[1:], config)
        if not issubclass(next_input_schema, RootModel):
            # 它是一个字典，正如预期的那样
            return create_model_v2(
                "RunnableSequenceInput",
                field_definitions={
                    k: (v.annotation, v.default)
                    for k, v in next_input_schema.model_fields.items()
                    if k not in first.mapper.steps__
                },
            )
    elif isinstance(first, RunnablePick):
        return _seq_input_schema(steps[1:], config)

    return first.get_input_schema(config)


def _seq_output_schema(
    steps: list[Runnable[Any, Any]], config: RunnableConfig | None
) -> type[BaseModel]:
    # 在本地导入以避免循环导入
    from langchain_core.runnables.passthrough import (  # noqa: PLC0415
        RunnableAssign,
        RunnablePick,
    )

    last = steps[-1]
    if len(steps) == 1:
        return last.get_input_schema(config)
    if isinstance(last, RunnableAssign):
        mapper_output_schema = last.mapper.get_output_schema(config)
        prev_output_schema = _seq_output_schema(steps[:-1], config)
        if not issubclass(prev_output_schema, RootModel):
            # 它是一个字典，正如预期的那样
            return create_model_v2(
                "RunnableSequenceOutput",
                field_definitions={
                    **{
                        k: (v.annotation, v.default)
                        for k, v in prev_output_schema.model_fields.items()
                    },
                    **{
                        k: (v.annotation, v.default)
                        for k, v in mapper_output_schema.model_fields.items()
                    },
                },
            )
    elif isinstance(last, RunnablePick):
        prev_output_schema = _seq_output_schema(steps[:-1], config)
        if not issubclass(prev_output_schema, RootModel):
            # 它是一个字典，正如预期的那样
            if isinstance(last.keys, list):
                return create_model_v2(
                    "RunnableSequenceOutput",
                    field_definitions={
                        k: (v.annotation, v.default)
                        for k, v in prev_output_schema.model_fields.items()
                        if k in last.keys
                    },
                )
            field = prev_output_schema.model_fields[last.keys]
            return create_model_v2(
                "RunnableSequenceOutput", root=(field.annotation, field.default)
            )

    return last.get_output_schema(config)


_RUNNABLE_SEQUENCE_MIN_STEPS = 2


class RunnableSequence(RunnableSerializable[Input, Output]):
    """可运行单元（Runnable）对象的序列，其中一个的输出是下一个的输入。

    **`RunnableSequence`**（顺序运行单元）是 LangChain 中最重要的组合运算符，
    因为它在几乎每个链中都有使用。

    `RunnableSequence` 可以直接实例化，但更常见的是使用 `|` 运算符，
    其中左操作数或右操作数（或两者）必须是 `Runnable`。

    任何 `RunnableSequence` 自动支持同步、异步和批量调用。

    `batch` 和 `abatch` 的默认实现利用线程池和 asyncio gather，
    比对 IO 密集型 `Runnable` 进行朴素调用 `invoke` 或 `ainvoke` 更快。

    批量调用通过按顺序对 `RunnableSequence` 的每个组件调用批量方法来实现。

    `RunnableSequence` 保留其组件的流式属性，因此如果序列的所有组件都实现了
    `transform` 方法——这是实现将流式输入映射到流式输出的逻辑的方法——
    那么序列就能够将输入流式传输到输出！

    如果序列的任何组件没有实现 transform，那么流式传输
    只会在该组件运行后开始。如果有多个阻塞组件，
    流式传输会在最后一个组件之后开始。

    !!! note
        `RunnableLambda` 默认不支持 `transform`！因此，如果需要使用
        `RunnableLambda`，请注意它们在 `RunnableSequence` 中的位置
        （如果需要使用 `stream`/`astream` 方法）。

        如果需要任意逻辑和流式传输，可以子类化 Runnable，
        并为所需的逻辑实现 `transform`。

    以下是一个使用简单函数说明 `RunnableSequence` 用法的简单示例：

        ```python
        from langchain_core.runnables import RunnableLambda


        def add_one(x: int) -> int:
            return x + 1


        def mul_two(x: int) -> int:
            return x * 2


        runnable_1 = RunnableLambda(add_one)
        runnable_2 = RunnableLambda(mul_two)
        sequence = runnable_1 | runnable_2
        # Or equivalently:
        # sequence = RunnableSequence(first=runnable_1, last=runnable_2)
        sequence.invoke(1)
        await sequence.ainvoke(1)

        sequence.batch([1, 2, 3])
        await sequence.abatch([1, 2, 3])
        ```

    以下是一个使用流式传输 LLM 生成的 JSON 输出的示例：

        ```python
        from langchain_core.output_parsers.json import SimpleJsonOutputParser
        from langchain_openai import ChatOpenAI

        prompt = PromptTemplate.from_template(
            "In JSON format, give me a list of {topic} and their "
            "corresponding names in French, Spanish and in a "
            "Cat Language."
        )

        model = ChatOpenAI()
        chain = prompt | model | SimpleJsonOutputParser()

        async for chunk in chain.astream({"topic": "colors"}):
            print("-")  # noqa: T201
            print(chunk, sep="", flush=True)  # noqa: T201
        ```
    """

    # 将步骤分为 first、middle 和 last，仅用于类型检查。
    # 它允许在第一个类型上指定 `Input`，在最后一个类型上指定 `Output`。
    first: Runnable[Input, Any]
    """序列中的第一个可运行单元（Runnable）。"""
    middle: list[Runnable[Any, Any]] = Field(default_factory=list)
    """序列中的中间可运行单元（Runnable）。"""
    last: Runnable[Any, Output]
    """序列中的最后一个可运行单元（Runnable）。"""

    def __init__(
        self,
        *steps: RunnableLike,
        name: str | None = None,
        first: Runnable[Any, Any] | None = None,
        middle: list[Runnable[Any, Any]] | None = None,
        last: Runnable[Any, Any] | None = None,
    ) -> None:
        """创建新的 `RunnableSequence`（顺序运行单元）。

        Args:
            steps: 要包含在序列中的步骤。
            name: 可运行单元（Runnable）的名称。
            first: 序列中的第一个可运行单元（Runnable）。
            middle: 序列中的中间可运行单元（Runnable）对象。
            last: 序列中的最后一个可运行单元（Runnable）。

        Raises:
            ValueError: 如果序列少于 2 个步骤。
        """
        steps_flat: list[Runnable] = []
        if not steps and first is not None and last is not None:
            steps_flat = [first] + (middle or []) + [last]
        for step in steps:
            if isinstance(step, RunnableSequence):
                steps_flat.extend(step.steps)
            else:
                steps_flat.append(coerce_to_runnable(step))
        if len(steps_flat) < _RUNNABLE_SEQUENCE_MIN_STEPS:
            msg = (
                f"RunnableSequence must have at least {_RUNNABLE_SEQUENCE_MIN_STEPS} "
                f"steps, got {len(steps_flat)}"
            )
            raise ValueError(msg)
        super().__init__(
            first=steps_flat[0],
            middle=list(steps_flat[1:-1]),
            last=steps_flat[-1],
            name=name,
        )

    @classmethod
    @override
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "schema", "runnable"]`
        """
        return ["langchain", "schema", "runnable"]

    @property
    def steps(self) -> list[Runnable[Any, Any]]:
        """组成序列的所有可运行单元（Runnable），按顺序排列。

        Returns:
            可运行单元（Runnable）的列表。
        """
        return [self.first, *self.middle, self.last]

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        """返回 `True`，因为此类是可序列化的。"""
        return True

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @property
    @override
    def InputType(self) -> type[Input]:
        """可运行单元（Runnable）的输入类型。"""
        return self.first.InputType

    @property
    @override
    def OutputType(self) -> type[Output]:
        """可运行单元（Runnable）的输出类型。"""
        return self.last.OutputType

    @override
    def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
        """获取可运行单元（Runnable）的输入模式。

        Args:
            config: 要使用的配置。

        Returns:
            可运行单元（Runnable）的输入模式。

        """
        return _seq_input_schema(self.steps, config)

    @override
    def get_output_schema(
        self, config: RunnableConfig | None = None
    ) -> type[BaseModel]:
        """获取可运行单元（Runnable）的输出模式。

        Args:
            config: 要使用的配置。

        Returns:
            可运行单元（Runnable）的输出模式。

        """
        return _seq_output_schema(self.steps, config)

    @property
    @override
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        """获取可运行单元（Runnable）的配置规范。

        Returns:
            可运行单元（Runnable）的配置规范。

        """
        # 在本地导入以避免循环导入
        return get_unique_config_specs(
            [spec for step in self.steps for spec in step.config_specs]
        )

    @override
    def get_graph(self, config: RunnableConfig | None = None) -> Graph:
        """获取可运行单元（Runnable）的图形表示。

        Args:
            config: 要使用的配置。

        Returns:
            可运行单元（Runnable）的图形表示。

        Raises:
            ValueError: 如果可运行单元（Runnable）没有第一个或最后一个节点。

        """
        # 在本地导入以避免循环导入
        from langchain_core.runnables.graph import Graph  # noqa: PLC0415

        graph = Graph()
        for step in self.steps:
            current_last_node = graph.last_node()
            step_graph = step.get_graph(config)
            if step is not self.first:
                step_graph.trim_first_node()
            if step is not self.last:
                step_graph.trim_last_node()
            step_first_node, _ = graph.extend(step_graph)
            if not step_first_node:
                msg = f"Runnable {step} has no first node"
                raise ValueError(msg)
            if current_last_node:
                graph.add_edge(current_last_node, step_first_node)

        return graph

    @override
    def __repr__(self) -> str:
        return "\n| ".join(
            repr(s) if i == 0 else indent_lines_after_first(repr(s), "| ")
            for i, s in enumerate(self.steps)
        )

    @override
    def __or__(
        self,
        other: Runnable[Any, Other]
        | Callable[[Iterator[Any]], Iterator[Other]]
        | Callable[[AsyncIterator[Any]], AsyncIterator[Other]]
        | Callable[[Any], Other]
        | Mapping[str, Runnable[Any, Other] | Callable[[Any], Other] | Any],
    ) -> RunnableSerializable[Input, Other]:
        if isinstance(other, RunnableSequence):
            return RunnableSequence(
                self.first,
                *self.middle,
                self.last,
                other.first,
                *other.middle,
                other.last,
                name=self.name or other.name,
            )
        return RunnableSequence(
            self.first,
            *self.middle,
            self.last,
            coerce_to_runnable(other),
            name=self.name,
        )

    @override
    def __ror__(
        self,
        other: Runnable[Other, Any]
        | Callable[[Iterator[Other]], Iterator[Any]]
        | Callable[[AsyncIterator[Other]], AsyncIterator[Any]]
        | Callable[[Other], Any]
        | Mapping[str, Runnable[Other, Any] | Callable[[Other], Any] | Any],
    ) -> RunnableSerializable[Other, Output]:
        if isinstance(other, RunnableSequence):
            return RunnableSequence(
                other.first,
                *other.middle,
                other.last,
                self.first,
                *self.middle,
                self.last,
                name=other.name or self.name,
            )
        return RunnableSequence(
            coerce_to_runnable(other),
            self.first,
            *self.middle,
            self.last,
            name=self.name,
        )

    @override
    def invoke(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        # 设置回调和上下文
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        # 启动根运行
        run_manager = callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        input_ = input

        # 按顺序调用所有步骤
        try:
            for i, step in enumerate(self.steps):
                # 将每个步骤标记为子运行
                config = patch_config(
                    config, callbacks=run_manager.get_child(f"seq:step:{i + 1}")
                )
                with set_config_context(config) as context:
                    if i == 0:
                        input_ = context.run(step.invoke, input_, config, **kwargs)
                    else:
                        input_ = context.run(step.invoke, input_, config)
        # 完成根运行
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        else:
            run_manager.on_chain_end(input_)
            return cast("Output", input_)

    @override
    async def ainvoke(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Output:
        # 设置回调和上下文
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        # 启动根运行
        run_manager = await callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        input_ = input

        # 按顺序调用所有步骤
        try:
            for i, step in enumerate(self.steps):
                # 将每个步骤标记为子运行
                config = patch_config(
                    config, callbacks=run_manager.get_child(f"seq:step:{i + 1}")
                )
                with set_config_context(config) as context:
                    if i == 0:
                        part = functools.partial(step.ainvoke, input_, config, **kwargs)
                    else:
                        part = functools.partial(step.ainvoke, input_, config)
                    input_ = await coro_with_context(part(), context, create_task=True)
            # 完成根运行
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        else:
            await run_manager.on_chain_end(input_)
            return cast("Output", input_)

    @override
    def batch(
        self,
        inputs: list[Input],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> list[Output]:
        if not inputs:
            return []

        # 设置回调和上下文
        configs = get_config_list(config, len(inputs))
        callback_managers = [
            CallbackManager.configure(
                inheritable_callbacks=config.get("callbacks"),
                local_callbacks=None,
                verbose=False,
                inheritable_tags=config.get("tags"),
                local_tags=None,
                inheritable_metadata=config.get("metadata"),
                local_metadata=None,
            )
            for config in configs
        ]
        # 启动根运行，每个输入一个
        run_managers = [
            cm.on_chain_start(
                None,
                input_,
                name=config.get("run_name") or self.get_name(),
                run_id=config.pop("run_id", None),
            )
            for cm, input_, config in zip(
                callback_managers, inputs, configs, strict=False
            )
        ]

        # 调用
        try:
            if return_exceptions:
                # 跟踪到目前为止失败（按索引）的输入
                # 如果某个输入失败了，它会出现在此映射中，
                # 值将是抛出的异常。
                failed_inputs_map: dict[int, Exception] = {}
                for stepidx, step in enumerate(self.steps):
                    # 组合剩余输入的原始索引
                    # （即尚未失败的输入）
                    remaining_idxs = [
                        i for i in range(len(configs)) if i not in failed_inputs_map
                    ]
                    # 在剩余输入上调用该步骤
                    inputs = step.batch(
                        [
                            inp
                            for i, inp in zip(remaining_idxs, inputs, strict=False)
                            if i not in failed_inputs_map
                        ],
                        [
                            # 每个步骤是对应根运行的子运行
                            patch_config(
                                config,
                                callbacks=rm.get_child(f"seq:step:{stepidx + 1}"),
                            )
                            for i, (rm, config) in enumerate(
                                zip(run_managers, configs, strict=False)
                            )
                            if i not in failed_inputs_map
                        ],
                        return_exceptions=return_exceptions,
                        **(kwargs if stepidx == 0 else {}),
                    )
                    # 如果某个输入失败了，将其添加到映射中
                    failed_inputs_map.update(
                        {
                            i: inp
                            for i, inp in zip(remaining_idxs, inputs, strict=False)
                            if isinstance(inp, Exception)
                        }
                    )
                    inputs = [inp for inp in inputs if not isinstance(inp, Exception)]
                    # 如果所有输入都失败了，停止处理
                    if len(failed_inputs_map) == len(configs):
                        break

                # 重新组装输出，为失败的输入插入异常
                inputs_copy = inputs.copy()
                inputs = []
                for i in range(len(configs)):
                    if i in failed_inputs_map:
                        inputs.append(cast("Input", failed_inputs_map[i]))
                    else:
                        inputs.append(inputs_copy.pop(0))
            else:
                for i, step in enumerate(self.steps):
                    inputs = step.batch(
                        inputs,
                        [
                            # 每个步骤是对应根运行的子运行
                            patch_config(
                                config, callbacks=rm.get_child(f"seq:step:{i + 1}")
                            )
                            for rm, config in zip(run_managers, configs, strict=False)
                        ],
                        return_exceptions=return_exceptions,
                        **(kwargs if i == 0 else {}),
                    )

        # 完成根运行
        except BaseException as e:
            for rm in run_managers:
                rm.on_chain_error(e)
            if return_exceptions:
                return cast("list[Output]", [e for _ in inputs])
            raise
        else:
            first_exception: Exception | None = None
            for run_manager, out in zip(run_managers, inputs, strict=False):
                if isinstance(out, Exception):
                    first_exception = first_exception or out
                    run_manager.on_chain_error(out)
                else:
                    run_manager.on_chain_end(out)
            if return_exceptions or first_exception is None:
                return cast("list[Output]", inputs)
            raise first_exception

    @override
    async def abatch(
        self,
        inputs: list[Input],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> list[Output]:
        if not inputs:
            return []

        # 设置回调和上下文
        configs = get_config_list(config, len(inputs))
        callback_managers = [
            AsyncCallbackManager.configure(
                inheritable_callbacks=config.get("callbacks"),
                local_callbacks=None,
                verbose=False,
                inheritable_tags=config.get("tags"),
                local_tags=None,
                inheritable_metadata=config.get("metadata"),
                local_metadata=None,
            )
            for config in configs
        ]
        # 启动根运行，每个输入一个
        run_managers: list[AsyncCallbackManagerForChainRun] = await asyncio.gather(
            *(
                cm.on_chain_start(
                    None,
                    input_,
                    name=config.get("run_name") or self.get_name(),
                    run_id=config.pop("run_id", None),
                )
                for cm, input_, config in zip(
                    callback_managers, inputs, configs, strict=False
                )
            )
        )

        # 在每个步骤上调用 .batch()
        # 这利用了 Runnable 子类中的批量优化，如 LLM
        try:
            if return_exceptions:
                # 跟踪到目前为止失败（按索引）的输入
                # 如果某个输入失败了，它会出现在此映射中，
                # 值将是抛出的异常。
                failed_inputs_map: dict[int, Exception] = {}
                for stepidx, step in enumerate(self.steps):
                    # 组合剩余输入的原始索引
                    # （即尚未失败的输入）
                    remaining_idxs = [
                        i for i in range(len(configs)) if i not in failed_inputs_map
                    ]
                    # 在剩余输入上调用该步骤
                    inputs = await step.abatch(
                        [
                            inp
                            for i, inp in zip(remaining_idxs, inputs, strict=False)
                            if i not in failed_inputs_map
                        ],
                        [
                            # 每个步骤是对应根运行的子运行
                            patch_config(
                                config,
                                callbacks=rm.get_child(f"seq:step:{stepidx + 1}"),
                            )
                            for i, (rm, config) in enumerate(
                                zip(run_managers, configs, strict=False)
                            )
                            if i not in failed_inputs_map
                        ],
                        return_exceptions=return_exceptions,
                        **(kwargs if stepidx == 0 else {}),
                    )
                    # 如果有输入失败，将其添加到映射中
                    failed_inputs_map.update(
                        {
                            i: inp
                            for i, inp in zip(remaining_idxs, inputs, strict=False)
                            if isinstance(inp, Exception)
                        }
                    )
                    inputs = [inp for inp in inputs if not isinstance(inp, Exception)]
                    # 如果所有输入都失败了，停止处理
                    if len(failed_inputs_map) == len(configs):
                        break

                # 重新组装输出，为失败的输入插入异常
                inputs_copy = inputs.copy()
                inputs = []
                for i in range(len(configs)):
                    if i in failed_inputs_map:
                        inputs.append(cast("Input", failed_inputs_map[i]))
                    else:
                        inputs.append(inputs_copy.pop(0))
            else:
                for i, step in enumerate(self.steps):
                    inputs = await step.abatch(
                        inputs,
                        [
                            # 每个步骤是对应根运行的子运行
                            patch_config(
                                config, callbacks=rm.get_child(f"seq:step:{i + 1}")
                            )
                            for rm, config in zip(run_managers, configs, strict=False)
                        ],
                        return_exceptions=return_exceptions,
                        **(kwargs if i == 0 else {}),
                    )
        # 完成根运行
        except BaseException as e:
            await asyncio.gather(*(rm.on_chain_error(e) for rm in run_managers))
            if return_exceptions:
                return cast("list[Output]", [e for _ in inputs])
            raise
        else:
            first_exception: Exception | None = None
            coros: list[Awaitable[None]] = []
            for run_manager, out in zip(run_managers, inputs, strict=False):
                if isinstance(out, Exception):
                    first_exception = first_exception or out
                    coros.append(run_manager.on_chain_error(out))
                else:
                    coros.append(run_manager.on_chain_end(out))
            await asyncio.gather(*coros)
            if return_exceptions or first_exception is None:
                return cast("list[Output]", inputs)
            raise first_exception

    def _transform(
        self,
        inputs: Iterator[Input],
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Iterator[Output]:
        steps = [self.first, *self.middle, self.last]
        # 使用下一个步骤转换每个步骤的输入流
        # 不原生支持转换输入流的步骤会将输入缓存在内存中，
        # 直到所有输入都可用后再开始输出
        final_pipeline = cast("Iterator[Output]", inputs)
        for idx, step in enumerate(steps):
            config = patch_config(
                config, callbacks=run_manager.get_child(f"seq:step:{idx + 1}")
            )
            if idx == 0:
                final_pipeline = step.transform(final_pipeline, config, **kwargs)
            else:
                final_pipeline = step.transform(final_pipeline, config)

        yield from final_pipeline

    async def _atransform(
        self,
        inputs: AsyncIterator[Input],
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> AsyncIterator[Output]:
        steps = [self.first, *self.middle, self.last]
        # 流式处理最后的步骤
        # 使用下一个步骤转换每个步骤的输入流
        # 不原生支持转换输入流的步骤会将输入缓存在内存中，
        # 直到所有输入都可用后再开始输出
        final_pipeline = cast("AsyncIterator[Output]", inputs)
        for idx, step in enumerate(steps):
            config = patch_config(
                config,
                callbacks=run_manager.get_child(f"seq:step:{idx + 1}"),
            )
            if idx == 0:
                final_pipeline = step.atransform(final_pipeline, config, **kwargs)
            else:
                final_pipeline = step.atransform(final_pipeline, config)
        async for output in final_pipeline:
            yield output

    @override
    def transform(
        self,
        input: Iterator[Input],
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Output]:
        yield from self._transform_stream_with_config(
            input,
            self._transform,
            patch_config(config, run_name=(config or {}).get("run_name") or self.name),
            **kwargs,
        )

    @override
    def stream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Output]:
        yield from self.transform(iter([input]), config, **kwargs)

    @override
    async def atransform(
        self,
        input: AsyncIterator[Input],
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Output]:
        async for chunk in self._atransform_stream_with_config(
            input,
            self._atransform,
            patch_config(config, run_name=(config or {}).get("run_name") or self.name),
            **kwargs,
        ):
            yield chunk

    @override
    async def astream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Output]:
        async def input_aiter() -> AsyncIterator[Input]:
            yield input

        async for chunk in self.atransform(input_aiter(), config, **kwargs):
            yield chunk


class RunnableParallel(RunnableSerializable[Input, dict[str, Any]]):
    """并行运行 `Runnable` 映射的可运行单元。

    返回它们输出的映射。

    `RunnableParallel` 是两个主要的组合原语之一，另一个是 `RunnableSequence`。
    它并发地调用 `Runnable`，向每个 `Runnable` 提供相同的输入。

    `RunnableParallel` 可以直接实例化，也可以在序列中使用字典字面量来创建。

    下面是一个使用函数来说明 `RunnableParallel` 用法的简单示例：

        ```python
        from langchain_core.runnables import RunnableLambda


        def add_one(x: int) -> int:
            return x + 1


        def mul_two(x: int) -> int:
            return x * 2


        def mul_three(x: int) -> int:
            return x * 3


        runnable_1 = RunnableLambda(add_one)
        runnable_2 = RunnableLambda(mul_two)
        runnable_3 = RunnableLambda(mul_three)

        sequence = runnable_1 | {  # 这个字典会被强制转换为 RunnableParallel
            "mul_two": runnable_2,
            "mul_three": runnable_3,
        }
        # 或者等效地：
        # sequence = runnable_1 | RunnableParallel(
        #     {"mul_two": runnable_2, "mul_three": runnable_3}
        # )
        # 也是等效的：
        # sequence = runnable_1 | RunnableParallel(
        #     mul_two=runnable_2,
        #     mul_three=runnable_3,
        # )

        sequence.invoke(1)
        await sequence.ainvoke(1)

        sequence.batch([1, 2, 3])
        await sequence.abatch([1, 2, 3])
        ```

    `RunnableParallel` 使得并行运行 `Runnable` 变得容易。在下面的示例中，
    我们同时从两个不同的 `Runnable` 对象流式输出：

        ```python
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnableParallel
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI()
        joke_chain = (
            ChatPromptTemplate.from_template("tell me a joke about {topic}") | model
        )
        poem_chain = (
            ChatPromptTemplate.from_template("write a 2-line poem about {topic}")
            | model
        )

        runnable = RunnableParallel(joke=joke_chain, poem=poem_chain)

        # 显示流式输出
        output = {key: "" for key, _ in runnable.output_schema()}
        for chunk in runnable.stream({"topic": "bear"}):
            for key in chunk:
                output[key] = output[key] + chunk[key].content
            print(output)  # noqa: T201
        ```
    """

    steps__: Mapping[str, Runnable[Input, Any]]

    def __init__(
        self,
        steps__: Mapping[
            str,
            Runnable[Input, Any]
            | Callable[[Input], Any]
            | Mapping[str, Runnable[Input, Any] | Callable[[Input], Any]],
        ]
        | None = None,
        **kwargs: Runnable[Input, Any]
        | Callable[[Input], Any]
        | Mapping[str, Runnable[Input, Any] | Callable[[Input], Any]],
    ) -> None:
        """Create a `RunnableParallel`.

        Args:
            steps__: The steps to include.
            **kwargs: Additional steps to include.

        """
        merged = {**steps__} if steps__ is not None else {}
        merged.update(kwargs)
        super().__init__(
            steps__={key: coerce_to_runnable(r) for key, r in merged.items()}
        )

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        """Return `True` as this class is serializable."""
        return True

    @classmethod
    @override
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "schema", "runnable"]`
        """
        return ["langchain", "schema", "runnable"]

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @override
    def get_name(self, suffix: str | None = None, *, name: str | None = None) -> str:
        """Get the name of the `Runnable`.

        Args:
            suffix: The suffix to use.
            name: The name to use.

        Returns:
            The name of the `Runnable`.

        """
        name = name or self.name or f"RunnableParallel<{','.join(self.steps__.keys())}>"
        return super().get_name(suffix, name=name)

    @property
    @override
    def InputType(self) -> Any:
        """The type of the input to the `Runnable`."""
        for step in self.steps__.values():
            if step.InputType:
                return step.InputType

        return Any

    @override
    def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
        """Get the input schema of the `Runnable`.

        Args:
            config: The config to use.

        Returns:
            The input schema of the `Runnable`.

        """
        if all(
            s.get_input_schema(config).model_json_schema().get("type", "object")
            == "object"
            for s in self.steps__.values()
        ):
            for step in self.steps__.values():
                fields = step.get_input_schema(config).model_fields
                root_field = fields.get("root")
                if root_field is not None and root_field.annotation != Any:
                    return super().get_input_schema(config)

            # This is correct, but pydantic typings/mypy don't think so.
            return create_model_v2(
                self.get_name("Input"),
                field_definitions={
                    k: (v.annotation, v.default)
                    for step in self.steps__.values()
                    for k, v in step.get_input_schema(config).model_fields.items()
                    if k != "__root__"
                },
            )

        return super().get_input_schema(config)

    @override
    def get_output_schema(
        self, config: RunnableConfig | None = None
    ) -> type[BaseModel]:
        """Get the output schema of the `Runnable`.

        Args:
            config: The config to use.

        Returns:
            The output schema of the `Runnable`.

        """
        fields = {k: (v.OutputType, ...) for k, v in self.steps__.items()}
        return create_model_v2(self.get_name("Output"), field_definitions=fields)

    @property
    @override
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        """Get the config specs of the `Runnable`.

        Returns:
            The config specs of the `Runnable`.

        """
        return get_unique_config_specs(
            spec for step in self.steps__.values() for spec in step.config_specs
        )

    @override
    def get_graph(self, config: RunnableConfig | None = None) -> Graph:
        """Get the graph representation of the `Runnable`.

        Args:
            config: The config to use.

        Returns:
            The graph representation of the `Runnable`.

        Raises:
            ValueError: If a `Runnable` has no first or last node.

        """
        # Import locally to prevent circular import
        from langchain_core.runnables.graph import Graph  # noqa: PLC0415

        graph = Graph()
        input_node = graph.add_node(self.get_input_schema(config))
        output_node = graph.add_node(self.get_output_schema(config))
        for step in self.steps__.values():
            step_graph = step.get_graph()
            step_graph.trim_first_node()
            step_graph.trim_last_node()
            if not step_graph:
                graph.add_edge(input_node, output_node)
            else:
                step_first_node, step_last_node = graph.extend(step_graph)
                if not step_first_node:
                    msg = f"Runnable {step} has no first node"
                    raise ValueError(msg)
                if not step_last_node:
                    msg = f"Runnable {step} has no last node"
                    raise ValueError(msg)
                graph.add_edge(input_node, step_first_node)
                graph.add_edge(step_last_node, output_node)

        return graph

    @override
    def __repr__(self) -> str:
        map_for_repr = ",\n  ".join(
            f"{k}: {indent_lines_after_first(repr(v), '  ' + k + ': ')}"
            for k, v in self.steps__.items()
        )
        return "{\n  " + map_for_repr + "\n}"

    @override
    def invoke(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        # 设置回调
        config = ensure_config(config)
        callback_manager = CallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            local_callbacks=None,
            verbose=False,
            inheritable_tags=config.get("tags"),
            local_tags=None,
            inheritable_metadata=config.get("metadata"),
            local_metadata=None,
        )
        # 启动根运行
        run_manager = callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )

        def _invoke_step(
            step: Runnable[Input, Any], input_: Input, config: RunnableConfig, key: str
        ) -> Any:
            child_config = patch_config(
                config,
                # 将每个步骤标记为子运行
                callbacks=run_manager.get_child(f"map:key:{key}"),
            )
            with set_config_context(child_config) as context:
                return context.run(
                    step.invoke,
                    input_,
                    child_config,
                )

        # 收集所有步骤的结果
        try:
            # 复制以避免调用者在调用过程中修改步骤时出现问题
            steps = dict(self.steps__)

            with get_executor_for_config(config) as executor:
                futures = [
                    executor.submit(_invoke_step, step, input, config, key)
                    for key, step in steps.items()
                ]
                output = {
                    key: future.result()
                    for key, future in zip(steps, futures, strict=False)
                }
        # 完成根运行
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        else:
            run_manager.on_chain_end(output)
            return output

    @override
    async def ainvoke(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> dict[str, Any]:
        # 设置回调
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        # 启动根运行
        run_manager = await callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )

        async def _ainvoke_step(
            step: Runnable[Input, Any], input_: Input, config: RunnableConfig, key: str
        ) -> Any:
            child_config = patch_config(
                config,
                callbacks=run_manager.get_child(f"map:key:{key}"),
            )
            with set_config_context(child_config) as context:
                return await coro_with_context(
                    step.ainvoke(input_, child_config), context, create_task=True
                )

        # 收集所有步骤的结果
        try:
            # 复制以避免调用者在调用过程中修改步骤时出现问题
            steps = dict(self.steps__)
            results = await asyncio.gather(
                *(
                    _ainvoke_step(
                        step,
                        input,
                        # 将每个步骤标记为子运行
                        config,
                        key,
                    )
                    for key, step in steps.items()
                )
            )
            output = dict(zip(steps, results, strict=False))
        # 完成根运行
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        else:
            await run_manager.on_chain_end(output)
            return output

    def _transform(
        self,
        inputs: Iterator[Input],
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
    ) -> Iterator[AddableDict]:
        # 浅拷贝步骤以忽略进行中的修改
        steps = dict(self.steps__)
        # 每个步骤获得输入迭代器的副本，
        # 该副本在单独的线程中被并行消费
        input_copies = list(safetee(inputs, len(steps), lock=threading.Lock()))
        with get_executor_for_config(config) as executor:
            # 为每个步骤创建 transform() 生成器
            named_generators = [
                (
                    name,
                    step.transform(
                        input_copies.pop(),
                        patch_config(
                            config, callbacks=run_manager.get_child(f"map:key:{name}")
                        ),
                    ),
                )
                for name, step in steps.items()
            ]
            # 启动每个生成器的第一次迭代
            futures = {
                executor.submit(next, generator): (step_name, generator)
                for step_name, generator in named_generators
            }
            # 当每个生成器有可用输出时产生块，
            # 并启动生成该块的生成器的下一次迭代。
            # 当所有生成器耗尽时停止。
            while futures:
                completed_futures, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed_futures:
                    (step_name, generator) = futures.pop(future)
                    try:
                        chunk = AddableDict({step_name: future.result()})
                        yield chunk
                        futures[executor.submit(next, generator)] = (
                            step_name,
                            generator,
                        )
                    except StopIteration:
                        pass

    @override
    def transform(
        self,
        input: Iterator[Input],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        yield from self._transform_stream_with_config(
            input, self._transform, config, **kwargs
        )

    @override
    def stream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[dict[str, Any]]:
        yield from self.transform(iter([input]), config)

    async def _atransform(
        self,
        inputs: AsyncIterator[Input],
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
    ) -> AsyncIterator[AddableDict]:
        # 浅拷贝步骤以忽略进行中的修改
        steps = dict(self.steps__)
        # 每个步骤获得输入迭代器的副本，
        # 该副本在单独的线程中被并行消费
        input_copies = list(atee(inputs, len(steps), lock=asyncio.Lock()))
        # 为每个步骤创建 transform() 生成器
        named_generators = [
            (
                name,
                step.atransform(
                    input_copies.pop(),
                    patch_config(
                        config, callbacks=run_manager.get_child(f"map:key:{name}")
                    ),
                ),
            )
            for name, step in steps.items()
        ]

        # 用协程包装以满足 linter 要求
        async def get_next_chunk(generator: AsyncIterator) -> Output | None:
            return await anext(generator)

        # 启动每个生成器的第一次迭代
        tasks = {
            asyncio.create_task(get_next_chunk(generator)): (step_name, generator)
            for step_name, generator in named_generators
        }
        # 当每个生成器有可用输出时产生块，
        # 并启动生成该块的生成器的下一次迭代。
        # 当所有生成器耗尽时停止。
        while tasks:
            completed_tasks, _ = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in completed_tasks:
                (step_name, generator) = tasks.pop(task)
                try:
                    chunk = AddableDict({step_name: task.result()})
                    yield chunk
                    new_task = asyncio.create_task(get_next_chunk(generator))
                    tasks[new_task] = (step_name, generator)
                except StopAsyncIteration:
                    pass

    @override
    async def atransform(
        self,
        input: AsyncIterator[Input],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        async for chunk in self._atransform_stream_with_config(
            input, self._atransform, config, **kwargs
        ):
            yield chunk

    @override
    async def astream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[dict[str, Any]]:
        async def input_aiter() -> AsyncIterator[Input]:
            yield input

        async for chunk in self.atransform(input_aiter(), config):
            yield chunk


# We support both names
RunnableMap = RunnableParallel


class RunnableGenerator(Runnable[Input, Output]):
    """运行生成器函数的 `Runnable`。

    `RunnableGenerator` 可以直接实例化，也可以在序列中使用生成器来创建。

    `RunnableGenerator` 可用于实现自定义行为，例如自定义输出解析器，
    同时保留流式调用能力。对于签名 为 `Iterator[A] -> Iterator[B]` 的生成器函数，
    将其包装在 `RunnableGenerator` 中可以使其在从上一个步骤流式输入时立即发出输出块。

    !!! note
        如果生成器函数的签名为 `A -> Iterator[B]`，即需要上一个步骤完成才能开始发出块
        （例如，大多数 LLM 需要整个提示可用才能开始生成），则可以改用 `RunnableLambda` 包装。

    下面是一个展示 `RunnableGenerator` 基本机制的示例：

        ```python
        from typing import Any, AsyncIterator, Iterator

        from langchain_core.runnables import RunnableGenerator


        def gen(input: Iterator[Any]) -> Iterator[str]:
            for token in ["Have", " a", " nice", " day"]:
                yield token


        runnable = RunnableGenerator(gen)
        runnable.invoke(None)  # "Have a nice day"
        list(runnable.stream(None))  # ["Have", " a", " nice", " day"]
        runnable.batch([None, None])  # ["Have a nice day", "Have a nice day"]


        # 异步版本：
        async def agen(input: AsyncIterator[Any]) -> AsyncIterator[str]:
            for token in ["Have", " a", " nice", " day"]:
                yield token


        runnable = RunnableGenerator(agen)
        await runnable.ainvoke(None)  # "Have a nice day"
        [p async for p in runnable.astream(None)]  # ["Have", " a", " nice", " day"]
        ```

    `RunnableGenerator` 使得在流式调用上下文中实现自定义行为变得容易。下面是一个示例：

        ```python
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnableGenerator, RunnableLambda
        from langchain_openai import ChatOpenAI
        from langchain_core.output_parsers import StrOutputParser


        model = ChatOpenAI()
        chant_chain = (
            ChatPromptTemplate.from_template("Give me a 3 word chant about {topic}")
            | model
            | StrOutputParser()
        )


        def character_generator(input: Iterator[str]) -> Iterator[str]:
            for token in input:
                if "," in token or "." in token:
                    yield "👏" + token
                else:
                    yield token


        runnable = chant_chain | character_generator
        assert type(runnable.last) is RunnableGenerator
        "".join(runnable.stream({"topic": "waste"}))  # Reduce👏, Reuse👏, Recycle👏.


        # 注意，RunnableLambda 可用于延迟序列中某个步骤的流式输出，
        # 直到前一个步骤完成：
        def reverse_generator(input: str) -> Iterator[str]:
            # 以逆序产生输入的字符
            for character in input[::-1]:
                yield character


        runnable = chant_chain | RunnableLambda(reverse_generator)
        "".join(runnable.stream({"topic": "waste"}))  # ".elcycer ,esuer ,ecudeR"
        ```
    """

    def __init__(
        self,
        transform: Callable[[Iterator[Input]], Iterator[Output]]
        | Callable[[AsyncIterator[Input]], AsyncIterator[Output]],
        atransform: Callable[[AsyncIterator[Input]], AsyncIterator[Output]]
        | None = None,
        *,
        name: str | None = None,
    ) -> None:
        """Initialize a `RunnableGenerator`.

        Args:
            transform: The transform function.
            atransform: The async transform function.
            name: The name of the `Runnable`.

        Raises:
            TypeError: If the transform is not a generator function.

        """
        if atransform is not None:
            self._atransform = atransform
            func_for_name: Callable = atransform

        if is_async_generator(transform):
            self._atransform = transform
            func_for_name = transform
        elif inspect.isgeneratorfunction(transform):
            self._transform = transform
            func_for_name = transform
        else:
            msg = (
                "Expected a generator function type for `transform`."
                f"Instead got an unsupported type: {type(transform)}"
            )
            raise TypeError(msg)

        try:
            self.name = name or func_for_name.__name__
        except AttributeError:
            self.name = "RunnableGenerator"

    @property
    @override
    def InputType(self) -> Any:
        func = getattr(self, "_transform", None) or self._atransform
        try:
            params = inspect.signature(func).parameters
            first_param = next(iter(params.values()), None)
            if first_param and first_param.annotation != inspect.Parameter.empty:
                return getattr(first_param.annotation, "__args__", (Any,))[0]
        except ValueError:
            pass
        return Any

    @override
    def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
        # Override the default implementation.
        # For a runnable generator, we need to bring to provide the
        # module of the underlying function when creating the model.
        root_type = self.InputType

        func = getattr(self, "_transform", None) or self._atransform
        module = getattr(func, "__module__", None)

        if (
            inspect.isclass(root_type)
            and not isinstance(root_type, GenericAlias)
            and issubclass(root_type, BaseModel)
        ):
            return root_type

        return create_model_v2(
            self.get_name("Input"),
            root=root_type,
            # 为了创建模式，我们需要提供底层函数定义的模块。
            # 这允许 pydantic 适当解析类型注解。
            module_name=module,
        )

    @property
    @override
    def OutputType(self) -> Any:
        func = getattr(self, "_transform", None) or self._atransform
        try:
            sig = inspect.signature(func)
            return (
                getattr(sig.return_annotation, "__args__", (Any,))[0]
                if sig.return_annotation != inspect.Signature.empty
                else Any
            )
        except ValueError:
            return Any

    @override
    def get_output_schema(
        self, config: RunnableConfig | None = None
    ) -> type[BaseModel]:
        # Override the default implementation.
        # For a runnable generator, we need to bring to provide the
        # module of the underlying function when creating the model.
        root_type = self.OutputType
        func = getattr(self, "_transform", None) or self._atransform
        module = getattr(func, "__module__", None)

        if (
            inspect.isclass(root_type)
            and not isinstance(root_type, GenericAlias)
            and issubclass(root_type, BaseModel)
        ):
            return root_type

        return create_model_v2(
            self.get_name("Output"),
            root=root_type,
            # To create the schema, we need to provide the module
            # where the underlying function is defined.
            # This allows pydantic to resolve type annotations appropriately.
            module_name=module,
        )

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, RunnableGenerator):
            if hasattr(self, "_transform") and hasattr(other, "_transform"):
                return self._transform == other._transform
            if hasattr(self, "_atransform") and hasattr(other, "_atransform"):
                return self._atransform == other._atransform
            return False
        return False

    __hash__ = None  # type: ignore[assignment]

    @override
    def __repr__(self) -> str:
        return f"RunnableGenerator({self.name})"

    @override
    def transform(
        self,
        input: Iterator[Input],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Output]:
        if not hasattr(self, "_transform"):
            msg = f"{self!r} only supports async methods."
            raise NotImplementedError(msg)
        return self._transform_stream_with_config(
            input,
            self._transform,  # type: ignore[arg-type]
            config,
            defers_inputs=True,
            **kwargs,
        )

    @override
    def stream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Output]:
        return self.transform(iter([input]), config, **kwargs)

    @override
    def invoke(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        final: Output | None = None
        for output in self.stream(input, config, **kwargs):
            final = output if final is None else final + output  # type: ignore[operator]
        return cast("Output", final)

    @override
    def atransform(
        self,
        input: AsyncIterator[Input],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Output]:
        if not hasattr(self, "_atransform"):
            msg = f"{self!r} only supports sync methods."
            raise NotImplementedError(msg)

        return self._atransform_stream_with_config(
            input, self._atransform, config, defers_inputs=True, **kwargs
        )

    @override
    def astream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Output]:
        async def input_aiter() -> AsyncIterator[Input]:
            yield input

        return self.atransform(input_aiter(), config, **kwargs)

    @override
    async def ainvoke(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        final: Output | None = None
        async for output in self.astream(input, config, **kwargs):
            final = output if final is None else final + output  # type: ignore[operator]
        return cast("Output", final)


class RunnableLambda(Runnable[Input, Output]):
    """`RunnableLambda` 将 Python 可调用对象转换为 `Runnable`。

    将可调用对象包装在 `RunnableLambda` 中使其可以在同步或异步上下文中使用。

    `RunnableLambda` 可以像其他 `Runnable` 一样组合，并提供与 LangChain 追踪的无缝集成。

    `RunnableLambda` 最适合不需要支持流式调用的代码。如果需要支持流式调用
    （即能够处理输入块并产生输出块），请改用 `RunnableGenerator`。

    请注意，如果 `RunnableLambda` 返回 `Runnable` 的实例，则该实例在执行期间会被调用（或流式调用）。

    示例：
        ```python
        # 这是一个 RunnableLambda
        from langchain_core.runnables import RunnableLambda


        def add_one(x: int) -> int:
            return x + 1


        runnable = RunnableLambda(add_one)

        runnable.invoke(1)  # 返回 2
        runnable.batch([1, 2, 3])  # 返回 [2, 3, 4]

        # 默认支持异步，委托给同步实现
        await runnable.ainvoke(1)  # 返回 2
        await runnable.abatch([1, 2, 3])  # 返回 [2, 3, 4]


        # 或者，可以同时提供同步和异步实现
        async def add_one_async(x: int) -> int:
            return x + 1


        runnable = RunnableLambda(add_one, afunc=add_one_async)
        runnable.invoke(1)  # 使用 add_one
        await runnable.ainvoke(1)  # 使用 add_one_async
        ```
    """

    @overload
    def __init__(
        self,
        func: Callable[[Input, RunnableConfig], Awaitable[Output]],
        afunc: None = None,
        name: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        func: Callable[[Input], Awaitable[Output]],
        afunc: None = None,
        name: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        func: Callable[[Input], AsyncIterator[Output]],
        afunc: None = None,
        name: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        func: Callable[[Input, AsyncCallbackManagerForChainRun], Awaitable[Output]],
        afunc: None = None,
        name: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        func: Callable[
            [Input, AsyncCallbackManagerForChainRun, RunnableConfig], Awaitable[Output]
        ],
        afunc: None = None,
        name: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        func: Callable[[Input, RunnableConfig], Output],
        afunc: Callable[[Input], Awaitable[Output]]
        | Callable[[Input], AsyncIterator[Output]]
        | Callable[[Input, RunnableConfig], Awaitable[Output]]
        | Callable[[Input, AsyncCallbackManagerForChainRun], Awaitable[Output]]
        | Callable[
            [Input, AsyncCallbackManagerForChainRun, RunnableConfig], Awaitable[Output]
        ]
        | None = None,
        name: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        func: Callable[[Input], Iterator[Output]],
        afunc: Callable[[Input], Awaitable[Output]]
        | Callable[[Input], AsyncIterator[Output]]
        | Callable[[Input, RunnableConfig], Awaitable[Output]]
        | Callable[[Input, AsyncCallbackManagerForChainRun], Awaitable[Output]]
        | Callable[
            [Input, AsyncCallbackManagerForChainRun, RunnableConfig], Awaitable[Output]
        ]
        | None = None,
        name: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        func: Callable[[Input], Runnable[Input, Output]],
        afunc: Callable[[Input], Awaitable[Output]]
        | Callable[[Input], AsyncIterator[Output]]
        | Callable[[Input, RunnableConfig], Awaitable[Output]]
        | Callable[[Input, AsyncCallbackManagerForChainRun], Awaitable[Output]]
        | Callable[
            [Input, AsyncCallbackManagerForChainRun, RunnableConfig], Awaitable[Output]
        ]
        | None = None,
        name: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        func: Callable[[Input, CallbackManagerForChainRun], Output],
        afunc: Callable[[Input], Awaitable[Output]]
        | Callable[[Input], AsyncIterator[Output]]
        | Callable[[Input, RunnableConfig], Awaitable[Output]]
        | Callable[[Input, AsyncCallbackManagerForChainRun], Awaitable[Output]]
        | Callable[
            [Input, AsyncCallbackManagerForChainRun, RunnableConfig], Awaitable[Output]
        ]
        | None = None,
        name: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        func: Callable[[Input, CallbackManagerForChainRun, RunnableConfig], Output],
        afunc: Callable[[Input], Awaitable[Output]]
        | Callable[[Input], AsyncIterator[Output]]
        | Callable[[Input, RunnableConfig], Awaitable[Output]]
        | Callable[[Input, AsyncCallbackManagerForChainRun], Awaitable[Output]]
        | Callable[
            [Input, AsyncCallbackManagerForChainRun, RunnableConfig], Awaitable[Output]
        ]
        | None = None,
        name: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        func: Callable[[Input], Output],
        afunc: Callable[[Input], Awaitable[Output]]
        | Callable[[Input], AsyncIterator[Output]]
        | Callable[[Input, RunnableConfig], Awaitable[Output]]
        | Callable[[Input, AsyncCallbackManagerForChainRun], Awaitable[Output]]
        | Callable[
            [Input, AsyncCallbackManagerForChainRun, RunnableConfig], Awaitable[Output]
        ]
        | None = None,
        name: str | None = None,
    ) -> None: ...

    def __init__(
        self,
        func: Callable[[Input], Iterator[Output]]
        | Callable[[Input], Runnable[Input, Output]]
        | Callable[[Input], Output]
        | Callable[[Input, RunnableConfig], Output]
        | Callable[[Input, CallbackManagerForChainRun], Output]
        | Callable[[Input, CallbackManagerForChainRun, RunnableConfig], Output]
        | Callable[[Input], Awaitable[Output]]
        | Callable[[Input], AsyncIterator[Output]]
        | Callable[[Input, RunnableConfig], Awaitable[Output]]
        | Callable[[Input, AsyncCallbackManagerForChainRun], Awaitable[Output]]
        | Callable[
            [Input, AsyncCallbackManagerForChainRun, RunnableConfig], Awaitable[Output]
        ],
        afunc: Callable[[Input], Awaitable[Output]]
        | Callable[[Input], AsyncIterator[Output]]
        | Callable[[Input, RunnableConfig], Awaitable[Output]]
        | Callable[[Input, AsyncCallbackManagerForChainRun], Awaitable[Output]]
        | Callable[
            [Input, AsyncCallbackManagerForChainRun, RunnableConfig], Awaitable[Output]
        ]
        | None = None,
        name: str | None = None,
    ) -> None:
        """从可调用对象、异步可调用对象或两者创建 `RunnableLambda`。

        接受同步和异步变体，以允许为同步和异步执行提供高效实现。

        Args:
            func: 同步或异步可调用对象
            afunc: 接受输入并返回输出的异步可调用对象。

            name: `Runnable` 的名称。

        Raises:
            TypeError: 如果 `func` 不是可调用类型。
            TypeError: 如果同时提供了 `func` 和 `afunc`。

        """
        if afunc is not None:
            self.afunc = afunc
            func_for_name: Callable = afunc

        if is_async_callable(func) or is_async_generator(func):
            if afunc is not None:
                msg = (
                    "Func was provided as a coroutine function, but afunc was "
                    "also provided. If providing both, func should be a regular "
                    "function to avoid ambiguity."
                )
                raise TypeError(msg)
            self.afunc = func
            func_for_name = func
        elif callable(func):
            self.func = cast("Callable[[Input], Output]", func)
            func_for_name = func
        else:
            msg = (
                "Expected a callable type for `func`."
                f"Instead got an unsupported type: {type(func)}"
            )
            raise TypeError(msg)

        try:
            if name is not None:
                self.name = name
            elif func_for_name.__name__ != "<lambda>":
                self.name = func_for_name.__name__
        except AttributeError:
            pass

        self._repr: str | None = None

    @property
    @override
    def InputType(self) -> Any:
        """The type of the input to this `Runnable`."""
        func = getattr(self, "func", None) or self.afunc
        try:
            params = inspect.signature(func).parameters
            first_param = next(iter(params.values()), None)
            if first_param and first_param.annotation != inspect.Parameter.empty:
                return first_param.annotation
        except ValueError:
            pass
        return Any

    @override
    def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
        """The Pydantic schema for the input to this `Runnable`.

        Args:
            config: The config to use.

        Returns:
            The input schema for this `Runnable`.

        """
        func = getattr(self, "func", None) or self.afunc

        if isinstance(func, itemgetter):
            # 这很糟糕，但据我所知，无法访问 itemgetter 对象上的 _items，
            # 所以我们必须解析 repr
            items = str(func).replace("operator.itemgetter(", "")[:-1].split(", ")
            if all(
                item[0] == "'" and item[-1] == "'" and item != "''" for item in items
            ):
                fields = {item[1:-1]: (Any, ...) for item in items}
                # 这是一个字典
                return create_model_v2(self.get_name("Input"), field_definitions=fields)
            module = getattr(func, "__module__", None)
        return create_model_v2(
            self.get_name("Output"),
            root=root_type,
            # 为了创建模式，我们需要提供底层函数定义的模块。
            # 这允许 pydantic 适当解析类型注解。
            module_name=module,
        )

        if self.InputType != Any:
            return super().get_input_schema(config)

        if dict_keys := get_function_first_arg_dict_keys(func):
            return create_model_v2(
                self.get_name("Input"),
                field_definitions=dict.fromkeys(dict_keys, (Any, ...)),
            )

        return super().get_input_schema(config)

    @property
    @override
    def OutputType(self) -> Any:
        """The type of the output of this `Runnable` as a type annotation.

        Returns:
            The type of the output of this `Runnable`.

        """
        func = getattr(self, "func", None) or self.afunc
        try:
            sig = inspect.signature(func)
            if sig.return_annotation != inspect.Signature.empty:
                # unwrap iterator types
                if getattr(sig.return_annotation, "__origin__", None) in {
                    collections.abc.Iterator,
                    collections.abc.AsyncIterator,
                }:
                    return getattr(sig.return_annotation, "__args__", (Any,))[0]
                return sig.return_annotation
        except ValueError:
            pass
        return Any

    @override
    def get_output_schema(
        self, config: RunnableConfig | None = None
    ) -> type[BaseModel]:
        # 重写默认实现。
        # 对于可运行 lambda，我们需要在创建模型时提供底层函数的模块。
        root_type = self.OutputType
        func = getattr(self, "func", None) or self.afunc
        module = getattr(func, "__module__", None)

        if (
            inspect.isclass(root_type)
            and not isinstance(root_type, GenericAlias)
            and issubclass(root_type, BaseModel)
        ):
            return root_type

        return create_model_v2(
            self.get_name("Output"),
            root=root_type,
            # 为了创建模式，我们需要提供底层函数定义的模块。
            # 这允许 pydantic 适当解析类型注解。
            module_name=module,
        )

    @functools.cached_property
    def deps(self) -> list[Runnable]:
        """The dependencies of this `Runnable`.

        Returns:
            The dependencies of this `Runnable`. If the function has nonlocal
            variables that are `Runnable`s, they are considered dependencies.

        """
        if hasattr(self, "func"):
            objects = get_function_nonlocals(self.func)
        elif hasattr(self, "afunc"):
            objects = get_function_nonlocals(self.afunc)
        else:
            objects = []

        deps: list[Runnable] = []
        for obj in objects:
            if isinstance(obj, Runnable):
                deps.append(obj)
            elif isinstance(getattr(obj, "__self__", None), Runnable):
                deps.append(obj.__self__)
        return deps

    @property
    @override
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return get_unique_config_specs(
            spec for dep in self.deps for spec in dep.config_specs
        )

    @override
    def get_graph(self, config: RunnableConfig | None = None) -> Graph:
        if deps := self.deps:
            # Import locally to prevent circular import
            from langchain_core.runnables.graph import Graph  # noqa: PLC0415

            graph = Graph()
            input_node = graph.add_node(self.get_input_schema(config))
            output_node = graph.add_node(self.get_output_schema(config))
            for dep in deps:
                dep_graph = dep.get_graph()
                dep_graph.trim_first_node()
                dep_graph.trim_last_node()
                if not dep_graph:
                    graph.add_edge(input_node, output_node)
                else:
                    dep_first_node, dep_last_node = graph.extend(dep_graph)
                    if not dep_first_node:
                        msg = f"Runnable {dep} has no first node"
                        raise ValueError(msg)
                    if not dep_last_node:
                        msg = f"Runnable {dep} has no last node"
                        raise ValueError(msg)
                    graph.add_edge(input_node, dep_first_node)
                    graph.add_edge(dep_last_node, output_node)
        else:
            graph = super().get_graph(config)

        return graph

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, RunnableLambda):
            if hasattr(self, "func") and hasattr(other, "func"):
                return self.func == other.func
            if hasattr(self, "afunc") and hasattr(other, "afunc"):
                return self.afunc == other.afunc
            return False
        return False

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        """Return a string representation of this `Runnable`."""
        if self._repr is None:
            if hasattr(self, "func") and isinstance(self.func, itemgetter):
                self._repr = f"RunnableLambda({str(self.func)[len('operator.') :]})"
            elif hasattr(self, "func"):
                self._repr = f"RunnableLambda({get_lambda_source(self.func) or '...'})"
            elif hasattr(self, "afunc"):
                self._repr = (
                    f"RunnableLambda(afunc={get_lambda_source(self.afunc) or '...'})"
                )
            else:
                self._repr = "RunnableLambda(...)"
        return self._repr

    def _invoke(
        self,
        input_: Input,
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Output:
        if inspect.isgeneratorfunction(self.func):
            output: Output | None = None
            for chunk in call_func_with_variable_args(
                cast("Callable[[Input], Iterator[Output]]", self.func),
                input_,
                config,
                run_manager,
                **kwargs,
            ):
                if output is None:
                    output = chunk
                else:
                    try:
                        output = output + chunk  # type: ignore[operator]
                    except TypeError:
                        output = chunk
        else:
            output = call_func_with_variable_args(
                self.func, input_, config, run_manager, **kwargs
            )
        # 如果输出是 Runnable，则调用它
        if isinstance(output, Runnable):
            recursion_limit = config["recursion_limit"]
            if recursion_limit <= 0:
                msg = (
                    f"Recursion limit reached when invoking {self} with input {input_}."
                )
                raise RecursionError(msg)
            output = output.invoke(
                input_,
                patch_config(
                    config,
                    callbacks=run_manager.get_child(),
                    recursion_limit=recursion_limit - 1,
                ),
            )
        return cast("Output", output)

    async def _ainvoke(
        self,
        value: Input,
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Output:
        if hasattr(self, "afunc"):
            afunc = self.afunc
        else:
            if inspect.isgeneratorfunction(self.func):

                def func(
                    value: Input,
                    run_manager: AsyncCallbackManagerForChainRun,
                    config: RunnableConfig,
                    **kwargs: Any,
                ) -> Output:
                    output: Output | None = None
                    for chunk in call_func_with_variable_args(
                        cast("Callable[[Input], Iterator[Output]]", self.func),
                        value,
                        config,
                        run_manager.get_sync(),
                        **kwargs,
                    ):
                        if output is None:
                            output = chunk
                        else:
                            try:
                                output = output + chunk  # type: ignore[operator]
                            except TypeError:
                                output = chunk
                    return cast("Output", output)

            else:

                def func(
                    value: Input,
                    run_manager: AsyncCallbackManagerForChainRun,
                    config: RunnableConfig,
                    **kwargs: Any,
                ) -> Output:
                    return call_func_with_variable_args(
                        self.func, value, config, run_manager.get_sync(), **kwargs
                    )

            @wraps(func)
            async def f(*args: Any, **kwargs: Any) -> Any:
                return await run_in_executor(config, func, *args, **kwargs)

            afunc = f

        if is_async_generator(afunc):
            output: Output | None = None
            async with aclosing(
                cast(
                    "AsyncGenerator[Any, Any]",
                    acall_func_with_variable_args(
                        cast("Callable", afunc),
                        value,
                        config,
                        run_manager,
                        **kwargs,
                    ),
                )
            ) as stream:
                async for chunk in cast(
                    "AsyncIterator[Output]",
                    stream,
                ):
                    if output is None:
                        output = chunk
                    else:
                        try:
                            output = output + chunk  # type: ignore[operator]
                        except TypeError:
                            output = chunk
        else:
            output = await acall_func_with_variable_args(
                cast("Callable", afunc), value, config, run_manager, **kwargs
            )
        # 如果输出是 Runnable，则调用它
        if isinstance(output, Runnable):
            recursion_limit = config["recursion_limit"]
            if recursion_limit <= 0:
                msg = (
                    f"Recursion limit reached when invoking {self} with input {value}."
                )
                raise RecursionError(msg)
            output = await output.ainvoke(
                value,
                patch_config(
                    config,
                    callbacks=run_manager.get_child(),
                    recursion_limit=recursion_limit - 1,
                ),
            )
        return cast("Output", output)

    @override
    def invoke(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Output:
        """Invoke this `Runnable` synchronously.

        Args:
            input: The input to this `Runnable`.
            config: The config to use.
            **kwargs: Additional keyword arguments.

        Returns:
            The output of this `Runnable`.

        Raises:
            TypeError: If the `Runnable` is a coroutine function.

        """
        if hasattr(self, "func"):
            return self._call_with_config(
                self._invoke,
                input,
                ensure_config(config),
                **kwargs,
            )
        msg = "Cannot invoke a coroutine function synchronously.Use `ainvoke` instead."
        raise TypeError(msg)

    @override
    async def ainvoke(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Output:
        """Invoke this `Runnable` asynchronously.

        Args:
            input: The input to this `Runnable`.
            config: The config to use.
            **kwargs: Additional keyword arguments.

        Returns:
            The output of this `Runnable`.

        """
        return await self._acall_with_config(
            self._ainvoke,
            input,
            ensure_config(config),
            **kwargs,
        )

    def _transform(
        self,
        chunks: Iterator[Input],
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Iterator[Output]:
        final: Input
        got_first_val = False
        for ichunk in chunks:
            # 根据定义，RunnableLambda 在发出输出之前会消费所有输入。
            # 如果输入不可添加，那么我们假设只能操作最后一个块。
            # 所以我们会迭代直到到达最后一个块！
            if not got_first_val:
                final = ichunk
                got_first_val = True
            else:
                try:
                    final = final + ichunk  # type: ignore[operator]
                except TypeError:
                    final = ichunk

        if inspect.isgeneratorfunction(self.func):
            output: Output | None = None
            for chunk in call_func_with_variable_args(
                self.func, final, config, run_manager, **kwargs
            ):
                yield chunk
                if output is None:
                    output = chunk
                else:
                    try:
                        output = output + chunk
                    except TypeError:
                        output = chunk
        else:
            output = call_func_with_variable_args(
                self.func, final, config, run_manager, **kwargs
            )

        # 如果输出是 Runnable，使用它的流式输出
        if isinstance(output, Runnable):
            recursion_limit = config["recursion_limit"]
            if recursion_limit <= 0:
                msg = (
                    f"Recursion limit reached when invoking {self} with input {final}."
                )
                raise RecursionError(msg)
            for chunk in output.stream(
                final,
                patch_config(
                    config,
                    callbacks=run_manager.get_child(),
                    recursion_limit=recursion_limit - 1,
                ),
            ):
                yield chunk
        elif not inspect.isgeneratorfunction(self.func):
            # 否则，直接产生它
            yield cast("Output", output)

    @override
    def transform(
        self,
        input: Iterator[Input],
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Output]:
        if hasattr(self, "func"):
            yield from self._transform_stream_with_config(
                input,
                self._transform,
                ensure_config(config),
                **kwargs,
            )
        else:
            msg = (
                "Cannot stream a coroutine function synchronously."
                "Use `astream` instead."
            )
            raise TypeError(msg)

    @override
    def stream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Output]:
        return self.transform(iter([input]), config, **kwargs)

    async def _atransform(
        self,
        chunks: AsyncIterator[Input],
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> AsyncIterator[Output]:
        final: Input
        got_first_val = False
        async for ichunk in chunks:
            # 根据定义，RunnableLambda 在发出输出之前会消费所有输入。
            # 如果输入不可添加，那么我们假设只能操作最后一个块。
            # 所以我们会迭代直到到达最后一个块！
            if not got_first_val:
                final = ichunk
                got_first_val = True
            else:
                try:
                    final = final + ichunk  # type: ignore[operator]
                except TypeError:
                    final = ichunk

        if hasattr(self, "afunc"):
            afunc = self.afunc
        else:
            if inspect.isgeneratorfunction(self.func):
                msg = (
                    "Cannot stream from a generator function asynchronously."
                    "Use .stream() instead."
                )
                raise TypeError(msg)

            def func(
                input_: Input,
                run_manager: AsyncCallbackManagerForChainRun,
                config: RunnableConfig,
                **kwargs: Any,
            ) -> Output:
                return call_func_with_variable_args(
                    self.func, input_, config, run_manager.get_sync(), **kwargs
                )

            @wraps(func)
            async def f(*args: Any, **kwargs: Any) -> Any:
                return await run_in_executor(config, func, *args, **kwargs)

            afunc = f

        if is_async_generator(afunc):
            output: Output | None = None
            async for chunk in cast(
                "AsyncIterator[Output]",
                acall_func_with_variable_args(
                    cast("Callable", afunc),
                    final,
                    config,
                    run_manager,
                    **kwargs,
                ),
            ):
                yield chunk
                if output is None:
                    output = chunk
                else:
                    try:
                        output = output + chunk  # type: ignore[operator]
                    except TypeError:
                        output = chunk
        else:
            output = await acall_func_with_variable_args(
                cast("Callable", afunc),
                final,
                config,
                run_manager,
                **kwargs,
            )

        # 如果输出是 Runnable，使用它的异步流式输出
        if isinstance(output, Runnable):
            recursion_limit = config["recursion_limit"]
            if recursion_limit <= 0:
                msg = (
                    f"Recursion limit reached when invoking {self} with input {final}."
                )
                raise RecursionError(msg)
            async for chunk in output.astream(
                final,
                patch_config(
                    config,
                    callbacks=run_manager.get_child(),
                    recursion_limit=recursion_limit - 1,
                ),
            ):
                yield chunk
        elif not is_async_generator(afunc):
            # 否则，直接产生它
            yield cast("Output", output)

    @override
    async def atransform(
        self,
        input: AsyncIterator[Input],
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Output]:
        async for output in self._atransform_stream_with_config(
            input,
            self._atransform,
            ensure_config(config),
            **kwargs,
        ):
            yield output

    @override
    async def astream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Output]:
        async def input_aiter() -> AsyncIterator[Input]:
            yield input

        async for chunk in self.atransform(input_aiter(), config, **kwargs):
            yield chunk


class RunnableEachBase(RunnableSerializable[list[Input], list[Output]]):
    """RunnableEachBase 类。

    对输入序列的每个元素调用另一个 `Runnable` 的 `Runnable`。

    仅在创建具有不同 `__init__` 参数的新 `RunnableEach` 子类时使用。

    有关详细信息请参阅 `RunnableEach` 的文档。

    """

    bound: Runnable[Input, Output]

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @property
    @override
    def InputType(self) -> Any:
        return list[self.bound.InputType]  # type: ignore[name-defined]

    @override
    def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
        return create_model_v2(
            self.get_name("Input"),
            root=(
                list[self.bound.get_input_schema(config)],  # type: ignore[misc]
                None,
            ),
            # 创建模型需要访问适当的类型注解才能构造 Pydantic 模型。
            # 当我们创建模型时，我们传递有关创建模型的命名空间的信息，
            # 以便类型注解也可以被正确解析。
            # self.__class__.__module__ 处理 Runnable 在不同模块中被子类化的情况。
            module_name=self.__class__.__module__,
        )

    @property
    @override
    def OutputType(self) -> type[list[Output]]:
        return list[self.bound.OutputType]  # type: ignore[name-defined]

    @override
    def get_output_schema(
        self, config: RunnableConfig | None = None
    ) -> type[BaseModel]:
        schema = self.bound.get_output_schema(config)
        return create_model_v2(
            self.get_name("Output"),
            root=list[schema],  # type: ignore[valid-type]
            # 创建模型需要访问适当的类型注解才能构造 Pydantic 模型。
            # 当我们创建模型时，我们传递有关创建模型的命名空间的信息，
            # 以便类型注解也可以被正确解析。
            # self.__class__.__module__ 处理 Runnable 在不同模块中被子类化的情况。
            module_name=self.__class__.__module__,
        )

    @property
    @override
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return self.bound.config_specs

    @override
    def get_graph(self, config: RunnableConfig | None = None) -> Graph:
        return self.bound.get_graph(config)

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        """Return `True` as this class is serializable."""
        return True

    @classmethod
    @override
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "schema", "runnable"]`
        """
        return ["langchain", "schema", "runnable"]

    def _invoke(
        self,
        inputs: list[Input],
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> list[Output]:
        configs = [
            patch_config(config, callbacks=run_manager.get_child()) for _ in inputs
        ]
        return self.bound.batch(inputs, configs, **kwargs)

    @override
    def invoke(
        self, input: list[Input], config: RunnableConfig | None = None, **kwargs: Any
    ) -> list[Output]:
        return self._call_with_config(self._invoke, input, config, **kwargs)

    async def _ainvoke(
        self,
        inputs: list[Input],
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> list[Output]:
        configs = [
            patch_config(config, callbacks=run_manager.get_child()) for _ in inputs
        ]
        return await self.bound.abatch(inputs, configs, **kwargs)

    @override
    async def ainvoke(
        self, input: list[Input], config: RunnableConfig | None = None, **kwargs: Any
    ) -> list[Output]:
        return await self._acall_with_config(self._ainvoke, input, config, **kwargs)

    @override
    async def astream_events(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[StreamEvent]:
        def _error_stream_event(message: str) -> StreamEvent:
            raise NotImplementedError(message)

        for _ in range(1):
            yield _error_stream_event(
                "RunnableEach does not support astream_events yet."
            )


class RunnableEach(RunnableEachBase[Input, Output]):
    """RunnableEach 类。

    对输入序列的每个元素调用另一个 `Runnable` 的 `Runnable`。

    它允许使用绑定的 `Runnable` 调用多个输入。

    `RunnableEach` 使得为 `Runnable` 运行多个输入变得容易。
    在下面的示例中，我们关联并运行三个输入
    与一个 `Runnable`：

        ```python
        from langchain_core.runnables.base import RunnableEach
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        prompt = ChatPromptTemplate.from_template("Tell me a short joke about
        {topic}")
        model = ChatOpenAI()
        output_parser = StrOutputParser()
        runnable = prompt | model | output_parser
        runnable_each = RunnableEach(bound=runnable)
        output = runnable_each.invoke([{'topic':'Computer Science'},
                                    {'topic':'Art'},
                                    {'topic':'Biology'}])
        print(output)  # noqa: T201

        ```
    """

    @override
    def get_name(self, suffix: str | None = None, *, name: str | None = None) -> str:
        name = name or self.name or f"RunnableEach<{self.bound.get_name()}>"
        return super().get_name(suffix, name=name)

    @override
    def bind(self, **kwargs: Any) -> RunnableEach[Input, Output]:
        return RunnableEach(bound=self.bound.bind(**kwargs))

    @override
    def with_config(
        self, config: RunnableConfig | None = None, **kwargs: Any
    ) -> RunnableEach[Input, Output]:
        return RunnableEach(bound=self.bound.with_config(config, **kwargs))

    @override
    def with_listeners(
        self,
        *,
        on_start: Callable[[Run], None]
        | Callable[[Run, RunnableConfig], None]
        | None = None,
        on_end: Callable[[Run], None]
        | Callable[[Run, RunnableConfig], None]
        | None = None,
        on_error: Callable[[Run], None]
        | Callable[[Run, RunnableConfig], None]
        | None = None,
    ) -> RunnableEach[Input, Output]:
        """Bind lifecycle listeners to a `Runnable`, returning a new `Runnable`.

        The `Run` object contains information about the run, including its `id`,
        `type`, `input`, `output`, `error`, `start_time`, `end_time`, and
        any tags or metadata added to the run.

        Args:
            on_start: Called before the `Runnable` starts running, with the `Run`
                object.
            on_end: Called after the `Runnable` finishes running, with the `Run`
                object.
            on_error: Called if the `Runnable` throws an error, with the `Run`
                object.

        Returns:
            A new `Runnable` with the listeners bound.

        """
        return RunnableEach(
            bound=self.bound.with_listeners(
                on_start=on_start, on_end=on_end, on_error=on_error
            )
        )

    def with_alisteners(
        self,
        *,
        on_start: AsyncListener | None = None,
        on_end: AsyncListener | None = None,
        on_error: AsyncListener | None = None,
    ) -> RunnableEach[Input, Output]:
        """绑定异步生命周期监听器到 `Runnable`。

        返回一个新的 `Runnable`。

        `Run` 对象包含有关运行的信息，包括其 `id`、
        `type`、`input`、`output`、`error`、`start_time`、`end_time` 以及
        添加到运行中的任何标签或元数据。

        Args:
            on_start: 在 `Runnable` 开始运行之前异步调用，传入 `Run` 对象。
            on_end: 在 `Runnable` 运行完成后异步调用，传入 `Run` 对象。
            on_error: 如果 `Runnable` 抛出错误，则异步调用，传入 `Run` 对象。

        Returns:
            绑定了监听器的新 `Runnable`。

        """
        return RunnableEach(
            bound=self.bound.with_alisteners(
                on_start=on_start, on_end=on_end, on_error=on_error
            )
        )


class RunnableBindingBase(RunnableSerializable[Input, Output]):  # type: ignore[no-redef]
    """使用一组 `**kwargs` 将调用委托给另一个 `Runnable` 的 `Runnable`。

    仅在创建具有不同 `__init__` 参数的新 `RunnableBinding` 子类时使用。

    有关详细信息请参阅 `RunnableBinding` 的文档。

    """

    bound: Runnable[Input, Output]
    """The underlying `Runnable` that this `Runnable` delegates to."""

    kwargs: Mapping[str, Any] = Field(default_factory=dict)
    """kwargs to pass to the underlying `Runnable` when running.

    For example, when the `Runnable` binding is invoked the underlying
    `Runnable` will be invoked with the same input but with these additional
    kwargs.

    """

    config: RunnableConfig = Field(default_factory=RunnableConfig)
    """The config to bind to the underlying `Runnable`."""

    config_factories: list[Callable[[RunnableConfig], RunnableConfig]] = Field(
        default_factory=list
    )
    """The config factories to bind to the underlying `Runnable`."""

    # Union[Type[Input], BaseModel] + things like list[str]
    custom_input_type: Any | None = None
    """Override the input type of the underlying `Runnable` with a custom type.

    The type can be a Pydantic model, or a type annotation (e.g., `list[str]`).
    """
    # Union[Type[Output], BaseModel] + things like list[str]
    custom_output_type: Any | None = None
    """Override the output type of the underlying `Runnable` with a custom type.

    The type can be a Pydantic model, or a type annotation (e.g., `list[str]`).
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def __init__(
        self,
        *,
        bound: Runnable[Input, Output],
        kwargs: Mapping[str, Any] | None = None,
        config: RunnableConfig | None = None,
        config_factories: list[Callable[[RunnableConfig], RunnableConfig]]
        | None = None,
        custom_input_type: type[Input] | BaseModel | None = None,
        custom_output_type: type[Output] | BaseModel | None = None,
        **other_kwargs: Any,
    ) -> None:
        """Create a `RunnableBinding` from a `Runnable` and kwargs.

        Args:
            bound: The underlying `Runnable` that this `Runnable` delegates calls
                to.
            kwargs: optional kwargs to pass to the underlying `Runnable`, when running
                the underlying `Runnable` (e.g., via `invoke`, `batch`,
                `transform`, or `stream` or async variants)

            config: optional config to bind to the underlying `Runnable`.

            config_factories: optional list of config factories to apply to the
                config before binding to the underlying `Runnable`.

            custom_input_type: Specify to override the input type of the underlying
                `Runnable` with a custom type.
            custom_output_type: Specify to override the output type of the underlying
                `Runnable` with a custom type.
            **other_kwargs: Unpacked into the base class.
        """
        super().__init__(
            bound=bound,
            kwargs=kwargs or {},
            config=config or {},
            config_factories=config_factories or [],
            custom_input_type=custom_input_type,
            custom_output_type=custom_output_type,
            **other_kwargs,
        )
        # 如果我们不在这里将 config 显式设置为 TypedDict，
        # 上面的 pydantic init 将剥离所有 "extra" 字段，
        # 即使 TypedDict 上的 total=False。
        self.config = config or {}

    @override
    def get_name(self, suffix: str | None = None, *, name: str | None = None) -> str:
        return self.bound.get_name(suffix, name=name)

    @property
    @override
    def InputType(self) -> type[Input]:
        return (
            cast("type[Input]", self.custom_input_type)
            if self.custom_input_type is not None
            else self.bound.InputType
        )

    @property
    @override
    def OutputType(self) -> type[Output]:
        return (
            cast("type[Output]", self.custom_output_type)
            if self.custom_output_type is not None
            else self.bound.OutputType
        )

    @override
    def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
        if self.custom_input_type is not None:
            return super().get_input_schema(config)
        return self.bound.get_input_schema(merge_configs(self.config, config))

    @override
    def get_output_schema(
        self, config: RunnableConfig | None = None
    ) -> type[BaseModel]:
        if self.custom_output_type is not None:
            return super().get_output_schema(config)
        return self.bound.get_output_schema(merge_configs(self.config, config))

    @property
    @override
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return self.bound.config_specs

    @override
    def get_graph(self, config: RunnableConfig | None = None) -> Graph:
        return self.bound.get_graph(self._merge_configs(config))

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        """Return `True` as this class is serializable."""
        return True

    @classmethod
    @override
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "schema", "runnable"]`
        """
        return ["langchain", "schema", "runnable"]

    def _merge_configs(self, *configs: RunnableConfig | None) -> RunnableConfig:
        config = merge_configs(self.config, *configs)
        return merge_configs(config, *(f(config) for f in self.config_factories))

    @override
    def invoke(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Output:
        return self.bound.invoke(
            input,
            self._merge_configs(config),
            **{**self.kwargs, **kwargs},
        )

    @override
    async def ainvoke(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Output:
        return await self.bound.ainvoke(
            input,
            self._merge_configs(config),
            **{**self.kwargs, **kwargs},
        )

    @override
    def batch(
        self,
        inputs: list[Input],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> list[Output]:
        if isinstance(config, list):
            configs = cast(
                "list[RunnableConfig]",
                [self._merge_configs(conf) for conf in config],
            )
        else:
            configs = [self._merge_configs(config) for _ in range(len(inputs))]
        return self.bound.batch(
            inputs,
            configs,
            return_exceptions=return_exceptions,
            **{**self.kwargs, **kwargs},
        )

    @override
    async def abatch(
        self,
        inputs: list[Input],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> list[Output]:
        if isinstance(config, list):
            configs = cast(
                "list[RunnableConfig]",
                [self._merge_configs(conf) for conf in config],
            )
        else:
            configs = [self._merge_configs(config) for _ in range(len(inputs))]
        return await self.bound.abatch(
            inputs,
            configs,
            return_exceptions=return_exceptions,
            **{**self.kwargs, **kwargs},
        )

    @overload
    def batch_as_completed(
        self,
        inputs: Sequence[Input],
        config: RunnableConfig | Sequence[RunnableConfig] | None = None,
        *,
        return_exceptions: Literal[False] = False,
        **kwargs: Any,
    ) -> Iterator[tuple[int, Output]]: ...

    @overload
    def batch_as_completed(
        self,
        inputs: Sequence[Input],
        config: RunnableConfig | Sequence[RunnableConfig] | None = None,
        *,
        return_exceptions: Literal[True],
        **kwargs: Any,
    ) -> Iterator[tuple[int, Output | Exception]]: ...

    @override
    def batch_as_completed(
        self,
        inputs: Sequence[Input],
        config: RunnableConfig | Sequence[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> Iterator[tuple[int, Output | Exception]]:
        if isinstance(config, Sequence):
            configs = cast(
                "list[RunnableConfig]",
                [self._merge_configs(conf) for conf in config],
            )
        else:
            configs = [self._merge_configs(config) for _ in range(len(inputs))]
        # 哈哈 mypy
        if return_exceptions:
            yield from self.bound.batch_as_completed(
                inputs,
                configs,
                return_exceptions=return_exceptions,
                **{**self.kwargs, **kwargs},
            )
        else:
            yield from self.bound.batch_as_completed(
                inputs,
                configs,
                return_exceptions=return_exceptions,
                **{**self.kwargs, **kwargs},
            )

    @overload
    def abatch_as_completed(
        self,
        inputs: Sequence[Input],
        config: RunnableConfig | Sequence[RunnableConfig] | None = None,
        *,
        return_exceptions: Literal[False] = False,
        **kwargs: Any | None,
    ) -> AsyncIterator[tuple[int, Output]]: ...

    @overload
    def abatch_as_completed(
        self,
        inputs: Sequence[Input],
        config: RunnableConfig | Sequence[RunnableConfig] | None = None,
        *,
        return_exceptions: Literal[True],
        **kwargs: Any | None,
    ) -> AsyncIterator[tuple[int, Output | Exception]]: ...

    @override
    async def abatch_as_completed(
        self,
        inputs: Sequence[Input],
        config: RunnableConfig | Sequence[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> AsyncIterator[tuple[int, Output | Exception]]:
        if isinstance(config, Sequence):
            configs = cast(
                "list[RunnableConfig]",
                [self._merge_configs(conf) for conf in config],
            )
        else:
            configs = [self._merge_configs(config) for _ in range(len(inputs))]
        if return_exceptions:
            async for item in self.bound.abatch_as_completed(
                inputs,
                configs,
                return_exceptions=return_exceptions,
                **{**self.kwargs, **kwargs},
            ):
                yield item
        else:
            async for item in self.bound.abatch_as_completed(
                inputs,
                configs,
                return_exceptions=return_exceptions,
                **{**self.kwargs, **kwargs},
            ):
                yield item

    @override
    def stream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Output]:
        yield from self.bound.stream(
            input,
            self._merge_configs(config),
            **{**self.kwargs, **kwargs},
        )

    @override
    async def astream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Output]:
        async for item in self.bound.astream(
            input,
            self._merge_configs(config),
            **{**self.kwargs, **kwargs},
        ):
            yield item

    @override
    async def astream_events(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[StreamEvent]:
        async for item in self.bound.astream_events(
            input, self._merge_configs(config), **{**self.kwargs, **kwargs}
        ):
            yield item

    @override
    def transform(
        self,
        input: Iterator[Input],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Output]:
        yield from self.bound.transform(
            input,
            self._merge_configs(config),
            **{**self.kwargs, **kwargs},
        )

    @override
    async def atransform(
        self,
        input: AsyncIterator[Input],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Output]:
        async for item in self.bound.atransform(
            input,
            self._merge_configs(config),
            **{**self.kwargs, **kwargs},
        ):
            yield item


class RunnableBinding(RunnableBindingBase[Input, Output]):  # type: ignore[no-redef]
    """用附加功能包装 `Runnable`。

    `RunnableBinding` 可以被看作是一个"可运行单元装饰器"，它保留了 `Runnable` 的基本特性；
    即批量调用、流式调用和异步支持，同时添加了额外的功能。

    任何继承自 `Runnable` 的类都可以绑定到 `RunnableBinding`。
    可运行单元暴露了一组标准方法，用于创建 `RunnableBinding`
    或 `RunnableBinding` 的子类（如 `RunnableRetry`、`RunnableWithFallbacks`），
    以添加额外的功能。

    这些方法包括：

    - `bind`: 绑定要传递给底层 `Runnable` 的 kwargs。
    - `with_config`: 绑定要传递给底层 `Runnable` 的配置。
    - `with_listeners`: 绑定生命周期监听器到底层 `Runnable`。
    - `with_types`: 覆盖底层 `Runnable` 的输入和输出类型。
    - `with_retry`: 绑定重试策略到底层 `Runnable`。
    - `with_fallbacks`: 绑定后备策略到底层 `Runnable`。

    示例：
    `bind`: 绑定要传递给底层 `Runnable` 的 kwargs。

        ```python
        # 创建一个 Runnable binding，在运行聊天模型时
        # 添加额外的 kwarg `stop=['-']`
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI()
        model.invoke('Say "Parrot-MAGIC"', stop=["-"])  # Should return `Parrot`
        # 通过 `bind` 方法的简单方式来使用，它返回一个新的
        # RunnableBinding
        runnable_binding = model.bind(stop=["-"])
        runnable_binding.invoke('Say "Parrot-MAGIC"')  # Should return `Parrot`
        ```
        也可以通过直接实例化 `RunnableBinding` 来完成（不推荐）：

        ```python
        from langchain_core.runnables import RunnableBinding

        runnable_binding = RunnableBinding(
            bound=model,
            kwargs={"stop": ["-"]},  # <-- 注意额外的 kwargs
        )
        runnable_binding.invoke('Say "Parrot-MAGIC"')  # Should return `Parrot`
        ```
    """

    @override
    def bind(self, **kwargs: Any) -> Runnable[Input, Output]:
        """Bind additional kwargs to a `Runnable`, returning a new `Runnable`.

        Args:
            **kwargs: The kwargs to bind to the `Runnable`.

        Returns:
            A new `Runnable` with the same type and config as the original,
            but with the additional kwargs bound.

        """
        return self.__class__(
            bound=self.bound,
            config=self.config,
            config_factories=self.config_factories,
            kwargs={**self.kwargs, **kwargs},
            custom_input_type=self.custom_input_type,
            custom_output_type=self.custom_output_type,
        )

    @override
    def with_config(
        self,
        config: RunnableConfig | None = None,
        # Sadly Unpack is not well supported by mypy so this will have to be untyped
        **kwargs: Any,
    ) -> Runnable[Input, Output]:
        return self.__class__(
            bound=self.bound,
            kwargs=self.kwargs,
            config=cast("RunnableConfig", {**self.config, **(config or {}), **kwargs}),
            config_factories=self.config_factories,
            custom_input_type=self.custom_input_type,
            custom_output_type=self.custom_output_type,
        )

    @override
    def with_listeners(
        self,
        *,
        on_start: Callable[[Run], None]
        | Callable[[Run, RunnableConfig], None]
        | None = None,
        on_end: Callable[[Run], None]
        | Callable[[Run, RunnableConfig], None]
        | None = None,
        on_error: Callable[[Run], None]
        | Callable[[Run, RunnableConfig], None]
        | None = None,
    ) -> Runnable[Input, Output]:
        """Bind lifecycle listeners to a `Runnable`, returning a new `Runnable`.

        The `Run` object contains information about the run, including its `id`,
        `type`, `input`, `output`, `error`, `start_time`, `end_time`, and
        any tags or metadata added to the run.

        Args:
            on_start: Called before the `Runnable` starts running, with the `Run`
                object.
            on_end: Called after the `Runnable` finishes running, with the `Run`
                object.
            on_error: Called if the `Runnable` throws an error, with the `Run`
                object.

        Returns:
            A new `Runnable` with the listeners bound.
        """

        def listener_config_factory(config: RunnableConfig) -> RunnableConfig:
            return {
                "callbacks": [
                    RootListenersTracer(
                        config=config,
                        on_start=on_start,
                        on_end=on_end,
                        on_error=on_error,
                    )
                ],
            }

        return self.__class__(
            bound=self.bound,
            kwargs=self.kwargs,
            config=self.config,
            config_factories=[listener_config_factory, *self.config_factories],
            custom_input_type=self.custom_input_type,
            custom_output_type=self.custom_output_type,
        )

    @override
    def with_types(
        self,
        input_type: type[Input] | BaseModel | None = None,
        output_type: type[Output] | BaseModel | None = None,
    ) -> Runnable[Input, Output]:
        return self.__class__(
            bound=self.bound,
            kwargs=self.kwargs,
            config=self.config,
            config_factories=self.config_factories,
            custom_input_type=(
                input_type if input_type is not None else self.custom_input_type
            ),
            custom_output_type=(
                output_type if output_type is not None else self.custom_output_type
            ),
        )

    @override
    def with_retry(self, **kwargs: Any) -> Runnable[Input, Output]:
        return self.__class__(
            bound=self.bound.with_retry(**kwargs),
            kwargs=self.kwargs,
            config=self.config,
            config_factories=self.config_factories,
        )

    @override
    def __getattr__(self, name: str) -> Any:  # type: ignore[misc]
        attr = getattr(self.bound, name)

        if callable(attr) and (
            config_param := inspect.signature(attr).parameters.get("config")
        ):
            if config_param.kind == inspect.Parameter.KEYWORD_ONLY:

                @wraps(attr)
                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    return attr(
                        *args,
                        config=merge_configs(self.config, kwargs.pop("config", None)),
                        **kwargs,
                    )

                return wrapper
            if config_param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                idx = list(inspect.signature(attr).parameters).index("config")

                @wraps(attr)
                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    if len(args) >= idx + 1:
                        argsl = list(args)
                        argsl[idx] = merge_configs(self.config, argsl[idx])
                        return attr(*argsl, **kwargs)
                    return attr(
                        *args,
                        config=merge_configs(self.config, kwargs.pop("config", None)),
                        **kwargs,
                    )

                return wrapper

        return attr


class _RunnableCallableSync(Protocol[Input, Output]):
    def __call__(self, _in: Input, /, *, config: RunnableConfig) -> Output: ...


class _RunnableCallableAsync(Protocol[Input, Output]):
    def __call__(
        self, _in: Input, /, *, config: RunnableConfig
    ) -> Awaitable[Output]: ...


class _RunnableCallableIterator(Protocol[Input, Output]):
    def __call__(
        self, _in: Iterator[Input], /, *, config: RunnableConfig
    ) -> Iterator[Output]: ...


class _RunnableCallableAsyncIterator(Protocol[Input, Output]):
    def __call__(
        self, _in: AsyncIterator[Input], /, *, config: RunnableConfig
    ) -> AsyncIterator[Output]: ...


RunnableLike = (
    Runnable[Input, Output]
    | Callable[[Input], Output]
    | Callable[[Input], Awaitable[Output]]
    | Callable[[Iterator[Input]], Iterator[Output]]
    | Callable[[AsyncIterator[Input]], AsyncIterator[Output]]
    | _RunnableCallableSync[Input, Output]
    | _RunnableCallableAsync[Input, Output]
    | _RunnableCallableIterator[Input, Output]
    | _RunnableCallableAsyncIterator[Input, Output]
    | Mapping[str, Any]
)


def coerce_to_runnable(thing: RunnableLike) -> Runnable[Input, Output]:
    """将类 `Runnable` 对象强制转换为 `Runnable`。

    Args:
        thing: 类 `Runnable` 对象。

    Returns:
        一个 `Runnable`。

    Raises:
        TypeError: 如果对象不是类 `Runnable`。
    """
    if isinstance(thing, Runnable):
        return thing
    if is_async_generator(thing) or inspect.isgeneratorfunction(thing):
        return RunnableGenerator(thing)
    if callable(thing):
        return RunnableLambda(cast("Callable[[Input], Output]", thing))
    if isinstance(thing, dict):
        return cast("Runnable[Input, Output]", RunnableParallel(thing))
    msg = (
        f"Expected a Runnable, callable or dict."
        f"Instead got an unsupported type: {type(thing)}"
    )
    raise TypeError(msg)


@overload
def chain(
    func: Callable[[Input], Coroutine[Any, Any, Output]],
) -> Runnable[Input, Output]: ...


@overload
def chain(
    func: Callable[[Input], Iterator[Output]],
) -> Runnable[Input, Output]: ...


@overload
def chain(
    func: Callable[[Input], AsyncIterator[Output]],
) -> Runnable[Input, Output]: ...


@overload
def chain(
    func: Callable[[Input], Output],
) -> Runnable[Input, Output]: ...


def chain(
    func: Callable[[Input], Output]
    | Callable[[Input], Iterator[Output]]
    | Callable[[Input], Coroutine[Any, Any, Output]]
    | Callable[[Input], AsyncIterator[Output]],
) -> Runnable[Input, Output]:
    """装饰函数以使其成为 `Runnable`。

    将 `Runnable` 的名称设置为函数的名称。
    函数调用的任何 runnable 将被追踪为依赖项。

    Args:
        func: 一个 `Callable`。

    Returns:
        一个 `Runnable`。

    Example:
        ```python
        from langchain_core.runnables import chain
        from langchain_core.prompts import PromptTemplate
        from langchain_openai import OpenAI


        @chain
        def my_func(fields):
            prompt = PromptTemplate("Hello, {name}!")
            model = OpenAI()
            formatted = prompt.invoke(**fields)

            for chunk in model.stream(formatted):
                yield chunk
        ```
    """
    return RunnableLambda(func)
