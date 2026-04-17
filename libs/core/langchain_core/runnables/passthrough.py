"""直通链（RunnablePassthrough）的实现。"""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Awaitable, Callable
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

from pydantic import BaseModel, RootModel
from typing_extensions import override

from langchain_core.runnables.base import (
    Other,
    Runnable,
    RunnableParallel,
    RunnableSerializable,
)
from langchain_core.runnables.config import (
    RunnableConfig,
    acall_func_with_variable_args,
    call_func_with_variable_args,
    ensure_config,
    get_executor_for_config,
    patch_config,
)
from langchain_core.runnables.utils import (
    AddableDict,
    ConfigurableFieldSpec,
)
from langchain_core.utils.aiter import atee
from langchain_core.utils.iter import safetee
from langchain_core.utils.pydantic import create_model_v2

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Mapping

    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForChainRun,
        CallbackManagerForChainRun,
    )
    from langchain_core.runnables.graph import Graph


def identity(x: Other) -> Other:
    """恒等函数。

    Args:
        x: 输入。

    Returns:
        输出。
    """
    return x


async def aidentity(x: Other) -> Other:
    """异步恒等函数。

    Args:
        x: 输入。

    Returns:
        输出。
    """
    return x


class RunnablePassthrough(RunnableSerializable[Other, Other]):
    """直通链（RunnablePassthrough）：原样传递输入，或添加额外的键。

    此 Runnable 的行为与恒等函数几乎相同，只是如果输入是字典，
    则可以配置为向输出添加额外的键。

    以下示例演示了此 Runnable 如何通过几个简单的链工作。
    这些链依赖于简单的 lambda，以便于执行和实验。

    Examples:
        ```python
        from langchain_core.runnables import (
            RunnableLambda,
            RunnableParallel,
            RunnablePassthrough,
        )

        runnable = RunnableParallel(
            origin=RunnablePassthrough(), modified=lambda x: x + 1
        )

        runnable.invoke(1)  # {'origin': 1, 'modified': 2}


        def fake_llm(prompt: str) -> str:  # 示例用的假LLM
            return "completion"


        chain = RunnableLambda(fake_llm) | {
            "original": RunnablePassthrough(),  # 原始 LLM 输出
            "parsed": lambda text: text[::-1],  # 解析逻辑
        }

        chain.invoke("hello")  # {'original': 'completion', 'parsed': 'noitelpmoc'}
        ```

    在某些情况下，可能需要在传递输入的同时向输出添加一些键。
    在这种情况下，可以使用 `assign` 方法：

        ```python
        from langchain_core.runnables import RunnablePassthrough


        def fake_llm(prompt: str) -> str:  # 示例用的假LLM
            return "completion"


        runnable = {
            "llm1": fake_llm,
            "llm2": fake_llm,
        } | RunnablePassthrough.assign(
            total_chars=lambda inputs: len(inputs["llm1"] + inputs["llm2"])
        )

        runnable.invoke("hello")
        # {'llm1': 'completion', 'llm2': 'completion', 'total_chars': 20}
        ```
    """

    input_type: type[Other] | None = None

    func: Callable[[Other], None] | Callable[[Other, RunnableConfig], None] | None = (
        None
    )

    afunc: (
        Callable[[Other], Awaitable[None]]
        | Callable[[Other, RunnableConfig], Awaitable[None]]
        | None
    ) = None

    @override
    def __repr_args__(self) -> Any:
        # 没有这个 repr(self) 会引发 RecursionError
        # 见 https://github.com/pydantic/pydantic/issues/7327
        return []

    def __init__(
        self,
        func: Callable[[Other], None]
        | Callable[[Other, RunnableConfig], None]
        | Callable[[Other], Awaitable[None]]
        | Callable[[Other, RunnableConfig], Awaitable[None]]
        | None = None,
        afunc: Callable[[Other], Awaitable[None]]
        | Callable[[Other, RunnableConfig], Awaitable[None]]
        | None = None,
        *,
        input_type: type[Other] | None = None,
        **kwargs: Any,
    ) -> None:
        """创建直通链（RunnablePassthrough）。

        Args:
            func: 要使用输入调用的函数。
            afunc: 要使用输入调用的异步函数。
            input_type: 输入的类型。
        """
        if inspect.iscoroutinefunction(func):
            afunc = func
            func = None

        super().__init__(func=func, afunc=afunc, input_type=input_type, **kwargs)

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        """返回 `True`，因为此类可序列化。"""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            `["langchain", "schema", "runnable"]`
        """
        return ["langchain", "schema", "runnable"]

    @property
    @override
    def InputType(self) -> Any:
        return self.input_type or Any

    @property
    @override
    def OutputType(self) -> Any:
        return self.input_type or Any

    @classmethod
    @override
    def assign(
        cls,
        **kwargs: Runnable[dict[str, Any], Any]
        | Callable[[dict[str, Any]], Any]
        | Mapping[str, Runnable[dict[str, Any], Any] | Callable[[dict[str, Any]], Any]],
    ) -> RunnableAssign:
        """将 Dict 输入与映射参数生成的输出合并。

        Args:
            **kwargs: `Runnable`、`Callable` 或从键到 `Runnable` 对象或 `Callable` 的映射。

        Returns:
            一个将 dict 输入与映射参数生成的输出合并的 Runnable。
        """
        return RunnableAssign(RunnableParallel[dict[str, Any]](kwargs))

    @override
    def invoke(
        self, input: Other, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Other:
        if self.func is not None:
            call_func_with_variable_args(
                self.func, input, ensure_config(config), **kwargs
            )
        return self._call_with_config(identity, input, config)

    @override
    async def ainvoke(
        self,
        input: Other,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Other:
        if self.afunc is not None:
            await acall_func_with_variable_args(
                self.afunc, input, ensure_config(config), **kwargs
            )
        elif self.func is not None:
            call_func_with_variable_args(
                self.func, input, ensure_config(config), **kwargs
            )
        return await self._acall_with_config(aidentity, input, config)

    @override
    def transform(
        self,
        input: Iterator[Other],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Other]:
        if self.func is None:
            for chunk in self._transform_stream_with_config(input, identity, config):
                yield chunk
        else:
            final: Other
            got_first_chunk = False

            for chunk in self._transform_stream_with_config(input, identity, config):
                yield chunk

                if not got_first_chunk:
                    final = chunk
                    got_first_chunk = True
                else:
                    try:
                        final = final + chunk  # type: ignore[operator]
                    except TypeError:
                        final = chunk

            if got_first_chunk:
                call_func_with_variable_args(
                    self.func, final, ensure_config(config), **kwargs
                )

    @override
    async def atransform(
        self,
        input: AsyncIterator[Other],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Other]:
        if self.afunc is None and self.func is None:
            async for chunk in self._atransform_stream_with_config(
                input, identity, config
            ):
                yield chunk
        else:
            got_first_chunk = False

            async for chunk in self._atransform_stream_with_config(
                input, identity, config
            ):
                yield chunk

                # 根据定义，函数将对聚合的输入进行操作。
                # 因此，我们将聚合输入，直到到达最后一个块。
                # 如果输入不可添加，那么我们将假设只能对最后一个块进行操作。
                if not got_first_chunk:
                    final = chunk
                    got_first_chunk = True
                else:
                    try:
                        final = final + chunk  # type: ignore[operator]
                    except TypeError:
                        final = chunk

            if got_first_chunk:
                config = ensure_config(config)
                if self.afunc is not None:
                    await acall_func_with_variable_args(
                        self.afunc, final, config, **kwargs
                    )
                elif self.func is not None:
                    call_func_with_variable_args(self.func, final, config, **kwargs)

    @override
    def stream(
        self,
        input: Other,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Other]:
        return self.transform(iter([input]), config, **kwargs)

    @override
    async def astream(
        self,
        input: Other,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Other]:
        async def input_aiter() -> AsyncIterator[Other]:
            yield input

        async for chunk in self.atransform(input_aiter(), config, **kwargs):
            yield chunk


_graph_passthrough: RunnablePassthrough = RunnablePassthrough()


class RunnableAssign(RunnableSerializable[dict[str, Any], dict[str, Any]]):
    """赋值链（RunnableAssign）：向 `dict[str, Any]` 输入分配键值对。

    赋值链类通过 `RunnableParallel`（并行链）实例获取输入字典，
    应用转换，然后将这些转换与原始数据组合，
    根据映射器的逻辑引入新的键值对。

    Examples:
        ```python
        # 这是一个 RunnableAssign
        from langchain_core.runnables.passthrough import (
            RunnableAssign,
            RunnableParallel,
        )
        from langchain_core.runnables.base import RunnableLambda


        def add_ten(x: dict[str, int]) -> dict[str, int]:
            return {"added": x["input"] + 10}


        mapper = RunnableParallel(
            {
                "add_step": RunnableLambda(add_ten),
            }
        )

        runnable_assign = RunnableAssign(mapper)

        # 同步示例
        runnable_assign.invoke({"input": 5})
        # 返回 {'input': 5, 'add_step': {'added': 15}}

        # 异步示例
        await runnable_assign.ainvoke({"input": 5})
        # 返回 {'input': 5, 'add_step': {'added': 15}}
        ```
    """

    mapper: RunnableParallel

    def __init__(self, mapper: RunnableParallel[dict[str, Any]], **kwargs: Any) -> None:
        """创建赋值链（RunnableAssign）。

        Args:
            mapper: 一个 `RunnableParallel`（并行链）实例，用于转换输入字典。
        """
        super().__init__(mapper=mapper, **kwargs)

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        """返回 `True`，因为此类可序列化。"""
        return True

    @classmethod
    @override
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            `["langchain", "schema", "runnable"]`
        """
        return ["langchain", "schema", "runnable"]

    @override
    def get_name(self, suffix: str | None = None, *, name: str | None = None) -> str:
        name = (
            name
            or self.name
            or f"RunnableAssign<{','.join(self.mapper.steps__.keys())}>"
        )
        return super().get_name(suffix, name=name)

    @override
    def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
        map_input_schema = self.mapper.get_input_schema(config)
        if not issubclass(map_input_schema, RootModel):
            # ie. it's a dict
            return map_input_schema

        return super().get_input_schema(config)

    @override
    def get_output_schema(
        self, config: RunnableConfig | None = None
    ) -> type[BaseModel]:
        map_input_schema = self.mapper.get_input_schema(config)
        map_output_schema = self.mapper.get_output_schema(config)
        if not issubclass(map_input_schema, RootModel) and not issubclass(
            map_output_schema, RootModel
        ):
            fields = {}

            for name, field_info in map_input_schema.model_fields.items():
                fields[name] = (field_info.annotation, field_info.default)

            for name, field_info in map_output_schema.model_fields.items():
                fields[name] = (field_info.annotation, field_info.default)

            return create_model_v2("RunnableAssignOutput", field_definitions=fields)
        if not issubclass(map_output_schema, RootModel):
            # 即只有映射输出是字典
            # 即输入类型未知或推断不正确
            return map_output_schema

        return super().get_output_schema(config)

    @property
    @override
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return self.mapper.config_specs

    @override
    def get_graph(self, config: RunnableConfig | None = None) -> Graph:
        # 从映射器获取图
        graph = self.mapper.get_graph(config)
        # 添加直通节点和边
        input_node = graph.first_node()
        output_node = graph.last_node()
        if input_node is not None and output_node is not None:
            passthrough_node = graph.add_node(_graph_passthrough)
            graph.add_edge(input_node, passthrough_node)
            graph.add_edge(passthrough_node, output_node)
        return graph

    def _invoke(
        self,
        value: dict[str, Any],
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not isinstance(value, dict):
            msg = "The input to RunnablePassthrough.assign() must be a dict."
            raise ValueError(msg)  # noqa: TRY004

        return {
            **value,
            **self.mapper.invoke(
                value,
                patch_config(config, callbacks=run_manager.get_child()),
                **kwargs,
            ),
        }

    @override
    def invoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._call_with_config(self._invoke, input, config, **kwargs)

    async def _ainvoke(
        self,
        value: dict[str, Any],
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not isinstance(value, dict):
            msg = "The input to RunnablePassthrough.assign() must be a dict."
            raise ValueError(msg)  # noqa: TRY004

        return {
            **value,
            **await self.mapper.ainvoke(
                value,
                patch_config(config, callbacks=run_manager.get_child()),
                **kwargs,
            ),
        }

    @override
    async def ainvoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._acall_with_config(self._ainvoke, input, config, **kwargs)

    def _transform(
        self,
        values: Iterator[dict[str, Any]],
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        # 收集映射器的键
        mapper_keys = set(self.mapper.steps__.keys())
        # 创建两个流，一个用于映射，一个用于直通
        for_passthrough, for_map = safetee(values, 2, lock=threading.Lock())

        # 创建映射输出流
        map_output = self.mapper.transform(
            for_map,
            patch_config(
                config,
                callbacks=run_manager.get_child(),
            ),
            **kwargs,
        )

        # 获取执行器以在后台启动映射输出流
        with get_executor_for_config(config) as executor:
            # 启动映射输出流
            first_map_chunk_future = executor.submit(
                next,
                map_output,
                None,
            )
            # 消费直通流
            for chunk in for_passthrough:
                if not isinstance(chunk, dict):
                    msg = "The input to RunnablePassthrough.assign() must be a dict."
                    raise ValueError(msg)  # noqa: TRY004
                # 从直通块中移除映射器的键，以便被映射覆盖
                filtered = AddableDict(
                    {k: v for k, v in chunk.items() if k not in mapper_keys}
                )
                if filtered:
                    yield filtered
            # 生成映射输出
            yield cast("dict[str, Any]", first_map_chunk_future.result())
            for chunk in map_output:
                yield chunk

    @override
    def transform(
        self,
        input: Iterator[dict[str, Any]],
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[dict[str, Any]]:
        yield from self._transform_stream_with_config(
            input, self._transform, config, **kwargs
        )

    async def _atransform(
        self,
        values: AsyncIterator[dict[str, Any]],
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        # 收集映射器的键
        mapper_keys = set(self.mapper.steps__.keys())
        # 创建两个流，一个用于映射，一个用于直通
        for_passthrough, for_map = atee(values, 2, lock=asyncio.Lock())
        # 创建映射输出流
        map_output = self.mapper.atransform(
            for_map,
            patch_config(
                config,
                callbacks=run_manager.get_child(),
            ),
            **kwargs,
        )
        # 启动映射输出流
        first_map_chunk_task: asyncio.Task = asyncio.create_task(
            anext(map_output, None),
        )
        # 消费直通流
        async for chunk in for_passthrough:
            if not isinstance(chunk, dict):
                msg = "The input to RunnablePassthrough.assign() must be a dict."
                raise ValueError(msg)  # noqa: TRY004

            # 从直通块中移除映射器的键，以便被映射输出覆盖
            filtered = AddableDict(
                {k: v for k, v in chunk.items() if k not in mapper_keys}
            )
            if filtered:
                yield filtered
        # 生成映射输出
        yield await first_map_chunk_task
        async for chunk in map_output:
            yield chunk

    @override
    async def atransform(
        self,
        input: AsyncIterator[dict[str, Any]],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        async for chunk in self._atransform_stream_with_config(
            input, self._atransform, config, **kwargs
        ):
            yield chunk

    @override
    def stream(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        return self.transform(iter([input]), config, **kwargs)

    @override
    async def astream(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        async def input_aiter() -> AsyncIterator[dict[str, Any]]:
            yield input

        async for chunk in self.atransform(input_aiter(), config, **kwargs):
            yield chunk


class RunnablePick(RunnableSerializable[dict[str, Any], Any]):
    """选择链（RunnablePick）：从 `dict[str, Any]` 输入中选择键。

    选择链类表示一个 Runnable，它可以有选择地从字典输入中选取键。
    它允许您指定一个或多个键来从输入字典中提取。

    !!! note "返回类型行为"
        返回类型取决于 `keys` 参数：

        - 当 `keys` 是 `str` 时：返回与该键关联的单个值
        - 当 `keys` 是 `list` 时：返回一个仅包含所选键的字典

    Example:
        ```python
        from langchain_core.runnables.passthrough import RunnablePick

        input_data = {
            "name": "John",
            "age": 30,
            "city": "New York",
            "country": "USA",
        }

        # 单个键 - 直接返回值
        runnable_single = RunnablePick(keys="name")
        result_single = runnable_single.invoke(input_data)
        print(result_single)  # 输出: "John"

        # 多个键 - 返回字典
        runnable_multiple = RunnablePick(keys=["name", "age"])
        result_multiple = runnable_multiple.invoke(input_data)
        print(result_multiple)  # 输出: {'name': 'John', 'age': 30}
        ```
    """

    keys: str | list[str]

    def __init__(self, keys: str | list[str], **kwargs: Any) -> None:
        """创建选择链（RunnablePick）。

        Args:
            keys: 要从输入字典中选取的单个键或键列表。
        """
        super().__init__(keys=keys, **kwargs)

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        """返回 `True`，因为此类可序列化。"""
        return True

    @classmethod
    @override
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            `["langchain", "schema", "runnable"]`
        """
        return ["langchain", "schema", "runnable"]

    @override
    def get_name(self, suffix: str | None = None, *, name: str | None = None) -> str:
        name = (
            name
            or self.name
            or "RunnablePick"
            f"<{','.join([self.keys] if isinstance(self.keys, str) else self.keys)}>"
        )
        return super().get_name(suffix, name=name)

    def _pick(self, value: dict[str, Any]) -> Any:
        if not isinstance(value, dict):
            msg = "The input to RunnablePassthrough.assign() must be a dict."
            raise ValueError(msg)  # noqa: TRY004

        if isinstance(self.keys, str):
            return value.get(self.keys)
        picked = {k: value.get(k) for k in self.keys if k in value}
        if picked:
            return AddableDict(picked)
        return None

    @override
    def invoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        return self._call_with_config(self._pick, input, config, **kwargs)

    async def _ainvoke(
        self,
        value: dict[str, Any],
    ) -> Any:
        return self._pick(value)

    @override
    async def ainvoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        return await self._acall_with_config(self._ainvoke, input, config, **kwargs)

    def _transform(
        self,
        chunks: Iterator[dict[str, Any]],
    ) -> Iterator[Any]:
        for chunk in chunks:
            picked = self._pick(chunk)
            if picked is not None:
                yield picked

    @override
    def transform(
        self,
        input: Iterator[dict[str, Any]],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        yield from self._transform_stream_with_config(
            input, self._transform, config, **kwargs
        )

    async def _atransform(
        self,
        chunks: AsyncIterator[dict[str, Any]],
    ) -> AsyncIterator[Any]:
        async for chunk in chunks:
            picked = self._pick(chunk)
            if picked is not None:
                yield picked

    @override
    async def atransform(
        self,
        input: AsyncIterator[dict[str, Any]],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        async for chunk in self._atransform_stream_with_config(
            input, self._atransform, config, **kwargs
        ):
            yield chunk

    @override
    def stream(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        return self.transform(iter([input]), config, **kwargs)

    @override
    async def astream(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        async def input_aiter() -> AsyncIterator[dict[str, Any]]:
            yield input

        async for chunk in self.atransform(input_aiter(), config, **kwargs):
            yield chunk
