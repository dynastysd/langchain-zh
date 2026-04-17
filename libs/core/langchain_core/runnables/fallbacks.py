"""当失败时可以降级到其他 Runnable 对象的 Runnable。"""

import asyncio
import inspect
import typing
from collections.abc import AsyncIterator, Iterator, Sequence
from functools import wraps
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ConfigDict
from typing_extensions import override

from langchain_core.callbacks.manager import AsyncCallbackManager, CallbackManager
from langchain_core.runnables.base import Runnable, RunnableSerializable
from langchain_core.runnables.config import (
    RunnableConfig,
    ensure_config,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
    get_config_list,
    patch_config,
    set_config_context,
)
from langchain_core.runnables.utils import (
    ConfigurableFieldSpec,
    Input,
    Output,
    coro_with_context,
    get_unique_config_specs,
)

if TYPE_CHECKING:
    from langchain_core.callbacks.manager import AsyncCallbackManagerForChainRun


class RunnableWithFallbacks(RunnableSerializable[Input, Output]):
    """当失败时可以降级到其他 Runnable 对象的 Runnable。

    外部 API（例如语言模型的 API）有时可能会出现性能下降甚至停机。

    在这些情况下，有一个降级 Runnable 会很有用，它可以用来替代原始
    Runnable（例如降级到另一个 LLM 提供者）。

    降级可以在单个 Runnable 级别或 Runnable 链的级别定义。降级按顺序尝试，
    直到一个成功或全部失败。

    虽然你可以直接实例化 `RunnableWithFallbacks`，但通常更方便的是在
    Runnable 上使用 `with_fallbacks` 方法。

    示例：
        ```python
        from langchain_core.chat_models.openai import ChatOpenAI
        from langchain_core.chat_models.anthropic import ChatAnthropic

        model = ChatAnthropic(model="claude-sonnet-4-6").with_fallbacks(
            [ChatOpenAI(model="gpt-5.4-mini")]
        )
        # 通常使用 ChatAnthropic，但如果 ChatAnthropic 失败，
        # 则降级到 ChatOpenAI。
        model.invoke("hello")

        # 也可以在链的级别使用降级。
        # 如果两个 LLM 提供者都失败，我们将降级到一个好的硬编码响应。

        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parser import StrOutputParser
        from langchain_core.runnables import RunnableLambda


        def when_all_is_lost(inputs):
            return (
                "Looks like our LLM providers are down. "
                "Here's a nice 🦜️ emoji for you instead."
            )


        chain_with_fallback = (
            PromptTemplate.from_template("Tell me a joke about {topic}")
            | model
            | StrOutputParser()
        ).with_fallbacks([RunnableLambda(when_all_is_lost)])
        ```
    """

    runnable: Runnable[Input, Output]
    """首先运行的 Runnable。"""
    fallbacks: Sequence[Runnable[Input, Output]]
    """要尝试的降级序列。"""
    exceptions_to_handle: tuple[type[BaseException], ...] = (Exception,)
    """应该尝试降级的异常类型。

    任何不是这些异常子类的异常都会立即重新抛出。
    """
    exception_key: str | None = None
    """如果指定了字符串，则将处理的异常作为输入
    的一部分传递给降级，放在指定的键下。

    如果为 `None`，异常不会传递给降级。

    如果使用，基础 Runnable 及其降级必须接受字典作为输入。
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @property
    @override
    def InputType(self) -> type[Input]:
        return self.runnable.InputType

    @property
    @override
    def OutputType(self) -> type[Output]:
        return self.runnable.OutputType

    @override
    def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
        return self.runnable.get_input_schema(config)

    @override
    def get_output_schema(
        self, config: RunnableConfig | None = None
    ) -> type[BaseModel]:
        return self.runnable.get_output_schema(config)

    @property
    @override
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return get_unique_config_specs(
            spec
            for step in [self.runnable, *self.fallbacks]
            for spec in step.config_specs
        )

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        """返回 `True`，因为此类是可序列化的。"""
        return True

    @classmethod
    @override
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        返回:
            `["langchain", "schema", "runnable"]`
        """
        return ["langchain", "schema", "runnable"]

    @property
    def runnables(self) -> Iterator[Runnable[Input, Output]]:
        """Runnable 及其降级的迭代器。

        产出:
            Runnable 及其降级。
        """
        yield self.runnable
        yield from self.fallbacks

    @override
    def invoke(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        if self.exception_key is not None and not isinstance(input, dict):
            msg = (
                "If 'exception_key' is specified then input must be a dictionary."
                f"However found a type of {type(input)} for input"
            )
            raise ValueError(msg)
        # 设置回调
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        # 启动根运行
        run_manager = callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        first_error = None
        last_error = None
        for runnable in self.runnables:
            try:
                if self.exception_key and last_error is not None:
                    input[self.exception_key] = last_error  # type: ignore[index]
                child_config = patch_config(config, callbacks=run_manager.get_child())
                with set_config_context(child_config) as context:
                    output = context.run(
                        runnable.invoke,
                        input,
                        config,
                        **kwargs,
                    )
            except self.exceptions_to_handle as e:
                if first_error is None:
                    first_error = e
                last_error = e
            except BaseException as e:
                run_manager.on_chain_error(e)
                raise
            else:
                run_manager.on_chain_end(output)
                return output
        if first_error is None:
            msg = "No error stored at end of fallbacks."
            raise ValueError(msg)
        run_manager.on_chain_error(first_error)
        raise first_error

    @override
    async def ainvoke(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Output:
        if self.exception_key is not None and not isinstance(input, dict):
            msg = (
                "If 'exception_key' is specified then input must be a dictionary."
                f"However found a type of {type(input)} for input"
            )
            raise ValueError(msg)
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

        first_error = None
        last_error = None
        for runnable in self.runnables:
            try:
                if self.exception_key and last_error is not None:
                    input[self.exception_key] = last_error  # type: ignore[index]
                child_config = patch_config(config, callbacks=run_manager.get_child())
                with set_config_context(child_config) as context:
                    coro = context.run(runnable.ainvoke, input, config, **kwargs)
                    output = await coro_with_context(coro, context)
            except self.exceptions_to_handle as e:
                if first_error is None:
                    first_error = e
                last_error = e
            except BaseException as e:
                await run_manager.on_chain_error(e)
                raise
            else:
                await run_manager.on_chain_end(output)
                return output
        if first_error is None:
            msg = "No error stored at end of fallbacks."
            raise ValueError(msg)
        await run_manager.on_chain_error(first_error)
        raise first_error

    @override
    def batch(
        self,
        inputs: list[Input],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> list[Output]:
        if self.exception_key is not None and not all(
            isinstance(input_, dict) for input_ in inputs
        ):
            msg = (
                "If 'exception_key' is specified then inputs must be dictionaries."
                f"However found a type of {type(inputs[0])} for input"
            )
            raise ValueError(msg)

        if not inputs:
            return []

        # 设置回调
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
                input_ if isinstance(input_, dict) else {"input": input_},
                name=config.get("run_name") or self.get_name(),
                run_id=config.pop("run_id", None),
            )
            for cm, input_, config in zip(
                callback_managers, inputs, configs, strict=False
            )
        ]

        to_return: dict[int, Any] = {}
        run_again = dict(enumerate(inputs))
        handled_exceptions: dict[int, BaseException] = {}
        first_to_raise = None
        for runnable in self.runnables:
            outputs = runnable.batch(
                [input_ for _, input_ in sorted(run_again.items())],
                [
                    # each step a child run of the corresponding root run
                    patch_config(configs[i], callbacks=run_managers[i].get_child())
                    for i in sorted(run_again)
                ],
                return_exceptions=True,
                **kwargs,
            )
            for (i, input_), output in zip(
                sorted(run_again.copy().items()), outputs, strict=False
            ):
                if isinstance(output, BaseException) and not isinstance(
                    output, self.exceptions_to_handle
                ):
                    if not return_exceptions:
                        first_to_raise = first_to_raise or output
                    else:
                        handled_exceptions[i] = output
                    run_again.pop(i)
                elif isinstance(output, self.exceptions_to_handle):
                    if self.exception_key:
                        input_[self.exception_key] = output  # type: ignore[index]
                    handled_exceptions[i] = output
                else:
                    run_managers[i].on_chain_end(output)
                    to_return[i] = output
                    run_again.pop(i)
                    handled_exceptions.pop(i, None)
            if first_to_raise:
                raise first_to_raise
            if not run_again:
                break

        sorted_handled_exceptions = sorted(handled_exceptions.items())
        for i, error in sorted_handled_exceptions:
            run_managers[i].on_chain_error(error)
        if not return_exceptions and sorted_handled_exceptions:
            raise sorted_handled_exceptions[0][1]
        to_return.update(handled_exceptions)
        return [output for _, output in sorted(to_return.items())]

    @override
    async def abatch(
        self,
        inputs: list[Input],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> list[Output]:
        if self.exception_key is not None and not all(
            isinstance(input_, dict) for input_ in inputs
        ):
            msg = (
                "If 'exception_key' is specified then inputs must be dictionaries."
                f"However found a type of {type(inputs[0])} for input"
            )
            raise ValueError(msg)

        if not inputs:
            return []

        # 设置回调
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

        to_return: dict[int, Output | BaseException] = {}
        run_again = dict(enumerate(inputs))
        handled_exceptions: dict[int, BaseException] = {}
        first_to_raise = None
        for runnable in self.runnables:
            outputs = await runnable.abatch(
                [input_ for _, input_ in sorted(run_again.items())],
                [
                    # 每个步骤是对应根运行的子运行
                    patch_config(configs[i], callbacks=run_managers[i].get_child())
                    for i in sorted(run_again)
                ],
                return_exceptions=True,
                **kwargs,
            )

            for (i, input_), output in zip(
                sorted(run_again.copy().items()), outputs, strict=False
            ):
                if isinstance(output, BaseException) and not isinstance(
                    output, self.exceptions_to_handle
                ):
                    if not return_exceptions:
                        first_to_raise = first_to_raise or output
                    else:
                        handled_exceptions[i] = output
                    run_again.pop(i)
                elif isinstance(output, self.exceptions_to_handle):
                    if self.exception_key:
                        input_[self.exception_key] = output  # type: ignore[index]
                    handled_exceptions[i] = output
                else:
                    to_return[i] = output
                    await run_managers[i].on_chain_end(output)
                    run_again.pop(i)
                    handled_exceptions.pop(i, None)

            if first_to_raise:
                raise first_to_raise
            if not run_again:
                break

        sorted_handled_exceptions = sorted(handled_exceptions.items())
        await asyncio.gather(
            *(
                run_managers[i].on_chain_error(error)
                for i, error in sorted_handled_exceptions
            )
        )
        if not return_exceptions and sorted_handled_exceptions:
            raise sorted_handled_exceptions[0][1]
        to_return.update(handled_exceptions)
        return [cast("Output", output) for _, output in sorted(to_return.items())]

    @override
    def stream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Output]:
        if self.exception_key is not None and not isinstance(input, dict):
            msg = (
                "If 'exception_key' is specified then input must be a dictionary."
                f"However found a type of {type(input)} for input"
            )
            raise ValueError(msg)
        # 设置回调
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        # 启动根运行
        run_manager = callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        first_error = None
        last_error = None
        for runnable in self.runnables:
            try:
                if self.exception_key and last_error is not None:
                    input[self.exception_key] = last_error  # type: ignore[index]
                child_config = patch_config(config, callbacks=run_manager.get_child())
                with set_config_context(child_config) as context:
                    stream = context.run(
                        runnable.stream,
                        input,
                        **kwargs,
                    )
                    chunk: Output = context.run(next, stream)
            except self.exceptions_to_handle as e:
                first_error = e if first_error is None else first_error
                last_error = e
            except BaseException as e:
                run_manager.on_chain_error(e)
                raise
            else:
                first_error = None
                break
        if first_error:
            run_manager.on_chain_error(first_error)
            raise first_error

        yield chunk
        output: Output | None = chunk
        try:
            for chunk in stream:
                yield chunk
                try:
                    output = output + chunk  # type: ignore[operator]
                except TypeError:
                    output = None
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        run_manager.on_chain_end(output)

    @override
    async def astream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Output]:
        if self.exception_key is not None and not isinstance(input, dict):
            msg = (
                "If 'exception_key' is specified then input must be a dictionary."
                f"However found a type of {type(input)} for input"
            )
            raise ValueError(msg)
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
        first_error = None
        last_error = None
        for runnable in self.runnables:
            try:
                if self.exception_key and last_error is not None:
                    input[self.exception_key] = last_error  # type: ignore[index]
                child_config = patch_config(config, callbacks=run_manager.get_child())
                with set_config_context(child_config) as context:
                    stream = runnable.astream(
                        input,
                        child_config,
                        **kwargs,
                    )
                    chunk = await coro_with_context(anext(stream), context)
            except self.exceptions_to_handle as e:
                first_error = e if first_error is None else first_error
                last_error = e
            except BaseException as e:
                await run_manager.on_chain_error(e)
                raise
            else:
                first_error = None
                break
        if first_error:
            await run_manager.on_chain_error(first_error)
            raise first_error

        yield chunk
        output: Output | None = chunk
        try:
            async for chunk in stream:
                yield chunk
                try:
                    output = output + chunk  # type: ignore[operator]
                except TypeError:
                    output = None
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        await run_manager.on_chain_end(output)

    def __getattr__(self, name: str) -> Any:
        """从包装的 Runnable 及其降级中获取属性。

        返回:
            如果属性不是输出 Runnable 的方法，则返回
            `getattr(self.runnable, name)`。如果属性是返回新
            Runnable 的方法（例如 `model.bind_tools([...])` 输出新的
            `RunnableBinding`），则 `self.runnable` 和 `self.fallbacks`
            中的每个 runnable 都会被替换为 `getattr(x, name)`。

        示例：
            ```python
            from langchain_openai import ChatOpenAI
            from langchain_anthropic import ChatAnthropic

            gpt_4o = ChatOpenAI(model="gpt-4o")
            claude_3_sonnet = ChatAnthropic(model="claude-sonnet-4-5-20250929")
            model = gpt_4o.with_fallbacks([claude_3_sonnet])

            model.model_name
            # -> "gpt-4o"

            # .bind_tools() 会在 ChatOpenAI 和 ChatAnthropic 上调用
            # 等价于：
            # gpt_4o.bind_tools([...]).with_fallbacks([claude_3_sonnet.bind_tools([...])])
            model.bind_tools([...])
            # -> RunnableWithFallbacks(
                runnable=RunnableBinding(bound=ChatOpenAI(...), kwargs={"tools": [...]}),
                fallbacks=[RunnableBinding(bound=ChatAnthropic(...), kwargs={"tools": [...]})],
            )
            ```
        """  # noqa: E501
        attr = getattr(self.runnable, name)
        if _returns_runnable(attr):

            @wraps(attr)
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                new_runnable = attr(*args, **kwargs)
                new_fallbacks = []
                for fallback in self.fallbacks:
                    fallback_attr = getattr(fallback, name)
                    new_fallbacks.append(fallback_attr(*args, **kwargs))

                return self.__class__(
                    **{
                        **self.model_dump(),
                        "runnable": new_runnable,
                        "fallbacks": new_fallbacks,
                    }
                )

            return wrapped

        return attr


def _returns_runnable(attr: Any) -> bool:
    if not callable(attr):
        return False
    return_type = typing.get_type_hints(attr).get("return")
    return bool(return_type and _is_runnable_type(return_type))


def _is_runnable_type(type_: Any) -> bool:
    if inspect.isclass(type_):
        return issubclass(type_, Runnable)
    origin = getattr(type_, "__origin__", None)
    if inspect.isclass(origin):
        return issubclass(origin, Runnable)
    if origin is typing.Union:
        return all(_is_runnable_type(t) for t in type_.__args__)
    return False
