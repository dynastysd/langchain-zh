"""根据条件选择运行哪个分支的分支链（RunnableBranch）。"""

from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Mapping,
    Sequence,
)
from typing import (
    Any,
    cast,
)

from pydantic import BaseModel, ConfigDict
from typing_extensions import override

from langchain_core.runnables.base import (
    Runnable,
    RunnableLike,
    RunnableSerializable,
    coerce_to_runnable,
)
from langchain_core.runnables.config import (
    RunnableConfig,
    ensure_config,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
    patch_config,
)
from langchain_core.runnables.utils import (
    ConfigurableFieldSpec,
    Input,
    Output,
    get_unique_config_specs,
)

_MIN_BRANCHES = 2


class RunnableBranch(RunnableSerializable[Input, Output]):
    """根据条件选择运行哪个分支的分支链（RunnableBranch）。

    初始化时传入一组 `(condition, Runnable)` 对（即分支条件及其对应的可运行单元）和一个默认分支。

    当处理输入时，从第一个条件开始逐个评估，第一个评估为 True 的条件被选中，
    然后对应的 Runnable 会对该输入执行。

    如果没有条件评估为 True，则运行默认分支。

    示例：
        ```python
        from langchain_core.runnables import RunnableBranch

        branch = RunnableBranch(
            (lambda x: isinstance(x, str), lambda x: x.upper()),
            (lambda x: isinstance(x, int), lambda x: x + 1),
            (lambda x: isinstance(x, float), lambda x: x * 2),
            lambda x: "goodbye",
        )

        branch.invoke("hello")  # "HELLO"
        branch.invoke(None)  # "goodbye"
        ```
    """

    branches: Sequence[tuple[Runnable[Input, bool], Runnable[Input, Output]]]
    """`(condition, Runnable)` 对的列表，表示分支条件及其对应的可运行单元。"""
    default: Runnable[Input, Output]
    """当没有条件满足时运行的 Runnable（即可运行单元）。"""

    def __init__(
        self,
        *branches: tuple[
            Runnable[Input, bool]
            | Callable[[Input], bool]
            | Callable[[Input], Awaitable[bool]],
            RunnableLike,
        ]
        | RunnableLike,
    ) -> None:
        """根据条件运行两个分支之一的可运行单元（Runnable）。

        参数：
            *branches: 一个 `(condition, Runnable)` 对的列表。
                最后一个元素作为默认的 Runnable，当没有条件满足时运行。

        异常：
            ValueError: 如果分支数量少于 2 个。
            TypeError: 如果默认分支不是 Runnable、Callable 或 Mapping。
            TypeError: 如果某个分支不是 tuple 或 list。
            ValueError: 如果某个分支的长度不为 2。
        """
        if len(branches) < _MIN_BRANCHES:
            msg = "RunnableBranch requires at least two branches"
            raise ValueError(msg)

        default = branches[-1]

        if not isinstance(
            default,
            (Runnable, Callable, Mapping),  # type: ignore[arg-type]
        ):
            msg = "RunnableBranch default must be Runnable, callable or mapping."
            raise TypeError(msg)

        default_ = cast(
            "Runnable[Input, Output]", coerce_to_runnable(cast("RunnableLike", default))
        )

        branches_ = []

        for branch in branches[:-1]:
            if not isinstance(branch, (tuple, list)):
                msg = (
                    f"RunnableBranch branches must be "
                    f"tuples or lists, not {type(branch)}"
                )
                raise TypeError(msg)

            if len(branch) != _MIN_BRANCHES:
                msg = (
                    f"RunnableBranch branches must be "
                    f"tuples or lists of length 2, not {len(branch)}"
                )
                raise ValueError(msg)
            condition, runnable = branch
            condition = cast("Runnable[Input, bool]", coerce_to_runnable(condition))
            runnable = coerce_to_runnable(runnable)
            branches_.append((condition, runnable))

        super().__init__(
            branches=branches_,
            default=default_,
        )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """返回 `True`，因为此类可序列化。"""
        return True

    @classmethod
    @override
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        返回值：
            `["langchain", "schema", "runnable"]`
        """
        return ["langchain", "schema", "runnable"]

    @override
    def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
        runnables = (
            [self.default]
            + [r for _, r in self.branches]
            + [r for r, _ in self.branches]
        )

        for runnable in runnables:
            if (
                runnable.get_input_schema(config).model_json_schema().get("type")
                is not None
            ):
                return runnable.get_input_schema(config)

        return super().get_input_schema(config)

    @property
    @override
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return get_unique_config_specs(
            spec
            for step in (
                [self.default]
                + [r for _, r in self.branches]
                + [r for r, _ in self.branches]
            )
            for spec in step.config_specs
        )

    @override
    def invoke(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        """首先评估条件，然后委托给 `True` 或 `False` 分支。

        参数：
            input: 可运行单元的输入。
            config: 可运行单元的配置。
            **kwargs: 传递给可运行单元的其他关键字参数。

        返回值：
            所运行分支的输出。
        """
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        run_manager = callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )

        try:
            for idx, branch in enumerate(self.branches):
                condition, runnable = branch

                expression_value = condition.invoke(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag=f"condition:{idx + 1}"),
                    ),
                )

                if expression_value:
                    output = runnable.invoke(
                        input,
                        config=patch_config(
                            config,
                            callbacks=run_manager.get_child(tag=f"branch:{idx + 1}"),
                        ),
                        **kwargs,
                    )
                    break
            else:
                output = self.default.invoke(
                    input,
                    config=patch_config(
                        config, callbacks=run_manager.get_child(tag="branch:default")
                    ),
                    **kwargs,
                )
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        run_manager.on_chain_end(output)
        return output

    @override
    async def ainvoke(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        run_manager = await callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        try:
            for idx, branch in enumerate(self.branches):
                condition, runnable = branch

                expression_value = await condition.ainvoke(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag=f"condition:{idx + 1}"),
                    ),
                )

                if expression_value:
                    output = await runnable.ainvoke(
                        input,
                        config=patch_config(
                            config,
                            callbacks=run_manager.get_child(tag=f"branch:{idx + 1}"),
                        ),
                        **kwargs,
                    )
                    break
            else:
                output = await self.default.ainvoke(
                    input,
                    config=patch_config(
                        config, callbacks=run_manager.get_child(tag="branch:default")
                    ),
                    **kwargs,
                )
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        await run_manager.on_chain_end(output)
        return output

    @override
    def stream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Output]:
        """首先评估条件，然后委托给 `True` 或 `False` 分支。

        参数：
            input: 可运行单元的输入。
            config: 可运行单元的配置。
            **kwargs: 传递给可运行单元的其他关键字参数。

        生成：
            所运行分支的输出。
        """
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        run_manager = callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        final_output: Output | None = None
        final_output_supported = True

        try:
            for idx, branch in enumerate(self.branches):
                condition, runnable = branch

                expression_value = condition.invoke(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag=f"condition:{idx + 1}"),
                    ),
                )

                if expression_value:
                    for chunk in runnable.stream(
                        input,
                        config=patch_config(
                            config,
                            callbacks=run_manager.get_child(tag=f"branch:{idx + 1}"),
                        ),
                        **kwargs,
                    ):
                        yield chunk
                        if final_output_supported:
                            if final_output is None:
                                final_output = chunk
                            else:
                                try:
                                    final_output = final_output + chunk  # type: ignore[operator]
                                except TypeError:
                                    final_output = None
                                    final_output_supported = False
                    break
            else:
                for chunk in self.default.stream(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag="branch:default"),
                    ),
                    **kwargs,
                ):
                    yield chunk
                    if final_output_supported:
                        if final_output is None:
                            final_output = chunk
                        else:
                            try:
                                final_output = final_output + chunk  # type: ignore[operator]
                            except TypeError:
                                final_output = None
                                final_output_supported = False
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        run_manager.on_chain_end(final_output)

    @override
    async def astream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Output]:
        """首先评估条件，然后委托给 `True` 或 `False` 分支。

        参数：
            input: 可运行单元的输入。
            config: 可运行单元的配置。
            **kwargs: 传递给可运行单元的其他关键字参数。

        生成：
            所运行分支的输出。
        """
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        run_manager = await callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        final_output: Output | None = None
        final_output_supported = True

        try:
            for idx, branch in enumerate(self.branches):
                condition, runnable = branch

                expression_value = await condition.ainvoke(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag=f"condition:{idx + 1}"),
                    ),
                )

                if expression_value:
                    async for chunk in runnable.astream(
                        input,
                        config=patch_config(
                            config,
                            callbacks=run_manager.get_child(tag=f"branch:{idx + 1}"),
                        ),
                        **kwargs,
                    ):
                        yield chunk
                        if final_output_supported:
                            if final_output is None:
                                final_output = chunk
                            else:
                                try:
                                    final_output = final_output + chunk  # type: ignore[operator]
                                except TypeError:
                                    final_output = None
                                    final_output_supported = False
                    break
            else:
                async for chunk in self.default.astream(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag="branch:default"),
                    ),
                    **kwargs,
                ):
                    yield chunk
                    if final_output_supported:
                        if final_output is None:
                            final_output = chunk
                        else:
                            try:
                                final_output = final_output + chunk  # type: ignore[operator]
                            except TypeError:
                                final_output = None
                                final_output_supported = False
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        await run_manager.on_chain_end(final_output)
