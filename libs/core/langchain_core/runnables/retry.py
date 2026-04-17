"""失败时重试 Runnable 的 Runnable。"""

from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    cast,
)

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
from typing_extensions import TypedDict, override

from langchain_core.runnables.base import RunnableBindingBase
from langchain_core.runnables.config import RunnableConfig, patch_config
from langchain_core.runnables.utils import Input, Output

if TYPE_CHECKING:
    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForChainRun,
        CallbackManagerForChainRun,
    )

    T = TypeVar("T", CallbackManagerForChainRun, AsyncCallbackManagerForChainRun)
U = TypeVar("U")


class ExponentialJitterParams(TypedDict, total=False):
    """`tenacity.wait_exponential_jitter` 的参数。"""

    initial: float
    """初始等待时间。"""
    max: float
    """最大等待时间。"""
    exp_base: float
    """指数退避的基数。"""
    jitter: float
    """从 random.uniform(0, jitter) 中采样的随机额外等待时间。"""


class RunnableRetry(RunnableBindingBase[Input, Output]):  # type: ignore[no-redef]
    """失败时重试 Runnable。

    RunnableRetry 可用于向任何子类化基础 Runnable 的对象添加重试逻辑。

    这种重试对于可能因暂时性错误而失败的网络调用特别有用。

    RunnableRetry 实现为 RunnableBinding。最简单的方法是通过所有
    Runnable 上的 `.with_retry()` 方法使用它。

    示例：
    这是一个使用 RunnableLambda 抛出异常的示例

        ```python
        import time


        def foo(input) -> None:
            '''抛出异常的假函数。'''
            raise ValueError(f"调用 foo 失败。时间：{time.time()}")


        runnable = RunnableLambda(foo)

        runnable_with_retries = runnable.with_retry(
            retry_if_exception_type=(ValueError,),  # 只在 ValueError 上重试
            wait_exponential_jitter=True,  # 向指数退避添加抖动
            stop_after_attempt=2,  # 重试两次
            exponential_jitter_params={"initial": 2},  # 如有需要，自定义退避
        )

        # 上面的方法调用等价于下面的较长形式：

        runnable_with_retries = RunnableRetry(
            bound=runnable,
            retry_exception_types=(ValueError,),
            max_attempt_number=2,
            wait_exponential_jitter=True,
            exponential_jitter_params={"initial": 2},
        )
        ```

    此逻辑可用于重试任何 Runnable，包括 Runnable 链，但通常最好将
    重试范围尽可能小。例如，如果你有一个 Runnable 链，你应该只重试
    可能失败的 Runnable，而不是整个链。

    示例：
        ```python
        from langchain_core.chat_models import ChatOpenAI
        from langchain_core.prompts import PromptTemplate

        template = PromptTemplate.from_template("tell me a joke about {topic}.")
        model = ChatOpenAI(temperature=0.5)

        # 好
        chain = template | model.with_retry()

        # 不好
        chain = template | model
        retryable_chain = chain.with_retry()
        ```
    """

    retry_exception_types: tuple[type[BaseException], ...] = (Exception,)
    """要重试的异常类型。默认重试所有异常。

    通常你应该只重试可能是暂时性的异常，例如网络错误。

    适合重试的异常包括所有服务器错误（5xx）和选定的客户端
    错误（4xx），如 429 请求过多。
    """

    wait_exponential_jitter: bool = True
    """是否向指数退避添加抖动。"""

    exponential_jitter_params: ExponentialJitterParams | None = None
    """`tenacity.wait_exponential_jitter` 的参数。即：`initial`、
    `max`、`exp_base` 和 `jitter`（都是 `float` 值）。
    """

    max_attempt_number: int = 3
    """重试 Runnable 的最大尝试次数。"""

    @property
    def _kwargs_retrying(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}

        if self.max_attempt_number:
            kwargs["stop"] = stop_after_attempt(self.max_attempt_number)

        if self.wait_exponential_jitter:
            kwargs["wait"] = wait_exponential_jitter(
                **(self.exponential_jitter_params or {})
            )

        if self.retry_exception_types:
            kwargs["retry"] = retry_if_exception_type(self.retry_exception_types)

        return kwargs

    def _sync_retrying(self, **kwargs: Any) -> Retrying:
        return Retrying(**self._kwargs_retrying, **kwargs)

    def _async_retrying(self, **kwargs: Any) -> AsyncRetrying:
        return AsyncRetrying(**self._kwargs_retrying, **kwargs)

    @staticmethod
    def _patch_config(
        config: RunnableConfig,
        run_manager: "T",
        retry_state: RetryCallState,
    ) -> RunnableConfig:
        attempt = retry_state.attempt_number
        tag = f"retry:attempt:{attempt}" if attempt > 1 else None
        return patch_config(config, callbacks=run_manager.get_child(tag))

    def _patch_config_list(
        self,
        config: list[RunnableConfig],
        run_manager: list["T"],
        retry_state: RetryCallState,
    ) -> list[RunnableConfig]:
        return [
            self._patch_config(c, rm, retry_state)
            for c, rm in zip(config, run_manager, strict=False)
        ]

    def _invoke(
        self,
        input_: Input,
        run_manager: "CallbackManagerForChainRun",
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Output:
        for attempt in self._sync_retrying(reraise=True):
            with attempt:
                result = super().invoke(
                    input_,
                    self._patch_config(config, run_manager, attempt.retry_state),
                    **kwargs,
                )
            if attempt.retry_state.outcome and not attempt.retry_state.outcome.failed:
                attempt.retry_state.set_result(result)
        return result

    @override
    def invoke(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        return self._call_with_config(self._invoke, input, config, **kwargs)

    async def _ainvoke(
        self,
        input_: Input,
        run_manager: "AsyncCallbackManagerForChainRun",
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Output:
        async for attempt in self._async_retrying(reraise=True):
            with attempt:
                result = await super().ainvoke(
                    input_,
                    self._patch_config(config, run_manager, attempt.retry_state),
                    **kwargs,
                )
            if attempt.retry_state.outcome and not attempt.retry_state.outcome.failed:
                attempt.retry_state.set_result(result)
        return result

    @override
    async def ainvoke(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        return await self._acall_with_config(self._ainvoke, input, config, **kwargs)

    def _batch(
        self,
        inputs: list[Input],
        run_manager: list["CallbackManagerForChainRun"],
        config: list[RunnableConfig],
        **kwargs: Any,
    ) -> list[Output | Exception]:
        results_map: dict[int, Output] = {}

        not_set: list[Output] = []
        result = not_set
        try:
            for attempt in self._sync_retrying():
                with attempt:
                    # Retry for inputs that have not yet succeeded
                    # Determine which original indices remain.
                    remaining_indices = [
                        i for i in range(len(inputs)) if i not in results_map
                    ]
                    if not remaining_indices:
                        break
                    pending_inputs = [inputs[i] for i in remaining_indices]
                    pending_configs = [config[i] for i in remaining_indices]
                    pending_run_managers = [run_manager[i] for i in remaining_indices]
                    # Invoke underlying batch only on remaining elements.
                    result = super().batch(
                        pending_inputs,
                        self._patch_config_list(
                            pending_configs, pending_run_managers, attempt.retry_state
                        ),
                        return_exceptions=True,
                        **kwargs,
                    )
                    # Register the results of the inputs that have succeeded, mapping
                    # back to their original indices.
                    first_exception = None
                    for offset, r in enumerate(result):
                        if isinstance(r, Exception):
                            if not first_exception:
                                first_exception = r
                            continue
                        orig_idx = remaining_indices[offset]
                        results_map[orig_idx] = r
                    # If any exception occurred, raise it, to retry the failed ones
                    if first_exception:
                        raise first_exception
                if (
                    attempt.retry_state.outcome
                    and not attempt.retry_state.outcome.failed
                ):
                    attempt.retry_state.set_result(result)
        except RetryError as e:
            if result is not_set:
                result = cast("list[Output]", [e] * len(inputs))

        outputs: list[Output | Exception] = []
        for idx in range(len(inputs)):
            if idx in results_map:
                outputs.append(results_map[idx])
            else:
                outputs.append(result.pop(0))
        return outputs

    @override
    def batch(
        self,
        inputs: list[Input],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[Output]:
        return self._batch_with_config(
            self._batch, inputs, config, return_exceptions=return_exceptions, **kwargs
        )

    async def _abatch(
        self,
        inputs: list[Input],
        run_manager: list["AsyncCallbackManagerForChainRun"],
        config: list[RunnableConfig],
        **kwargs: Any,
    ) -> list[Output | Exception]:
        results_map: dict[int, Output] = {}

        not_set: list[Output] = []
        result = not_set
        try:
            async for attempt in self._async_retrying():
                with attempt:
                    # Retry for inputs that have not yet succeeded
                    # Determine which original indices remain.
                    remaining_indices = [
                        i for i in range(len(inputs)) if i not in results_map
                    ]
                    if not remaining_indices:
                        break
                    pending_inputs = [inputs[i] for i in remaining_indices]
                    pending_configs = [config[i] for i in remaining_indices]
                    pending_run_managers = [run_manager[i] for i in remaining_indices]
                    result = await super().abatch(
                        pending_inputs,
                        self._patch_config_list(
                            pending_configs, pending_run_managers, attempt.retry_state
                        ),
                        return_exceptions=True,
                        **kwargs,
                    )
                    # Register the results of the inputs that have succeeded, mapping
                    # back to their original indices.
                    first_exception = None
                    for offset, r in enumerate(result):
                        if isinstance(r, Exception):
                            if not first_exception:
                                first_exception = r
                            continue
                        orig_idx = remaining_indices[offset]
                        results_map[orig_idx] = r
                    # If any exception occurred, raise it, to retry the failed ones
                    if first_exception:
                        raise first_exception
                if (
                    attempt.retry_state.outcome
                    and not attempt.retry_state.outcome.failed
                ):
                    attempt.retry_state.set_result(result)
        except RetryError as e:
            if result is not_set:
                result = cast("list[Output]", [e] * len(inputs))

        outputs: list[Output | Exception] = []
        for idx in range(len(inputs)):
            if idx in results_map:
                outputs.append(results_map[idx])
            else:
                outputs.append(result.pop(0))
        return outputs

    @override
    async def abatch(
        self,
        inputs: list[Input],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[Output]:
        return await self._abatch_with_config(
            self._abatch, inputs, config, return_exceptions=return_exceptions, **kwargs
        )

    # stream() 和 transform() 不会被重试，因为重试流式输出不太直观。
