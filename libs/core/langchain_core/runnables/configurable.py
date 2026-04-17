"""可动态配置的可运行单元（Runnable）对象。"""

from __future__ import annotations

import enum
import threading
from abc import abstractmethod
from collections.abc import (
    AsyncIterator,
    Callable,
    Iterator,
    Sequence,
)
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)
from weakref import WeakValueDictionary

from pydantic import BaseModel, ConfigDict
from typing_extensions import override

from langchain_core.runnables.base import Runnable, RunnableSerializable
from langchain_core.runnables.config import (
    RunnableConfig,
    ensure_config,
    get_config_list,
    get_executor_for_config,
    merge_configs,
)
from langchain_core.runnables.utils import (
    AnyConfigurableField,
    ConfigurableField,
    ConfigurableFieldMultiOption,
    ConfigurableFieldSingleOption,
    ConfigurableFieldSpec,
    Input,
    Output,
    gather_with_concurrency,
    get_unique_config_specs,
)

if TYPE_CHECKING:
    from langchain_core.runnables.graph import Graph


class DynamicRunnable(RunnableSerializable[Input, Output]):
    """可动态配置的可序列化可运行单元（Runnable）。

    DynamicRunnable 应通过 Runnable 的 `configurable_fields` 或
    `configurable_alternatives` 方法来创建。
    """

    default: RunnableSerializable[Input, Output]
    """默认使用的可运行单元（Runnable）。"""

    config: RunnableConfig | None = None
    """要使用的配置。"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @classmethod
    @override
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

    @property
    @override
    def InputType(self) -> type[Input]:
        return self.default.InputType

    @property
    @override
    def OutputType(self) -> type[Output]:
        return self.default.OutputType

    @override
    def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
        runnable, config = self.prepare(config)
        return runnable.get_input_schema(config)

    @override
    def get_output_schema(
        self, config: RunnableConfig | None = None
    ) -> type[BaseModel]:
        runnable, config = self.prepare(config)
        return runnable.get_output_schema(config)

    @override
    def get_graph(self, config: RunnableConfig | None = None) -> Graph:
        runnable, config = self.prepare(config)
        return runnable.get_graph(config)

    @override
    def with_config(
        self,
        config: RunnableConfig | None = None,
        # Sadly Unpack is not well supported by mypy so this will have to be untyped
        **kwargs: Any,
    ) -> Runnable[Input, Output]:
        return self.__class__(
            **{**self.__dict__, "config": ensure_config(merge_configs(config, kwargs))}  # type: ignore[arg-type]
        )

    def prepare(
        self, config: RunnableConfig | None = None
    ) -> tuple[Runnable[Input, Output], RunnableConfig]:
        """准备用于调用的可运行单元（Runnable）。

        参数：
            config: 要使用的配置。

        返回值：
            准备好的可运行单元（Runnable）和配置。
        """
        runnable: Runnable[Input, Output] = self
        while isinstance(runnable, DynamicRunnable):
            runnable, config = runnable._prepare(merge_configs(runnable.config, config))  # noqa: SLF001
        return runnable, cast("RunnableConfig", config)

    @abstractmethod
    def _prepare(
        self, config: RunnableConfig | None = None
    ) -> tuple[Runnable[Input, Output], RunnableConfig]: ...

    @override
    def invoke(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        runnable, config = self.prepare(config)
        return runnable.invoke(input, config, **kwargs)

    @override
    async def ainvoke(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        runnable, config = self.prepare(config)
        return await runnable.ainvoke(input, config, **kwargs)

    @override
    def batch(
        self,
        inputs: list[Input],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> list[Output]:
        configs = get_config_list(config, len(inputs))
        prepared = [self.prepare(c) for c in configs]

        if all(p is self.default for p, _ in prepared):
            return self.default.batch(
                inputs,
                [c for _, c in prepared],
                return_exceptions=return_exceptions,
                **kwargs,
            )

        if not inputs:
            return []

        def invoke(
            prepared: tuple[Runnable[Input, Output], RunnableConfig],
            input_: Input,
        ) -> Output | Exception:
            bound, config = prepared
            if return_exceptions:
                try:
                    return bound.invoke(input_, config, **kwargs)
                except Exception as e:
                    return e
            else:
                return bound.invoke(input_, config, **kwargs)

        # 如果只有一个输入，则无需使用执行器
        if len(inputs) == 1:
            return cast("list[Output]", [invoke(prepared[0], inputs[0])])

        with get_executor_for_config(configs[0]) as executor:
            return cast("list[Output]", list(executor.map(invoke, prepared, inputs)))

    @override
    async def abatch(
        self,
        inputs: list[Input],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> list[Output]:
        configs = get_config_list(config, len(inputs))
        prepared = [self.prepare(c) for c in configs]

        if all(p is self.default for p, _ in prepared):
            return await self.default.abatch(
                inputs,
                [c for _, c in prepared],
                return_exceptions=return_exceptions,
                **kwargs,
            )

        if not inputs:
            return []

        async def ainvoke(
            prepared: tuple[Runnable[Input, Output], RunnableConfig],
            input_: Input,
        ) -> Output | Exception:
            bound, config = prepared
            if return_exceptions:
                try:
                    return await bound.ainvoke(input_, config, **kwargs)
                except Exception as e:
                    return e
            else:
                return await bound.ainvoke(input_, config, **kwargs)

        coros = map(ainvoke, prepared, inputs)
        return await gather_with_concurrency(configs[0].get("max_concurrency"), *coros)

    @override
    def stream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Output]:
        runnable, config = self.prepare(config)
        return runnable.stream(input, config, **kwargs)

    @override
    async def astream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Output]:
        runnable, config = self.prepare(config)
        async for chunk in runnable.astream(input, config, **kwargs):
            yield chunk

    @override
    def transform(
        self,
        input: Iterator[Input],
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Output]:
        runnable, config = self.prepare(config)
        return runnable.transform(input, config, **kwargs)

    @override
    async def atransform(
        self,
        input: AsyncIterator[Input],
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Output]:
        runnable, config = self.prepare(config)
        async for chunk in runnable.atransform(input, config, **kwargs):
            yield chunk

    @override
    def __getattr__(self, name: str) -> Any:  # type: ignore[misc]
        attr = getattr(self.default, name)
        if callable(attr):

            @wraps(attr)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                for key, arg in kwargs.items():
                    if key == "config" and (
                        isinstance(arg, dict)
                        and "configurable" in arg
                        and isinstance(arg["configurable"], dict)
                    ):
                        runnable, config = self.prepare(cast("RunnableConfig", arg))
                        kwargs = {**kwargs, "config": config}
                        return getattr(runnable, name)(*args, **kwargs)

                for idx, arg in enumerate(args):
                    if (
                        isinstance(arg, dict)
                        and "configurable" in arg
                        and isinstance(arg["configurable"], dict)
                    ):
                        runnable, config = self.prepare(cast("RunnableConfig", arg))
                        argsl = list(args)
                        argsl[idx] = config
                        return getattr(runnable, name)(*argsl, **kwargs)

                if self.config:
                    runnable, config = self.prepare()
                    return getattr(runnable, name)(*args, **kwargs)

                return attr(*args, **kwargs)

            return wrapper

        return attr


class RunnableConfigurableFields(DynamicRunnable[Input, Output]):
    """可动态配置的可运行单元（Runnable）。

    RunnableConfigurableFields 应通过 Runnable 的
    `configurable_fields` 方法来创建。

    以下是结合 LLM 使用 RunnableConfigurableFields 的示例：

        ```python
        from langchain_core.prompts import PromptTemplate
        from langchain_core.runnables import ConfigurableField
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(temperature=0).configurable_fields(
            temperature=ConfigurableField(
                id="temperature",
                name="LLM Temperature",
                description="The temperature of the LLM",
            )
        )
        # 这为聊天模型创建了一个 RunnableConfigurableFields。

        # 当调用所创建的 RunnableSequence 时，可以传入
        # ConfigurableField 的 id 值，这里会改变 temperature

        prompt = PromptTemplate.from_template("Pick a random number above {x}")
        chain = prompt | model

        chain.invoke({"x": 0})
        chain.invoke({"x": 0}, config={"configurable": {"temperature": 0.9}})
        ```

    以下是结合 HubRunnables 使用 RunnableConfigurableFields 的示例：

        ```python
        from langchain_core.prompts import PromptTemplate
        from langchain_core.runnables import ConfigurableField
        from langchain_openai import ChatOpenAI
        from langchain.runnables.hub import HubRunnable

        prompt = HubRunnable("rlm/rag-prompt").configurable_fields(
            owner_repo_commit=ConfigurableField(
                id="hub_commit",
                name="Hub Commit",
                description="The Hub commit to pull from",
            )
        )

        prompt.invoke({"question": "foo", "context": "bar"})

        # 使用 `with_config` 方法调用 prompt

        prompt.invoke(
            {"question": "foo", "context": "bar"},
            config={"configurable": {"hub_commit": "rlm/rag-prompt-llama"}},
        )
        ```
    """

    fields: dict[str, AnyConfigurableField]
    """要使用的可配置字段。"""

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        """获取 RunnableConfigurableFields 的配置规范。

        返回值：
            配置规范列表。
        """
        config_specs = []

        default_fields = type(self.default).model_fields
        for field_name, spec in self.fields.items():
            if isinstance(spec, ConfigurableField):
                config_specs.append(
                    ConfigurableFieldSpec(
                        id=spec.id,
                        name=spec.name,
                        description=spec.description
                        or default_fields[field_name].description,
                        annotation=spec.annotation
                        or default_fields[field_name].annotation,
                        default=getattr(self.default, field_name),
                        is_shared=spec.is_shared,
                    )
                )
            else:
                config_specs.append(
                    make_options_spec(spec, default_fields[field_name].description)
                )

        config_specs.extend(self.default.config_specs)

        return get_unique_config_specs(config_specs)

    @override
    def configurable_fields(
        self, **kwargs: AnyConfigurableField
    ) -> RunnableSerializable[Input, Output]:
        return self.default.configurable_fields(**{**self.fields, **kwargs})

    def _prepare(
        self, config: RunnableConfig | None = None
    ) -> tuple[Runnable[Input, Output], RunnableConfig]:
        config = ensure_config(config)
        specs_by_id = {spec.id: (key, spec) for key, spec in self.fields.items()}
        configurable_fields = {
            specs_by_id[k][0]: v
            for k, v in config.get("configurable", {}).items()
            if k in specs_by_id and isinstance(specs_by_id[k][1], ConfigurableField)
        }
        configurable_single_options = {
            k: v.options[(config.get("configurable", {}).get(v.id) or v.default)]
            for k, v in self.fields.items()
            if isinstance(v, ConfigurableFieldSingleOption)
        }
        configurable_multi_options = {
            k: [
                v.options[o]
                for o in config.get("configurable", {}).get(v.id, v.default)
            ]
            for k, v in self.fields.items()
            if isinstance(v, ConfigurableFieldMultiOption)
        }
        configurable = {
            **configurable_fields,
            **configurable_single_options,
            **configurable_multi_options,
        }

        if configurable:
            init_params = {
                k: v
                for k, v in self.default.__dict__.items()
                if k in type(self.default).model_fields
            }
            return (
                self.default.__class__(**{**init_params, **configurable}),
                config,
            )
        return (self.default, config)


# Python 3.11 之前原生 StrEnum 不可用
class StrEnum(str, enum.Enum):
    """字符串枚举。"""


_enums_for_spec: WeakValueDictionary[
    ConfigurableFieldSingleOption | ConfigurableFieldMultiOption | ConfigurableField,
    type[StrEnum],
] = WeakValueDictionary()

_enums_for_spec_lock = threading.Lock()


class RunnableConfigurableAlternatives(DynamicRunnable[Input, Output]):
    """可动态配置的可运行单元（Runnable）。

    RunnableConfigurableAlternatives 应通过 Runnable 的
    `configurable_alternatives` 方法来创建，也可以直接实例化。

    以下是一个使用替代提示（alternative prompts）来说明其功能的
    RunnableConfigurableAlternatives 示例：

        ```python
        from langchain_core.runnables import ConfigurableField
        from langchain_openai import ChatOpenAI

        # 这为提示词可运行单元（Prompt Runnable）创建了一个
        # RunnableConfigurableAlternatives，包含两个替代选项。
        prompt = PromptTemplate.from_template(
            "Tell me a joke about {topic}"
        ).configurable_alternatives(
            ConfigurableField(id="prompt"),
            default_key="joke",
            poem=PromptTemplate.from_template("Write a short poem about {topic}"),
        )

        # 当调用所创建的 RunnableSequence 时，可以传入
        # ConfigurableField 的 id 值，这里可以是 `joke` 或 `poem`。
        chain = prompt | ChatOpenAI(model="gpt-5.4-mini")

        # `with_config` 方法可以将所需的提示词可运行单元
        # 添加到你的 Runnable Sequence 中。
        chain.with_config(configurable={"prompt": "poem"}).invoke({"topic": "bears"})
        ```

    同样，你也可以直接实例化 RunnableConfigurableAlternatives，
    并以相同的方式在 LCEL 中使用：

        ```python
        from langchain_core.runnables import ConfigurableField
        from langchain_core.runnables.configurable import (
            RunnableConfigurableAlternatives,
        )
        from langchain_openai import ChatOpenAI

        prompt = RunnableConfigurableAlternatives(
            which=ConfigurableField(id="prompt"),
            default=PromptTemplate.from_template("Tell me a joke about {topic}"),
            default_key="joke",
            prefix_keys=False,
            alternatives={
                "poem": PromptTemplate.from_template("Write a short poem about {topic}")
            },
        )
        chain = prompt | ChatOpenAI(model="gpt-5.4-mini")
        chain.with_config(configurable={"prompt": "poem"}).invoke({"topic": "bears"})
        ```
    """

    which: ConfigurableField
    """用于在替代选项之间进行选择的 ConfigurableField。"""

    alternatives: dict[
        str,
        Runnable[Input, Output] | Callable[[], Runnable[Input, Output]],
    ]
    """可选择的替代选项字典。"""

    default_key: str = "default"
    """默认选项使用的枚举值。"""

    prefix_keys: bool
    """是否为每个替代选项的可配置字段添加命名空间前缀，
    格式为 <which.id>==<alternative_key>，例如：名为 "gpt3" 的替代选项使用的
    名为 "temperature" 的键会变成 "model==gpt3/temperature"。
    """

    @property
    @override
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        with _enums_for_spec_lock:
            if which_enum := _enums_for_spec.get(self.which):
                pass
            else:
                which_enum = StrEnum(  # type: ignore[call-overload]
                    self.which.name or self.which.id,
                    (
                        (v, v)
                        for v in [*list(self.alternatives.keys()), self.default_key]
                    ),
                )
                _enums_for_spec[self.which] = cast("type[StrEnum]", which_enum)
        return get_unique_config_specs(
            # 选择哪个替代选项
            [
                ConfigurableFieldSpec(
                    id=self.which.id,
                    name=self.which.name,
                    description=self.which.description,
                    annotation=which_enum,
                    default=self.default_key,
                    is_shared=self.which.is_shared,
                ),
            ]
            # 默认选项的配置规范
            + (
                [
                    prefix_config_spec(s, f"{self.which.id}=={self.default_key}")
                    for s in self.default.config_specs
                ]
                if self.prefix_keys
                else self.default.config_specs
            )
            # 替代选项的配置规范
            + [
                (
                    prefix_config_spec(s, f"{self.which.id}=={alt_key}")
                    if self.prefix_keys
                    else s
                )
                for alt_key, alt in self.alternatives.items()
                if isinstance(alt, RunnableSerializable)
                for s in alt.config_specs
            ]
        )

    @override
    def configurable_fields(
        self, **kwargs: AnyConfigurableField
    ) -> RunnableSerializable[Input, Output]:
        return self.__class__(
            which=self.which,
            default=self.default.configurable_fields(**kwargs),
            alternatives=self.alternatives,
            default_key=self.default_key,
            prefix_keys=self.prefix_keys,
        )

    def _prepare(
        self, config: RunnableConfig | None = None
    ) -> tuple[Runnable[Input, Output], RunnableConfig]:
        config = ensure_config(config)
        which = config.get("configurable", {}).get(self.which.id, self.default_key)
        # 为所选替代选项重新映射可配置键
        if self.prefix_keys:
            config = cast(
                "RunnableConfig",
                {
                    **config,
                    "configurable": {
                        _strremoveprefix(k, f"{self.which.id}=={which}/"): v
                        for k, v in config.get("configurable", {}).items()
                    },
                },
            )
        # 返回所选替代选项
        if which == self.default_key:
            return (self.default, config)
        if which in self.alternatives:
            alt = self.alternatives[which]
            if isinstance(alt, Runnable):
                return (alt, config)
            return (alt(), config)
        msg = f"Unknown alternative: {which}"
        raise ValueError(msg)


def _strremoveprefix(s: str, prefix: str) -> str:
    """`str.removeprefix()` 仅在 Python 3.9+ 可用。"""
    return s.replace(prefix, "", 1) if s.startswith(prefix) else s


def prefix_config_spec(
    spec: ConfigurableFieldSpec, prefix: str
) -> ConfigurableFieldSpec:
    """为 ConfigurableFieldSpec 的 id 添加前缀。

    当一个 RunnableConfigurableAlternatives 用作另一个
    RunnableConfigurableAlternatives 的 ConfigurableField 时，此功能很有用。

    参数：
        spec: 要添加前缀的 ConfigurableFieldSpec。
        prefix: 要添加的前缀。

    返回值：
        添加前缀后的 ConfigurableFieldSpec。
    """
    return (
        ConfigurableFieldSpec(
            id=f"{prefix}/{spec.id}",
            name=spec.name,
            description=spec.description,
            annotation=spec.annotation,
            default=spec.default,
            is_shared=spec.is_shared,
        )
        if not spec.is_shared
        else spec
    )


def make_options_spec(
    spec: ConfigurableFieldSingleOption | ConfigurableFieldMultiOption,
    description: str | None,
) -> ConfigurableFieldSpec:
    """创建选项规范。

    为 ConfigurableFieldSingleOption 或 ConfigurableFieldMultiOption
    创建一个 ConfigurableFieldSpec。

    参数：
        spec: ConfigurableFieldSingleOption 或 ConfigurableFieldMultiOption。
        description: 如果规范没有描述，则使用此描述。

    返回值：
        ConfigurableFieldSpec。
    """
    with _enums_for_spec_lock:
        if enum := _enums_for_spec.get(spec):
            pass
        else:
            enum = StrEnum(  # type: ignore[call-overload]
                spec.name or spec.id,
                ((v, v) for v in list(spec.options.keys())),
            )
            _enums_for_spec[spec] = cast("type[StrEnum]", enum)
    if isinstance(spec, ConfigurableFieldSingleOption):
        return ConfigurableFieldSpec(
            id=spec.id,
            name=spec.name,
            description=spec.description or description,
            annotation=enum,
            default=spec.default,
            is_shared=spec.is_shared,
        )
    return ConfigurableFieldSpec(
        id=spec.id,
        name=spec.name,
        description=spec.description or description,
        annotation=Sequence[enum],  # type: ignore[valid-type]
        default=spec.default,
        is_shared=spec.is_shared,
    )
