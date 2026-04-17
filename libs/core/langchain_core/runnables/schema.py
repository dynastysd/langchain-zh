"""此模块包含与 `Runnable`（可运行单元）对象一起使用的类型定义。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from typing_extensions import NotRequired, TypedDict

if TYPE_CHECKING:
    from collections.abc import Sequence


class EventData(TypedDict, total=False):
    """与流式事件关联的数据。"""

    input: Any
    """传递给生成事件的 `Runnable`（可运行单元）的输入。

    输入有时在 `Runnable` 的*开始*时可用，有时在 `Runnable` 的*结束*时才可用。

    如果 `Runnable` 能够流式传输其输入，则其输入根据定义
    在 `Runnable` 完成流式传输其输入之前不会已知，即在*结束*时才能确定。
    """
    error: NotRequired[BaseException]
    """执行 `Runnable`（可运行单元）时发生的错误。

    此字段仅在 `Runnable` 抛出异常时可用。

    !!! version-added "Added in `langchain-core` 1.0.0"
    """
    output: Any
    """生成事件的 `Runnable`（可运行单元）的输出。

    输出仅在 `Runnable` 的*结束*时可用。

    对于大多数 `Runnable` 对象，可以从 `chunk` 字段推断此字段，
    尽管某些特殊情况的 `Runnable`（例如聊天模型）可能有例外，
    它们可能返回更多信息。
    """
    chunk: Any
    """生成事件的输出流式块。

    块通常支持加法运算，将它们相加应该会得到
    生成事件的 `Runnable` 的输出。
    """
    tool_call_id: NotRequired[str | None]
    """与工具执行关联的工具调用 ID。

    此字段可用于 `on_tool_error` 事件，可用于
    在无状态代理实现中将错误链接到特定的工具调用。
    """


class BaseStreamEvent(TypedDict):
    """流式事件。

    从 `astream_events` 方法生成的流式事件的 Schema。

    示例:
        ```python
        from langchain_core.runnables import RunnableLambda


        async def reverse(s: str) -> str:
            return s[::-1]


        chain = RunnableLambda(func=reverse)

        events = [event async for event in chain.astream_events("hello")]

        # 将生成以下事件
        # (为简洁起见，某些字段已省略):
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
    """

    event: str
    """事件名称格式为: `on_[runnable_type]_(start|stream|end)`。

    Runnable 类型包括:

    - **llm** - 非聊天模型使用
    - **chat_model** - 聊天模型使用
    - **prompt** -- 例如 `ChatPromptTemplate`
    - **tool** -- 通过 `@tool` 装饰器定义或继承自
        `Tool`/`BaseTool` 的工具
    - **chain** - 大多数 `Runnable` 对象属于此类型

    此外，事件分为以下类别:

    - **start** - 当 `Runnable` 启动时
    - **stream** - 当 `Runnable` 正在流式传输时
    - **end** - 当 `Runnable` 结束时

    start、stream 和 end 与略微不同的 `data` 负载相关联。

    有关更多详细信息，请参阅 `EventData` 的文档。
    """
    run_id: str
    """用于跟踪给定 `Runnable`（可运行单元）执行随机生成的 ID。

    作为父 `Runnable` 执行的一部分调用的每个子 `Runnable`
    都会被分配自己的唯一 ID。
    """
    tags: NotRequired[list[str]]
    """与生成此事件的 `Runnable`（可运行单元）关联的标签。

    标签始终从父 `Runnable` 对象继承。

    标签可以通过 `.with_config({"tags": ["hello"]})` 绑定到 `Runnable`，
    也可以在运行时通过 `.astream_events(..., {"tags": ["hello"]})` 传递。
    """
    metadata: NotRequired[dict[str, Any]]
    """与生成此事件的 `Runnable`（可运行单元）关联的元数据。

    元数据可以通过以下方式绑定到 `Runnable`:

        `.with_config({"metadata": { "foo": "bar" }})`

    或在运行时通过以下方式传递:

        `.astream_events(..., {"metadata": {"foo": "bar"}})`.
    """

    parent_ids: Sequence[str]
    """与此事件关联的父 ID 列表。

    根事件将有一个空列表。

    例如，如果 `Runnable` A 调用 `Runnable` B，则
    `Runnable` B 生成的事件将在 `parent_ids` 字段中包含 `Runnable` A 的 ID。

    父 ID 的顺序是从根父级到直接父级。

    仅在 astream events API 的 v2 版本中支持。v1 将返回空列表。
    """


class StandardStreamEvent(BaseStreamEvent):
    """遵循 LangChain 事件数据约定的标准流式事件。"""

    data: EventData
    """事件数据。

    事件数据的内容取决于事件类型。
    """
    name: str
    """生成事件的 `Runnable`（可运行单元）的名称。"""


class CustomStreamEvent(BaseStreamEvent):
    """用户创建的自定义流式事件。"""

    # 重写 event 字段以更具体
    event: Literal["on_custom_event"]  # type: ignore[misc]
    """事件类型。"""
    name: str
    """事件的用户定义名称。"""
    data: Any
    """与事件关联的数据。形式自由，可以是任何内容。"""


StreamEvent = StandardStreamEvent | CustomStreamEvent
