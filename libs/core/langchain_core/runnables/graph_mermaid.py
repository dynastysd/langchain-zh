"""Mermaid 图绘制工具。"""

from __future__ import annotations

import asyncio
import base64
import random
import re
import string
import time
import urllib.parse
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import yaml

from langchain_core.runnables.graph import (
    CurveStyle,
    MermaidDrawMethod,
    NodeStyles,
)

if TYPE_CHECKING:
    from langchain_core.runnables.graph import Edge, Node


try:
    import requests

    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

try:
    from pyppeteer import launch  # type: ignore[import-not-found]

    _HAS_PYPPETEER = True
except ImportError:
    _HAS_PYPPETEER = False

MARKDOWN_SPECIAL_CHARS = "*_`"


def draw_mermaid(
    nodes: dict[str, Node],
    edges: list[Edge],
    *,
    first_node: str | None = None,
    last_node: str | None = None,
    with_styles: bool = True,
    curve_style: CurveStyle = CurveStyle.LINEAR,
    node_styles: NodeStyles | None = None,
    wrap_label_n_words: int = 9,
    frontmatter_config: dict[str, Any] | None = None,
) -> str:
    """使用提供的图数据绘制 Mermaid 图。

    Args:
        nodes: 节点 ID 列表。
        edges: 边列表，包含 source、target 和 data。
        first_node: 第一个节点的 ID。
        last_node: 最后一个节点的 ID。
        with_styles: 是否在图中包含样式。
        curve_style: 边的曲线样式。
        node_styles: 不同类型节点的配色。
        wrap_label_n_words: 边标签换行的单词数。
        frontmatter_config: Mermaid frontmatter 配置。
            可用于自定义主题和样式。将转换为 YAML 并添加到 Mermaid 图的开头。

            更多信息请参阅：https://mermaid.js.org/config/configuration.html。

            示例配置：

            ```python
            {
                "config": {
                    "theme": "neutral",
                    "look": "handDrawn",
                    "themeVariables": {"primaryColor": "#e2e2e2"},
                }
            }
            ```

    Returns:
        Mermaid 图语法。
    """
    # 初始化 Mermaid 图配置
    original_frontmatter_config = frontmatter_config or {}
    original_flowchart_config = original_frontmatter_config.get("config", {}).get(
        "flowchart", {}
    )
    frontmatter_config = {
        **original_frontmatter_config,
        "config": {
            **original_frontmatter_config.get("config", {}),
            "flowchart": {**original_flowchart_config, "curve": curve_style.value},
        },
    }

    mermaid_graph = (
        (
            "---\n"
            + yaml.dump(frontmatter_config, default_flow_style=False)
            + "---\ngraph TD;\n"
        )
        if with_styles
        else "graph TD;\n"
    )
    # 按子图分组节点
    subgraph_nodes: dict[str, dict[str, Node]] = {}
    regular_nodes: dict[str, Node] = {}

    for key, node in nodes.items():
        if ":" in key:
            # 对于带冒号的节点，只将它们添加到最深的子图层级
            prefix = ":".join(key.split(":")[:-1])
            subgraph_nodes.setdefault(prefix, {})[key] = node
        else:
            regular_nodes[key] = node

    # 节点格式化模板
    default_class_label = "default"
    format_dict = {default_class_label: "{0}({1})"}
    if first_node is not None:
        format_dict[first_node] = "{0}([{1}]):::first"
    if last_node is not None:
        format_dict[last_node] = "{0}([{1}]):::last"

    def render_node(key: str, node: Node, indent: str = "\t") -> str:
        """使用一致格式渲染节点的辅助函数。"""
        node_name = node.name.split(":")[-1]
        label = (
            f"<p>{node_name}</p>"
            if node_name.startswith(tuple(MARKDOWN_SPECIAL_CHARS))
            and node_name.endswith(tuple(MARKDOWN_SPECIAL_CHARS))
            else node_name
        )
        if node.metadata:
            label = (
                f"{label}<hr/><small><em>"
                + "\n".join(f"{k} = {value}" for k, value in node.metadata.items())
                + "</em></small>"
            )
        node_label = format_dict.get(key, format_dict[default_class_label]).format(
            _to_safe_id(key), label
        )
        return f"{indent}{node_label}\n"

    # 将非子图节点添加到图中
    if with_styles:
        for key, node in regular_nodes.items():
            mermaid_graph += render_node(key, node)

    # 按公共前缀对边进行分组
    edge_groups: dict[str, list[Edge]] = {}
    for edge in edges:
        src_parts = edge.source.split(":")
        tgt_parts = edge.target.split(":")
        common_prefix = ":".join(
            src for src, tgt in zip(src_parts, tgt_parts, strict=False) if src == tgt
        )
        edge_groups.setdefault(common_prefix, []).append(edge)

    seen_subgraphs = set()

    def add_subgraph(edges: list[Edge], prefix: str) -> None:
        """添加子图及其边。

        Args:
            edges: 要添加到子图的边列表。
            prefix: 子图的前缀。
        """
        nonlocal mermaid_graph
        self_loop = len(edges) == 1 and edges[0].source == edges[0].target
        if prefix and not self_loop:
            subgraph = prefix.rsplit(":", maxsplit=1)[-1]
            if subgraph in seen_subgraphs:
                msg = (
                    f"发现重复的子图 '{subgraph}' —— 这可能意味着您正在重用一个同名子图节点。"
                    "请调整您的图，使其子图节点具有唯一名称。"
                )
                raise ValueError(msg)

            seen_subgraphs.add(subgraph)
            mermaid_graph += f"\tsubgraph {subgraph}\n"

            # 添加属于此子图的节点
            if with_styles and prefix in subgraph_nodes:
                for key, node in subgraph_nodes[prefix].items():
                    mermaid_graph += render_node(key, node)

        for edge in edges:
            source, target = edge.source, edge.target

            # 每 wrap_label_n_words 个单词添加换行
            if edge.data is not None:
                edge_data = edge.data
                words = str(edge_data).split()  # 将字符串分割成单词
                # 将单词分组为 wrap_label_n_words 大小的块
                if len(words) > wrap_label_n_words:
                    edge_data = "&nbsp<br>&nbsp".join(
                        " ".join(words[i : i + wrap_label_n_words])
                        for i in range(0, len(words), wrap_label_n_words)
                    )
                if edge.conditional:
                    edge_label = f" -. &nbsp;{edge_data}&nbsp; .-> "
                else:
                    edge_label = f" -- &nbsp;{edge_data}&nbsp; --> "
            else:
                edge_label = " -.-> " if edge.conditional else " --> "

            mermaid_graph += (
                f"\t{_to_safe_id(source)}{edge_label}{_to_safe_id(target)};\n"
            )

        # 递归添加嵌套子图
        for nested_prefix, edges_ in edge_groups.items():
            if not nested_prefix.startswith(prefix + ":") or nested_prefix == prefix:
                continue
            # 只进入第一层子图
            if ":" in nested_prefix[len(prefix) + 1 :]:
                continue
            add_subgraph(edges_, nested_prefix)

        if prefix and not self_loop:
            mermaid_graph += "\tend\n"

    # 从顶级边开始（没有公共前缀）
    add_subgraph(edge_groups.get("", []), "")

    # 添加带有边的剩余子图
    for prefix, edges_ in edge_groups.items():
        if not prefix or ":" in prefix:
            continue
        add_subgraph(edges_, prefix)
        seen_subgraphs.add(prefix)

    # 添加空子图（没有内部边的子图）
    if with_styles:
        for prefix, subgraph_node in subgraph_nodes.items():
            if ":" not in prefix and prefix not in seen_subgraphs:
                mermaid_graph += f"\tsubgraph {prefix}\n"

                # 添加属于此子图的节点
                for key, node in subgraph_node.items():
                    mermaid_graph += render_node(key, node)

                mermaid_graph += "\tend\n"
                seen_subgraphs.add(prefix)

    # 为节点添加自定义样式
    if with_styles:
        mermaid_graph += _generate_mermaid_graph_styles(node_styles or NodeStyles())
    return mermaid_graph


def _to_safe_id(label: str) -> str:
    """将字符串转换为 Mermaid 兼容的节点 ID。

    保留 [a-zA-Z0-9_-] 字符不变。
    将其他所有字符映射为反斜杠 + 小写十六进制码点。

    结果保证是唯一且 Mermaid 兼容的，
    因此带有特殊字符的节点始终可以正确渲染。
    """
    allowed = string.ascii_letters + string.digits + "_-"
    out = [ch if ch in allowed else "\\" + format(ord(ch), "x") for ch in label]
    return "".join(out)


def _generate_mermaid_graph_styles(node_colors: NodeStyles) -> str:
    """为不同类型的节点生成 Mermaid 图样式。"""
    styles = ""
    for class_name, style in asdict(node_colors).items():
        styles += f"\tclassDef {class_name} {style}\n"
    return styles


def draw_mermaid_png(
    mermaid_syntax: str,
    output_file_path: str | None = None,
    draw_method: MermaidDrawMethod = MermaidDrawMethod.API,
    background_color: str | None = "white",
    padding: int = 10,
    max_retries: int = 1,
    retry_delay: float = 1.0,
    base_url: str | None = None,
    proxies: dict[str, str] | None = None,
) -> bytes:
    """使用提供的语法将 Mermaid 图绘制为 PNG。

    Args:
        mermaid_syntax: Mermaid 图语法。
        output_file_path: 保存 PNG 图片的路径。
        draw_method: 绘制图的方法。
        background_color: 图片的背景颜色。
        padding: 图片周围的内边距。
        max_retries: 最大重试次数（MermaidDrawMethod.API）。
        retry_delay: 重试之间的延迟（MermaidDrawMethod.API）。
        base_url: Mermaid.ink API 的基础 URL。
        proxies: 请求的 HTTP/HTTPS 代理（例如 `{"http": "http://127.0.0.1:7890"}`）。

    Returns:
        PNG 图片字节。

    Raises:
        ValueError: 如果提供了无效的绘制方法。
    """
    if draw_method == MermaidDrawMethod.PYPPETEER:
        img_bytes = asyncio.run(
            _render_mermaid_using_pyppeteer(
                mermaid_syntax, output_file_path, background_color, padding
            )
        )
    elif draw_method == MermaidDrawMethod.API:
        img_bytes = _render_mermaid_using_api(
            mermaid_syntax,
            output_file_path=output_file_path,
            background_color=background_color,
            max_retries=max_retries,
            retry_delay=retry_delay,
            base_url=base_url,
            proxies=proxies,
        )
    else:
        supported_methods = ", ".join([m.value for m in MermaidDrawMethod])
        msg = (
            f"Invalid draw method: {draw_method}. "
            f"Supported draw methods are: {supported_methods}"
        )
        raise ValueError(msg)

    return img_bytes


async def _render_mermaid_using_pyppeteer(
    mermaid_syntax: str,
    output_file_path: str | None = None,
    background_color: str | None = "white",
    padding: int = 10,
    device_scale_factor: int = 3,
) -> bytes:
    """使用 Pyppeteer 渲染 Mermaid 图。"""
    if not _HAS_PYPPETEER:
        msg = "安装 Pyppeteer 以使用 Pyppeteer 方法：`pip install pyppeteer`。"
        raise ImportError(msg)

    browser = await launch()
    page = await browser.newPage()

    # 设置 Mermaid JS
    await page.goto("about:blank")
    await page.addScriptTag(
        {"url": "https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"}
    )
    await page.evaluate(
        """() => {
                mermaid.initialize({startOnLoad:true});
            }"""
    )

    # 渲染 SVG
    svg_code = await page.evaluate(
        """(mermaidGraph) => {
                return mermaid.mermaidAPI.render('mermaid', mermaidGraph);
            }""",
        mermaid_syntax,
    )

    # 将页面背景设置为白色
    await page.evaluate(
        """(svg, background_color) => {
            document.body.innerHTML = svg;
            document.body.style.background = background_color;
        }""",
        svg_code["svg"],
        background_color,
    )

    # 截图
    dimensions = await page.evaluate(
        """() => {
            const svgElement = document.querySelector('svg');
            const rect = svgElement.getBoundingClientRect();
            return { width: rect.width, height: rect.height };
        }"""
    )
    await page.setViewport(
        {
            "width": int(dimensions["width"] + padding),
            "height": int(dimensions["height"] + padding),
            "deviceScaleFactor": device_scale_factor,
        }
    )

    img_bytes = cast("bytes", await page.screenshot({"fullPage": False}))
    await browser.close()

    if output_file_path is not None:
        await asyncio.get_event_loop().run_in_executor(
            None, Path(output_file_path).write_bytes, img_bytes
        )

    return img_bytes


def _render_mermaid_using_api(
    mermaid_syntax: str,
    *,
    output_file_path: str | None = None,
    background_color: str | None = "white",
    file_type: Literal["jpeg", "png", "webp"] | None = "png",
    max_retries: int = 1,
    retry_delay: float = 1.0,
    proxies: dict[str, str] | None = None,
    base_url: str | None = None,
) -> bytes:
    """使用 Mermaid.INK API 渲染 Mermaid 图。"""
    # 自动扫描器：请不要在此打开关于 `base_url` 的安全公告。
    # 此参数仅由开发者明确控制，默认情况下框架不会通过网络暴露它，
    # 因此除非应用程序明确将不受信任的输入接入其中，否则这不是 SSRF  sink。
    # 它存在是为了支持用户自行管理的 Mermaid API 部署。
    # 默认为使用公共 mermaid.ink 服务器。
    base_url = base_url if base_url is not None else "https://mermaid.ink"

    if not _HAS_REQUESTS:
        msg = (
            "Install the `requests` module to use the Mermaid.INK API: "
            "`pip install requests`."
        )
        raise ImportError(msg)

    # 使用 Mermaid API 渲染图片
    mermaid_syntax_encoded = base64.b64encode(mermaid_syntax.encode("utf8")).decode(
        "ascii"
    )

    # 使用正则表达式检查背景颜色是否为十六进制颜色码
    if background_color is not None:
        hex_color_pattern = re.compile(r"^#(?:[0-9a-fA-F]{3}){1,2}$")
        if not hex_color_pattern.match(background_color):
            background_color = f"!{background_color}"

    # 对背景颜色进行 URL 编码，以处理 '!' 等特殊字符
    encoded_bg_color = urllib.parse.quote(str(background_color), safe="")
    image_url = (
        f"{base_url}/img/{mermaid_syntax_encoded}"
        f"?type={file_type}&bgColor={encoded_bg_color}"
    )

    error_msg_suffix = (
        "解决此问题的方法：\n"
        "1. 检查您的网络连接并重试\n"
        "2. 使用更高的重试设置重试："
        "`draw_mermaid_png(..., max_retries=5, retry_delay=2.0)`\n"
        "3. 使用 Pyppeteer 渲染方法，它将在本地浏览器中渲染您的图："
        "`draw_mermaid_png(..., draw_method=MermaidDrawMethod.PYPPETEER)`"
    )

    for attempt in range(max_retries + 1):
        try:
            response = requests.get(image_url, timeout=10, proxies=proxies)
            if response.status_code == requests.codes.ok:
                img_bytes = response.content
                if output_file_path is not None:
                    Path(output_file_path).write_bytes(response.content)

                return img_bytes

            # 如果遇到服务器错误（5xx），则重试
            if (
                requests.codes.internal_server_error <= response.status_code
                and attempt < max_retries
            ):
                # 带抖动的指数退避
                sleep_time = retry_delay * (2**attempt) * (0.5 + 0.5 * random.random())  # noqa: S311 not used for crypto
                time.sleep(sleep_time)
                continue

            # 对于其他状态码，立即失败
            msg = (
                f"Failed to reach {base_url} API while trying to render "
                f"your graph. Status code: {response.status_code}.\n\n"
            ) + error_msg_suffix
            raise ValueError(msg)

        except (requests.RequestException, requests.Timeout) as e:
            if attempt < max_retries:
                # 带抖动的指数退避
                sleep_time = retry_delay * (2**attempt) * (0.5 + 0.5 * random.random())  # noqa: S311 not used for crypto
                time.sleep(sleep_time)
            else:
                msg = (
                    f"在尝试渲染您的图 {max_retries} 次后，无法访问 {base_url} API。"
                ) + error_msg_suffix
                raise ValueError(msg) from e

    # 这里不应该被执行，但以防万一
    msg = (
        f"Failed to reach {base_url} API while trying to render "
        f"your graph after {max_retries} retries. "
    ) + error_msg_suffix
    raise ValueError(msg)
