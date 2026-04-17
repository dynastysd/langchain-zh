"""使用 ASCII 字符绘制有向无环图（DAG）。

改编自 https://github.com/iterative/dvc/blob/main/dvc/dagascii.py。
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Any

try:
    from grandalf.graphs import Edge, Graph, Vertex  # type: ignore[import-untyped]
    from grandalf.layouts import SugiyamaLayout  # type: ignore[import-untyped]
    from grandalf.routing import route_with_lines  # type: ignore[import-untyped]

    _HAS_GRANDALF = True
except ImportError:
    _HAS_GRANDALF = False

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from langchain_core.runnables.graph import Edge as LangEdge


class VertexViewer:
    """顶点查看器类。

    用于定义顶点的边界框，在 grandalf 构建图时会被考虑。
    """

    HEIGHT = 3  # 顶部和底部边框 + 文本
    """边界框的高度。"""

    def __init__(self, name: str) -> None:
        """创建顶点查看器。

        Args:
            name: 顶点的名称。
        """
        self._h = self.HEIGHT  # 顶部和底部边框 + 文本
        self._w = len(name) + 2  # 左右边框 + 文本

    @property
    def h(self) -> int:
        """边界框的高度。"""
        return self._h

    @property
    def w(self) -> int:
        """边界框的宽度。"""
        return self._w


class AsciiCanvas:
    """用于 ASCII 绘图的画布类。"""

    TIMEOUT = 10

    def __init__(self, cols: int, lines: int) -> None:
        """创建 ASCII 画布。

        Args:
            cols: 画布的列数。应该大于 1。
            lines: 画布的行数。应该大于 1。

        Raises:
            ValueError: 如果画布尺寸无效。
        """
        if cols <= 1 or lines <= 1:
            msg = "Canvas dimensions should be > 1"
            raise ValueError(msg)

        self.cols = cols
        self.lines = lines

        self.canvas = [[" "] * cols for line in range(lines)]

    def draw(self) -> str:
        """在屏幕上绘制 ASCII 画布。

        Returns:
            ASCII 画布字符串。
        """
        lines = map("".join, self.canvas)
        return os.linesep.join(lines)

    def point(self, x: int, y: int, char: str) -> None:
        """在 ASCII 画布上创建一个点。

        Args:
            x: x 坐标。应该 >= 0 且 < 画布的列数。
            y: y 坐标。应该 >= 0 且 < 画布的行数。
            char: 要放置在指定位置的字符。

        Raises:
            ValueError: 如果 char 不是单个字符或坐标超出范围。
        """
        if len(char) != 1:
            msg = "char should be a single character"
            raise ValueError(msg)
        if x >= self.cols or x < 0:
            msg = "x should be >= 0 and < number of columns"
            raise ValueError(msg)
        if y >= self.lines or y < 0:
            msg = "y should be >= 0 and < number of lines"
            raise ValueError(msg)

        self.canvas[y][x] = char

    def line(self, x0: int, y0: int, x1: int, y1: int, char: str) -> None:
        """在 ASCII 画布上创建一条线。

        Args:
            x0: 线条起点的 x 坐标。
            y0: 线条起点的 y 坐标。
            x1: 线条终点的 x 坐标。
            y1: 线条终点的 y 坐标。
            char: 用于绘制线条的字符。
        """
        if x0 > x1:
            x1, x0 = x0, x1
            y1, y0 = y0, y1

        dx = x1 - x0
        dy = y1 - y0

        if dx == 0 and dy == 0:
            self.point(x0, y0, char)
        elif abs(dx) >= abs(dy):
            for x in range(x0, x1 + 1):
                y = y0 if dx == 0 else y0 + round((x - x0) * dy / float(dx))
                self.point(x, y, char)
        elif y0 < y1:
            for y in range(y0, y1 + 1):
                x = x0 if dy == 0 else x0 + round((y - y0) * dx / float(dy))
                self.point(x, y, char)
        else:
            for y in range(y1, y0 + 1):
                x = x0 if dy == 0 else x1 + round((y - y1) * dx / float(dy))
                self.point(x, y, char)

    def text(self, x: int, y: int, text: str) -> None:
        """在 ASCII 画布上打印文本。

        Args:
            x: 文本起始位置的 x 坐标。
            y: 文本起始位置的 y 坐标。
            text: 要打印的字符串。
        """
        for i, char in enumerate(text):
            self.point(x + i, y, char)

    def box(self, x0: int, y0: int, width: int, height: int) -> None:
        """在 ASCII 画布上创建一个边框。

        Args:
            x0: 边框角落的 x 坐标。
            y0: 边框角落的 y 坐标。
            width: 边框宽度。
            height: 边框高度。

        Raises:
            ValueError: 如果边框尺寸无效。
        """
        if width <= 1 or height <= 1:
            msg = "Box dimensions should be > 1"
            raise ValueError(msg)

        width -= 1
        height -= 1

        for x in range(x0, x0 + width):
            self.point(x, y0, "-")
            self.point(x, y0 + height, "-")

        for y in range(y0, y0 + height):
            self.point(x0, y, "|")
            self.point(x0 + width, y, "|")

        self.point(x0, y0, "+")
        self.point(x0 + width, y0, "+")
        self.point(x0, y0 + height, "+")
        self.point(x0 + width, y0 + height, "+")


class _EdgeViewer:
    def __init__(self) -> None:
        self.pts: list[tuple[float]] = []

    def setpath(self, pts: list[tuple[float]]) -> None:
        self.pts = pts


def _build_sugiyama_layout(
    vertices: Mapping[str, str], edges: Sequence[LangEdge]
) -> Any:
    """构建杉山布局（Sugiyama layout）。

    Args:
        vertices: 图中的顶点列表。
        edges: 图中的边列表。

    Returns:
        杉山布局对象。
    """
    if not _HAS_GRANDALF:
        msg = "安装 grandalf 以绘制图：`pip install grandalf`。"
        raise ImportError(msg)

    #
    # 坐标命名约定的提醒：
    # +------------X
    # |
    # |
    # |
    # |
    # Y
    #

    vertices_ = {id_: Vertex(f" {data} ") for id_, data in vertices.items()}
    edges_ = [Edge(vertices_[s], vertices_[e], data=cond) for s, e, _, cond in edges]
    vertices_list = vertices_.values()
    graph = Graph(vertices_list, edges_)

    for vertex in vertices_list:
        vertex.view = VertexViewer(vertex.data)

    # 注意：确定最小边界框长度以创建最佳布局
    minw = min(v.view.w for v in vertices_list)

    for edge in edges_:
        edge.view = _EdgeViewer()

    sug = SugiyamaLayout(graph.C[0])
    graph = graph.C[0]
    roots = list(filter(lambda x: len(x.e_in()) == 0, graph.sV))

    sug.init_all(roots=roots, optimize=True)

    sug.yspace = VertexViewer.HEIGHT
    sug.xspace = minw
    sug.route_edge = route_with_lines

    sug.draw()

    return sug


def draw_ascii(vertices: Mapping[str, str], edges: Sequence[LangEdge]) -> str:
    """构建有向无环图（DAG）并使用 ASCII 字符绘制。

    Args:
        vertices: 图中的顶点列表。
        edges: 图中的边列表。

    Raises:
        ValueError: 如果画布尺寸无效或边坐标无效。

    Returns:
        ASCII 表示形式

    示例:
        ```python
        from langchain_core.runnables.graph_ascii import draw_ascii

        vertices = {1: "1", 2: "2", 3: "3", 4: "4"}
        edges = [
            (source, target, None, None)
            for source, target in [(1, 2), (2, 3), (2, 4), (1, 4)]
        ]


        print(draw_ascii(vertices, edges))
        ```

        ```txt

                 +---+
                 | 1 |
                 +---+
                 *    *
                *     *
               *       *
           +---+       *
           | 2 |       *
           +---+**     *
             *    **   *
             *      ** *
             *        **
           +---+     +---+
           | 3 |     | 4 |
           +---+     +---+
        ```
    """
    # 注意：坐标可能是负数，因此需要在绘制前
    # 将所有内容平移到正平面。
    xlist: list[float] = []
    ylist: list[float] = []

    sug = _build_sugiyama_layout(vertices, edges)

    for vertex in sug.g.sV:
        # 注意：将边界框向左移动 w/2
        xlist.extend(
            (
                vertex.view.xy[0] - vertex.view.w / 2.0,
                vertex.view.xy[0] + vertex.view.w / 2.0,
            )
        )
        ylist.extend((vertex.view.xy[1], vertex.view.xy[1] + vertex.view.h))

    for edge in sug.g.sE:
        for x, y in edge.view.pts:
            xlist.append(x)
            ylist.append(y)

    minx = min(xlist)
    miny = min(ylist)
    maxx = max(xlist)
    maxy = max(ylist)

    canvas_cols = math.ceil(math.ceil(maxx) - math.floor(minx)) + 1
    canvas_lines = round(maxy - miny)

    canvas = AsciiCanvas(canvas_cols, canvas_lines)

    # 注意：先绘制边，这样节点边界框可以覆盖它们
    for edge in sug.g.sE:
        if len(edge.view.pts) <= 1:
            msg = "没有足够的点来绘制边"
            raise ValueError(msg)
        for index in range(1, len(edge.view.pts)):
            start = edge.view.pts[index - 1]
            end = edge.view.pts[index]

            start_x = round(start[0] - minx)
            start_y = round(start[1] - miny)
            end_x = round(end[0] - minx)
            end_y = round(end[1] - miny)

            if start_x < 0 or start_y < 0 or end_x < 0 or end_y < 0:
                msg = (
                    "无效的边坐标："
                    f"start_x={start_x}, "
                    f"start_y={start_y}, "
                    f"end_x={end_x}, "
                    f"end_y={end_y}"
                )
                raise ValueError(msg)

            canvas.line(start_x, start_y, end_x, end_y, "." if edge.data else "*")

    for vertex in sug.g.sV:
        # 注意：将边界框向左移动 w/2
        x = vertex.view.xy[0] - vertex.view.w / 2.0
        y = vertex.view.xy[1]

        canvas.box(
            round(x - minx),
            round(y - miny),
            vertex.view.w,
            vertex.view.h,
        )

        canvas.text(round(x - minx) + 1, round(y - miny) + 1, vertex.data)

    return canvas.draw()
