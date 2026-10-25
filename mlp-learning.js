/* ==========================================================
   学习，搭建 MLP · mlp-learning.js
   1) 生成暖色泡泡背景
   2) 实现可点击生长的双向知识谱图（SVG）
      - 左图：数学推导（向右生长）
      - 右图：Python 代码实现（向左生长）
      - 双向条件箭头 + 左右对应末端跨图连线
   ========================================================== */

/* ==========================================================
   1. 背景泡泡
   ========================================================== */
(function generateBubbles() {
  const container = document.querySelector(".bubbles");
  if (!container) return;
  const COUNT = 22;
  for (let i = 0; i < COUNT; i++) {
    const b = document.createElement("div");
    b.className = "bubble";
    const size = 30 + Math.random() * 140; // 大小不一
    b.style.width = size + "px";
    b.style.height = size + "px";
    b.style.left = Math.random() * 100 + "vw";
    b.style.top = Math.random() * 100 + "vh";
    b.style.setProperty("--dur", 14 + Math.random() * 18 + "s");
    b.style.setProperty("--delay", -Math.random() * 20 + "s");
    b.style.setProperty("--dx", (Math.random() * 80 - 40) + "px");
    container.appendChild(b);
  }
})();

/* ==========================================================
   2. 知识谱图
   ========================================================== */

// ---------- 数据结构 ----------
// 节点格式：
// { id, label, side: 'L' | 'R', depth, x, y, w, h,
//   children: [id...], expanded, isCode, isRoot, visible }
//
// 用户提供的"尾生"关系：
// 左图（数学，向右）：
//   M (root) 数学上解决简单异或问题
//     -> A 初始参数设置
//        -> A1 输入点设置
//        -> A2 隐藏点设置
//        -> A3 w, b_hidden, v, b_output 的设置
//     -> B 输入到隐藏的特征投射
//        -> B1 仿射变换
//        -> B2 非线性激活
//     -> C 隐藏到输出的特征投射
//        -> C1 计算对数几率
//     -> D 优化算法
//        -> D1 二分类交叉熵计算损失
//        -> D2 Adam 梯度下降逐步优化
//     -> E 最终输出
//        -> E1 最终参数代入
//        -> E2 二值化处理
//   B 与 D 之间：当二者都被尾生出来后，出现两个单向箭头互指
//
// 右图（代码，向左）：
//   P (root) python代码实现
//     -> Pa 张量
//        -> Pa1 x,target 的 tf.constant()
//        -> Pa2 模型参数的 tf.Variable()
//     -> Pb 设置
//        -> Pb1 keras 精度设置
//        -> Pb2 learn_rate 设置
//     -> Pc 模型
//        -> Pc1 hidden = tf.sigmoid(tf.matmul(x,W)+b_hidden)
//        -> Pc2 logits = tf.matmul(hidden,v)+b_output
//     -> Pd 优化
//        -> Pd1 loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=logits))
//        -> Pd2 gradients = tape.gradient(loss, parameters)
//        -> Pd3 optimizer.apply_gradients(zip(gradients, parameters))
//     -> Pe 最终输出
//        -> Pe1 hidden = tf.sigmoid(tf.matmul(x,W)+b_hidden)
//             prob  = tf.sigmoid(tf.matmul(hidden,v)+b_output)   # 0~1 概率
//             print(tf.cast(prob >= 0.5, tf.int32).numpy().ravel())
//
// 左右对应末端连线（语义对应）：
//   A  <-> Pa   (初始参数 对应 张量)
//   B  <-> Pc   (输入到隐藏 对应 模型)
//   D  <-> Pd   (优化算法 对应 优化)
//   E  <-> Pe   (最终输出 对应 最终输出)
//   C1 <-> Pc2  (计算对数几率 对应 logits)

const SVG_NS = "http://www.w3.org/2000/svg";
const SVG_W = 1600;
const SVG_H = 900;

// 把可能很长的标签按字符数切成多行
function wrapLabel(text, maxCharsPerLine) {
  if (text.length <= maxCharsPerLine) return [text];
  const lines = [];
  let cur = "";
  for (const ch of text) {
    cur += ch;
    if (cur.length >= maxCharsPerLine) {
      lines.push(cur);
      cur = "";
    }
  }
  if (cur) lines.push(cur);
  return lines;
}

// ---------- 节点定义 ----------
const NODE_DEFS = [
  // ---- 左图（数学） ----
  { id: "M",  label: "数学上解决简单异或问题", side: "L", depth: 0, parent: null, isRoot: true },
  { id: "A",  label: "初始参数设置",        side: "L", depth: 1, parent: "M" },
  { id: "B",  label: "输入到隐藏的特征投射", side: "L", depth: 1, parent: "M" },
  { id: "C",  label: "隐藏到输出的特征投射", side: "L", depth: 1, parent: "M" },
  { id: "D",  label: "优化算法",            side: "L", depth: 1, parent: "M" },
  { id: "E",  label: "最终输出",            side: "L", depth: 1, parent: "M" },

  { id: "A1", label: "输入点设置",          side: "L", depth: 2, parent: "A" },
  { id: "A2", label: "隐藏点设置",          side: "L", depth: 2, parent: "A" },
  { id: "A3", label: "w, b_hidden, v, b_output 的设置", side: "L", depth: 2, parent: "A" },

  { id: "B1", label: "仿射变换",            side: "L", depth: 2, parent: "B" },
  { id: "B2", label: "非线性激活",          side: "L", depth: 2, parent: "B" },

  { id: "C1", label: "计算对数几率",        side: "L", depth: 2, parent: "C" },

  { id: "D1", label: "二分类交叉熵计算损失", side: "L", depth: 2, parent: "D" },
  { id: "D2", label: "Adam 梯度下降逐步优化", side: "L", depth: 2, parent: "D" },

  { id: "E1", label: "最终参数代入",        side: "L", depth: 2, parent: "E" },
  { id: "E2", label: "二值化处理",          side: "L", depth: 2, parent: "E" },

  // ---- 右图（代码） ----
  { id: "P",  label: "python 代码实现",     side: "R", depth: 0, parent: null, isRoot: true, isCode: true },
  { id: "Pa", label: "张量",                side: "R", depth: 1, parent: "P", isCode: true },
  { id: "Pb", label: "设置",                side: "R", depth: 1, parent: "P", isCode: true },
  { id: "Pc", label: "模型",                side: "R", depth: 1, parent: "P", isCode: true },
  { id: "Pd", label: "优化",                side: "R", depth: 1, parent: "P", isCode: true },
  { id: "Pe", label: "最终输出",            side: "R", depth: 1, parent: "P", isCode: true },

  { id: "Pa1", label: "x, target 的 tf.constant()",   side: "R", depth: 2, parent: "Pa", isCode: true },
  { id: "Pa2", label: "模型参数的 tf.Variable()",     side: "R", depth: 2, parent: "Pa", isCode: true },

  { id: "Pb1", label: "keras 精度设置",                side: "R", depth: 2, parent: "Pb", isCode: true },
  { id: "Pb2", label: "learn_rate 设置",               side: "R", depth: 2, parent: "Pb", isCode: true },

  { id: "Pc1", label: "hidden = tf.sigmoid(tf.matmul(x, W) + b_hidden)", side: "R", depth: 2, parent: "Pc", isCode: true },
  { id: "Pc2", label: "logits = tf.matmul(hidden, v) + b_output",        side: "R", depth: 2, parent: "Pc", isCode: true },

  { id: "Pd1", label: "loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=logits))", side: "R", depth: 2, parent: "Pd", isCode: true, wrap: 30 },
  { id: "Pd2", label: "gradients = tape.gradient(loss, parameters)",         side: "R", depth: 2, parent: "Pd", isCode: true },
  { id: "Pd3", label: "optimizer.apply_gradients(zip(gradients, parameters))", side: "R", depth: 2, parent: "Pd", isCode: true },

  { id: "Pe1", label: "hidden = tf.sigmoid(tf.matmul(x, W) + b_hidden)\nprob  = tf.sigmoid(tf.matmul(hidden, v) + b_output)  # 0~1 概率\nprint(tf.cast(prob >= 0.5, tf.int32).numpy().ravel())", side: "R", depth: 2, parent: "Pe", isCode: true, rawLines: true },
];

// 跨图末端对应关系（左 -> 右）：左右各 10 个最末端节点一一对应
// 左：A1 A2 A3 B1 B2 C1 D1 D2 E1 E2
// 右：Pa1 Pa2 Pb1 Pb2 Pc1 Pc2 Pd1 Pd2 Pd3 Pe1
const CROSS_LINKS = [
  ["A1", "Pa1"],
  ["A2", "Pa2"],
  ["A3", "Pb1"],
  ["B1", "Pb2"],
  ["B2", "Pc1"],
  ["C1", "Pc2"],
  ["D1", "Pd1"],
  ["D2", "Pd2"],
  ["E1", "Pd3"],
  ["E2", "Pe1"],
];

// ---------- 节点状态初始化 ----------
const nodes = {};
NODE_DEFS.forEach(def => {
  nodes[def.id] = {
    ...def,
    expanded: false,
    visible: def.parent === null, // 只有根节点初始可见
    x: 0, y: 0, w: 0, h: 0,
  };
});

const svg = document.getElementById("knowledge-graph");
if (svg) {
  // 让根节点默认展开一层（即显示出根节点），用户点击根节点后才展开下一层
  layoutGraph();
  renderGraph();
  bindReset();
}

// ---------- 布局算法 ----------
// 左侧：根节点 x=80，每展开一层 x += 260
// 右侧：根节点 x=1520，每展开一层 x -= 320（代码标签更长，所以层距更大）
function layoutGraph() {
  const LEFT_X0 = 110;
  const LEFT_DX = 250;
  const RIGHT_X0 = SVG_W - 110;
  const RIGHT_DX = 280;

  // 给每一边做递归布局
  layoutSide("M", LEFT_X0, LEFT_DX, +1);
  layoutSide("P", RIGHT_X0, RIGHT_DX, -1);
}

// side 方向布局：递归地计算每个可见节点的 y
function layoutSide(rootId, x0, dx, sign) {
  // 收集可见子树，按当前展开状态做纵向分布
  const rootNode = nodes[rootId];
  // 先测量：每个可见节点占多少高度
  const visibleNodes = Object.values(nodes).filter(n => n.visible && n.side === rootNode.side);
  if (visibleNodes.length === 0) return;

  // 计算每个节点的尺寸
  Object.values(nodes).forEach(n => {
    const wrap = n.wrap || 14;
    let lines;
    if (n.rawLines) {
      lines = n.label.split("\n");
    } else {
      lines = wrapLabel(n.label, wrap);
    }
    n._lines = lines;
    const lineHeight = 18;
    const padX = 14;
    const padY = 8;
    // 估算字符宽度：中文 14px，英文 8px
    let maxW = 0;
    lines.forEach(l => {
      let w = 0;
      for (const ch of l) {
        w += ch.charCodeAt(0) > 255 ? 14 : 8;
      }
      if (w > maxW) maxW = w;
    });
    n.w = Math.min(Math.max(maxW + padX * 2, 90), 460);
    n.h = lines.length * lineHeight + padY * 2;
  });

  // 递归布局：返回当前子树占用的总高度，并把节点放进 y 范围
  const V_SPACING = 24;

  function place(nodeId, centerX, topY) {
    const node = nodes[nodeId];
    if (!node.visible) return topY;

    // 计算可见子节点
    const childIds = NODE_DEFS.filter(d => d.parent === nodeId).map(d => d.id);
    const visibleChildren = node.expanded ? childIds.filter(cid => nodes[cid].visible) : [];

    if (visibleChildren.length === 0) {
      node.x = centerX;
      node.y = topY + node.h / 2;
      return topY + node.h + V_SPACING;
    }

    // 先布局所有子节点，计算子树总高度
    let childTop = topY;
    const childX = centerX + (node.side === "L" ? 1 : -1) * (node.side === "L" ? 250 : 280);
    visibleChildren.forEach(cid => {
      childTop = place(cid, childX, childTop);
    });
    const subtreeHeight = childTop - topY - V_SPACING;
    node.x = centerX;
    node.y = topY + subtreeHeight / 2 + node.h / 2 - V_SPACING / 2;
    return childTop;
  }

  place(rootId, x0, 60);
}

// ---------- 渲染 ----------
function renderGraph() {
  if (!svg) return;
  // 清空
  while (svg.firstChild) svg.removeChild(svg.firstChild);

  // 定义箭头 marker
  const defs = document.createElementNS(SVG_NS, "defs");
  const marker = document.createElementNS(SVG_NS, "marker");
  marker.setAttribute("id", "arrowhead");
  marker.setAttribute("markerWidth", "10");
  marker.setAttribute("markerHeight", "10");
  marker.setAttribute("refX", "9");
  marker.setAttribute("refY", "3");
  marker.setAttribute("orient", "auto");
  marker.setAttribute("markerUnits", "strokeWidth");
  const arrowPath = document.createElementNS(SVG_NS, "path");
  arrowPath.setAttribute("d", "M0,0 L0,6 L9,3 z");
  arrowPath.classList.add("kg-arrow-head");
  marker.appendChild(arrowPath);
  defs.appendChild(marker);

  svg.appendChild(defs);

  // 先画边（在节点之下）
  // 1) 普通树边
  NODE_DEFS.forEach(d => {
    if (!d.parent) return;
    const child = nodes[d.id];
    const parent = nodes[d.parent];
    if (!child.visible || !parent.visible) return;
    // 父已展开才有这条边
    if (!parent.expanded) return;
    drawEdge(svg, parent, child);
  });

  // 2) 跨图末端连线（仅当两端都已展开显示时才绘制）
  CROSS_LINKS.forEach(([lid, rid]) => {
    const L = nodes[lid];
    const R = nodes[rid];
    if (L && R && L.visible && R.visible) {
      drawLinkEdge(svg, L, R);
    }
  });

  // 再画节点
  Object.values(nodes).forEach(n => {
    if (!n.visible) return;
    drawNode(svg, n);
  });
}

// ---------- 绘制原语 ----------
function drawNode(svg, node) {
  const g = document.createElementNS(SVG_NS, "g");
  g.classList.add("kg-node");
  if (node.isRoot) g.classList.add(node.side === "R" ? "kg-right-root" : "kg-root");
  if (node.isCode) g.classList.add("kg-code");

  const rect = document.createElementNS(SVG_NS, "rect");
  rect.setAttribute("x", node.x - node.w / 2);
  rect.setAttribute("y", node.y - node.h / 2);
  rect.setAttribute("width", node.w);
  rect.setAttribute("height", node.h);
  rect.setAttribute("rx", 10);
  rect.setAttribute("ry", 10);
  g.appendChild(rect);

  const text = document.createElementNS(SVG_NS, "text");
  text.setAttribute("x", node.x);
  text.setAttribute("text-anchor", "middle");
  text.setAttribute("dominant-baseline", "middle");
  const lines = node._lines || [node.label];
  const lineHeight = 18;
  const totalH = lines.length * lineHeight;
  const startY = node.y - totalH / 2 + lineHeight / 2;
  lines.forEach((line, i) => {
    const tspan = document.createElementNS(SVG_NS, "tspan");
    tspan.setAttribute("x", node.x);
    tspan.setAttribute("y", startY + i * lineHeight);
    tspan.textContent = line;
    text.appendChild(tspan);
  });
  g.appendChild(text);

  g.addEventListener("click", () => toggleNode(node.id));
  svg.appendChild(g);
}

function drawEdge(svg, parent, child) {
  const path = document.createElementNS(SVG_NS, "path");
  const x1 = parent.side === "L" ? parent.x + parent.w / 2 : parent.x - parent.w / 2;
  const y1 = parent.y;
  const x2 = child.side === "L" ? child.x - child.w / 2 : child.x + child.w / 2;
  const y2 = child.y;
  const mx = (x1 + x2) / 2;
  const d = `M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`;
  path.setAttribute("d", d);
  path.classList.add("kg-edge");
  path.setAttribute("marker-end", "url(#arrowhead)");
  svg.appendChild(path);
}

function drawLinkEdge(svg, L, R) {
  const path = document.createElementNS(SVG_NS, "path");
  const x1 = L.x + L.w / 2;
  const y1 = L.y;
  const x2 = R.x - R.w / 2;
  const y2 = R.y;
  const mx = (x1 + x2) / 2;
  const d = `M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`;
  path.setAttribute("d", d);
  path.classList.add("kg-edge", "kg-edge-link");
  svg.appendChild(path);
}

// ---------- 交互 ----------
function toggleNode(id) {
  const node = nodes[id];
  if (!node) return;
  const childIds = NODE_DEFS.filter(d => d.parent === id).map(d => d.id);
  if (childIds.length === 0) {
    // 没有子节点，无操作
    return;
  }
  if (!node.expanded) {
    // 展开
    node.expanded = true;
    childIds.forEach(cid => {
      nodes[cid].visible = true;
    });
  } else {
    // 折叠：递归把整棵子树隐藏
    node.expanded = false;
    collapseSubtree(id);
  }
  layoutGraph();
  renderGraph();
}

function collapseSubtree(id) {
  const childIds = NODE_DEFS.filter(d => d.parent === id).map(d => d.id);
  childIds.forEach(cid => {
    const c = nodes[cid];
    c.visible = false;
    c.expanded = false;
    collapseSubtree(cid);
  });
}

// 重置按钮
function bindReset() {
  const btn = document.getElementById("graph-reset");
  if (!btn) return;
  btn.addEventListener("click", () => {
    Object.values(nodes).forEach(n => {
      n.expanded = false;
      n.visible = n.parent === null;
    });
    layoutGraph();
    renderGraph();
  });
}

/* ==========================================================
   3. 作品版本迭代 · 翻页
   点击左 1/4 翻上一页，右 1/4 翻下一页
   ========================================================== */
(function initVersionCarousel() {
  const carousel = document.getElementById("version-carousel");
  const track = document.getElementById("version-track");
  const dotsBox = document.getElementById("version-dots");
  if (!carousel || !track) return;

  const cards = track.querySelectorAll(".version-card");
  const total = cards.length;
  if (total === 0) return;

  const btnPrev = carousel.querySelector(".version-nav-prev");
  const btnNext = carousel.querySelector(".version-nav-next");

  let current = 0;

  // 生成页码圆点（可点击跳转）
  for (let i = 0; i < total; i++) {
    const d = document.createElement("button");
    d.type = "button";
    d.className = "dot" + (i === 0 ? " active" : "");
    d.setAttribute("aria-label", "第 " + (i + 1) + " 版");
    d.addEventListener("click", (e) => {
      e.stopPropagation();
      goTo(i);
    });
    dotsBox.appendChild(d);
  }
  const dots = dotsBox.querySelectorAll(".dot");

  function goTo(idx) {
    // 循环翻页：到尾后再向右回到第 0 页，反之亦然
    if (idx < 0) idx = total - 1;
    if (idx > total - 1) idx = 0;
    current = idx;
    track.style.transform = "translateX(-" + current * 100 + "%)";
    dots.forEach((d, i) => d.classList.toggle("active", i === current));
  }

  btnPrev.addEventListener("click", () => goTo(current - 1));
  btnNext.addEventListener("click", () => goTo(current + 1));

  // 键盘左右方向键也可翻页
  carousel.setAttribute("tabindex", "0");
  carousel.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") goTo(current - 1);
    if (e.key === "ArrowRight") goTo(current + 1);
  });

  goTo(0);
})();
