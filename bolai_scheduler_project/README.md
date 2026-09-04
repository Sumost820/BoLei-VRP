# 伯镭调度 Gurobi 模型

本项目实现《伯镭调度数据模型 - 差异化时间和能耗》的最终 MILP。

## 目录

```text
src/bolaiScheduling/
  modelData.py      # 模型参数对象和 JSON 读写
  gurobiModel.py    # Gurobi 模型
  mockData.py       # mock 数据生成器

tests/
  testModelData.py
  testMockData.py
  testGurobiModel.py

data/
  mock25.json
  mock50.json
  mock75.json
  mock100.json
```

## 节点定义

- `0`：起点车场节点
- `C = {1, ..., |C|}`：任务节点
- `S = {|C|+1, ..., |C|+R}`：换电服务副本节点
- `n = |C| + R`
- `n+1`：终点车场节点，与节点 0 为同一物理车场，但使用独立下标

mock25 / 50 / 75 / 100 中的数字表示任务节点 `C` 的数量。

## 安装

```bash
pip install -e ".[test]"
```

需要有效的 Gurobi 安装和许可证。

## 测试

```bash
pytest -q
```

如果环境中没有 `gurobipy`，数据相关测试仍可运行，Gurobi 测试会自动跳过。

## 使用

```python
from bolaiScheduling.modelData import loadData
from bolaiScheduling.gurobiModel import GurobiScheduler

modelData = loadData("data/mock25.json")
scheduler = GurobiScheduler(modelData)
model = scheduler.solveModel(timeLimit=3600, mipGap=0.001)

print("objective:", model.ObjVal)
for route in scheduler.getRoutes():
    print(route)
```

## 模型变量

代码严格保留文档中的核心符号：

- `x[i,j]`：弧变量
- `T[i]`：节点离开时间
- `E[i]`：节点服务完成后的剩余电量
- `p[i]`、`t[i,j]`、`q[i]`、`e[i,j]`
- `Q`、`QMin`、`K`、`M`

目标直接使用 `T[n+1]`。
