from dataclasses import dataclass
import json


@dataclass
class ModelData:
    C: list
    S: list
    V: list
    A: list
    p: dict
    t: dict
    q: dict
    e: dict
    Q: float
    QMin: float
    K: int
    M: float
    R: int
    n: int
    startNode: int = 0
    endNode: int = 0

    def validate(self):
        if self.startNode != 0:
            raise ValueError("模型文档规定起点下标必须为 0")
        if self.endNode != self.n + 1:
            raise ValueError("终点下标必须为 n + 1")
        if 0 in self.C:
            raise ValueError("C 不能包含 0；0 是起点车场节点")
        if self.endNode in self.C or self.endNode in self.S:
            raise ValueError("终点车场节点不能属于 C 或 S")
        if self.QMin < 0 or self.QMin > self.Q:
            raise ValueError("要求 0 <= QMin <= Q")
        if self.K <= 0:
            raise ValueError("K 必须为正整数")
        if self.M <= 0:
            raise ValueError("M 必须为正数")
        if self.R != len(self.S):
            raise ValueError("R 必须等于换电服务节点数量")

        expectedV = [self.startNode] + self.C + self.S + [self.endNode]
        if self.V != expectedV:
            raise ValueError("V 必须按照 {0} ∪ C ∪ S ∪ {n+1} 构造")

        for i, j in self.A:
            if (i, j) not in self.t:
                raise ValueError(f"弧 {(i, j)} 缺少 t_ij")
            if (i, j) not in self.e:
                raise ValueError(f"弧 {(i, j)} 缺少 e_ij")

        for i in [self.startNode] + self.C + self.S:
            if i not in self.p:
                raise ValueError(f"节点 {i} 缺少 p_i")
            if i not in self.q:
                raise ValueError(f"节点 {i} 缺少 q_i")

    def toJson(self, filePath):
        data = {
            "C": self.C,
            "S": self.S,
            "V": self.V,
            "A": [[i, j] for i, j in self.A],
            "p": {str(i): value for i, value in self.p.items()},
            "t": [[i, j, value] for (i, j), value in self.t.items()],
            "q": {str(i): value for i, value in self.q.items()},
            "e": [[i, j, value] for (i, j), value in self.e.items()],
            "Q": self.Q,
            "QMin": self.QMin,
            "K": self.K,
            "M": self.M,
            "R": self.R,
            "n": self.n,
            "startNode": self.startNode,
            "endNode": self.endNode,
        }
        with open(filePath, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)


def loadData(filePath):
    with open(filePath, "r", encoding="utf-8") as file:
        raw = json.load(file)

    p = {int(i): value for i, value in raw["p"].items()}
    q = {int(i): value for i, value in raw["q"].items()}
    t = {(i, j): value for i, j, value in raw["t"]}
    e = {(i, j): value for i, j, value in raw["e"]}

    modelData = ModelData(
        C=raw["C"],
        S=raw["S"],
        V=raw["V"],
        A=[tuple(arc) for arc in raw["A"]],
        p=p,
        t=t,
        q=q,
        e=e,
        Q=raw["Q"],
        QMin=raw["QMin"],
        K=raw["K"],
        M=raw["M"],
        R=raw["R"],
        n=raw["n"],
        startNode=raw["startNode"],
        endNode=raw["endNode"],
    )
    modelData.validate()
    return modelData
