# 集合
- $$\mathcal{C}=\{1,2,\cdots, |\mathcal{C}| \}：任务节点集合$$
- $$\mathcal{S}=\{S^1=|\mathcal{C}|+1,\cdots,S^{R}=|\mathcal{C}|+R=n \}：虚拟换电站节点$$
- $$\mathcal{V}=\{0\} \cup \mathcal{C} \cup \mathcal{S} \cup \{n+1 \}：节点集合$$
  - $$0表示起点，n+1表示终点$$
- $$\mathcal{A}=\mathcal{A}^{OC}\cup\mathcal{A}^{CC}\cup\mathcal{A}^{CS}\cup\mathcal{A}^{SC}\cup\mathcal{A}^{CF}\cup\mathcal{A}^{SF}：弧集合$$
# 参数
- $$p_i：节点自身的执行时间$$
  - $$对于任务节点：即从装料点到卸料点的行程时间$$
  - $$对于换电站节点：即换电时间$$
- $$t_{ij}：节点间的行程时间$$
  - $$对于任务节点至任务节点：即上一任务的卸料点至下一任务的装料点的行程时间$$
  - $$对于任务节点至换电站节点：即任务的卸料点至换电站的行程时间$$
  - $$对于换电站节点至任务节点：即换电站至任务的装料点的行程时间$$
- $$q_i：节点自身的能耗$$
  - $$对于任务节点：即从装料点到卸料点的能耗$$
  - $$对于换电站节点：0$$
- $$e_{ij}：节点间的行程能耗$$
  - $$对于任务节点至任务节点：即上一任务的卸料点至下一任务的装料点的能耗$$
  - $$对于任务节点至换电站节点：即任务的卸料点至换电站的能耗$$
  - $$对于换电站节点至任务节点：即换电站至任务的装料点的能耗$$
- $$Q,Q^{min}：满电电量和安全电量$$
- $$K：出勤车辆上限$$
# 决策
1. $$弧变量：表示是否从节点i前往节点j$$
$$x_{ij} \in \{0,1\}$$
2. $$离开时间变量：表示节点i服务完成时刻$$
$$T_i \geq 0 $$
3. $$剩余电量变量：表示节点i服务完成后的剩余变量$$
$$E_{i}\geq 0$$
# 目标
- $$最小化完工时间（回到终点为止）$$
$$\text{min}\quad T_{n+1}$$
# 约束
1. $$发车数量约束 + 出发/返回车辆相等约束$$
$$\sum_{j:(0,j)\in \mathcal{A}}x_{0j} = \sum_{i:(i,n+1)\in \mathcal{A}}x_{i,n+1} \leq K$$
2. $$任务访问约束$$
$$\sum_{j:(i,j)\in \mathcal{A}} x_{ij} = \sum_{j:(j,i)\in \mathcal{A}} x_{ji} = 1，\quad \forall i \in \mathcal{C}$$
3. $$换电站访问约束$$
$$\sum_{j:(i,j)\in \mathcal{A}} x_{ij} = \sum_{j:(j,i)\in \mathcal{A}} x_{ji} \leq 1，\quad \forall i \in \mathcal{S}$$
4. $$节点离开时间约束$$
$$T_j \geq T_i+t_{ij}+p_j-M(1-x_{ij}), \quad \forall (i,j) \in \mathcal{A},j \in \mathcal{C}\cup\mathcal{S}$$
$$T_j \leq T_i+t_{ij}+p_j+M(1-x_{ij}), \quad \forall (i,j) \in \mathcal{A},j \in \mathcal{C}$$
5. $$未使用的换电站，离开时间设定为0$$
$$T_{j} \leq M\sum_{i:(i,j)\in \mathcal{A}}x_{ij}, \quad \forall j \in \mathcal{S}$$
6. $$返回终点时间约束$$
$$T_{n+1}\geq T_i+t_{i,n+1}-M(1-x_{i,n+1}),\quad \forall (i,n+1)\in \mathcal{A}$$
7. $$电量合法性性约束$$
$$Q^{min} \leq E_i \leq Q, \quad\forall i \in \mathcal{C}$$
8. $$电量递推约束$$
$$E_i-e_{ij}-q_j-M(1-x_{ij}) \leq E_j \leq E_i-e_{ij}-q_j+M(1-x_{ij}), \quad \forall (i,j) \in \mathcal{A}, j \in \mathcal{C}$$
$$E_i-e_{ij} \geq Q_{min}-M(1-x_{ij}),\quad  \forall (i,j) \in \mathcal{A}, j \in \mathcal{S}\cup\{ n+1\}$$
$$E_{j}=Q\sum_{i:(i,j)\in \mathcal{A}}x_{ij},  \quad \forall j \in \mathcal{S}$$
9. $$换电站对称性消除$$
$$\sum_{i:(i,S^{r+1})\in \mathcal{A}}x_{iS^{r+1}} \leq \sum_{i:(i,S^{r})\in \mathcal{A}}x_{iS^{r}}, \quad \forall r=1,2\cdots,R-1$$
10. $$换电站排队约束$$
$$T_{S^{r+1}}\geq T_{S^{r}}+p_{S^{r+1}}-M(1-\sum_{i:(i,S^{r+1})\in \mathcal{A}}x_{iS^{r+1}}), \quad \forall r=1,2\cdots,R-1$$
11. $$起点初始化约束$$
$$T_0=0,E_0=Q$$
12. $$完工时间下界约束$$
$$K\cdot T_{n+1} \geq \sum_{i\in \mathcal{C}}p_i+\sum_{(i,j) \in \mathcal{A}}t_{ij}x_{ij}+\sum_{r=1}^{R}p_{S^r}\sum_{i:(i,S^r)\in \mathcal{A}}x_{iS^r}$$
