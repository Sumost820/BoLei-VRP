# Arc-based 模型
## 集合
- $$\mathcal{C}=\{1,2,\cdots, |\mathcal{C}| \}：任务节点集合$$
- $$\mathcal{S}=\{S^1=|\mathcal{C}|+1,\cdots,S^{R}=|\mathcal{C}|+R=n \}：虚拟换电站节点$$
- $$\mathcal{V}=\{0\} \cup \mathcal{C} \cup \mathcal{S} \cup \{n+1 \}：节点集合$$
  - $$0表示起点，n+1表示终点$$
- $$\mathcal{A}=\mathcal{A}^{OC}\cup\mathcal{A}^{CC}\cup\mathcal{A}^{CS}\cup\mathcal{A}^{SC}\cup\mathcal{A}^{CF}\cup\mathcal{A}^{SF}：弧集合$$
## 参数
- $$p_i：节点自身的执行时间（服务时间）$$
  - $$对于任务节点：即从装料点到卸料点的行程时间$$
  - $$对于换电站节点：即换电时间$$
  - $$对于起点和终点：0$$
- $$t_{ij}：节点间的行程时间$$
  - $$对于任务节点至任务节点：即上一任务的卸料点至下一任务的装料点的行程时间$$
  - $$对于任务节点至换电站节点：即任务的卸料点至换电站的行程时间$$
  - $$对于换电站节点至任务节点：即换电站至任务的装料点的行程时间$$
- $$q_i：节点自身的能耗$$
  - $$对于任务节点：即从装料点到卸料点的能耗$$
  - $$对于换电站节点、起点和终点：0$$
- $$e_{ij}：节点间的行程能耗$$
  - $$对于任务节点至任务节点：即上一任务的卸料点至下一任务的装料点的能耗$$
  - $$对于任务节点至换电站节点：即任务的卸料点至换电站的能耗$$
  - $$对于换电站节点至任务节点：即换电站至任务的装料点的能耗$$
- $$Q,Q^{min}：满电电量和安全电量$$
- $$K：出勤车辆上限$$
## 决策
1. $$弧变量：表示是否从节点i前往节点j$$
$$x_{ij} \in \{0,1\}$$
2. $$离开时间变量：表示节点i服务完成时刻$$
$$T_i \geq 0 $$
3. $$剩余电量变量：表示节点i服务完成后的剩余变量$$
$$E_{i}\geq 0$$
## 目标
- $$最小化完工时间（回到终点为止）$$
$$\text{min}\quad T_{n+1}$$
## 约束
1. $$发车数量约束 + 出发/返回车辆相等约束$$
$$\sum_{j:(0,j)\in \mathcal{A}}x_{0j} = \sum_{i:(i,n+1)\in \mathcal{A}}x_{i,n+1} \leq K$$
2. $$任务访问约束$$
$$\sum_{j:(i,j)\in \mathcal{A}} x_{ij} = \sum_{j:(j,i)\in \mathcal{A}} x_{ji} = 1，\quad \forall i \in \mathcal{C}$$
3. $$换电站访问约束$$
$$\sum_{j:(i,j)\in \mathcal{A}} x_{ij} = \sum_{j:(j,i)\in \mathcal{A}} x_{ji} \leq 1，\quad \forall i \in \mathcal{S}$$
4. $$节点离开时间约束$$
$$T_j \geq T_i+t_{ij}+p_j-M(1-x_{ij}), \quad \forall (i,j) \in \mathcal{A},j \in \mathcal{C}\cup\mathcal{S}\cup \{n+1\}$$
5. $$未使用的换电站，离开时间设定为0$$
$$T_{j} \leq M\sum_{i:(i,j)\in \mathcal{A}}x_{ij}, \quad \forall j \in \mathcal{S}$$
6. $$电量合法性性约束$$
$$Q^{min} \leq E_i \leq Q, \quad\forall i \in \mathcal{C}$$
7. $$电量递推约束$$
$$E_i-e_{ij}-q_j-M(1-x_{ij}) \leq E_j \leq E_i-e_{ij}-q_j+M(1-x_{ij}), \quad \forall (i,j) \in \mathcal{A}, j \in \mathcal{C}$$
$$E_i-e_{ij} \geq Q_{min}-M(1-x_{ij}),\quad  \forall (i,j) \in \mathcal{A}, j \in \mathcal{S}\cup\{ n+1\}$$
$$E_{j}=Q\sum_{i:(i,j)\in \mathcal{A}}x_{ij},  \quad \forall j \in \mathcal{S}$$
8. $$换电站对称性消除$$
$$\sum_{i:(i,S^{r+1})\in \mathcal{A}}x_{iS^{r+1}} \leq \sum_{i:(i,S^{r})\in \mathcal{A}}x_{iS^{r}}, \quad \forall r=1,2\cdots,R-1$$
9. $$换电站排队约束$$
$$T_{S^{r+1}}\geq T_{S^{r}}+p_{S^{r+1}}-M(1-\sum_{i:(i,S^{r+1})\in \mathcal{A}}x_{iS^{r+1}}), \quad \forall r=1,2\cdots,R-1$$
10. $$起点初始化约束$$
$$T_0=0,E_0=Q$$
11. $$完工时间下界约束$$
$$K\cdot T_{n+1} \geq \sum_{i\in \mathcal{C}}p_i+\sum_{(i,j) \in \mathcal{A}}t_{ij}x_{ij}+\sum_{r=1}^{R}p_{S^r}\sum_{i:(i,S^r)\in \mathcal{A}}x_{iS^r}$$

---
# Path-based 模型
- arc-based模型中，换电站需要copy多少份是人为指定的参数，影响如下：
  - copy过少：限制了换电次数，可能无法求出最优
  - copy过多：网络节点数增长，求解时间明显增加
- 通常会取客户（任务）数量作为一个保守的值
- 为了避免人为指定参数，下面提出path-based模型
## 集合
- $$\mathcal{C}=\{ 1,2,\cdots,n  \} ：任务节点集合$$
- $$\mathcal{V}=\{ 0\} \cup \mathcal{C} \cup \{n+1 \} ：图节点集合$$
  - $$\mathcal{V}^-=\{0\} \cup C：路径起点集合$$
  - $$\mathcal{V}^+=C \cup \{n+1\}：路径终点集合$$
- $$对于i \in \mathcal{V}^-,j \in \mathcal{V}^+, i \neq j,定义两类path$$
  - $$普通路径：p^0_{ij}=(i,j)$$
  - $$换电路径：p^S_{ij}=(i,S,j)$$
  - $$P^S=\bigcup_{i,j}p^S_{ij},P^0=\bigcup_{i,j}p^0_{ij},P=P^0 \cup P^S$$
  - $$额外定义：P^S_{0j} = \emptyset ,\forall j \in \mathcal{V}^+  ; P^0_{0,n+1}= \emptyset$$
- $$H=\{ (i,k):i,k \in C,i<k  \}：换电排序任务对集合$$
## 参数
- $$行驶时间t_{ij}，换电路径t(p^S_{ij})=t_{iS}+t_{Sj}$$
- $$行驶能耗e_{ij}$$
- $$任务时间p_i，任务能耗q_i（起终点均为0）$$
- $$电池容量Q，最低允许电量Q_{min}$$
- $$换电时间\tau_S$$
- $$最大发车数量K$$
## 决策
- $$x_p\in \{0,1\}, p\in P ：是否选择路径p$$
  - $$o(p),d(p)：路径p的起点和终点$$
- $$T_i：任务i完成时间$$
- $$E_i：任务i完成后剩余电量$$
- $$y_i \in \{0,1\},i\in C：是否在任务i后换电$$
- $$s_i,i \in C：完成任务i后，如果前往换电，则表示换电开始的时间$$
- $$z_{ik}\in \{0,1\},(i,k) \in H：换电顺序变量$$
## 目标
- $$最小化完工时间（回到终点为止）$$
$$\text{min}\quad T_{n+1}$$
## 约束
1. 任务完成约束
$$\sum_{p\in P:o(p)=i}x_p=1,\quad \forall i \in C$$
2. 流平衡约束
$$\sum_{p \in P:d(p)=i}x_p = \sum_{p \in P:o(p)=i}x_p,\quad \forall i\in C$$
3. 发车数量约束
$$\sum_{p\in P:o(p)=0}x_p=\sum_{p\in P:d(p)=n+1}x_p \leq K$$
4. 换电事件定义
$$y_i=\sum_{p\in P^S,o(p)=i}x_p,\quad \forall i \in C$$
$$0 \leq s_i \leq My_i, \quad \forall i \in C$$
5. 时间传播约束
$$T_{d(p)}\geq T_{o(p)}+t_{o(p),d(p)}+p_{d(p)}-M(1-x_p),\quad \forall p \in P^0$$
$$s_{i}\geq T_{i}+t_{i,S}-M(1-y_i), \quad \forall i \in C$$
$$T_{d(p)} \geq s_{o(p)}+\tau_S+t_{S,d(p)}+p_{d(p)}-M(1-x_p), \quad \forall p \in P^S$$
6. 电量传播约束
$$E_{o(p)}-e_{o(p),d(p)}-q_{d(p)}-M(1-x_p) \leq E_{d(p)} \leq E_{o(p)}-e_{o(p),d(p)}-q_{d(p)}+M(1-x_p),\: \forall p \in P^0,d(p)\in C$$
$$E_{o(p)}-e_{o(p),d(p)}-q_{d(p)}\geq Q^{min} - M(1-x_p),\quad \forall p \in P^0,d(p)=n+1$$
$$E_{o(p)}-e_{o(p),S} \geq Q^{min} - M(1-x_p),\quad \forall p \in P^S$$
$$Q-e_{S,d(p)}-q_{d(p)}-M(1-x_p) \leq E_{d(p)} \leq Q-e_{S,d(p)}-q_{d(p)}+M(1-x_p),\quad \forall p \in P^S,d(p)\in C$$
$$Q-e_{S,d(p)}-q_{d(p)}\geq Q^{min}-M(1-x_p),\quad \forall p \in P^S,d(p)\in n+1$$
7. 电量上下限约束
$$Q^{min}\leq E_i \leq Q, \quad \forall i \in C$$
8. 起点初始化
$$T_0=0,E_0=Q$$
9. 单换电站排队模型
  1. 排序变量仅对发生的两个换电事件生效
$$z_{ik}\leq y_i, z_{ik}\leq y_k,\quad \forall(i,k) \in H$$
  2. 换电顺序时间约束
$$s_k\geq s_i + \tau_S - M(1-z_{ik})-M(2-y_i-y_k),\quad \forall (i,k)\in H$$
$$s_i\geq s_k + \tau_S - Mz_{ik}-M(2-y_i-y_k),\quad \forall (i,k)\in H$$
10. $$完工时间下界约束$$
$$K\cdot T_{n+1} \geq \sum_{i\in \mathcal{C}}p_i+\sum_{p \in P^0}t_{o(p),d(p)}x_{p}+\sum_{p\in P^S}(t_{o(p),S}+\tau_S+t_{S,d(p)})x_{p}$$
