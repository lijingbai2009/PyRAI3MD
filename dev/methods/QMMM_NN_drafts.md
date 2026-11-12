# PyRAI3MD NN-QM/MM 设计草案

**统一 total model：孤立分子势能分支 + 环境响应分支**

**日期：** 2026-10-16  
**更新：** 2026-10-30 — 明确零阶 ESPF 的 V/E 职责、完整梯度训练与 NN/LAMMPS 梯度计数约定。  
**状态：** 方法与增量接口设计，不代表当前已经实现。  
**基础文档：** [QMMM_DEVELOPMENT.md](QMMM_DEVELOPMENT.md)。  
**输入依据：** 用户提供的 ESP/电场条件化 NN 讨论，以及本次明确提出的“面向后续多种 NN 模型，不局限于 `nn_pes`”。

本文采用讨论最后确定的目标：**一个 total model 内部包含 `Eiso` 与 `Eenv`，同时支持孤立体系和环境体系的训练、预测及动力学。** 网络不必显式预测 QM 原子电荷；环境对 QM 能量的影响通过外部静电势及其空间导数进入模型。

框架只规定物理语义、输入输出与导数责任，不规定 TensorFlow、PyTorch、MLP、图网络或等变网络。下文出现的类名、字段和新模块路径均为建议，不是现有 API。

## 1. 与原 QM/MM 草案的关系

### 1.1 保留的软件原则

- 采用 additive QM/MM，每一种物理贡献只有一个 owner。
- `Calculator` 继续唯一管理运行状态、backend 生命周期、求值身份及结果提交。
- MM 原子坐标、周期盒、力场拓扑和环境电荷由经典层维护。
- QM/NN 只接收所需环境表示；统一 assembler 负责最终能量、梯度和性质组装。
- 固定分区、无跨界共价键、一个 QM 电子中心优先；NPT、复杂边界和自洽可极化 MM 后置。
- 能量与力必须来自同一个模型，不能以 NN 预测取代缺失的物理导数契约。

### 1.2 必须改变的物理归属

原草案最初的主路线（下式暂省略后来加入的 QM 周期镜像项）为：

$$
U_s^{\mathrm{old}}=E_{\mathrm{internal},s}[\Psi_{\mathrm{emb},s}]
+E_{\mathrm{MM-MM}}+E_{\mathrm{QM-MM,Coul}}^{\mathrm{proxy}}
+E_{\mathrm{QM-MM,LJ}}.
$$

它把完整 classical QM-MM Coulomb 留给 MM，因此需要 QM proxy charges。

本草案的推荐路线为：

$$
\boxed{
U_s=E_s^{\mathrm{iso,NN}}(Z_Q,R_Q)
+E_s^{\mathrm{env,NN}}(Z_Q,R_Q,\Phi_M)
+E_{\mathrm{MM-MM}}(R_M,h)
+E_{\mathrm{QM-MM,LJ}}(R_Q,R_M,h)
}
$$

其中环境分支学习**完整静电嵌入修正**：

$$
E_s^{\mathrm{env,ref}}=E_s^{\mathrm{emb,ref}}-E_s^{\mathrm{iso,ref}}.
$$

`Eemb` 包含 QM 在外场中的电子结构能量、QM 核与外场作用及自洽极化效应，但不含 MM-MM 和外加的 QM-MM LJ。若原始 QM 输出带有环境自能或其他常数，参考数据 adapter 必须先剥离并记录。

**新方案不再叠加 classical QM-MM Coulomb 能量，也不要求独立预测 QM proxy charges。** 但允许将能量对外势的响应电荷送回 LAMMPS，用辅助 Coulomb 求值实现同一能量的力回传；辅助 Coulomb 能量不加入总能量，详见 §5.4。 环境分支不是原草案狭义的 polarization-only 内部能修正。讨论中早期将两者说成“完全对上”的表述不应作为实现依据；后续讨论已经明确了这一区别。

| 物理项 | 原草案主路线 | 本草案推荐路线 |
|---|---|---|
| 孤立 QM 内部能 | QM 内部能的一部分 | `Eiso` 分支 |
| QM 对环境的电子响应 | polarization-only QM | `Eenv` 分支 |
| QM-MM 直接静电作用 | MM + QM proxy charges | `Eenv` 分支 |
| MM-MM 作用 | MM | MM |
| QM-MM LJ | MM | MM；后续可显式加入短程修正 |
| QM-QM 经典项 | 排除 | 排除 |

建议用独立语义标识 `electrostatic_embedding_total` 和 ownership 标识 `nn_embedding_mm_lj`，与原草案 `polarization_only` 并存。旧模型不能仅改标签迁移，新模型也不能复用旧 proxy-Coulomb assembler 配置。

## 2. 统一 total model 的定义

### 2.1 一个模型，两类职责

$$
E_s^{\mathrm{model}}(R_Q,\Phi_M)
=E_s^{\mathrm{iso}}(R_Q)+E_s^{\mathrm{env}}(R_Q,\Phi_M).
$$

这里 `model total` 指 NN 负责的 embedded QM 能量，不是包含 MM-MM/LJ 的全体系总能量 `U`。接口、训练文件与日志必须区分这两个“total”。

推荐的逻辑结构：

```text
TotalEmbeddedModel
├── geometry encoder                 # 可共享，也可独立
├── intrinsic branch → Eiso[S]
├── environment-response branch → Eenv[S]
└── energy composition → Eembedded[S]
    └── derivatives → coordinate / potential / field responses
```

一个 total model 是一个可部署、可版本化的模型契约，不要求只有一个神经网络或一个权重文件。已有真空模型可以作为内部子模型接入，但必须与响应分支一起导出完整元数据、状态映射和版本指纹。

框架将其视为一个 embedded-QM provider，不将 `Eiso` 和 `Eenv` 注册成两个普通 theory backend，以免误触发 ONIOM 或通用加权求和。

### 2.2 严格的零环境边界

对任意几何和态，要求：

$$
E_s^{\mathrm{env}}(R_Q,0)=0.
$$

建议以结构约束实现：

$$
E_s^{\mathrm{env}}(R_Q,\Phi)
=g_s(R_Q,\Phi)-g_s(R_Q,0).
$$

两次调用使用相同权重和几何。不能停止 `g(R,0)` 对几何或权重的求导；否则梯度或训练目标不再对应实际能量。使用 dropout 等随机层时，两项须共享一致随机实现或采用确定性的响应路径；独立随机掩码会破坏逐样本零场恒等式。

“环境层不设 bias”不是普适保证：共享几何特征、非零激活、归一化偏置或残差旁路仍可能产生零场输出。可选的响应基展开或显式场因子门控，必须同样证明恒等式成立。

在零环境下，固定环境特征的几何偏导也应为零；但通常

$$
\left.\frac{\partial E_s^{\mathrm{env}}}{\partial\Phi}\right|_{\Phi=0}\ne0,
$$

因为未受扰动分子仍可具有永久静电响应。**零环境能量不意味着零环境响应导数。** 更不能把“某一几何上特征恰为零”误当作“环境不存在”：前者的坐标 Jacobian 仍可能非零。

### 2.3 三种训练模式，一种推理形式

| 模式 | 主要样本 | 参数策略 | 目的 |
|---|---|---|---|
| `train_iso` | 真空 | 训练 intrinsic/geometry；响应分支待训练 | 建立真空基线 |
| `train_env` | 环境及配对真空 | 冻结基线，训练响应参数 | 扩展既有模型 |
| `train_joint` | 真空 + 环境 | 联合更新，保留真空锚点 | 统一优化 |

若响应分支与基线共享 encoder，`train_env` 必须冻结所有影响 `Eiso` 的共享参数，或采用独立 adapter；只冻结最后一个能量 head 并不能冻结真空势能面。

仅有环境能量标签时，两分支在已采样区域可能不可辨识；零场结构约束本身不能替代真空数据。推荐同一 QM 几何下的真空/环境配对样本，并在联合训练中持续保留真空损失。

## 3. ESP 条件化与 ESPF 的边界

### 3.1 本方案首先需要环境势，而非预测 QM 电荷

以正试探电荷的势约定，在原子单位下：

$$
V_i=\sum_{A\in M}\frac{q_A}{r_{iA}},\qquad
\mathbf{\mathcal E}_i=-\nabla_{R_i}V_i
=\sum_{A\in M}q_A\frac{R_i-R_A}{r_{iA}^3}.
$$

MM 的固定电荷仍然需要存在；剥离的是 **NN 显式学习 QM 原子电荷作为 Coulomb 中间变量** 的要求，而不是删除整个模拟的电荷数据。

通用扩展的环境描述符（首版零阶 ESPF 采用下面区分的 V-only 能量输入）：

$$
\Phi_M=\{V_i,\mathbf{\mathcal E}_i\}_{i\in Q}.
$$

**首版细化：** `ΔE_s=f_s(R_Q,V)`，仅将 `V` 作为独立环境特征。LAMMPS 仍返回电场 `ℰ=-∇V`，用于计算坐标全导数，但它不是这个 profile 的独立 NN 输入。这样能量对 `V` 的导数可直接提供 MM 静电力所需的响应电荷。将电场也独立输入 NN 的 `V+ℰ` 模型作为后续扩展，必须额外实现电场响应及其坐标回传。

若采用等变网络，`V` 是标量，电场是极向量；在常见 O(3) 表示中分别对应 `0e` 和 `1o`。其他架构可采用电场与几何方向的内积等不变量，但不能把实验室坐标系的 xyz 分量直接当普通标量而失去旋转一致性。

### 3.2 不把 ESP 采样统称为 ESPF 算法

- **ESP/field descriptor：** 在 QM 位点采样 MM 势、电场或更高阶导数。
- **ESPF embedding：** 构建 electrostatic-potential-fitting 电荷/多极算符及相关拟合过程，用于定义 QM 外场耦合。
- **PME/PPPM/Ewald：** 周期静电求和方法，可以为场提供者提供实现基础。

三者职责不同。直接求 MM 点电荷势并输入 NN，不自动构成 ESPF 算法。建议公共名称采用 `ElectrostaticFieldProvider` 和 `EnvironmentResponseModel`；若参考 QM 使用 ESPF，在数据中记录具体算符阶数、拟合网格、约束、正则化和导数约定。

如果参考 Hamiltonian 是原子电荷型 ESPF、其外场依赖确实由原子势值参数化，则 `V` 可形成与该近似 Hamiltonian 匹配的输入；如果参考使用连续点电荷势的一电子积分，仅有有限个位点的 `V` 和电场一般不是完备表示。后者需要描述符收敛测试，而不能声称无损压缩。

### 3.3 预留高阶场，不提前要求所有模型使用

可扩展输入包括势 Hessian `H_i = ∇∇V_i` 或不可约多极张量。它们描述空间非均匀性，适用于强场、近邻接触及高阶响应。

必须区分两个问题：

1. NN 是否把 `H` 作为输入特征？首版可以不使用。
2. 输入电场的坐标导数是否需要 `H`？即使 NN 只输入 `V` 和电场，计算 QM 全梯度时也需要这个导数或等价 VJP。

NN 输入规模可保持与 `NQ` 成正比，但全体系场计算和 MM 反作用力仍需处理环境，不能因此宣称整个模拟成本与 `NM` 无关。

## 4. 势零点与对称性

### 4.1 Gauge 协变应由模型显式保证

在固定总 QM 电荷 `Q_s` 和固定电子态分支下，采用包含电子与核耦合的势约定：

$$
E_s^{\mathrm{emb}}(V+C,\mathcal E)
=E_s^{\mathrm{emb}}(V,\mathcal E)+Q_s C.
$$

同一电荷扇区的态能差不受常数移位影响。均匀势不是电场，不能被解释成中性体系发生额外极化的原因。

取固定权重、和为 1 的平均势，首版可用简单原子平均：

$$
\bar V=\frac1{N_Q}\sum_i V_i,\qquad \widetilde V_i=V_i-\bar V.
$$

推荐响应形式：

$$
E_s^{\mathrm{env}}
=Q_s\bar V+g_s(R_Q,\widetilde V,\mathcal E)
-g_s(R_Q,0,0).
$$

这同时保证零环境约束与 gauge 协变。训练时的特征标准化必须保留此结构；物理零场经过标准化后未必是数值全零，应先构造物理零场，再经过同一预处理。

### 4.2 对称性与响应解释

模型应满足能量的整体平移、旋转和约定范围内同种原子置换不变性，梯度及电场响应满足对应协变性。固定索引的旧分子 MLP 可以作为有限适用域 adapter，但不能宣称天然具备元素/分子大小迁移能力。

设

$$
a_{s,i}=\frac{\partial E_s^{\mathrm{model}}}{\partial V_i},\qquad
\mathbf b_{s,i}=\frac{\partial E_s^{\mathrm{model}}}{\partial\mathcal E_i}.
$$

上述 gauge 约束给出 `Σ_i a_s,i = Q_s`。这些量是模型对外势的共轭响应，可用于解释和回传力；它们不是 Mulliken/RESP 标签，也不应写入原有 `computed_charges` 更新链路，不能据此再次加入 QM-MM Coulomb 能量。它们可以通过 §5.4 的专用响应通道用于 LAMMPS 交叉静电力求值；这与更新一套独立的物理电荷模型不同。

只有对于明确的外场参数，能量导数才具有对应可观测响应的意义。例如对均匀外电场求导可定义总偶极响应；任意内部 descriptor 的偏导不能直接等同于唯一的物理原子电荷或原子偶极。

## 5. QM 与 MM 力：统一链式求导

### 5.1 同一个能量函数产生全部核梯度

将模型暂视作独立变量函数 `E(RQ,V,ℰ)`。固定外场输入的显式几何偏导记为 `g_exp`，则任意原子坐标分量 `x` 的导数为：

$$
\frac{dE_s^{\mathrm{model}}}{dx}
=\left.\frac{\partial E_s^{\mathrm{model}}}{\partial x}\right|_{V,\mathcal E}
+\sum_i a_{s,i}\frac{\partial V_i}{\partial x}
+\sum_{i\alpha}b_{s,i\alpha}\frac{\partial\mathcal E_{i\alpha}}{\partial x}.
$$

- `x∈Q`：显式项包含 `Eiso` 与 `Eenv` 对几何的偏导，还要加场采样点移动的贡献。
- `x∈M`：首版无 MM 几何直连 NN，显式项为零，但场链式项通常非零。
- 即使 MM 原子固定，只要 QM 原子移动，采样势和电场也会变；不能只用固定特征的几何偏导当 QM gradient。

全体系最终结果：

$$
G_{s,Q}=g_{s,Q}^{\mathrm{exp}}+G_{s,Q}^{\mathrm{field}}+G_Q^{\mathrm{LJ}},
\qquad
G_{s,M}=G_{s,M}^{\mathrm{field}}+G_M^{\mathrm{MM-MM}}+G_M^{\mathrm{LJ}}.
$$

输出为正梯度 `G=dU/dR`，动力学力为 `F=-G`。通过标量能量导数构造力保证保守形式，但有限时间步、数值精度、近邻不连续和模型误差仍影响实际 NVE 能量漂移；不能承诺数值模拟严格零漂移。

### 5.2 推荐 VJP，避免保存巨型 Jacobian

公共场接口建议提供向量-Jacobian 乘积：

```python
field = provider.evaluate(qm_coordinates, mm_snapshot, request)
response = model.evaluate(qm_input, field.features, target_states)
field_grad = provider.pullback(
    field.handle,
    denergy_dpotential=response.denergy_dpotential,
    denergy_dfield=response.denergy_dfield,
)
```

`pullback` 返回各态的 `gradient_qm`、`gradient_mm`，后续可返回 cell derivative。实现可为自动微分、解析导数或引擎提供的 adjoint；框架不依赖某一种 ML 张量库。

不要求长期保存 `[S,NQ,NM,3,3]` 等稠密矩阵。小体系 dense Jacobian 只用作 oracle；生产实现可重算、分块或调用周期求解器的伴随流程。

`handle` 只在拥有场计算资源的进程中有效；跨进程必须传递快照身份和可重建描述符，或通过有版本的服务调用返回 VJP，不能序列化 autograd graph、CUDA 指针或闭包。

### 5.3 两种求导责任只能选一种

| derivative contract | NN 返回 | 框架行为 |
|---|---|---|
| `partial_plus_field_vjp`，推荐 | 固定场几何偏导 + `a,b` | 场提供者回传一次，assembler 组合 |
| `full_coordinate_gradient` | 已包含全部场链式项的 Q/M 梯度 | 不再次调用同一场 VJP |

模型内部统一保留第一种导数原语；对接当前 Molcas/LAMMPS 组装流程时，推荐由 NN adapter 生成完整 QM 嵌入梯度，见 §5.7。第二种契约只有在确实返回完整 Q/M 梯度时才能声明；仅返回完整 QM 梯度的 adapter 仍需 MM 静电回传。禁止仅凭字段名 `gradient` 猜测它是否已包含环境链式贡献。

### 5.4 首版：用能量导数电荷替代 Molcas 的电荷输出

针对当前 Molcas 零阶 ESPF 流程，采用与其一致的原子势耦合：

$$
\hat H(R_Q,V)=\hat H_0(R_Q)+\sum_i\hat Q_i(R_Q)V_i.
$$

对于驻定电子态，或使用与该能量对应的完整 relaxed 导数定义，在固定 QM 几何、固定其余势分量时：

$$
q_{i,s}^{\mathrm{ESPF}}=\frac{\partial E_s^{\mathrm{embedded}}}{\partial V_i}.
$$

该关系是方法能力条件，不是任意电子结构输出电荷的通用性质。必须核验 Molcas 所选 method/state 输出的零阶 multipole 是否为相应能量导数，包括核电荷、符号、响应密度和拟合约束。非变分相关方法或特殊激发态方法不能只凭“ESPF charge”名称就认定一致。零阶 ESPF 的解析导数还涉及几何相关算符与拟合网格导数，参考实现应保持一致。[ESPF 解析能量、梯度与 Hessian](https://arxiv.org/abs/2010.06542)

由于孤立分子能量与外势无关：

$$
\boxed{
q_{i,s}^{\mathrm{response}}
=\frac{\partial\Delta E_s^{\mathrm{NN}}(R_Q,V)}{\partial V_i}
=\frac{\partial E_s^{\mathrm{embedded,NN}}}{\partial V_i}.
}
$$

因此孤立网络不需要电荷 head，环境网络也不必另设电荷 head；部署时输出标量能量及其导数即可。`q_response` 不是独立拟合后再拼接的电荷函数，而是该能量模型的共轭响应。

对于 MM 原子 A：

$$
G_{A,s}^{\mathrm{embedding}}
=\sum_i q_{i,s}^{\mathrm{response}}\frac{\partial V_i}{\partial R_A}.
$$

这正是同一 Coulomb kernel 下，固定 QM 响应电荷和固定 MM 电荷产生的交叉静电梯度。因而可复用 LAMMPS 的点电荷求力能力，作为通用 field VJP 的一种实现：

```text
LAMMPS / field provider：同一快照下得到 MM 源的 V、ℰ
    ↓
TotalEmbeddedModel：Eiso(R) + ΔE(R,V)
    ├─ ΔE、固定 V 的显式几何偏导
    └─ q_response = ∂ΔE/∂V
    ↓
LAMMPS：更新本次响应电荷，求 QM-MM 交叉静电力
        同时输出所配置的 MM-MM、QM-MM LJ 等经典贡献
    ↓
Assembler：NN 能量 + MM-MM/LJ 能量；完整梯度只组装一次
```

这保留用户现有“场 → QM 结果/电荷 → 经典力”的依赖顺序，将 Molcas 替换为 NN。经典贡献仍由统一配置的 LAMMPS 任务求值，不要求按分项分别启动引擎；但不能宣称响应电荷尚未产生前的一次旧力计算已经包含新的 NN 反作用力。

**辅助力求值与物理能量必须分开：**

$$
\Delta E_s^{\mathrm{NN}}\ne\sum_i q_{i,s}^{\mathrm{response}}V_i
\quad\text{（一般的非线性响应）}.
$$

LAMMPS 的 `q_response·V` 仅用于实现梯度回传，其能量不得替换或再次叠加到 NN 的 `ΔE`。这次辅助求力把 `q_response` 当作固定输入，不追加 `dq_response/dR` 项；链式法则已经是对 NN 原始能量的一阶导数，不是在求另一个 `q(R)V(R)` 势能的全导数。训练时仍需对响应电荷保留参数导数，不能与推理时的辅助固定电荷求力混淆。

该复用必须满足：场采样和辅助交叉力使用一致的边界、核函数、排除、势规范及单位。周期网格算法还需数值验证二者的离散导数一致性；不能仅因两边都称为 PPPM 就自动保证等价。

MM 原子力可以由交叉静电、MM-MM 和 LJ 合成；QM 原子力必须排除不属于目标能量的 QQ 内部或镜像辅助作用。初版暂忽略 QQ 周期修正时，不可直接拿仍含这些项的 QM 总力作为 NN 的 field pullback。已有 topology/exclusion 或已验证的交叉力输出负责这一筛选。

### 5.5 QM 梯度修正及 V+电场扩展的限制

对于固定 MM 电荷、MM-only 外势，并在对应 QM 位点采样：

$$
\Delta G_{i,s}
=\left.\frac{\partial\Delta E_s}{\partial R_i}\right|_V
+\sum_jq_{j,s}^{\mathrm{response}}\frac{\partial V_j}{\partial R_i}
=\left.\frac{\partial\Delta E_s}{\partial R_i}\right|_V
-q_{i,s}^{\mathrm{response}}\mathcal E_i^M.
$$

这里的固定场显式几何偏导包含模型学到的几何相关 ESPF 算符效应；末项来自采样点移动。参考梯度差 `Gembedded−Giso` 应与上述完整表达式比较，不能只训练固定 V 偏导。若标签使用力差 `Fembedded−Fiso`，整体取负号。

部署时可选“显式几何偏导 + LAMMPS 交叉力”或“已经包含链式项的完整 NN 梯度”，不能两者重复相加。原始 `V` 的响应须包括势中心化和 scaler 的导数；不能直接把对中心化势的偏导当作 `q_response`，否则可能丢失总电荷项。

若改成独立依赖 `f(R,V,ℰ)`，则还需：

$$
\sum_{i\alpha}\frac{\partial f}{\partial\mathcal E_{i\alpha}}
\frac{\partial\mathcal E_{i\alpha}}{\partial R_A}.
$$

只送 `∂f/∂V` 给 LAMMPS 会遗漏这一项。此时须采用完整 field VJP，或经验证的等价高阶响应实现；电荷 loss 也只能约束 V 通道，不能替代电场通道及全体系力验证。

### 5.6 零阶 ESPF 中 V 与电场的不同职责

本节限定参考耦合为 `MultipoleOrder = 0`，而非仅要求输出中存在零阶电荷。后者也可能来自包含高阶算符的计算。

在零阶耦合 `H = H0 + Σ qhat_i(R) V_i` 下，固定几何和全部原子势值时，电场不是额外独立的电子结构耦合参数。收敛电子结构、能量及零阶 ESPF 电荷由几何和 V 决定；电场作为电势的空间导数进入核梯度。真实外场变化通常也会改变各 QM 位点的 V，因此不能将这个结论理解为“物理电场不能极化分子”。均匀场的效应已可通过 `V_i = −ℰ·R_i + C` 进入零阶模型。

在驻定电子态或正确的 relaxed 导数处理中，耦合的显式几何导数包含：

$$
G_A^{\mathrm{int}}
=\sum_j V_j\left\langle\frac{\partial\hat q_j}{\partial R_A}\right\rangle
+\sum_j q_j\frac{\partial V_j}{\partial R_A}.
$$

第一项包括 ESPF 算符及拟合网格等几何依赖；第二项在固定 MM 源、对应 QM 位点采样的情形下给出 `G_i^field = −q_i ℰ_i`。这是数学分解，不要求 Molcas 内部按两次独立修正执行。电子结构方法所需的其他解析梯度/响应项仍须保留，不能用单独的 `−qℰ` 替代完整梯度。[OpenMolcas ESPF 文档](https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/espf.html)

因此首版应区分两个接口：

- **能量网络输入：** `R,V`；输出能量、固定 V 的几何偏导和 `q_response=∂ΔE/∂V`。
- **训练与部署的梯度组装输入：** 还需要电场 ℰ，由 LAMMPS 提供；通过明确公式得到完整 QM 梯度，不要求 NN 隐式学习电场链式项。

完整梯度修正损失应使用：

$$
L_{\Delta G_Q}
=\left\|
\left.\frac{\partial\Delta E_\theta}{\partial R_Q}\right|_V
-q^{\mathrm{response}}\mathcal E^M
-\left(G_Q^{\mathrm{emb,ref}}-G_Q^{\mathrm{iso,ref}}\right)
\right\|^2.
$$

式中的电荷与电场按原子、电子态逐项相乘。固定 V 的几何偏导不能直接拟合完整 Molcas 梯度差，否则会把已知的电场贡献错误地吸收到显式几何响应中。若在端到端计算图中已通过 `V(R)` 自动求全导数，则不再手动添加同一项。

V+ℰ 输入仍然可以作为独立 profile，但自由依赖 ℰ 会产生 §5.5 的额外电场响应，需要电场坐标导数或等价 VJP。对于严格零阶 ESPF 参考，固定 R,V 时不存在独立电场能量依赖；增加电场特征不能自动证明精度提高，需验证导数电荷与参考 ESPF 电荷的兼容性。若参考改为含偶极算符的高阶耦合，则电场可直接影响电子结构和零阶电荷，须同步升级模型与力接口。

### 5.7 兼容 Molcas 的完整梯度与 LAMMPS 重复项扣除

按照当前用户描述的工程流程，Molcas 返回已经包含电势及电场作用的 QM 能量和完整 QM 梯度；LAMMPS 计算经典贡献时，从 QM 原子结果中扣除已经由 Molcas 包含的交叉静电梯度。迁移到 NN 时保留这一流程，同时显式声明输出语义，而不在两侧设置互不约束的布尔开关。

建议新增如下语义字段（接口提案，不代表现有配置已支持）：

```ini
qm_gradient_convention = full_embedding
# 可选：fixed_external_potential
```

定义 `Gfield_Q = −q_response ℰM`，并将 LAMMPS 已筛选为目标 ownership 的梯度记为 `Gallowed_LAMMPS`，其中包含 QM-MM 交叉静电、LJ 和 MM-MM，不含未声明的 QQ 项：

| 约定 | QM provider 返回的 QM 梯度 | Assembler 对 LAMMPS 的处理 |
|---|---|---|
| `full_embedding`，当前流程默认 | `Giso + g_exp_env + Gfield_Q` | 仅从 QM 原子块扣除重复的 `Gfield_Q` |
| `fixed_external_potential` | `Giso + g_exp_env` | 保留 QM 原子块中的交叉静电梯度 |

完整嵌入模式的组装为：

$$
G_Q^{\mathrm{total}}
=G_Q^{\mathrm{provider,full}}
+G_{Q}^{\mathrm{allowed,LAMMPS}}-G_Q^{\mathrm{field}},
\qquad
G_M^{\mathrm{total}}=G_M^{\mathrm{allowed,LAMMPS}}.
$$

固定外势模式则为：

$$
G_Q^{\mathrm{total}}
=G_Q^{\mathrm{provider,partial}}
+G_Q^{\mathrm{allowed,LAMMPS}},
\qquad
G_M^{\mathrm{total}}=G_M^{\mathrm{allowed,LAMMPS}}.
$$

NN adapter 默认使用 MM 提供的电场和 NN 自身导数电荷生成完整 QM 梯度，与 Molcas 的外部契约一致；内部仍保留 `gradient_explicit` 和 `denergy_dpotential` 原语。`qm_gradient_convention` 只描述 QM 输出，不能替代 §5.3 对全体系 Q/M 导数责任的声明。

实现约束：

1. 扣除的是正梯度 `−qℰ`；若引擎返回力，先按 `G=−F` 转换。用带符号的梯度分量做扣除，不能笼统写成“减去 qE”而混淆符号。
2. 两种模式都必须保留 MM 原子上的静电反作用力。完整 QM 梯度不意味着已经返回 MM 梯度；若 provider 同时提供完整 MM 梯度，应走另行声明的全坐标契约。
3. NN、场和辅助求力使用同一态、同一快照的响应电荷和一致的静电核、边界、排除与单位。周期离散求解器中，`−qℰ` 与交叉力需经过一致性验证。
4. QM-MM、不同 QM 中心间以及自身周期镜像项分别声明来源和 owner；不得将全部静电力作为一个重复项整体扣除。上式首版只覆盖 MM 来源交叉项，多中心/镜像扩展须补充其能量扣重和对应梯度规则。
5. 能量 ownership 不随梯度约定切换：总能量仍使用 `Eiso+ΔE+MM-MM+LJ`，辅助 `qV` 能量不加入。
6. assembler 根据一个经过能力校验的约定决定扣除行为；缺失或冲突的约定应报错，不能由输出字段名推断。

验收时对同一模型、快照和电荷分别运行两种约定，要求最终能量及 Q/M 梯度一致，并与重算场后的能量有限差分比较。设置“完整梯度但不扣重复项”和“固定外势梯度却扣重复项”两个负对照，以捕获重复计入和漏项。

## 6. 短程非共价作用的扩展

势/电场主要编码静电环境，不足以一般性地辨识 dispersion、Pauli 排斥及其他短程量子效应。两个电荷分布近似相同的环境，可能具有不同的短程作用。

### 6.1 首版：MM 保留 LJ

`Eenv` 仅学习 electrostatic embedding correction；QM-MM LJ 保留在 MM。训练参考必须与此匹配，不能要求一个只见电场的模型吸收未知的全部 vdW。

### 6.2 后续：在 total model 内增加短程响应子通道

$$
E_s^{\mathrm{env}}=E_s^{\mathrm{elec}}(R_Q,\Phi_M)
+\Delta E_s^{\mathrm{sr}}(R_Q,\mathcal N_M^{\mathrm{local}}).
$$

局域输入提供近邻 MM 元素/类型、相对坐标和明确的 cutoff。第一阶段推荐学习相对 classical LJ 基线的残差，保留远程尾项及平滑衔接：

$$
U_s=E_s^{\mathrm{iso}}+E_s^{\mathrm{elec}}
+E_{\mathrm{MM-MM}}+E_{\mathrm{QM-MM,LJ}}+\Delta E_s^{\mathrm{sr}}.
$$

若未来完全替代 LJ，应建立另一 ownership 模式，关闭被替代的全部项并处理尾修正，不能仅添加一个网络。

短程标签需要包含相应物理信息的高层 cluster/interaction 参考及一致能量分解。普通点电荷 embedded QM 标签不含环境电子的显式交换/色散，不能凭这些标签训练出缺失作用。避免让短程网络重复学习已由 electrostatic 分支解释的氢键静电部分。

扩展后“零环境”是 `Φ=0` 且局域环境为空；只有零电荷而仍存在 LJ 原子，不要求总 `Eenv` 为零。共价边界仍需独立设计，不会因加入局域图而自动解决。

## 7. 周期环境与经典 backend

### 7.1 能量 ownership 与辅助电荷求力的区别

物理能量中的经典贡献为 MM-MM 与 QM-MM LJ，不再次加入 QM-MM Coulomb。通用场 VJP 实现可在物理经典任务中将 QM proxy 电荷置零；§5.4 的首版 adapter 则允许将 NN 导出的 `q_response` 注入辅助交叉静电求力。两种实现遵循相同物理能量，不能把“经典总能量不含交叉 Coulomb”误写成“经典引擎不允许计算交叉 Coulomb 力”。

采用辅助电荷路径时，必须从结果中分清 MM-MM、LJ、QM-MM 静电回传以及 QQ 项；不能直接将辅助任务总能量加到 NN 能量上。保留同一个常驻 LAMMPS 实例和统一拓扑，按求值阶段更新数据并取得所需输出。

### 7.2 周期场一致性

场 provider 必须固定并记录：cell/PBC、Ewald 边界、势零点、背景项、网格与插值、截断、平滑和求解精度。参考 QM、训练数据与在线推理使用同一物理定义；不能用有限 replicas 标签直接宣称训练出精确周期 embedding。

QM 分子使用连续、完整几何；周期映射不能撕裂分子。场在各 QM 位点求值，NN 内部可仍为分子模型，不能简单把所有 NN skill 的 `supports_pbc` 改为 true。

推荐首版中性、固定盒体系。带电 MM 分区和带电 QM/总体系需要单独规定背景、能量常数及 gauge；不能因为 MM-MM 引擎能计算非中性 PME，就假定 NN 与参考之间已经一致。

基础 QM/MM 草案在 2026-10-17 已更新为保留点电荷 QM 镜像静电。当前 NN 首版按本次讨论暂忽略这一修正，这是一项明确的参考模型差异，不是基础草案继续排除所有镜像作用。训练能量和梯度必须同时剥离对应修正，或由一套本就不含该项的参考计算产生；不能只改部署配置。

以后可以把镜像贡献纳入 NN 修正，但必须提供其所依赖的 cell/镜像场等信息，仅有 MM-only V 未必足够。若显式加入 `Eimage(q_response(R,V),R,h)`，其完整导数通常包含响应电荷变化项；若采用自洽驻定定义，则需对应的能量泛函和导数。不能将 §5.4 对辅助交叉力的固定电荷处理直接套用到新增物理镜像能量。本路线没有因此成为周期电子结构模型。

NPT/变胞优化需要场、网格、映像、NN 以及 LJ/MM 的全部 cell derivative；只得到 MM stress 时拒绝开放。未来压力标签不能从 MM stress 复制到全体系结果。

## 8. 模型无关的数据与接口契约

### 8.1 分层职责

```text
R_Q, R_M, h, fixed q_M
          │
          ▼
ElectrostaticFieldProvider ──► V, field, optional tensors
          │                              │
          │                       TotalEmbeddedModel
          │                       Eiso + Eenv → Eembedded
          │                              │
          ◄───────── field sensitivities a,b
          │
          ▼
Q/M field gradients + explicit QM gradient
          │
          ├── MM-MM + QM-MM LJ
          ▼
QMMMAssembler → standard Result → MD / optimize / single
```

### 8.2 输入契约

| 记录 | 最小内容 |
|---|---|
| `QMModelInput` | `Z[NQ]`、`coordinates[NQ,3]`、总电荷/自旋扇区、state map |
| `ElectrostaticFeatures` | `potential[NQ]`、`electric_field[NQ,3]`、可选张量及 mask |
| `FieldContext` | `evaluation_id`、partition/site mapping、field convention、单位、gauge、fingerprint |
| `MMModelSnapshot` | 完整坐标/原子映射、cell/PBC、电荷、拓扑/参数指纹 |

缺失环境不能自动当作真空：必须显式声明 `environment_mode=vacuum`，或提供完整环境特征。环境训练模型遇到缺场输入时不得静默退化为 `Eiso`。

框架坐标沿用 Angstrom；建议 field provider 和 NN adapter 约定 `V` 为 Hartree/e、电场为 Hartree/(e·Bohr)，内部求导明确所用坐标单位，最终梯度统一为 Hartree/Bohr。所有 scaler 的反变换及导数因子须纳入计算图或解析转换；不能将现有 NN 几何梯度换算再重复用于已为原子单位的 field VJP。

### 8.3 模型输出契约

```text
energy_semantics: electrostatic_embedding_total
ownership_id: nn_embedding_mm_lj
derivative_contract: partial_plus_field_vjp
qm_gradient_convention: full_embedding  # adapter 后的 QM 梯度语义，见 §5.7
gradient_qm[S,NQ,3]                 # 按上述约定输出，不能再与 explicit 相加
energy[S]                         # Eiso + Eenv，外部唯一能量入口
components.energy_iso[S]           # 诊断，不再重复组装
components.energy_env[S]
gradient_explicit[S,NQ,3]
denergy_dpotential[S,NQ]
response_charge[S,NQ]              # 与 denergy_dpotential 同一数据的语义别名
                                  # 不是另一套独立预测结果
denergy_dfield[S,NQ,3]             # 仅 V+电场扩展要求；V-only profile 不产生该通道
state_ids, evaluation_id, field_fingerprint, model_fingerprint
optional: property-specific uncertainty, electronic properties
```

最终 `Result.gradient[S,N,3]` 才是包含场与经典贡献的全体系梯度。诊断 components 不能被通用 assembler 当作独立任务再加一次。

### 8.4 模型能力声明

每个 backend/method/profile 分别声明：

```yaml
# 接口提案，当前配置尚不能直接运行
embedding_model:
  input_features: [potential]  # 首版；electric_field 留在求导通道
  energy_semantics: electrostatic_embedding_total
  derivative_contract: partial_plus_field_vjp
  qm_gradient_convention: full_embedding  # adapter 组装；可认证另一种约定
  zero_environment: exact
  potential_gauge: centered_plus_total_charge
  training_modes: [iso, env, joint]
  supports_field_sensitivities: true
  response_charge: energy_derivative
  supports_espf_charge_derivative_loss: true
  pullback_adapter: lammps_cross_coulomb  # 与通用 VJP 二选一实现
  supports_multistate_energy: true
  supports_environment_nac: false
  supports_cell_derivative: false
```

能力声明是认证结果，不是用户打开开关就能获得的能力。轻量的 schema/query/discovery 不导入重型 ML runtime；具体 engine 在 backend 初始化时加载。

新等变网络、图网络、通用势和现有 MLP 都实现同一契约；可以有不同的元素/大小/电荷/自旋适用域，不要求共享私有模型容器、权重格式或训练器。

## 9. 训练数据与损失函数

### 9.1 版本化数据集

新数据 schema 至少记录：

- 几何、原子类型与顺序、state/spin/pair maps、方法/基组和能量零点。
- `Eiso_ref`、`Eembedded_ref`、可选配对 `Eenv_ref`；用标签 mask 区分不存在的量，不能用零占位。
- 参考 QM 总梯度、参考 MM embedding 梯度及其语义；经典 LJ/MM-MM 梯度独立保存或可重建。
- `V/ℰ`、可选高阶场、field provider 版本、gauge 和单位；标记哪些是 NN 输入、哪些只用于求导。
- `espf_charge[nframe,S,NQ]`、标签 mask、态映射、总电荷、单位，以及 method/state 对应的电荷导数语义。
- `geometry_id`、`environment_id`、`trajectory_id`、partition/force-field fingerprints。
- 用于 field chain rule 的 Jacobian、可执行 VJP 描述，或者可重建它们的 MM 快照。

不能把 embedded QM gradient 标成固定场偏导；前者通常已经包含场随 QM 位置变化的贡献。

为了节省数据，环境特征可缓存，但“只保存 V/电场就足够训练所有力”不成立：QM 力训练需要采样点 Jacobian，MM 力训练需要对环境坐标的导数。可以分层保存小型训练张量与压缩原始快照，但应保留可追溯的原始 reference/环境数据。

### 9.2 参考标签构造

同一几何、方法和态身份下生成真空/嵌入标签；不能仅按能量排序对应两个计算中的态，因为环境可造成态序变化。

从完整 QM/MM 标签提取 NN 目标时：

$$
E^{\mathrm{emb,ref}}=U^{\mathrm{ref}}-E_{\mathrm{MM-MM}}-E_{\mathrm{QM-MM,LJ}},
$$

梯度同样减去一致的经典项。若原始数据采用原草案 polarization-only + proxy Coulomb，则不能直接按本式解释；必须重新构造满足 `electrostatic_embedding_total` 的参考。

ESPF 算符标签还要保留几何依赖的拟合算符导数。可控场扰动产生的响应标签，必须按同一 Hamiltonian 参数求导；不能假设各原子势/电场总能彼此独立地由真实电荷环境产生。

### 9.3 推荐损失

$$
\mathcal L=w_{\mathrm{iso}}L_{E^{\mathrm{iso}}}
+w_{\mathrm{emb}}L_{E^{\mathrm{emb}}}
+w_{\Delta}L_{\Delta E}
+w_Q L_{G_Q^{\mathrm{full}}}
+w_M L_{G_M^{\mathrm{embedding}}}
+w_q L_{q^{\mathrm{ESPF}}}
+w_{\mathrm{resp}}L_{\mathrm{response,other}}.
$$

各项按可用标签 mask 启用，按数量和量纲归一化；避免大量 MM 原子的近零力淹没近邻重要分量。配对差能与总能项是相关监督，不应未经调权重复放大同一信息。

QM/MM 力损失使用第 5 节链式计算得到的全导数，不能只训练 `gradient_explicit`。若没有 MM 力或场响应标签，模型的环境反作用力可能缺乏约束；应通过独立 MM-force 验证补足证据，不能只凭 QM RMSE 宣布可做自由环境 MD。

训练梯度损失需要模型参数与坐标/场之间的混合二阶导数。推荐平滑、可微的网络及 cutoff；后端需验证训练与部署模式下导数一致。

数据集按轨迹/几何族/环境分组拆分，同一几何的真空与不同环境配对放入同一数据划分，防止泄漏。额外留出溶剂、场强、场梯度和激发态区域作为外推测试；零环境边界不保证跨环境泛化。

### 9.4 用 Molcas 零阶 ESPF multipole 监督能量导数

**可以直接将 `q_response=∂ΔE/∂V` 与符合 §5.4 导数约定的 ESPF 0-order multipole 计算 loss。** 这约束能量对环境势的斜率，有助于环境响应和 MM 力精度，但不是精度一定改善的保证，需独立验证。

带有效标签 mask 的损失为：

$$
\mathcal L_q=\frac{1}{N_{\mathrm{valid}}}
\sum_{n,s,i}m_{nsi}
\left(
\frac{\partial\Delta E_s^{\mathrm{NN}}(R_n,V_n)}{\partial V_{ni}}
-q_{nsi}^{\mathrm{ESPF,ref}}
\right)^2.
$$

没有有效标签时跳过该项，不作零除，也不把缺失值填成中性零电荷。首版可采用 `LΔE + LΔGQ + Lq` 的加权目标，同时保留必要的真空训练/锚点。§9.3 的其他损失按实际数据启用，避免与此电荷响应重复计权。

标签与实现要求：

1. **完整电荷，不是极化电荷差。** 因为 `Eiso` 与 V 无关，`∂(Eembedded−Eiso)/∂V` 对应完整嵌入态的 ESPF 响应电荷，不能以 `qembedded−qvacuum` 监督。
2. **逐态、逐方法、逐原子映射。** 激发态响应不能统一匹配基态电荷；检查输出是含核贡献的净电荷，还是电子布居/其他 multipole 约定。对于非驻定能量方法，先认证相应 relaxed response。
3. **先验证参考恒等式。** 在可控势参数扰动下，固定 QM 几何并重收敛相同电子态，用能量差分验证 `∂Eembedded/∂V` 与导出 multipole 一致，再把它作为监督依据。只有可实现的联合扰动时，应比较相应方向导数而非虚构独立参数标签。
4. **单位和标准化。** 如果 `ΔE=σ_E y`、`V=μ_V+σ_V x`，则物理响应为 `(σ_E/σ_V)∂y/∂x`，再纳入势中心化等预处理的完整 Jacobian。以 e 为电荷单位、Hartree/e 为势单位核对结果。
5. **电荷守恒与 gauge。** 要求 `Σ_i q_response,i=Q_s`，优先使用 §4 的结构保证。检查参考 ESPF 标签是否守恒；若不一致，应先查明参考方法/拟合定义，不可悄悄修改标签或强行把不一致目标一起拟合。
6. **保持可微训练图。** `Lq` 的参数梯度包含 `∂²ΔE/(∂θ∂V)`，需要保留混合导数图，不能 detach、转换成不可微数组再计算 loss。推理时发送给 LAMMPS 的序列化电荷不受此训练限制。
7. **零场能量与零场响应不同。** `ΔE(R,0)=0` 不要求 `q_response(R,0)=0`。零场原子电荷刻画永久电荷分布；若有一致的真空 ESPF 响应标签，也可用于零场导数监督。

独立电荷 head 可以作为另一个研究基线，但其输出若不等于能量的势导数，不能直接替代能量一致的 MM 力源。当前推荐不新增该 head，而是用参考电荷监督已有标量能量的导数。

评估同时报告嵌入能量、QM 梯度、ESPF 电荷、总电荷偏差和实际 MM 力误差，并比较有/无 `Lq` 的消融实验。控制参考方法与数据划分不变，按验证集调整 loss 权重。低电荷 RMSE 不保证近距离 MM 力准确，因此仍需针对环境接触和强场样本检查全体系力。

## 10. 多态、NAMD 与主动学习

一个模型可输出 `Eiso[S]` 和 `Eenv[S]`；各态的响应 `a,b` 和 MM 梯度分别计算。固定 MM-MM/LJ 对各态广播，态能隙变化由嵌入模型产生。

不能由多个标量态能及其梯度自动得到 NAC/SOC。后续可采用独立且场条件化的耦合模型，或有依据的 diabatic Hamiltonian 模型；两者都需相位、态跟踪和 pair map 契约。

嵌入波函数依赖 MM 坐标，NAC 可有环境分量。旧真空 NAC head 在 QM 外补零只可作为标记清楚的近似。使用能隙/曲率型跳跃方法也应保证实际消费同一组最终嵌入态能，不能混用 vacuum gap。

建议 ensemble 的每个成员都是完整的 `Eiso + Eenv` 模型。逐成员完成 field pullback 和全体系组装后再统计分歧，以保留分支间相关性；不能简单相加两分支的标准差。

主动学习同时监测嵌入能量、态能隙、QM/MM 梯度和场响应分歧，并结合场强/描述符适用域。ensemble 低分歧不是可靠性的证明。参考回算必须保留同一环境快照，产生 embedded-QM 标签而非仅真空 QM 标签。

## 11. 对当前代码的增量落点

### 11.1 2026-10-16 的现状与原文档路径差异

本次检查确认：

- 已有 `pyrai3md/skills/nn_pes_skill/`，支持训练、预测、能量/梯度与部分耦合 profile。
- 当前 `NNPESBackend.calculate()` 对非空 `point_charges` 明确报错；manifest 声明 `supports_pbc: false`。
- 当前 `service/trainer_data.py` 读取几何及 energy/grad/nac/soc 标签，没有本草案场与环境导数 schema。
- 现有 `mlp_eg` 使用几何特征和 TensorFlow/Keras 自动微分；它不是现成的场条件化或等变模型。
- 组合、区域、电荷模块现位于 `workflow/multiscale/`；任务执行器现位于 `workflow/dispatch/terms.py`；偏差记录位于 `workflow/records/deviation.py`。原草案部分路径是较早布局。
- 现有 `planning/task.py` 仍有 QMMM 任务展开及 theory/reference 推断，不能据此认为本草案的 ownership 已落地。

原草案保持不变；新实现应按当前目录落点，而不是机械复制旧路径。

### 11.2 框架层：公共记录和物理编排

| 建议位置 | 增量职责 |
|---|---|
| `data/qmmm.py`，新增 | 分区、ownership、场描述符和版本化记录；不持有 ML tensor |
| `planning/qmmm.py`，新增 | 匹配模型/field/MM 能力，明确 embedded-total 模式 |
| `workflow/multiscale/qmmm/field.py`，新增 | field provider 协议、direct-field oracle、VJP 和映射 |
| `workflow/multiscale/qmmm/orchestrator.py`，新增 | 同一快照下场、NN、MM、pullback 的依赖执行 |
| `workflow/multiscale/qmmm/assembler.py`，新增 | ownership、导数语义、状态轴、单位和最终结果校验 |
| `workflow/calculator.py` | 委托上述编排，继续唯一提交 State |
| `workflow/dispatch/terms.py`、`term_worker.py` | 新可序列化请求和结果；跨进程 field 服务身份 |
| `schema/`、`kernel/maestro.py`、skill query 链路 | 发现与验证可选模型能力，不指定具体 NN 实现 |
| `data/runtime/serializer.py`、`workflow/records/` | 指纹、restart、分项及响应/力分歧记录 |

场计算与 pullback 可以由 MM skill 或专用 field provider 实现，但必须对应同一份场定义。默认计算流水线为：快照 → field → NN → field pullback → 组装；MM-MM/LJ 在没有依赖且执行安全时可独立调度。

若 NN、场或 MM 任一项失败，本次事务整体失败，不使用上一步场/梯度补齐。finalize constraints 仍只在完成物理拼接后执行一次。

### 11.3 模型层：允许多种 NN 实现

新模型 skill 自行提供 encoder、两分支结构、训练器、自动微分和部署产物，通过第 8 节契约接入。建议公共接口不使用 `NeuralNetPes` 等私有容器类型，不要求所有模型导入 `nn_pes_skill`。

`nn_pes` 可作为首个兼容性验证实例：新增独立方法/profile、场输入校验、数据读取和响应导数输出，保持旧 profile 行为。若新模型使用不同 runtime 或架构，应独立 skill 实现，不能为复用入口把它硬塞进旧 MLP 内核。

现有训练/预测工作流可以继续承担任务调度。联合训练、采样权重和模型优化步骤由 model skill 负责；框架不自行循环神经网络训练。

若采用 `nn_pes`，其开发文档明确 `nn_pes_skill` 内的 vendor 树是唯一来源，框架不得反向导入 legacy 树；新实现应落在正式 skill 中。该限制只说明当前模型边界，不将未来 NN 限定为这一模型家族。

### 11.4 发布与兼容

模型包至少绑定 energy semantics、ownership、qm_gradient_convention、字段阶数、单位、gauge、原子/元素范围、总电荷/自旋、state/pair map、两分支/encoder 指纹、scaler 和场定义版本。

加载时拒绝不匹配的模型与 field provider。旧真空模型只可作为 `Eiso` 子模型或显式真空模式使用，不能仅因输出形状相同就声明支持环境。

restart 保存可重建的参数/模型引用和快照，不保存 GPU graph、引擎 Context 或临时 VJP handle。更换任何物理模型指纹后，旧预测和响应缓存失效。

## 12. 开发阶段与验收

| 阶段 | 交付内容 | 出口条件 |
|---|---|---|
| A：公共契约 | 模型无关记录、ownership、mock total model、direct field/VJP | 不依赖 NN engine 完成正确能量与全梯度拼接 |
| B：单态非周期 | V-only NN、两分支、零场/gauge、ESPF 电荷导数 loss、LAMMPS 交叉力 adapter | 电荷参考恒等式、QM/MM 全坐标有限差分、真空恢复及训练通过 |
| C：动态环境 | MM 力监督或独立认证、短 NVE、固定盒优化、restart | 连续轨迹、反作用力及总能行为可信 |
| D：周期场 | 一致 Ewald/PME field 与 VJP、完整分子映射 | 周期参考比对、边界连续性、求解器精度收敛 |
| E：多态 | 态响应、态跟踪、特定 NAMD 方法和主动学习 | 环境态能隙、跳跃与参考动力学验证 |
| F：扩展 | 更多 NN adapter、短程残差、完整 cell derivative | 各项独立认证；随后才开放 NPT/复杂边界 |

建议首轮至少用两种独立 adapter 验证协议：解析 toy model 验证物理导数，一个真实 NN 验证训练与部署；后续第二种 NN 架构接入时不应修改 field provider 或 assembler 的物理定义。

### 12.1 必须通过的测试

1. **零环境与 gauge：** 随机几何/权重下 `Eenv(R,0)=0`；固定场几何偏导归零；常数势移位产生 `Q C`，同电荷态能隙不变，`Σ a=Q`。
2. **对称性：** 旋转/平移输入几何和场，能量不变、梯度协变；按模型声明检查置换。
3. **解析场：** 单点电荷与小体系 direct sum 对比；显式 Jacobian 与 VJP 对比，覆盖 QM 和 MM 坐标。
4. **端到端导数：** 位移任一 Q/M 原子后重算场及 NN，做能量有限差分；固定场偏导不完整的错误实现应被负对照捕获。
5. **物理计数：** 额外加入 proxy Coulomb 的错误实现被拒绝；MM-MM/LJ 分项单独验证，NN components 不重复加和。
6. **训练正确性：** 配对标签、缺失 mask、单位换算、真空/环境混合批次、共享 encoder 冻结和保存/加载后零场恒等式；电荷导数 loss 对参数的混合导数可用，完整 ESPF 电荷标签没有误换成电荷差。
   对比 V-only 的通用 VJP 与 LAMMPS 固定响应电荷交叉力，验证辅助 `qV` 能量未加入总能量、QM 链式项只加一次。对 V+电场扩展设置遗漏电场响应的负对照。
7. **环境作用力：** 无外加约束的有限孤立体系总力/总力矩检查；环境原子自由运动时短 NVE，随时间步与场精度收敛。
8. **周期：** cell translation、完整分子跨边界、倾斜盒、势规范、网格/VJP 一致性；NPT 未认证时提前拒绝。
9. **多态与分歧：** 每态 MM 响应不同、状态映射一致、模型成员完整组装后统计、真空耦合不能冒充嵌入耦合。
10. **框架兼容：** 旧真空 NN/纯 QM/普通 MM 路径回归，线程/远程身份一致，stale field 拒绝，失败不提交部分结果。

误差阈值按参考精度和体系尺度设定，分别报告能量、能隙、QM 力、MM 力及场响应；toy 测试通过不等于真实电子结构精度通过。没有环境力认证时，只发布范围明确的单点/固定环境能力，不宣布完整自由环境 MD 可用。

## 13. 文献背景与证据边界

讨论中的相关工作提供方法依据，但不能直接证明本设计已经实现，也不能据此宣称“这一组合尚无人提出”。本文是工程与方法设计，不是完整的新颖性检索。下列工作在本次撰写中核对了原始论文页面或作者公开论文：

| 工作 | 可借鉴内容 | 与本设计的关系 |
|---|---|---|
| [Lei、Yagi、Sugita，JCP 2024，10.1063/5.0205123](https://pubs.aip.org/aip/jcp/article-pdf/doi/10.1063/5.0205123/19972178/214109_1_5.0205123.pdf) | 高阶静电展开与等变多尺度模型 | 支持可扩展的场张量表示；不要求本项目使用相同网络 |
| [Mazzeo 等，Digital Discovery 2024，D4DD00295D](https://pubs.rsc.org/de-at/content/articlepdf/2024/dd/d4dd00295d) | 基态/激发态 electrostatic-embedding ML/MM | 参考环境条件化能量和力的整体设计 |
| [Tiefenbacher 等，2025，FieldSchNet 显式溶剂非绝热动力学](https://arxiv.org/abs/2501.16974) | 外部电场作为输入、多态显式环境轨迹 | 说明已有相关 NAMD 路线，不能把“场输入 + 激发态”本身当作首创 |
| [Electrostatic Embedding of Machine Learning Potentials，JCTC 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10061678/) | 真空势与 embedding 项的解耦 | 可借鉴模块边界，但其具体静电表示不能等同于本方案 |

附件还提及 Pan/DPMM、FIREANN 及未给完整标识的 2026 年工作。这些是后续专题综述的检索线索；本稿不采用其中未经逐篇核验的优先权、适用范围或性能结论。

## 14. 最终设计约束

**一个 total model，包含 `Eiso(R)` 与严格零环境的 `ΔE(R,V)`；首版通过 `q_response=∂ΔE/∂V` 替代 Molcas 的电荷输出，并用零阶 ESPF 电荷计算导数 loss。** LAMMPS 可用该响应电荷计算交叉静电力，但辅助 Coulomb 能量不重复加入 NN 嵌入能量。无需独立电荷预测 head；电荷是能量导数。电场用于首版坐标求导，独立 V+电场 NN 输入及其额外响应作为扩展。

公共契约面向多种未来 NN，而不是围绕当前 `nn_pes` 定义。短程非共价、周期应力和非绝热耦合分别扩展，只有在对应标签、导数和 ownership 同时明确后才开放。
