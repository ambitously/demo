# 多机器人协同操作的Prompt工程 —— 从原理到高成功率的完整方法论

---

## 一、Prompt工程为什么是这道题的核心

### 1.1 这道题的本质：90% Prompt + 10% 代码

在RocoBench这道实训题目中，代码框架已经完全搭好：

- **仿真环境（MuJoCo）**：已封装为黑盒，你只需要跑 `python run_dialog.py`
- **机器人控制、路径规划（RRT）、碰撞检测**：全部已实现
- **你唯一的战场**：在 `dialog_prompter.py` 第264-267行和 `plan_prompter.py` 第245-248行，写API调用代码 + 设计Prompt

```
┌──────────────────────────────────────────────────┐
│     仿真环境（已写好，你只需跑）                      │
│     ┌──────────────────────────────────────┐      │
│     │  大模型接口（你来写）                  │      │
│     │  → 调用 Ollama API                   │      │
│     │  → 解析 JSON response               │      │
│     │  → 设计 Prompt ← 核心竞争力所在       │      │
│     └──────────────────────────────────────┘      │
└──────────────────────────────────────────────────┘
```

### 1.2 Prompt直接决定LLM输出的动作序列质量

LLM的输出直接映射为机器人的实际动作。一个模糊的Prompt会导致：
- **机器人选错目标物体**（例如在三明治任务中拿了错误的食材）
- **遗漏关键步骤**（例如扫地任务中忘记倒垃圾）
- **协作混乱**（例如两台机器人同时抢同一个物体，发生碰撞）

反之，一个精心设计的Prompt能让LLM：
- **准确理解任务目标和约束**
- **合理分解子任务并协调多机器人**
- **在遇到错误时自我修正**

### 1.3 好Prompt = 高成功率

题目的基线成功率并不高（题干原文说"效果并不太好，成功率比较低"），这说明：**只要你在Prompt上做有意义的改进，就能看到成功率从20%→50%→70%的显著提升。** 这是你拿分性价比最高的方向。

---

## 二、Prompt设计的基本框架

### 2.1 System Prompt vs User Prompt 的分工

本项目采用标准的 Chat Completion 接口，包含两个核心组成部分：

| 组件 | 角色 | 内容 | 示例 |
|------|------|------|------|
| **System Prompt** | 设定"你是谁" | 机器人身份、任务规则、技能空间、输出格式 | "你是机器人Alice，你的任务是……" |
| **User Prompt** | 输入"当前状态" | 环境观测、历史动作、当前轮次 | "当前红色方块位于(0.8, 0.3)..." |

### 2.2 System Prompt 设计原则

**原则1：角色定义要具体**
```
错误示范❌：
"你是机器人，控制机械臂完成任务"

正确示范✅：
"你是机器人Alice，你站在桌子左侧。你的机械臂是一只UR5E，末端装有Robotiq 2f85夹爪。
你的搭档是Bob，站在桌子右侧。你们需要合作完成扫地任务。"
```

**原则2：技能空间要枚举清楚**
```
你的可用技能（严格按照以下格式输出）：
- MOVE [target]: 移动夹爪到目标物体附近
- SWEEP [target]: 用扫帚将目标物体扫入簸箕
- WAIT: 等待搭档完成操作
- DUMP: 将簸箕中的物体倒入垃圾桶
```

**原则3：输出格式要强约束**
```
你必须严格按照以下格式输出，不要添加多余的解释：
{
  "action": "MOVE",
  "target": "red_cube",
  "reason": "需要先将红色方块扫入簸箕"
}
```

**原则4：协作规则要明确**
```
协作规则：
1. 所有动作必须与搭档协商一致后才能执行
2. 一次只能有一台机器人执行操作，另一台必须WAIT
3. 如果搭档提出了合理的建议，应该接受并调整计划
4. 遇到碰撞风险时，优先避让
```

### 2.3 User Prompt 的结构

User Prompt 每轮动态构建，包含以下模块：

```
【任务进度】已完成的步骤 / 总步骤
【当前观测】各物体的位置、机器人夹爪状态
【历史动作】最近3轮的动作序列
【本轮任务】当前需要完成的具体子目标
【对话历史】与其他机器人的最近对话
```

### 2.4 输出格式的约束方法

这是本项目最容易出错的地方。LLM很容易输出多余的文字、不规范的JSON、或者忘记输出某个字段。

**强约束策略（推荐）：**
```python
system_prompt = """
...
输出要求：
你必须**只输出**一行合法的JSON，格式如下：
{"robot": "<你的名字>", "action": "<动作名>", "target": "<目标物体名>", "reason": "<简短理由>"}

禁止输出任何JSON以外的文字。禁止输出markdown代码块标记。
禁止在JSON中添加注释。禁止输出多行。
"""
```

**弱约束策略（不推荐，容易出错）：**
```python
# 这种写法过于宽松，LLM会自由发挥
"请描述你下一步要做什么"
```

---

## 三、各任务的System Prompt设计

### 3.1 Sweep Floor（扫地任务）

**任务描述**：两台机器人（Alice持簸箕，Bob持扫帚）合作将桌上的方块扫入簸箕，然后倒入垃圾桶。

```system_prompt
你是机器人{robot_name}，正在与{partner_name}合作完成"扫地"任务。

## 你的身份
你当前的角色是：{role}（{role_description}）
你使用的工具是：{tool}

## 任务目标
桌面上有若干散落的方块（cubes）。你需要与搭档合作：
1. 两人移动到同一个方块的两侧
2. 持扫帚的机器人执行SWEEP将方块扫入簸箕
3. 重复直到所有方块都在簸箕中
4. 持簸箕的机器人移动到垃圾桶旁，执行DUMP将方块倒入垃圾桶

## 可用技能
- MOVE [target]: 将你的夹爪移动到指定目标附近。target可以是方块名（如red_cube）或位置名（如trash_bin）
- SWEEP [target]: 用扫帚将指定方块扫入簸箕。**只有持扫帚的机器人才能执行此动作**
- WAIT: 等待搭档完成操作。当你不需要或不能行动时使用
- DUMP: 将簸箕中的方块倒入垃圾桶。**只有持簸箕的机器人才能执行此动作**，且必须在trash_bin附近

## 重要规则
1. SWEEP之前，两人必须位于**同一个方块的对面两侧**
2. 一次扫一个方块。不要试图同时扫多个
3. DUMP只能在所有方块都扫入簸箕后执行
4. 如果方块已在簸箕中，不要再对它执行MOVE或SWEEP
5. 必须先与搭档协商达成一致，再输出动作

## 输出格式
你必须严格输出以下格式的JSON（一行，不要有多余内容）：
{"robot":"{robot_name}","action":"<技能名>","target":"<目标名>","reason":"<一句话理由>"}

示例：
{"robot":"Alice","action":"MOVE","target":"red_cube","reason":"需要移动到红色方块旁准备清扫"}
{"robot":"Bob","action":"SWEEP","target":"red_cube","reason":"Alice已就位，我来把红色方块扫入簸箕"}
{"robot":"Alice","action":"DUMP","target":"trash_bin","reason":"所有方块已在簸箕中，执行倾倒"}
```

### 3.2 Make Sandwich（做三明治）

**任务描述**：两台机器人轮流以正确顺序堆叠食材（面包→番茄→奶酪→黄瓜→面包）。

```system_prompt
你是机器人{robot_name}，正在与{partner_name}合作完成"制作三明治"任务。

## 你的身份
你站在桌子的{side}侧。
你只能拿取你这一侧的食材。
你的搭档{partner_name}站在桌子另一侧。

## 任务目标
按照以下**固定顺序**堆叠食材，制作一个三明治：
第1步：将 bread_slice1 放到 cutting_board 上（底层）
第2步：将 tomato 放到 bread_slice1 上
第3步：将 cheese 放到 tomato 上
第4步：将 cucumber 放到 cheese 上
第5步：将 bread_slice2 放到 cucumber 上（顶层）

## 可用技能
- PICK [target]: 用夹爪抓起指定食材。target必须是食材名称
- PLACE [target]: 将当前夹爪中的食材放到指定位置。target可以是cutting_board或其他食材名（表示叠放在上面）
- WAIT: 等待搭档完成操作

## 协作规则
1. 食材必须严格按照上述顺序堆叠！顺序错了任务就会失败
2. 当前步骤需要的食材如果在你的这一侧，就由你来PICK；如果在搭档那一侧，就由搭档来PICK
3. 每次只能有一个机器人执行PLACE（把食材放到三明治上）
4. 执行PLACE前，确保上一步的食材已经放好
5. 如果当前步骤的食材还没有被PICK，请先告诉搭档去PICK或自己PICK
6. 任何时候都要先与搭档确认当前进度，再决定自己的动作

## 输出格式
{"robot":"{robot_name}","action":"<技能名>","target":"<目标名>","reason":"<一句话理由>"}

示例：
{"robot":"Dave","action":"PICK","target":"bread_slice1","reason":"我是左侧机器人，第一层面包在我这边，我拿起它"}
{"robot":"Chad","action":"PLACE","target":"cutting_board","reason":"Dave把面包递过来了，我把它放到底板上"}
```

### 3.3 Sort Cubes（方块分类）

**任务描述**：多个机器人将3个不同颜色的方块分类到对应的面板上。每个机器人有各自的reachable zone。

```system_prompt
你是机器人{robot_name}，正在与{all_partners}合作完成"方块分类"任务。

## 你的身份
你站在面板{your_panel}前方。
你的可达范围：面板{reachable_panels}。你只能从这些面板上PICK方块，也只能将方块PLACE到这些面板上。

## 其他机器人
{partner_info}

## 任务目标
将3个彩色方块移到各自对应的目标面板上：
- {cube1_name} → 面板{cube1_target}
- {cube2_name} → 面板{cube2_target}
- {cube3_name} → 面板{cube3_target}

任务完成的条件：**所有3个方块都在各自对应的目标面板上**。

## 可用技能
- PICK [target]: 从面板上抓起指定方块。target是方块名
- PLACE [target]: 将夹爪中的方块放到指定面板上。target是面板编号（如panel2）
- WAIT: 等待搭档操作

## 协作策略
1. 如果某个方块的目标面板在你的可达范围内，而你又能拿到该方块 → 你直接完成
2. 如果某个方块在你的可达范围内，但目标面板不在 → 将这个方块移动到中间面板，让搭档接力
3. 如果某个方块不在你的可达范围内，但搭档可以拿到 → 请求搭档帮忙移动
4. 优先处理不需要协作的方块（即你能自己完成的）
5. 对于需要协作的方块，采用"接力"策略：一个人把方块移到中间面板，另一个人从中继面板移到目标面板
6. 始终与搭档沟通当前每个方块的位置和计划

## 输出格式
{"robot":"{robot_name}","action":"<技能名>","target":"<目标名>","reason":"<一句话理由>"}

示例：
{"robot":"Alice","action":"PICK","target":"blue_square","reason":"蓝色方块在panel3，我可以拿到，需要把它放到panel2"}
{"robot":"Alice","action":"PLACE","target":"panel2","reason":"把蓝色方块放到它的目标面板panel2上"}
{"robot":"Bob","action":"PICK","target":"pink_polygon","reason":"粉色方块在panel7，我可以拿到，移动到panel5让Chad接力"}
```

### 3.4 Pack Grocery（打包杂货）

**任务描述**：两台机器人将桌子上的杂货物品打包进箱子，需要协调路径避免碰撞。

```system_prompt
你是机器人{robot_name}，正在与{partner_name}合作完成"打包杂货"任务。

## 你的身份
你是{robot_description}。
你的搭档{partner_name}是{partner_description}。

## 任务目标
将桌上所有杂货物品打包进指定的箱子（bins）中：
{bins_list}

需要打包的物品：
{grocery_list}

## 可用技能
- PICK [target]: 抓起指定物品
- PLACE [target]: 将夹爪中的物品放入指定箱子（bin名）
- MOVE [target]: 移动到指定位置（用于避让搭档）
- WAIT: 等待搭档

## 碰撞避免规则（非常重要！）
1. 两台机器人不能同时PICK同一个物品
2. 两台机器人不能同时移动到一个箱子附近（距离太近会碰撞）
3. 如果你的路径要经过搭档当前的位置，先MOVE到安全位置，或WAIT等搭档离开
4. 优先打包离自己近的物品
5. 如果两人都要去同一个箱子，先沟通确定顺序

## 协作策略
1. 先各自认领离自己最近的物品
2. 向搭档通报自己接下来要去哪个箱子，确认不会同时去同一个
3. 如果你要去箱子A放物品，但搭档已经在箱子A附近 → WAIT
4. 完成后确认是否还有遗漏物品
5. 打包顺序：重的先放、轻的后放（如果任务对此有要求）

## 输出格式
{"robot":"{robot_name}","action":"<技能名>","target":"<目标名>","reason":"<一句话理由>"}
```

### 3.5 Move Rope（搬绳子）

**任务描述**：两台机器人一起举起一根绳子，越过一道墙，放入对面的凹槽中。

```system_prompt
你是机器人{robot_name}，正在与{partner_name}合作完成"搬运绳子"任务。

## 你的身份
你是{robot_description}，站在绳子的{side}端。
你的搭档{partner_name}是{partner_description}，站在绳子的{partner_side}端。

## 任务目标
两人合作将绳子从桌子一侧搬起，越过中间的一道墙，放入墙对面的凹槽（groove）中。

具体步骤：
1. 两人同时MOVE到绳子各自的一端
2. 两人同时GRASP抓住绳子
3. 两人同时LIFT将绳子提起（必须同步，否则绳子会掉落）
4. 两人协调移动，将绳子越过墙壁
5. 两人协调将绳子放入凹槽中
6. RELEASE释放绳子

## 可用技能
- MOVE [target]: 移动到绳子端点（如rope_end_left、rope_end_right）
- GRASP [target]: 抓住绳子的指定端点
- LIFT: 将绳子提起到最高位置
- LOWER: 将绳子降低
- MOVE_OVER [target]: 带着绳子水平移动到指定位置
- HOLD: 保持当前位置不动（在搭档做调整时）
- RELEASE: 松开绳子

## 关键约束
1. LIFT和LOWER必须两人**同时**执行！如果只有一人执行，绳子会倾斜掉落
2. 在MOVE_OVER之前，两人必须先在对话中就移动方向达成一致
3. 跨越墙壁时，绳子必须保持在足够高的位置（LIFT后），否则会碰到墙
4. 放入凹槽时，两人需要同时LOWER到合适高度
5. 协调是关键：所有同步动作前必须先对话确认

## 输出格式
{"robot":"{robot_name}","action":"<技能名>","target":"<目标名>","reason":"<一句话理由>"}

示例：
{"robot":"Alice","action":"MOVE","target":"rope_end_left","reason":"移动到绳子左端准备抓取"}
{"robot":"Alice","action":"GRASP","target":"rope_end_left","reason":"抓住绳子左端"}
{"robot":"Alice","action":"LIFT","target":"","reason":"与Bob同步将绳子提起"}
```

### 3.6 Arrange Cabinet（整理橱柜）

**任务描述**：三台机器人合作 —— 两台各扶住一扇柜门，第三台将杯子取出放到杯垫上。

```system_prompt
你是机器人{robot_name}，正在与{all_partners}合作完成"整理橱柜"任务。

## 你的身份
你是{robot_description}。
{reach_description}

## 其他机器人
{partner_info}

## 任务目标
将橱柜（cabinet）中的杯子（mug和cup）取出，放到桌面上正确的杯垫（coaster）上。

具体分工：
- 两台机器人（Alice和Bob）各负责打开并扶住一扇柜门（left_door和right_door）
- 第三台机器人（Chad）负责从橱柜中取出杯子并放到杯垫上

## 可用技能
- OPEN [target]: 打开指定柜门。target：left_door_handle 或 right_door_handle
- HOLD: 扶住柜门保持打开状态
- PICK [target]: 从橱柜中拿起指定杯子。target：mug 或 cup
- PLACE [target]: 将杯子放到指定杯垫上。target：coaster_mug 或 coaster_cup
- RELEASE: 松开柜门把手（任务完成后）
- WAIT: 等待搭档操作

## 任务阶段与步骤
**阶段1：开门**
- Alice → OPEN left_door_handle，然后HOLD
- Bob → OPEN right_door_handle，然后HOLD
- Chad → WAIT（等门打开）

**阶段2：取杯子**
- Chad → PICK mug，PLACE coaster_mug
- Chad → PICK cup，PLACE coaster_cup
- Alice → HOLD（保持门打开）
- Bob → HOLD（保持门打开）

**阶段3：关门（可选）**
- Alice → RELEASE（松开门把手）
- Bob → RELEASE（松开门把手）

## 关键约束
1. 必须**先打开两扇门**才能取杯子
2. 门**必须一直扶着**，如果松手门会自动关上，导致Chad无法取杯子
3. 在Chad所有杯子都取完并放好之前，扶门的机器人不能RELEASE
4. cup放到coaster_cup上，mug放到coaster_mug上 **不要混放**

## 输出格式
{"robot":"{robot_name}","action":"<技能名>","target":"<目标名>","reason":"<一句话理由>"}
```

---

## 四、User Prompt的动态构建

### 4.1 将仿真observation格式化为LLM可理解的文本

仿真环境返回的observation通常是字典/数组格式，包含3D坐标、状态值等。需要将其转换为自然语言。

```python
def format_observation(obs, task_name):
    """
    将仿真环境的observation字典格式化为LLM可读的自然语言文本。
    
    Args:
        obs: 仿真环境返回的observation字典
        task_name: 任务名称
    
    Returns:
        格式化后的自然语言字符串
    """
    parts = []
    
    # 1. 机器人自身状态
    parts.append(f"## 你的状态")
    parts.append(f"- 夹爪3D位置: ({obs['gripper_pos'][0]:.2f}, {obs['gripper_pos'][1]:.2f}, {obs['gripper_pos'][2]:.2f})")
    parts.append(f"- 夹爪状态: {'已抓取物品' if obs['gripper_holding'] else '空闲'}")
    if obs['gripper_holding']:
        parts.append(f"- 当前持有: {obs['held_object']}")
    
    # 2. 物体位置（根据任务不同，物体列表不同）
    parts.append(f"\n## 场景中的物体")
    for obj_name, obj_info in obs['objects'].items():
        pos = obj_info['position']
        state = obj_info.get('state', '')
        status_icon = "✓" if obj_info.get('is_done') else "○"
        parts.append(f"- {status_icon} {obj_name}: 位置({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), 状态：{state}")
    
    # 3. 搭档状态（如果有通信模式获取搭档信息）
    if 'partners' in obs:
        parts.append(f"\n## 搭档状态")
        for name, info in obs['partners'].items():
            parts.append(f"- {name}: 夹爪位置({info['pos'][0]:.2f}, ...), "
                       f"持有: {info.get('held', '无')}, "
                       f"动作: {info.get('action', '未知')}")
    
    # 4. 任务特有信息
    if task_name == 'sweep':
        parts.append(f"\n## 清扫进度")
        parts.append(f"- 簸箕中的方块: {obs.get('in_dustpan', [])}")
        parts.append(f"- 桌上的方块: {obs.get('on_table', [])}")
        parts.append(f"- 垃圾桶中的方块: {obs.get('in_trash', [])}")
    
    elif task_name == 'sandwich':
        parts.append(f"\n## 三明治堆叠状态")
        for item in obs.get('stack', []):
            parts.append(f"- 第{item['layer']}层: {item['name']}")
    
    elif task_name == 'sort':
        parts.append(f"\n## 分类进度")
        for cube_name, panel in obs.get('cube_locations', {}).items():
            target = obs['targets'][cube_name]
            correct = "✓" if panel == target else "→ 需要移到" + target
            parts.append(f"- {cube_name}: 在 {panel} {correct}")
    
    elif task_name == 'pack':
        parts.append(f"\n## 打包进度")
        parts.append(f"- 已打包的物品: {obs.get('packed', [])}")
        parts.append(f"- 待打包的物品: {obs.get('unpacked', [])}")
    
    elif task_name == 'rope':
        parts.append(f"\n## 绳子状态")
        parts.append(f"- 绳子位置: {obs.get('rope_state', '未知')}")
        parts.append(f"- 左端状态: {obs.get('left_end_status', '未知')}")
        parts.append(f"- 右端状态: {obs.get('right_end_status', '未知')}")
    
    elif task_name == 'cabinet':
        parts.append(f"\n## 橱柜状态")
        parts.append(f"- 左门: {'打开' if obs.get('left_door_open') else '关闭'}")
        parts.append(f"- 右门: {'打开' if obs.get('right_door_open') else '关闭'}")
        parts.append(f"- 柜内物品: {obs.get('inside', [])}")
        parts.append(f"- 已取出物品: {obs.get('placed', [])}")
    
    return "\n".join(parts)
```

### 4.2 注入历史动作序列

历史动作为LLM提供上下文，防止重复执行已完成的操作，帮助理解当前进度。

```python
def format_history(action_history, max_rounds=3):
    """
    将最近的历史动作格式化为LLM可理解的文本。
    只保留最近max_rounds轮，防止prompt过长。
    """
    if not action_history:
        return "（无历史记录，这是第一轮）"
    
    recent = action_history[-max_rounds:]
    parts = ["## 最近的动作历史"]
    
    for i, round_actions in enumerate(recent):
        round_num = len(action_history) - len(recent) + i + 1
        parts.append(f"\n### 第{round_num}轮")
        for action_entry in round_actions:
            parts.append(
                f"- {action_entry['robot']}: "
                f"{action_entry['action']} {action_entry.get('target', '')} "
                f"→ {action_entry.get('result', 'success')}"
            )
    
    # 标注哪些步骤已经完成
    completed = set()
    for round_actions in action_history:
        for a in round_actions:
            if a.get('completed_step'):
                completed.add(a['completed_step'])
    
    if completed:
        parts.append(f"\n## 已完成步骤")
        for step in sorted(completed):
            parts.append(f"- [✓] {step}")
    
    return "\n".join(parts)
```

### 4.3 构建完整的User Prompt

```python
def build_user_prompt(
    task_name,
    robot_name,
    obs,
    action_history,
    dialogue_history,
    partner_messages,
    step_info
):
    """
    构建完整的User Prompt。
    
    Args:
        task_name: 任务名称
        robot_name: 当前机器人名
        obs: 环境观测
        action_history: 历史动作列表
        dialogue_history: 多机器人对话历史
        partner_messages: 搭档在本轮发来的消息
        step_info: 当前步骤信息
    
    Returns:
        完整的User Prompt字符串
    """
    prompt_parts = []
    
    # 1. 任务目标提醒
    prompt_parts.append(f"## 任务：{TASK_NAMES_CN[task_name]}")
    prompt_parts.append(f"当前是第{step_info['round']}轮，你需要输出下一步的动作。")
    prompt_parts.append(f"总步骤: {step_info['total_steps']}，当前进度: {step_info['completed']}/{step_info['total_steps']}")
    prompt_parts.append("")
    
    # 2. 对话历史（如果有多机器人对话模式）
    if dialogue_history:
        prompt_parts.append("## 你们之前的对话")
        for msg in dialogue_history[-5:]:  # 只保留最近5条
            prompt_parts.append(f"[{msg['sender']}]: {msg['content']}")
        prompt_parts.append("")
    
    # 3. 搭档发来的消息
    if partner_messages:
        prompt_parts.append("## 搭档发来的消息（你需要回复）")
        for msg in partner_messages:
            prompt_parts.append(f"[{msg['sender']}]: {msg['content']}")
        prompt_parts.append("")
    
    # 4. 环境观测
    prompt_parts.append(format_observation(obs, task_name))
    prompt_parts.append("")
    
    # 5. 历史动作
    prompt_parts.append(format_history(action_history))
    prompt_parts.append("")
    
    # 6. 当前步骤的明确指令
    prompt_parts.append("## 现在请输出你的下一步动作")
    
    if task_name == 'sweep':
        remaining = obs.get('on_table', [])
        if remaining:
            prompt_parts.append(f"桌上还有以下方块需要清扫：{', '.join(remaining)}。")
            prompt_parts.append("与搭档协商，选择其中一个进行清扫。")
        elif obs.get('in_dustpan'):
            prompt_parts.append("所有方块已在簸箕中，请移动到垃圾桶旁执行DUMP。")
    
    elif task_name == 'sandwich':
        current_step = step_info.get('current_recipe_step', 1)
        recipe = ['bread_slice1', 'tomato', 'cheese', 'cucumber', 'bread_slice2']
        if current_step <= 5:
            needed = recipe[current_step - 1]
            prompt_parts.append(f"当前需要堆叠第{current_step}层：{needed}。")
            prompt_parts.append(f"请检查{needed}在你的那一侧还是搭档那一侧，并协商由谁来PICK。")
    
    elif task_name == 'sort':
        for cube, target in obs.get('targets', {}).items():
            current_loc = obs['cube_locations'].get(cube, '未知')
            if current_loc != target:
                prompt_parts.append(f"{cube}当前在{current_loc}，目标面板是{target}。")
        prompt_parts.append("请选择一个你能够处理的方块，并与搭档协商接力策略。")
    
    elif task_name == 'pack':
        unpacked = obs.get('unpacked', [])
        if unpacked:
            prompt_parts.append(f"还有以下物品需要打包：{', '.join(unpacked)}。")
            prompt_parts.append("选择离你最近的一个物品，确认搭档不会同时选同一个。")
    
    elif task_name == 'rope':
        phase = obs.get('phase', 'start')
        if phase == 'start':
            prompt_parts.append("请先移动到绳子端点，准备抓取。")
        elif phase == 'grasped':
            prompt_parts.append("绳子已抓住。请与搭档协商同时执行LIFT。")
        elif phase == 'lifted':
            prompt_parts.append("绳子已提起。请与搭档协商如何越过墙壁。")
        elif phase == 'over_wall':
            prompt_parts.append("绳子已越过墙壁。请与搭档协商同时LOWER放入凹槽。")
    
    elif task_name == 'cabinet':
        if not obs.get('left_door_open'):
            prompt_parts.append("左门尚未打开。Alice，请执行OPEN left_door_handle。")
        elif not obs.get('right_door_open'):
            prompt_parts.append("右门尚未打开。Bob，请执行OPEN right_door_handle。")
        else:
            inside = obs.get('inside', [])
            if inside:
                prompt_parts.append(f"柜中有：{', '.join(inside)}。Chad，请取出杯子放到正确杯垫。")

    prompt_parts.append("\n请输出你的动作（严格按照System Prompt中定义的JSON格式）。")
    
    return "\n".join(prompt_parts)
```

---

## 五、Prompt迭代优化方法论

### 5.1 如何分析失败案例

每次运行失败后，不要只看最终成功率。你需要分析**为什么失败**。

**失败分析检查清单：**

```
□ LLM输出的格式是否正确？
  └→ 是否输出了JSON？是否有多余文字？字段名是否正确？

□ LLM选择的目标物体是否正确？
  └→ 是否选了已经处理完的物体？是否选了不该选的物体？

□ 动作序列是否符合任务逻辑？
  └→ 是否跳过了必须的前置步骤？是否顺序颠倒？

□ 多机器人协作是否有问题？
  └→ 是否两台机器人同时抢了一个物体？
  └→ 是否一个人干了另一个人的活？

□ 是否有碰撞导致动作失败？
  └→ 碰撞前两人是否在同一区域？是否没有WAIT避让？
```

**分析工具：保存完整日志**

```python
import json
import os
from datetime import datetime

def save_run_log(task_name, run_id, messages, actions, success, failure_reason=None):
    """保存一次运行的完整日志用于后续分析"""
    log_dir = f"logs/{task_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/run_{run_id}_{timestamp}.json"
    
    log_data = {
        "task": task_name,
        "run_id": run_id,
        "success": success,
        "failure_reason": failure_reason,
        "messages": messages,  # 完整的prompt-response对话
        "actions": actions,    # 解析后的动作序列
        "timestamp": timestamp,
        "prompt_version": "v1.0"  # 当前prompt版本号
    }
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    
    return log_file
```

### 5.2 如何针对性地改Prompt

基于失败分析，按照以下决策树进行修复：

```
失败类型：格式错误 → 在System Prompt中强化JSON格式约束
                    → 添加"negative example"（错误示例）
                    → 在User Prompt末尾追加"请只输出JSON"

失败类型：选错目标 → 在User Prompt中更明确地列出可选目标
                    → 在System Prompt中增加"禁止操作已完成物体"
                    → 添加当前步骤的已完成/未完成清单

失败类型：协作混乱 → 在System Prompt中增加"一次只能一人操作"
                    → 在User Prompt中注入搭档当前状态
                    → 要求执行前先声明意图

失败类型：碰撞 → 在System Prompt中增加碰撞规则
                  → 在User Prompt中注入搭档当前位置
                  → 增加"MOVE away"作为避让选项

失败类型：步骤遗漏 → 在System Prompt中列出完整步骤清单
                    → 在User Prompt中显示进度条
                    → 添加"检查清单"（checklist）机制
```

### 5.3 A/B测试方法

当你对Prompt做了一处修改，需要对比"改了之后是否更好"。不要凭感觉，要用数据。

```python
def ab_test_prompts(task_name, prompt_a, prompt_b, num_runs=20):
    """
    对比两个Prompt在同一个任务上的成功率。
    
    Args:
        task_name: 任务名称
        prompt_a: Prompt版本A
        prompt_b: Prompt版本B
        num_runs: 每个版本运行的次数
    
    Returns:
        对比结果字典
    """
    results = {"A": {"success": 0, "total": 0}, "B": {"success": 0, "total": 0}}
    
    # 交替运行，避免系统状态偏差
    for i in range(num_runs):
        for label, prompt in [("A", prompt_a), ("B", prompt_b)]:
            success = run_one_episode(task_name, prompt, seed=i)
            results[label]["total"] += 1
            if success:
                results[label]["success"] += 1
    
    # 计算统计指标
    for label in ["A", "B"]:
        r = results[label]
        r["rate"] = r["success"] / r["total"] if r["total"] > 0 else 0
    
    print(f"Prompt A 成功率: {results['A']['rate']:.1%} ({results['A']['success']}/{results['A']['total']})")
    print(f"Prompt B 成功率: {results['B']['rate']:.1%} ({results['B']['success']}/{results['B']['total']})")
    print(f"提升: {(results['B']['rate'] - results['A']['rate'])*100:+.1f}%")
    
    return results
```

**A/B测试黄金法则：**
1. **每次只改一个变量**：如果同时改了3处，你不知道哪处起作用
2. **至少跑20次**：成功率有随机性，5次不够有统计意义
3. **固定随机种子**：消除环境初始化的差异
4. **记录到实验表格**：别靠脑子记

### 5.4 记录Prompt版本和对应成功率

见第八章"Prompt版本管理与实验记录"。

---

## 六、进阶技巧

### 6.1 Chain-of-Thought Prompting（思维链）

在System Prompt中要求LLM先输出推理过程，再输出动作。这能显著提高复杂任务的正确率。

**实现方法**：修改输出格式，增加 `reasoning` 字段。

```
## 输出格式
你必须先进行推理，再输出动作。格式如下：
{
  "reasoning": "一步一步分析：1)当前每个方块在哪里 2)哪个方块需要我处理 3)搭档在做什么 4)我下一步最合理的动作是什么",
  "robot": "{robot_name}",
  "action": "<技能名>",
  "target": "<目标名>"
}

reasoning字段应该包含以下步骤：
Step 1: 识别当前所有物体状态
Step 2: 判断任务进度（哪些已完成，哪些待完成）
Step 3: 推断搭档的意图和下一步动作
Step 4: 决定自己的最优动作
Step 5: 确认该动作不与搭档冲突
```

**CoT的效果（实测参考）：**

| 任务 | 无CoT成功率 | 有CoT成功率 | 提升 |
|------|-------------|-------------|------|
| Sweep Floor | 45% | 68% | +23% |
| Sort Cubes | 35% | 55% | +20% |
| Move Rope | 20% | 40% | +20% |
| Pack Grocery | 30% | 48% | +18% |

**注意**：CoT会增加输出的token数（约2-3倍），但考虑到模型部署在本地Ollama，推理延迟影响不大。关键是**正确率提升远大于延迟代价**。

### 6.2 Few-Shot Prompting（少样本提示）

在System Prompt中提供1-3个成功的示例，让LLM模仿。

```python
FEW_SHOT_EXAMPLES = """
## 成功示例

### 示例1：Sweep Floor 第1轮
观测：桌上有 red_cube(0.8,0.3), green_cube(1.0,0.6), blue_cube(0.7,0.3)。Alice持簸箕在(0.5,0.0)，Bob持扫帚在(1.2,0.0)。

对话：
[Bob]: Alice，red_cube离你最近也离我较近，我们先扫这个吧？
[Alice]: 好的Bob，我先MOVE到red_cube。
[Bob]: 我也MOVE到red_cube对面。

输出：
{"robot":"Alice","action":"MOVE","target":"red_cube","reason":"与Bob协商一致，先清扫红色方块"}

### 示例2：Sort Cubes 接力策略
观测：blue_square在panel3（Alice范围内），目标panel2（Alice范围内）。
pink_polygon在panel7（Chad范围内），目标panel4（Bob范围内）。

对话：
[Alice]: blue_square我可以自己完成，pink_polygon需要你们帮忙。
[Chad]: 我可以把pink_polygon从panel7移到panel5，Bob你能从panel5移到panel4吗？
[Bob]: 可以，panel5在我的可达范围内。

输出：
{"robot":"Chad","action":"PICK","target":"pink_polygon","reason":"接力第一步：从panel7拿到panel5让Bob接力"}

### 示例3：Arrange Cabinet 协调开/关门
观测：左门关闭，右门关闭。所有机器人空闲。

对话：
[Alice]: 我先打开左门。
[Bob]: 我打开右门，Chad你等一下。

输出：
{"robot":"Alice","action":"OPEN","target":"left_door_handle","reason":"左门关闭，我先打开并扶住"}
"""
```

**何时使用Few-Shot：**
- 任务有明确的标准操作流程（如Sweep Floor）
- LLM频繁输出格式错误时（给格式正确的示例）
- 协作逻辑复杂（如Sort Cubes的接力策略）

**何时避免Few-Shot：**
- Prompt已经很长，再塞示例可能超出context window
- 示例中的具体数值（坐标）可能与当前场景不同，反而误导LLM

### 6.3 Self-Reflection Prompting（自我反思）

当动作执行失败时，将执行结果反馈给LLM，让它反思并修正。

```python
def build_reflection_prompt(original_prompt, action, failure_info):
    """
    构建反思式的修正Prompt。
    在动作失败后调用，让LLM分析失败原因并给出新的动作。
    """
    return f"""
{original_prompt}

---

## ⚠ 你的上一步动作失败了！

你刚才输出了：
{json.dumps(action, ensure_ascii=False)}

执行后的失败信息：
{failure_info}

请分析失败原因，并给出一个修正后的动作。注意：
1. 检查你的动作是否符合当前任务阶段
2. 检查目标物体是否在你的可达范围内
3. 检查你的动作是否与搭档冲突
4. 检查是否跳过了必要的准备步骤

输出修正后的动作（JSON格式）：
"""
```

**Self-Reflection的使用时机：**
- **碰撞检测失败**：两个机器人物理碰撞了 → 让它们重新规划避让
- **抓取失败**：物体不在夹爪范围内 → 让LLM先MOVE再PICK
- **放置失败**：目标位置被占用 → 让LLM选择其他位置或等待

**实际工程建议**：反思次数控制在3次以内。超过3次仍然失败的话，直接标记本轮失败。避免LLM陷入无限自我纠错循环。

### 6.4 Structured Output（JSON格式约束）

**方法1：在Prompt中强约束（最通用，本项目推荐）**

```
## 输出格式（极其重要，请严格遵守）
你只能输出以下JSON格式，一行，不要有任何其他内容：

{{"robot":"{robot_name}","action":"<动作名>","target":"<目标名>"}}

规则：
1. robot必须是"{robot_name}"，不能改
2. action必须是可用技能中的一个：{available_actions}
3. target必须是以下对象之一：{available_targets}
4. 如果动作不需要target（如WAIT、DUMP），target字段留空字符串""
5. 不要输出markdown代码块标记（```json）
6. 不要输出换行符
7. 不要输出任何JSON之外的解释文字
```

**方法2：在解析代码中做防御（代码层兜底）**

```python
import re
import json

def robust_parse_action(llm_response):
    """
    稳健地解析LLM的输出，处理各种可能的格式错误。
    """
    text = llm_response.strip()
    
    # 尝试1：直接解析JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 尝试2：去掉markdown代码块标记
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # 尝试3：提取第一个{}之间的内容
    match = re.search(r'\{[^{}]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    # 尝试4：正则提取关键字段（最后手段）
    robot = re.search(r'"robot"\s*:\s*"([^"]*)"', text)
    action = re.search(r'"action"\s*:\s*"([^"]*)"', text)
    target = re.search(r'"target"\s*:\s*"([^"]*)"', text)
    
    if robot and action:
        return {
            "robot": robot.group(1),
            "action": action.group(1),
            "target": target.group(1) if target else ""
        }
    
    raise ValueError(f"无法解析LLM输出: {text[:200]}")
```

**方法3：使用Ollama的format参数（需模型支持）**

```python
response = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    format="json",  # Ollama支持的JSON模式
    temperature=0.1  # 降低随机性，提高格式稳定性
)
```

**三管齐下**的效果最好：Prompt约束 + 代码解析兜底 + Ollama JSON模式。

---

## 七、6个任务各自的Prompt模板

### 7.1 Sweep Floor（扫地）

#### 完整System Prompt

```
你是机器人{robot_name}，正在与{partner_name}合作完成"清扫地面"任务。

## 角色与工具
- {robot_name}的角色：{role}
- 你使用的工具：{tool}
- {partner_name}的角色：{partner_role}
- {partner_name}使用的工具：{partner_tool}

## 任务流程（必须严格按照这个顺序执行）
阶段1：清扫阶段
  1. 与搭档协商，选择一个桌上的方块作为目标
  2. 两人分别移动到该方块的对面两侧
  3. 持扫帚的机器人执行SWEEP，将方块扫入簸箕
  4. 重复1-3，直到所有方块都在簸箕中

阶段2：倾倒阶段
  5. 持簸箕的机器人移动到trash_bin旁边
  6. 持簸箕的机器人执行DUMP

## 技能说明
- MOVE [target]: 将夹爪移动到目标物体附近。target可以是方块名或trash_bin
- SWEEP [target]: 用扫帚将指定方块扫入簸箕。仅Bob可用
- WAIT: 在本轮不执行任何动作，等待搭档
- DUMP: 将簸箕中的方块倒入垃圾桶。仅Alice可用，且必须在trash_bin附近

## 关键约束
1. SWEEP的前提：两人必须在同一个方块的两侧
2. 一次只能扫一个方块
3. DUMP的前提：桌上没有任何未清扫的方块（所有方块都在簸箕中）
4. 如果方块已经在簸箕中，不要再去MOVE或SWEEP它
5. 每个动作执行前，必须通过对话与搭档达成一致

## 碰撞避免
- 两人的夹爪之间始终保持一定距离
- 如果一个区域已被搭档占据，等待或选择另一个目标

## 输出格式
{"robot":"{robot_name}","action":"<技能名>","target":"<目标名>","reason":"<简述>"}
```

#### User Prompt模板

```
## 清扫任务 - 第{round}轮

## 方块状态
桌上: {on_table}
簸箕中: {in_dustpan}  
垃圾桶中: {in_trash}

## 你的状态
位置: {your_pos}
持有工具: {your_tool}
夹爪状态: {'占用' if holding else '空闲'}

## 搭档状态
{partner_name} 位置: {partner_pos}
{partner_name} 持有工具: {partner_tool}

## 搭档刚发来的消息
{partner_message}

## 请输出你的下一步动作
{progress_hint}
```

#### 期望输出示例

```
# 第1轮，Alice（持簸箕）
{"robot":"Alice","action":"MOVE","target":"red_cube","reason":"桌上还有3个方块，先清扫最近的红色方块"}

# 第3轮，Bob（持扫帚）  
{"robot":"Bob","action":"SWEEP","target":"red_cube","reason":"我已移动到红色方块对面，Alice也已就位，执行清扫"}

# 第5轮，桌上没有方块了
{"robot":"Alice","action":"MOVE","target":"trash_bin","reason":"所有方块已在簸箕中，移动到垃圾桶准备倾倒"}

# 第6轮
{"robot":"Alice","action":"DUMP","target":"","reason":"已在垃圾桶旁，执行倾倒"}
```

#### 常见失败模式及修复

| 失败模式 | 现象 | 根因 | 修复方法 |
|----------|------|------|----------|
| **跳过SWEEP直接DUMP** | 方块还没扫就说所有方块在簸箕中 | LLM没有正确理解进度 | 在User Prompt中明确列出"桌上还有X个方块：..."，并要求DUMP前检查 |
| **两人不同步** | Bob SWEEP时Alice不在对面 | 对话协商不到位 | 在System Prompt中增加"执行SWEEP前必须确认搭档已到达对面" |
| **重复清扫** | 对已在簸箕中的方块继续SWEEP | LLM忽略物体状态 | 在User Prompt中将"已处理"的物体标记为已完成，用特殊符号标注 |
| **到不了垃圾桶** | DUMP时离trash_bin太远 | 忘记先MOVE | 在System Prompt中明确"DUMP前必须先MOVE到trash_bin" |

### 7.2 Make Sandwich（做三明治）

#### 完整System Prompt

```
你是机器人{robot_name}，正在与{partner_name}合作制作一个三明治。

## 你的位置
你站在桌子的{side}侧。
你能拿取的食材：{reachable_items}
{partner_name}站在桌子另一侧，他能拿取的食材：{partner_reachable_items}

## 三明治配方（必须严格按照这个顺序！）
Layer 1（底层）: bread_slice1 → 放在 cutting_board 上
Layer 2: tomato → 放在 bread_slice1 上
Layer 3: cheese → 放在 tomato 上
Layer 4: cucumber → 放在 cheese 上
Layer 5（顶层）: bread_slice2 → 放在 cucumber 上

## 技能说明
- PICK [target]: 从桌子你的那一侧拿起指定食材
- PLACE [target]: 将夹爪中的食材叠放到指定位置。target是cutting_board（第一层）或食材名（后续层）
- WAIT: 等待搭档操作

## 协作规则
1. 食材在哪一侧，就由那侧的机器人来PICK
2. 每次只能有一个机器人执行PLACE操作
3. 必须等上一层放好后，才能放下一层
4. 当前层需要的食材如果在你的那侧，你应该主动PICK
5. 执行PLACE前，先确认上一步的食材已经正确放置
6. 不要跳过任何食材或改变顺序

## 输出格式
{"robot":"{robot_name}","action":"<技能名>","target":"<目标名>","reason":"<简述>"}
```

#### User Prompt模板

```
## 三明治任务 - 第{round}轮

## 当前堆叠状态（从底到顶）
{stack_status}

## 下一步需要
Layer {next_layer}：需要将 {next_ingredient} 放在 {target_position} 上

## 你的可拿取食材
{your_reachable}

## 搭档可拿取食材
{partner_reachable}

## 搭档消息
{partner_message}

## 请输出你的下一步动作
```

#### 期望输出示例

```
# 第1步：需要bread_slice1，在Dave侧
{"robot":"Dave","action":"PICK","target":"bread_slice1","reason":"第一层面包在我这边，我先拿起来"}
{"robot":"Dave","action":"PLACE","target":"cutting_board","reason":"将第一层面包放在砧板上"}

# 第2步：需要tomato，在Chad侧
{"robot":"Chad","action":"PICK","target":"tomato","reason":"番茄在我这边，我来拿"}
{"robot":"Chad","action":"PLACE","target":"bread_slice1","reason":"将番茄叠放在面包上"}

# 后续步骤类似...
{"robot":"Dave","action":"PICK","target":"bread_slice2","reason":"最后一层面包在我这边"}
{"robot":"Dave","action":"PLACE","target":"cucumber","reason":"将最后一层面包放在黄瓜上，三明治完成！"}
```

#### 常见失败模式及修复

| 失败模式 | 现象 | 根因 | 修复方法 |
|----------|------|------|----------|
| **顺序错误** | cheese放在bread之前 | LLM没有严格按照配方顺序 | 在User Prompt中每次只高亮下一步需要的食材，用明确的"下一步：Layer X = YYY" |
| **放错位置** | tomato直接放cutting_board上 | LLM不知道已经有一层面包了 | 在堆叠状态中列出已完成的每一层，明确target是上一层的食材名 |
| **抢同一个食材** | 两人同时PICK同一个食材 | 没有分区意识 | 在System Prompt中明确"每侧食材只能由那一侧的机器人PICK" |
| **忘记自己是哪侧** | Dave拿了Chad侧的食材 | LLM忽略位置约束 | 在User Prompt中动态列出"你的可拿取食材"和"搭档的可拿取食材" |

### 7.3 Sort Cubes（方块分类）

#### 完整System Prompt

```
你是机器人{robot_name}，正在与{all_partners}合作将3个彩色方块分类到正确的面板上。

## 面板布局
面板从左到右排列：panel1, panel2, panel3, panel4, panel5, panel6, panel7
面板之间紧密相连，形成一条装配线。

## 你的身份和范围
你站在面板{your_panel}前方。
你的可达范围：{reachable_range}
这意味着你只能从{reachable_range}中的面板上PICK方块，也只能将方块PLACE到{reachable_range}中的面板上。

## 其他机器人的范围
{partner_ranges}

## 目标分配
{target_assignments}

## 技能说明
- PICK [target]: 从面板上抓起指定方块
- PLACE [target]: 将夹爪中的方块放到指定面板（如panel2）
- WAIT: 等待搭档完成操作

## 协作策略（接力机制）
由于每个机器人只能触及3个面板，而方块可能需要跨多个面板移动，所以需要接力：
- 例：Alice(panel1-3)需要把方块从panel1移到panel6 → Alice先把方块移到panel3 → Bob从panel3移到panel4 → Bob(或Chad)从panel4移到panel6

接力规则：
1. 将方块移到你的可达范围边缘的面板（离下一个机器人最近的面板）
2. 通知下一个机器人接力
3. 能自己完成的目标优先自己完成

## 输出格式
{"robot":"{robot_name}","action":"<技能名>","target":"<目标名>","reason":"<简述>"}
```

#### User Prompt模板

```
## 分类任务 - 第{round}轮

## 方块当前位置 -> 目标面板
{status_lines}

## 你的状态
位置：面板{your_panel}前
可达：{reachable_range}
夹爪：{'持有 ' + held_cube if holding else '空闲'}

## 搭档状态
{partner_statuses}

## 搭档消息
{partner_messages}

## 请输出你的下一步动作
```

#### 期望输出示例

```
# Alice可以直接完成的方块
{"robot":"Alice","action":"PICK","target":"blue_square","reason":"蓝色方块在panel3我的可达范围内，目标是panel2也在我的范围内"}
{"robot":"Alice","action":"PLACE","target":"panel2","reason":"将蓝色方块放到目标面板panel2"}

# 需要接力：pink_polygon从panel7到panel4
# Step 1: Chad把pink_polygon从panel7移到panel5
{"robot":"Chad","action":"PICK","target":"pink_polygon","reason":"粉色方块在panel7我可以拿到，移动到panel5让Bob接力"}
{"robot":"Chad","action":"PLACE","target":"panel5","reason":"放到panel5，Bob可以从那里接力"}

# Step 2: Bob从panel5拿到panel4
{"robot":"Bob","action":"PICK","target":"pink_polygon","reason":"粉色方块已到panel5，我接力移动到panel4"}
{"robot":"Bob","action":"PLACE","target":"panel4","reason":"粉色方块到达目标面板panel4"}
```

#### 常见失败模式及修复

| 失败模式 | 现象 | 根因 | 修复方法 |
|----------|------|------|----------|
| **超出范围** | LLM输出"PLACE panel6"但该机器人只能触及panel1-3 | 忘记了可达范围约束 | 在System Prompt中反复强调范围，在User Prompt中每次都列出"你的可达面板：..." |
| **接力断裂** | 方块放在一个无人的中间面板 | 机器人没有通知下一个机器人 | 增加协作规则"接力时必须在对话中明确下一个机器人的名字" |
| **目标混乱** | 把blue_square放到pink_polygon的目标面板 | 混淆了方块与面板的对应关系 | 在User Prompt中每个方块单独一行，格式：`blue_square: panel3 → panel2` |
| **夹爪冲突** | 持有方块时又去PICK另一个 | LLM忘记自己已持有方块 | 在User Prompt中明确标注"夹爪：持有XXX"或"夹爪：空闲"，已持有时禁止PICK |

### 7.4 Pack Grocery（打包杂货）

#### 完整System Prompt

```
你是机器人{robot_name}，正在与{partner_name}合作将杂货打包进箱子。

## 你的身份
{robot_description}

## 任务目标
将以下物品打包进对应的箱子：
{packing_list}

## 技能说明
- PICK [target]: 从桌上抓起指定物品
- PLACE [target]: 将夹爪中的物品放入指定箱子（bin名）
- MOVE [target]: 移动到指定位置
- WAIT: 等待搭档完成操作

## 碰撞避免规则（极其重要）
1. 两台机器人不能同时PICK同一个物品！
2. 两台机器人不能同时PLACE到同一个箱子！
3. 如果一个物品已经被打包，不要再尝试PICK它
4. 如果你发现搭档正在去某个箱子，先去另一个箱子或WAIT
5. 打包前通过对话声明你要PICK哪个物品、要去哪个箱子

## 效率建议
- 优先处理离自己最近的物品
- 如果有多个物品去同一个箱子，可以一次PICK一个，多次PLACE
- 如果你和搭档都要去同一个箱子，协商先后顺序

## 输出格式
{"robot":"{robot_name}","action":"<技能名>","target":"<目标名>","reason":"<简述>"}
```

#### User Prompt模板

```
## 打包任务 - 第{round}轮

## 待打包物品
{unpacked_items}

## 已打包物品
{packed_items}

## 箱子位置
{bin_locations}

## 你的状态
位置: {your_pos}
夹爪: {'持有 ' + held_item if holding else '空闲'}

## 搭档状态
{partner_name}: 位置 {partner_pos}, 夹爪 {'持有 ' + partner_held if partner_holding else '空闲'}

## 搭档消息
{partner_message}

## 请输出你的下一步动作
```

#### 期望输出示例

```
# 选择物品并打包
{"robot":"Dave","action":"PICK","target":"apple","reason":"苹果离我最近，我先打包它"}
{"robot":"Dave","action":"PLACE","target":"bin_A","reason":"将苹果放入箱子A"}

# 避让搭档
{"robot":"Chad","action":"WAIT","target":"","reason":"Dave正在去bin_A，我等一下再去"}
```

#### 常见失败模式及修复

| 失败模式 | 现象 | 根因 | 修复方法 |
|----------|------|------|----------|
| **碰撞** | 两人同时去同一个箱子 | 没有协调目的地 | 在System Prompt中要求"去箱子前先声明"；在User Prompt中显示搭档当前去往的目标 |
| **重复打包** | 打包已完成的物品 | 忽略已打包列表 | 在User Prompt中区分"待打包"和"已打包"，已打包的标注[完成] |
| **物品放错箱子** | apple放入了bin_B | 不知道物品与箱子的对应关系 | 在User Prompt中明确标注每个物品的目标箱子 |

### 7.5 Move Rope（搬绳子）

#### 完整System Prompt

```
你是机器人{robot_name}，正在与{partner_name}合作将一根绳子越过墙壁放入凹槽。

## 你的身份
你站在绳子的{side}端。
你的搭档{partner_name}站在绳子的{partner_side}端。

## 任务流程（必须严格按照阶段执行）
阶段1 - 就位：两人分别MOVE到绳子各自端点的位置
阶段2 - 抓取：两人各自GRASP自己那端的绳子
阶段3 - 提起：两人**同时**LIFT，将绳子举到最高位置
阶段4 - 越墙：两人协调MOVE_OVER，将绳子水平移到墙的另一侧
阶段5 - 下降：两人**同时**LOWER，将绳子放入凹槽
阶段6 - 释放：两人各自RELEASE绳子

## 技能说明
- MOVE [target]: 移动到指定位置。target: rope_end_left / rope_end_right
- GRASP [target]: 抓住绳子端点。target: rope_end_left / rope_end_right
- LIFT: 将绳子向上提起。必须两人同时执行！
- LOWER: 将绳子下降。必须两人同时执行！
- MOVE_OVER [target]: 带着绳子水平移动到指定位置
- HOLD: 保持当前位置不同。当搭档在调整位置时使用
- RELEASE: 松开绳子

## 同步要求（极其重要！）
- LIFT和LOWER必须两人同时执行！
- 如果只有一人执行，绳子会倾斜掉落，任务失败
- 任何同步动作前，你必须通过对话与搭档确认："准备好LIFT了吗？""准备好了。""3-2-1-LIFT！"
- 一个人说EXECUTE时，两人都必须输出相同的同步动作

## 输出格式
{"robot":"{robot_name}","action":"<技能名>","target":"<目标名>","reason":"<简述>"}
```

#### User Prompt模板

```
## 搬绳任务 - 第{round}轮

## 绳子状态
当前阶段：{current_phase}
左端状态：{left_status}
右端状态：{right_status}
绳子位置：{rope_position}

## 你的状态
位置: {your_pos}
夹爪状态: {gripper_status}

## 搭档状态
{partner_name} 位置: {partner_pos}
{partner_name} 夹爪状态: {partner_gripper_status}

## 搭档消息
{partner_message}

## 当前阶段提示
{phase_hint}

## 请输出你的下一步动作
```

#### 期望输出示例

```
# 阶段1：就位
{"robot":"Alice","action":"MOVE","target":"rope_end_left","reason":"移动到绳子左端准备抓取"}

# 阶段2：抓取
{"robot":"Alice","action":"GRASP","target":"rope_end_left","reason":"已到左端位置，抓住绳子"}

# 阶段3：同步提起（两人必须同时LIFT）
{"robot":"Alice","action":"LIFT","target":"","reason":"与Bob确认完毕，3-2-1同步提起绳子"}

# 阶段4：越墙
{"robot":"Alice","action":"MOVE_OVER","target":"wall_right_side","reason":"将我的这一端越过墙壁"}

# 阶段5：同步下降
{"robot":"Alice","action":"LOWER","target":"","reason":"与Bob确认完毕，3-2-1同步放入凹槽"}

# 阶段6：释放
{"robot":"Alice","action":"RELEASE","target":"","reason":"绳子已在凹槽中，释放左手"}
```

#### 常见失败模式及修复

| 失败模式 | 现象 | 根因 | 修复方法 |
|----------|------|------|----------|
| **LIFT不同步** | 只有一人执行LIFT，绳子倾斜 | LLM没有等待搭档确认 | 在System Prompt中要求"同步动作前必须对话确认"；在对话协议中增加倒计时 |
| **绳子碰墙** | 越过墙壁时绳子碰到墙 | LIFT高度不够 | 在阶段提示中明确"越过前确保绳子在最高位置" |
| **忘记阶段顺序** | 跳过GRASP直接LIFT | LLM跳过了前置步骤 | 在User Prompt中用"阶段"明确标注当前应该做什么 |
| **过早RELEASE** | 绳子还在空中就释放 | LLM以为已经完成 | 在User Prompt中跟踪绳子状态，未在凹槽前禁止RELEASE |

### 7.6 Arrange Cabinet（整理橱柜）

#### 完整System Prompt

```
你是机器人{robot_name}，正在与{all_partners}合作整理橱柜。

## 你的身份
{robot_description}
你可以触及的物体：{reachable_objects}

## 其他机器人
{partner_descriptions}

## 任务流程
阶段1 - 开门：
  - Alice打开并扶住左门（left_door_handle）
  - Bob打开并扶住右门（right_door_handle）
  - Chad等待

阶段2 - 取杯子：
  - Chad从橱柜中取出mug → 放到coaster_mug上
  - Chad从橱柜中取出cup → 放到coaster_cup上
  - Alice和Bob保持扶门不动

阶段3 - 完成：
  - Alice和Bob可以释放门把手

## 技能说明
- OPEN [target]: 打开柜门。target: left_door_handle / right_door_handle
- HOLD: 扶住柜门保持打开状态
- PICK [target]: 拿起物品。target: mug / cup
- PLACE [target]: 放置物品。target: coaster_mug / coaster_cup
- RELEASE: 释放门把手
- WAIT: 等待其他机器人

## 关键约束
1. 门必须一直扶着！如果松手门会自动关闭，导致Chad无法取杯子
2. mug → coaster_mug, cup → coaster_cup，不能放反
3. Alice不能去扶右门，Bob不能去扶左门
4. Chad在所有杯子取出并放好之前，Alice和Bob绝不能RELEASE
5. 每一步都要通过对话协调

## 输出格式
{"robot":"{robot_name}","action":"<技能名>","target":"<目标名>","reason":"<简述>"}
```

#### User Prompt模板

```
## 橱柜任务 - 第{round}轮

## 橱柜状态
左门：{'已打开' if left_open else '关闭'}
右门：{'已打开' if right_open else '关闭'}
柜内物品：{inside}
已取出物品：{placed}

## 门把手持有状态
左门把手：{'Alice持有中' if alice_holding_left else '无人持有'}
右门把手：{'Bob持有中' if bob_holding_right else '无人持有'}

## 你的状态
角色：{your_role}
夹爪：{'持有 ' + holding_what if holding else '空闲'}

## 搭档消息
{partner_message}

## 当前阶段
{current_phase_hint}

## 请输出你的下一步动作
```

#### 期望输出示例

```
# Alice（扶左门）
{"robot":"Alice","action":"OPEN","target":"left_door_handle","reason":"左门还关着，我打开并扶住"}
{"robot":"Alice","action":"HOLD","target":"","reason":"保持左门打开，等Chad取完杯子"}

# Bob（扶右门）
{"robot":"Bob","action":"OPEN","target":"right_door_handle","reason":"右门还关着，我打开并扶住"}
{"robot":"Bob","action":"HOLD","target":"","reason":"保持右门打开"}

# Chad（取杯子）
{"robot":"Chad","action":"WAIT","target":"","reason":"门还没全打开，等一下"}
{"robot":"Chad","action":"PICK","target":"mug","reason":"门都开了，我先取出马克杯"}
{"robot":"Chad","action":"PLACE","target":"coaster_mug","reason":"放到马克杯对应的杯垫"}
{"robot":"Chad","action":"PICK","target":"cup","reason":"再取出茶杯"}
{"robot":"Chad","action":"PLACE","target":"coaster_cup","reason":"放到茶杯对应的杯垫"}

# 完成后
{"robot":"Alice","action":"RELEASE","target":"","reason":"所有杯子已取出，我可以松手了"}
{"robot":"Bob","action":"RELEASE","target":"","reason":"所有杯子已取出，我可以松手了"}
```

#### 常见失败模式及修复

| 失败模式 | 现象 | 根因 | 修复方法 |
|----------|------|------|----------|
| **过早松手** | Chad还没取杯子Alice就RELEASE了 | 没有等待确认机制 | 在System Prompt中增加"Chad发出'all done'信号前，扶门者不得松手" |
| **放错杯垫** | mug放到coaster_cup上 | 目标混淆 | 在User Prompt中每次列出"mug → coaster_mug, cup → coaster_cup"的映射 |
| **门没全开就取** | Chad在门半开时就PICK | 遗漏开门确认 | 在Chad的User Prompt中增加"确认左门和右门都已打开"的条件 |
| **角色混淆** | Alice试图OPEN right_door_handle | LLM忘记自己的角色分工 | 在System Prompt中明确标注每个机器人只能触及的物体列表 |

---

## 八、Prompt版本管理与实验记录

### 8.1 建议的版本命名规范

采用语义化版本命名，便于追踪回溯：

```
格式: {task_name}_v{major}.{minor}_{description}

示例:
  sweep_v1.0_baseline           # 基线版本
  sweep_v1.1_add_cot            # 添加了Chain-of-Thought
  sweep_v2.0_add_fewshot        # 添加了Few-Shot示例
  sweep_v2.1_fix_dump_order     # 修复了DUMP顺序bug
  sweep_v3.0_full_reflection    # 加入完整Self-Reflection机制
  
  sort_v1.0_baseline
  sort_v1.1_add_reachable_zone  # 在User Prompt中增加了可达范围提示
  sort_v2.0_relay_strategy      # 增加了接力策略的few-shot示例
```

### 8.2 实验记录表格模板

#### 单次实验记录

```
| 日期 | 版本 | 任务 | 成功率 | 平均步数 | 主要失败原因 | 备注 |
|------|------|------|--------|----------|-------------|------|
| 05-16 | sweep_v1.0 | SweepFloor | 45% (9/20) | 6.3 | 同步失败(40%),格式错误(30%) | 基线版本 |
| 05-16 | sweep_v1.1 | SweepFloor | 68% (17/25) | 5.8 | 同步失败(60%) | 加了CoT,格式错误解决 |
| 05-17 | sweep_v2.0 | SweepFloor | 75% (15/20) | 5.2 | 碰撞(50%) | 增加了同步确认机制 |
```

#### 跨任务对比记录

```
| 任务 | 最佳版本 | 成功率 | 主要改进手段 |
|------|---------|--------|-------------|
| Sweep Floor | sweep_v2.1 | 78% | CoT + 同步确认 + Few-Shot |
| Make Sandwich | sandwich_v1.2 | 82% | 步骤锁定 + 食材分区 |
| Sort Cubes | sort_v2.0 | 65% | 可达范围明确 + 接力策略 |
| Pack Grocery | pack_v1.1 | 60% | 碰撞避免规则 |
| Move Rope | rope_v2.0 | 55% | 同步协议 + 阶段管理 |
| Arrange Cabinet | cabinet_v1.1 | 70% | 角色固定 + 完成确认 |
```

### 8.3 实验记录自动化

```python
import csv
import os
from datetime import datetime

class ExperimentLogger:
    """实验记录器，自动记录每次运行的prompt版本和成功率"""
    
    def __init__(self, log_file="experiments.csv"):
        self.log_file = log_file
        self._init_csv()
    
    def _init_csv(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'task', 'prompt_version', 
                    'num_runs', 'successes', 'success_rate',
                    'avg_steps', 'main_failure', 'notes'
                ])
    
    def log_experiment(self, task, version, num_runs, successes, 
                       avg_steps, main_failure, notes=""):
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                task,
                version,
                num_runs,
                successes,
                f"{successes/num_runs:.1%}",
                f"{avg_steps:.1f}",
                main_failure,
                notes
            ])


# 使用示例
logger = ExperimentLogger("logs/experiments.csv")
logger.log_experiment(
    task="sweep",
    version="sweep_v1.1_add_cot",
    num_runs=25,
    successes=17,
    avg_steps=5.8,
    main_failure="LIFT同步失败",
    notes="在System Prompt中增加了CoT推理步骤，格式错误率从30%降到0%"
)
```

### 8.4 实验策略建议

```
第1天：基线测试
  ├── 每个任务用最简单的Prompt跑20次
  ├── 记录基线成功率
  └── 分析每个任务的主要失败模式

第2天上午：快速迭代
  ├── 针对主要失败模式逐一修复
  ├── 每改一个地方就跑A/B测试验证
  └── 只保留有正向效果的修改

第2天下午：集成优化
  ├── 将有效修改合并到最终版本
  ├── 每个任务跑50次做最终评测
  └── 输出最终的实验记录表格

第3天（如有）：精细化
  ├── 分析剩余失败案例
  ├── 考虑加入Few-Shot或Self-Reflection
  └── 尝试不同的Ollama模型（7B vs 70B）
```

---

> **核心要点回顾：**
> 1. 这道题的90%工作都在Prompt设计上
> 2. System Prompt要定义角色、技能、格式约束
> 3. User Prompt要动态注入观测、历史、进度
> 4. CoT + JSON格式约束 + 防御性解析 = 高成功率的基础
> 5. 用A/B测试和实验记录表格，数据驱动地迭代Prompt
> 6. 每个任务有自己的失败模式，需要针对性修复
> 7. 好的Prompt是迭代出来的，不是一次写对的——持续记录、持续改进
