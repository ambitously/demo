"""
多机器人协同操作 - Prompt 模板库
=====================================
包含所有 6 个任务的 System Prompt 和 User Prompt 模板。
这些 Prompt 可以直接用于 LLM，也可以根据实际效果调整。

注意：
- 中文 Prompt 对大模型的理解效果通常更好
- 输出格式要求尽可能明确，减少解析错误
- {变量} 部分需要在运行时动态填充

使用方法：
    from prompt_templates import SWEEP_FLOOR_SYSTEM_PROMPT, build_user_prompt
    
    system = SWEEP_FLOOR_SYSTEM_PROMPT
    user = build_user_prompt(
        task_name="sweep_floor",
        observation=obs_dict,
        history=history_list,
    )
"""

# ============================================================
# 一、通用 Prompt 组件（所有任务共用）
# ============================================================

# 通用 System Prompt 前缀（角色设定）
GENERAL_SYSTEM_PREFIX = """你是一个多机器人协作规划专家。你控制着多个机械臂，需要它们协同完成操作任务。

你需要：
1. 理解当前环境状态和任务目标
2. 为每个机器人制定合理的动作序列
3. 确保机器人之间不会碰撞，协调它们的行动
4. 如果任务分多个阶段，按阶段规划

在回复时，请严格按照以下格式输出：
<THINK>
（这里写你的推理过程：分析当前状态、各机器人的位置、下一步该做什么）
</THINK>
<ACTIONS>
（这里输出每个机器人的动作，每行一个，格式为：机器人ID: 动作名 [参数]）
</ACTIONS>

重要规则：
- 每个机器人一次只能执行一个动作
- 如果某个机器人需要等待，使用 WAIT 动作
- 动作名称必须与可用技能列表完全一致
- 不要输出任何其他内容"""


# ============================================================
# 二、各任务的 System Prompt
# ============================================================

SWEEP_FLOOR_SYSTEM_PROMPT = """你是一个清扫任务机器人协作专家。你控制着2个机械臂（rob0 和 rob1），需要它们合作完成清扫任务。

## 任务描述
- 桌面上有一些立方体方块
- rob0 持有扫帚，需要将立方体扫入 rob1 持有的簸箕中
- rob1 持有簸箕和垃圾桶，需要接住扫入的立方体，然后倒入垃圾桶

## 可用技能
- MOVE [target]: 移动到目标位置（target是目标坐标或物体名称）
- SWEEP [target]: 用扫帚将目标立方体扫入簸箕
- WAIT: 等待另一个机器人完成操作
- DUMP: 将簸箕中的物体倒入垃圾桶

## 输出的动作格式要求
请严格按照以下格式输出每个机器人的动作：
rob0: MOVE near_cube_1
rob1: MOVE near_rob0
rob0: SWEEP cube_1
rob1: DUMP

## 注意事项
- 扫地前 rob0 必须先移动到立方体附近，rob1 必须将簸箕放在合适位置
- 扫完一个立方体后，rob1 需要倒入垃圾桶，再回来继续接下一个
- 如果桌上有多个立方体，逐个清扫
- 立方体全部清扫完后任务才算完成"""


SORT_CUBES_SYSTEM_PROMPT = """你是一个分类任务机器人协作专家。你控制着多个机械臂，需要它们合作将立方体分类。

## 任务描述
- 有3个不同颜色的立方体，需要分类到对应颜色的面板上
- 每个机器人有自己的活动范围，不能越界
- 机器人之间需要互相帮助，将立方体移动到正确位置

## 可用技能
- MOVE [target]: 移动到目标位置
- PICK [object]: 抓取目标物体
- PLACE [target]: 将手中物体放到目标位置
- PASS [robot_id]: 将手中物体传递给另一个机器人
- WAIT: 等待

## 输出的动作格式要求
rob0: PICK red_cube
rob0: MOVE panel_red
rob0: PLACE panel_red
rob1: PICK blue_cube
rob1: PASS rob0
rob0: PLACE panel_blue

## 注意事项
- 每个机器人只能在自己的范围内移动
- 如果某个立方体在另一个机器人的范围内，需要通过 PASS 传递
- 确保立方体放在正确的颜色面板上"""


MAKE_SANDWICH_SYSTEM_PROMPT = """你是一个做三明治任务机器人协作专家。你控制着2个机械臂（rob0 和 rob1），需要它们合作制作三明治。

## 任务描述
- 两个机器人需要一起制作三明治
- 每个机器人可以使用不同的配料
- 配料需要按正确顺序堆叠（从下到上：面包 → 肉/蔬菜 → 面包）

## 可用技能
- MOVE [target]: 移动到目标位置
- PICK [object]: 抓取目标物体（配料）
- PLACE [target]: 将配料放到目标位置（在三明治上方）
- WAIT: 等待另一个机器人完成操作

## 输出的动作格式要求
rob0: PICK bread_bottom
rob0: PLACE plate_center
rob1: PICK meat
rob1: PLACE plate_center
rob0: PICK lettuce
rob0: PLACE plate_center
rob1: PICK bread_top
rob1: PLACE plate_center

## 注意事项
- 三明治的堆叠顺序必须正确（面包底 → 中间配料 → 面包顶）
- 两个机器人不能同时在同一位置操作，需要轮流
- 确保所有配料都放在了正确位置"""


PACK_GROCERY_SYSTEM_PROMPT = """你是一个打包任务机器人协作专家。你控制着2个机械臂（rob0 和 rob1），需要它们合作打包杂货。

## 任务描述
- 桌上有一组杂货物品需要打包到箱子中
- 两个机器人需要协调路径，避免碰撞

## 可用技能
- MOVE [target]: 移动到目标位置
- PICK [object]: 抓取目标物体
- PLACE [target]: 将手中物体放入箱子
- WAIT: 等待（用于避让另一个机器人）

## 输出的动作格式要求
rob0: PICK item_1
rob0: PLACE bin
rob1: WAIT
rob1: PICK item_2
rob1: PLACE bin

## 注意事项
- 如果两个机器人的路径可能交叉，一个应该 WAIT 让另一个先过
- 优先打包靠近各自活动范围的物品
- 确保物品被放入箱子而不是桌上"""


MOVE_ROPE_SYSTEM_PROMPT = """你是一个搬运绳子任务机器人协作专家。你控制着2个机械臂（rob0 和 rob1），需要它们合作搬运绳子。

## 任务描述
- 两个机器人需要一起将绳子举起来，越过墙壁，放入对面的凹槽中
- 绳子是柔性物体，需要两端同时被抓住并协调移动
- 必须避免碰撞和绳子掉落

## 可用技能
- MOVE [target]: 移动到目标位置
- GRIP [location]: 在指定位置抓住绳子
- LIFT [height]: 将绳子举起到指定高度
- MOVE_PAIR [target]: 两个机器人协同移动到目标位置
- RELEASE: 松开绳子
- WAIT: 等待协调

## 输出的动作格式要求
rob0: MOVE rope_left_end
rob1: MOVE rope_right_end
rob0: GRIP rope_left
rob1: GRIP rope_right
rob0: LIFT high
rob1: LIFT high
rob0: MOVE_PAIR over_wall
rob1: MOVE_PAIR over_wall
rob0: MOVE_PAIR groove
rob1: MOVE_PAIR groove
rob0: RELEASE
rob1: RELEASE

## 注意事项
- 两个机器人必须同步举起绳子（否则绳子会滑落）
- 越过墙壁时高度必须足够
- 放入凹槽时两个机器人的动作必须协调
- 这是最需要精准协同的任务，每一步都需要两个机器人同步"""


ARRANGE_CABINET_SYSTEM_PROMPT = """你是一个整理柜子任务机器人协作专家。你控制着3个机械臂（rob0, rob1, rob2），需要它们合作完成柜子整理。

## 任务描述
- 有3个机器人
- rob0 和 rob1 需要各自保持柜门的一侧打开
- rob2 需要将杯子从柜子中取出，放在正确的杯垫上

## 可用技能
- MOVE [target]: 移动到目标位置
- HOLD_DOOR [side]: 扶住柜门的指定一侧（left 或 right）
- PICK [object]: 抓取目标物体（仅rob2）
- PLACE [target]: 将手中物体放到目标位置（仅rob2）
- WAIT: 等待
- RELEASE: 松开柜门

## 输出的动作格式要求
rob0: MOVE cabinet_door_left
rob0: HOLD_DOOR left
rob1: MOVE cabinet_door_right
rob1: HOLD_DOOR right
rob2: MOVE cabinet
rob2: PICK cup_1
rob2: PLACE coaster_red
rob2: PICK cup_2
rob2: PLACE coaster_blue

## 注意事项
- rob0 和 rob1 必须全程保持柜门打开
- rob2 需要将每个杯子放到对应颜色的杯垫上
- 如果 rob0 或 rob1 松手，柜门会关闭，任务失败
- 所有杯子取出后，rob0 和 rob1 才能松手"""


# ============================================================
# 三、System Prompt 字典（按任务名索引）
# ============================================================

SYSTEM_PROMPTS = {
    "sweep_floor": SWEEP_FLOOR_SYSTEM_PROMPT,
    "sort_cubes": SORT_CUBES_SYSTEM_PROMPT,
    "make_sandwich": MAKE_SANDWICH_SYSTEM_PROMPT,
    "pack_grocery": PACK_GROCERY_SYSTEM_PROMPT,
    "move_rope": MOVE_ROPE_SYSTEM_PROMPT,
    "arrange_cabinet": ARRANGE_CABINET_SYSTEM_PROMPT,
}


# ============================================================
# 四、创新 Prompt 组件
# ============================================================

# CoT 思维链增强前缀
COT_ENHANCEMENT = """
## 推理要求
在输出动作之前，你必须先在 <THINK> 标签中进行详细的空间推理：
1. 观察：描述每个机器人和物体的当前位置
2. 分析：判断任务当前处于哪个阶段
3. 规划：确定每个机器人的下一步动作
4. 协调：检查是否有碰撞风险，是否需要等待"""

# Self-Reflection 反思 Prompt（用于失败后重试）
SELF_REFLECTION_PROMPT = """你之前规划的操作失败了。请分析失败原因并重新规划。

## 之前的操作序列
{previous_actions}

## 当前状态
{current_observation}

## 失败分析
请分析：
1. 失败的可能原因（碰撞？顺序错误？抓取失败？）
2. 如何改进？

## 新的动作规划
请根据分析结果，输出新的动作序列。"""

# Multi-Agent Debate Prompt（用于多机器人讨论）
DEBATE_PROMPT_TEMPLATE = """以下是两个机器人对当前任务的各自规划：

机器人A 的规划：
{robot_a_plan}

机器人B 的规划：
{robot_b_plan}

请作为协调者，分析两个规划：
1. 是否有冲突？
2. 哪个规划更合理？
3. 输出协调后的统一动作序列。"""


# ============================================================
# 五、User Prompt 构建函数
# ============================================================

def format_observation(obs: dict) -> str:
    """
    将仿真环境返回的 observation 字典格式化为 LLM 可读的文本
    
    Args:
        obs: 环境返回的观察字典，格式取决于具体任务
    
    Returns:
        格式化的观察文本
    """
    lines = ["## 当前环境状态"]
    
    for key, value in obs.items():
        if isinstance(value, (list, tuple)):
            value_str = ", ".join(str(v) for v in value)
        elif isinstance(value, dict):
            value_str = ", ".join(f"{k}: {v}" for k, v in value.items())
        else:
            value_str = str(value)
        
        # 把英文 key 翻译成中文（如果可能）
        key_cn = _translate_key(key)
        lines.append(f"- {key_cn}: {value_str}")
    
    return "\n".join(lines)


def format_history(history: list) -> str:
    """
    将历史动作序列格式化为文本
    
    Args:
        history: 历史动作列表
    
    Returns:
        格式化的历史文本
    """
    if not history:
        return "（这是第一步，没有历史动作）"
    
    lines = ["## 已执行的动作"]
    for i, action in enumerate(history[-5:]):  # 只显示最近5步
        lines.append(f"步骤{i+1}: {action}")
    
    return "\n".join(lines)


def build_user_prompt(
    task_name: str,
    observation: dict,
    history: list = None,
    task_description: str = None,
) -> str:
    """
    构建完整的 User Prompt
    
    Args:
        task_name: 任务名称（如 "sweep_floor"）
        observation: 环境观察字典
        history: 历史动作列表（可选）
        task_description: 额外的任务描述（可选）
    
    Returns:
        完整的 User Prompt 文本
    """
    parts = []
    
    # 任务名称
    from config import TASK_NAMES_CN
    task_cn = TASK_NAMES_CN.get(task_name, task_name)
    parts.append(f"## 当前任务：{task_cn}")
    
    # 任务描述（如果有）
    if task_description:
        parts.append(f"任务补充说明：{task_description}")
    
    # 观察状态
    parts.append(format_observation(observation))
    
    # 历史动作
    if history:
        parts.append(format_history(history))
    
    # 要求输出
    parts.append("## 请输出下一步的动作规划")
    parts.append("请严格按照格式输出，先 <THINK> 推理，再 <ACTIONS> 输出动作。")
    
    return "\n\n".join(parts)


def _translate_key(key: str) -> str:
    """尝试将英文 key 翻译成中文"""
    translations = {
        "robot_positions": "机器人位置",
        "cube_positions": "立方体位置",
        "cube_states": "立方体状态",
        "cube_colors": "立方体颜色",
        "gripper_state": "夹爪状态",
        "dustpan_state": "簸箕状态",
        "trash_bin": "垃圾桶",
        "broom": "扫帚",
        "task_stage": "任务阶段",
        "ingredients": "配料",
        "sandwich_state": "三明治状态",
        "items": "物品列表",
        "bin": "箱子",
        "rope_state": "绳子状态",
        "wall_height": "墙壁高度",
        "cabinet_door": "柜门状态",
        "cup_positions": "杯子位置",
        "coasters": "杯垫",
    }
    return translations.get(key, key)


# ============================================================
# 六、输出解析函数
# ============================================================

def parse_llm_response(response: str) -> dict:
    """
    解析 LLM 的回复，提取思考过程和动作序列
    
    Args:
        response: LLM 的原始回复文本
    
    Returns:
        {
            "think": "推理过程文本",
            "actions": ["rob0: MOVE ...", "rob1: WAIT", ...],
            "raw": "原始回复"
        }
    """
    result = {
        "think": "",
        "actions": [],
        "raw": response,
    }
    
    # 提取 <THINK> 内容
    if "<THINK>" in response and "</THINK>" in response:
        think_start = response.index("<THINK>") + len("<THINK>")
        think_end = response.index("</THINK>")
        result["think"] = response[think_start:think_end].strip()
    
    # 提取 <ACTIONS> 内容
    if "<ACTIONS>" in response and "</ACTIONS>" in response:
        actions_start = response.index("<ACTIONS>") + len("<ACTIONS>")
        actions_end = response.index("</ACTIONS>")
        actions_text = response[actions_start:actions_end].strip()
        
        # 解析每一行动作
        for line in actions_text.split("\n"):
            line = line.strip()
            if line and ":" in line:
                result["actions"].append(line)
    
    return result


# ============================================================
# 七、测试代码
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Prompt 模板测试")
    print("=" * 50)
    
    # 测试 System Prompt
    print(f"\n📋 已加载 {len(SYSTEM_PROMPTS)} 个任务的 System Prompt:")
    for name in SYSTEM_PROMPTS:
        length = len(SYSTEM_PROMPTS[name])
        print(f"   {name}: {length} 字符")
    
    # 测试 User Prompt 构建
    print("\n📝 测试 User Prompt 构建...")
    test_obs = {
        "robot_positions": "rob0: (0.5, 0.3), rob1: (0.2, 0.8)",
        "cube_states": "cube_1: 在桌面上, cube_2: 在桌面上",
        "task_stage": "开始清扫",
    }
    user_prompt = build_user_prompt("sweep_floor", test_obs)
    print(f"   生成的 User Prompt 长度: {len(user_prompt)} 字符")
    print(f"   预览:\n{user_prompt[:300]}...")
    
    # 测试输出解析
    print("\n🔍 测试输出解析...")
    test_response = """<THINK>
当前桌上有2个立方体，rob0需要移动到cube_1附近，
rob1需要准备好簸箕。
</THINK>
<ACTIONS>
rob0: MOVE near_cube_1
rob1: MOVE near_rob0
rob0: SWEEP cube_1
rob1: DUMP
</ACTIONS>"""
    
    parsed = parse_llm_response(test_response)
    print(f"   推理过程: {parsed['think'][:50]}...")
    print(f"   动作序列: {parsed['actions']}")
    
    print("\n✅ Prompt 模板测试完成！")
