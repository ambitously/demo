"""
角色分配器 - RoleAssigner
============================
借鉴：RoCo多机器人对话 / AutoGen role-based agents / Contract Net Protocol
创新点：每轮先决定机器人职责，减少抢物体/同时移动冲突

解决问题：
- 两个机器人同时抓同一物体→冲突
- 两个机器人同时移动到同一位置→碰撞
- 不知道谁该主动谁该等待→效率低

使用方法：
    from role_assigner import RoleAssigner
    ra = RoleAssigner("sweep_floor")
    roles = ra.assign_roles(observation, llm_client)
"""

from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


class RobotRole(Enum):
    PRIMARY = "primary"         # 主要操作员
    SECONDARY = "secondary"     # 辅助操作员
    ASSISTANT = "assistant"     # 助手
    WAITER = "waiter"           # 等待者
    LEFT_HOLDER = "left_holder" # 左门扶住者
    RIGHT_HOLDER = "right_holder" # 右门扶住者
    GRABBER = "grabber"         # 抓取者
    SWEEPER = "sweeper"         # 清扫者
    DUSTPAN = "dustpan"         # 簸箕持有者
    LEFT_GRIPPER = "left_gripper"   # 左侧抓绳者
    RIGHT_GRIPPER = "right_gripper" # 右侧抓绳者


class RoleAssigner:
    """动态角色分配器"""
    
    # 预定义角色模板（每任务、每机器人）
    PREDEFINED_ROLES = {
        "sweep_floor": {
            "rob0": {"role": "sweeper", "duty": "拿扫帚，移动到方块旁，执行SWEEP"},
            "rob1": {"role": "dustpan", "duty": "拿簸箕，移动到rob0对面，接住方块，执行DUMP"},
        },
        "sort_cubes": {
            "rob0": {"role": "sorter_left", "duty": "处理面板1-3范围内的立方体，可接收传递"},
            "rob1": {"role": "sorter_mid", "duty": "处理面板3-5范围内的立方体，可传递"},
            "rob2": {"role": "sorter_right", "duty": "处理面板5-7范围内的立方体，可传递"},
        },
        "make_sandwich": {
            "rob0": {"role": "chef_left", "duty": "拿取左边食材（面包底、蔬菜），轮流放置"},
            "rob1": {"role": "chef_right", "duty": "拿取右边食材（肉、奶酪、面包顶），轮流放置"},
        },
        "pack_grocery": {
            "rob0": {"role": "packer_left", "duty": "打包左侧物品到箱子，注意避让rob1"},
            "rob1": {"role": "packer_right", "duty": "打包右侧物品到箱子，注意避让rob0"},
        },
        "move_rope": {
            "rob0": {"role": "left_gripper", "duty": "移动到绳子左端，抓住，与rob1同步举起和移动"},
            "rob1": {"role": "right_gripper", "duty": "移动到绳子右端，抓住，与rob0同步举起和移动"},
        },
        "arrange_cabinet": {
            "rob0": {"role": "left_door_holder", "duty": "扶住柜子左门保持打开，直到所有杯子取出"},
            "rob1": {"role": "right_door_holder", "duty": "扶住柜子右门保持打开，直到所有杯子取出"},
            "rob2": {"role": "cup_retriever", "duty": "从柜子中取出杯子，放到对应颜色杯垫上"},
        },
    }
    
    def __init__(self, task_name: str, llm_client=None):
        """
        Args:
            task_name: 任务名称
            llm_client: OllamaClient实例(可选，用于LLM动态分配)
        """
        self.task_name = task_name
        self.llm_client = llm_client
        self.current_roles: Dict[str, Dict] = {}
        
        # 加载预定义角色
        if task_name in self.PREDEFINED_ROLES:
            self.current_roles = {
                rid: dict(role_info)
                for rid, role_info in self.PREDEFINED_ROLES[task_name].items()
            }
    
    def assign_roles(self, observation: Dict = None) -> Dict[str, Dict]:
        """
        分配角色
        
        先尝试预定义模板，如果观察变化较大则用LLM动态调整
        
        Returns:
            {robot_id: {"role": "sweeper", "duty": "..."}, ...}
        """
        # 如果观察变化大且LLM可用，动态调整
        if observation and self.llm_client:
            try:
                dynamic = self._dynamic_assign(observation)
                if dynamic:
                    self.current_roles = dynamic
            except Exception:
                pass
        
        return self.current_roles
    
    def _dynamic_assign(self, observation: Dict) -> Optional[Dict[str, Dict]]:
        """使用LLM动态分配角色"""
        if not self.llm_client:
            return None
        
        # 从预定义中获取机器人列表
        robots = list(self.PREDEFINED_ROLES.get(self.task_name, {}).keys())
        if not robots:
            return None
        
        prompt = f"""请根据当前状态为机器人分配角色。

任务：{self.task_name}
机器人：{', '.join(robots)}
当前状态：{observation}

请为每个机器人分配一个明确的角色和职责。
输出格式（JSON）：
{{
  "roles": {{
    "rob0": {{"role": "角色名", "duty": "具体职责", "priority": 1}},
    "rob1": {{"role": "角色名", "duty": "具体职责", "priority": 2}}
  }}
}}

注意：
- 优先级1最高，数字越小越优先执行
- 根据机器人当前位置分配最合适的角色"""

        try:
            response = self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300,
            )
            import json, re
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return data.get("roles", {})
        except Exception:
            pass
        
        return None
    
    def get_role(self, robot_id: str) -> Optional[Dict]:
        """获取指定机器人的角色"""
        return self.current_roles.get(robot_id)
    
    def get_duty(self, robot_id: str) -> str:
        """获取指定机器人的职责描述"""
        role_info = self.get_role(robot_id)
        if role_info:
            return role_info.get("duty", "无特定职责")
        return "自由行动"
    
    def get_priority(self, robot_id: str) -> int:
        """获取机器人的优先级（数字越小越高）"""
        role_info = self.get_role(robot_id)
        if role_info:
            return role_info.get("priority", 5)
        return 5
    
    def get_robots_by_role(self, role_name: str) -> List[str]:
        """根据角色名查找机器人"""
        return [
            rid for rid, info in self.current_roles.items()
            if info.get("role") == role_name
        ]
    
    def build_role_prompt_context(self) -> str:
        """构建包含角色信息的Prompt上下文"""
        if not self.current_roles:
            return ""
        
        lines = ["## 机器人角色分配"]
        for rid, info in self.current_roles.items():
            lines.append(f"- {rid}: **{info['role']}** — {info['duty']}")
        
        lines.append("\n注意：严格按照角色分工执行，不要越俎代庖。")
        return "\n".join(lines)
    
    def detect_role_conflict(self, action_dict: Dict[str, str]) -> List[str]:
        """
        检测角色冲突
        
        例如：两个机器人都想抓同一个物体
        """
        conflicts = []
        
        # 检查是否有两个机器人同时做相同动作
        targets = {}
        for rid, action in action_dict.items():
            role = self.get_role(rid)
            parts = action.split()
            if len(parts) >= 2:
                target = parts[-1]
                if target in targets:
                    conflicts.append(
                        f"冲突: {rid}({role['role'] if role else '?'}) 和 "
                        f"{targets[target][0]}({targets[target][1]}) "
                        f"都要操作 {target}"
                    )
                else:
                    targets[target] = (rid, role['role'] if role else '?')
        
        return conflicts
    
    def resolve_conflict(
        self,
        conflicts: List[str],
        action_dict: Dict[str, str],
        llm_client=None,
    ) -> Dict[str, str]:
        """
        解决角色冲突
        
        策略：优先级高的先执行，低的改为WAIT
        """
        if not conflicts:
            return action_dict
        
        # 简单策略：找出冲突的机器人对，让优先级低的等待
        resolved = dict(action_dict)
        
        for conflict in conflicts:
            # 从冲突描述中提取机器人ID
            for rid in self.current_roles:
                if rid in conflict:
                    # 检查是否有另一个机器人也在冲突中
                    for rid2 in self.current_roles:
                        if rid2 != rid and rid2 in conflict:
                            p1 = self.get_priority(rid)
                            p2 = self.get_priority(rid2)
                            if p1 > p2:
                                resolved[rid] = f"{rid}: WAIT"
                            elif p2 > p1:
                                resolved[rid2] = f"{rid2}: WAIT"
        
        return resolved
    
    def get_all_roles_summary(self) -> str:
        """获取所有角色摘要（用于报告）"""
        if not self.current_roles:
            return "未分配角色"
        
        lines = [f"任务 {self.task_name} 角色分配:"]
        for rid, info in self.current_roles.items():
            lines.append(f"  {rid} → {info['role']}: {info['duty']}")
        return "\n".join(lines)


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RoleAssigner 测试")
    print("=" * 60)
    
    for task_name in ["sweep_floor", "make_sandwich", "arrange_cabinet", "move_rope"]:
        print(f"\n{'='*40}")
        print(f"📋 任务: {task_name}")
        
        ra = RoleAssigner(task_name)
        roles = ra.assign_roles()
        
        for rid, info in roles.items():
            print(f"   {rid}: {info['role']} — {info['duty']}")
        
        # 测试冲突检测
        print(f"\n   角色Prompt上下文:")
        print(f"   {ra.build_role_prompt_context()[:200]}...")
    
    # 测试冲突检测
    print(f"\n{'='*40}")
    print("🔍 冲突检测测试:")
    ra = RoleAssigner("sweep_floor")
    ra.assign_roles()
    
    action_dict = {
        "rob0": "MOVE red_cube",
        "rob1": "MOVE red_cube",  # 冲突！两个都要去同一个方块
    }
    conflicts = ra.detect_role_conflict(action_dict)
    print(f"   冲突列表: {conflicts}")
    
    resolved = ra.resolve_conflict(conflicts, action_dict)
    print(f"   解决后: {resolved}")
    
    # 无冲突情况
    action_dict2 = {
        "rob0": "SWEEP red_cube",
        "rob1": "WAIT",
    }
    conflicts2 = ra.detect_role_conflict(action_dict2)
    print(f"   无冲突: {conflicts2}")
    
    print("\n✅ RoleAssigner 测试完成！")
