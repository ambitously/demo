"""
动作合法性过滤器 - ActionFilter (SayCan风格)
=============================================
借鉴：Google SayCan ("Do As I Can, Not As I Say")
创新点：不让LLM自由创造动作，而是从当前可执行动作列表中选

核心思想：
- LLM只负责"意图选择"（哪个动作最有用）
- 系统负责"可行性判断"（哪些动作可以做）
- 结合两者：有用 ∩ 可行 = 最终动作

解决问题：
- LLM会输出环境不支持的动作（如"飞到月球"）
- LLM不理解机器人的物理限制
- 减少幻觉动作导致的失败

使用方法：
    from action_filter import ActionFilter
    af = ActionFilter(task_name="sweep_floor")
    valid = af.get_valid_actions(observation)
    action_id = af.select_action(valid, observation, llm_client)
"""

from typing import List, Dict, Optional, Tuple, Any


class ActionFilter:
    """SayCan风格动作合法性过滤器"""
    
    # 各任务的预定义动作空间
    TASK_ACTION_SPACES = {
        "sweep_floor": [
            "MOVE {target}",    # 移动到目标位置
            "SWEEP {target}",   # 扫目标物体
            "WAIT",             # 等待
            "DUMP",             # 倒入垃圾桶
        ],
        "sort_cubes": [
            "MOVE {target}",
            "PICK {object}",
            "PLACE {target}",
            "PASS {robot_id}",
            "WAIT",
        ],
        "make_sandwich": [
            "MOVE {target}",
            "PICK {object}",
            "PLACE {target}",
            "WAIT",
        ],
        "pack_grocery": [
            "MOVE {target}",
            "PICK {object}",
            "PLACE {target}",
            "WAIT",
        ],
        "move_rope": [
            "MOVE {target}",
            "GRIP {location}",
            "LIFT {height}",
            "MOVE_PAIR {target}",
            "RELEASE",
            "WAIT",
        ],
        "arrange_cabinet": [
            "MOVE {target}",
            "HOLD_DOOR {side}",
            "PICK {object}",
            "PLACE {target}",
            "WAIT",
            "RELEASE",
        ],
    }
    
    # 机器人能力约束（哪个机器人能做什么）
    ROBOT_CAPABILITIES = {
        "sweep_floor": {
            "rob0": ["MOVE", "SWEEP", "WAIT"],     # 扫帚机器人
            "rob1": ["MOVE", "DUMP", "WAIT"],      # 簸箕机器人
        },
        "sort_cubes": {
            "rob0": ["MOVE", "PICK", "PLACE", "PASS", "WAIT"],
            "rob1": ["MOVE", "PICK", "PLACE", "PASS", "WAIT"],
            "rob2": ["MOVE", "PICK", "PLACE", "PASS", "WAIT"],
        },
        "make_sandwich": {
            "rob0": ["MOVE", "PICK", "PLACE", "WAIT"],
            "rob1": ["MOVE", "PICK", "PLACE", "WAIT"],
        },
        "pack_grocery": {
            "rob0": ["MOVE", "PICK", "PLACE", "WAIT"],
            "rob1": ["MOVE", "PICK", "PLACE", "WAIT"],
        },
        "move_rope": {
            "rob0": ["MOVE", "GRIP", "LIFT", "MOVE_PAIR", "RELEASE", "WAIT"],
            "rob1": ["MOVE", "GRIP", "LIFT", "MOVE_PAIR", "RELEASE", "WAIT"],
        },
        "arrange_cabinet": {
            "rob0": ["MOVE", "HOLD_DOOR", "WAIT", "RELEASE"],
            "rob1": ["MOVE", "HOLD_DOOR", "WAIT", "RELEASE"],
            "rob2": ["MOVE", "PICK", "PLACE", "WAIT"],
        },
    }
    
    def __init__(self, task_name: str):
        """
        Args:
            task_name: 任务名称 (sweep_floor, sort_cubes, etc.)
        """
        self.task_name = task_name
        self.action_space = self.TASK_ACTION_SPACES.get(task_name, [])
        self.capabilities = self.ROBOT_CAPABILITIES.get(task_name, {})
        self.filter_count = 0
        self.blocked_count = 0
    
    def get_action_templates(self) -> List[str]:
        """获取当前任务的动作模板"""
        return self.action_space
    
    def get_robot_actions(self, robot_id: str) -> List[str]:
        """获取指定机器人可执行的动作类型"""
        return self.capabilities.get(robot_id, [])
    
    def generate_valid_actions(
        self,
        observation: Dict,
        robot_id: str = None,
        max_actions: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        根据当前观察生成可执行动作候选列表
        
        这是SayCan的核心：把可执行动作列出来，让LLM从中选
        
        Args:
            observation: 环境观察字典
            robot_id: 指定机器人ID（None则为所有机器人）
            max_actions: 最大候选动作数
        
        Returns:
            [{"action_id": 0, "robot": "rob0", "action": "MOVE blue_cube", "feasibility": 0.9}, ...]
        """
        candidates = []
        action_id = 0
        
        robot_caps = self.capabilities
        if robot_id:
            robot_caps = {robot_id: self.capabilities.get(robot_id, [])}
        
        for rid, caps in robot_caps.items():
            # 1. MOVE动作：移动到环境中可到达的位置
            if "MOVE" in caps:
                targets = self._get_reachable_targets(observation, rid)
                for target in targets:
                    if action_id >= max_actions:
                        break
                    candidates.append({
                        "action_id": action_id,
                        "robot": rid,
                        "action": f"MOVE {target}",
                        "type": "MOVE",
                        "feasibility": 0.95,
                    })
                    action_id += 1
            
            # 2. WAIT动作：总是可行
            if "WAIT" in caps:
                candidates.append({
                    "action_id": action_id,
                    "robot": rid,
                    "action": "WAIT",
                    "type": "WAIT",
                    "feasibility": 1.0,
                })
                action_id += 1
            
            # 3. 任务特定动作
            if self.task_name == "sweep_floor":
                if "SWEEP" in caps:
                    for obj in self._get_objects(observation, "cube"):
                        candidates.append({
                            "action_id": action_id,
                            "robot": rid,
                            "action": f"SWEEP {obj}",
                            "type": "SWEEP",
                            "feasibility": 0.85,
                        })
                        action_id += 1
                if "DUMP" in caps:
                    candidates.append({
                        "action_id": action_id,
                        "robot": rid,
                        "action": "DUMP",
                        "type": "DUMP",
                        "feasibility": 0.9,
                    })
                    action_id += 1
            
            elif self.task_name in ("sort_cubes", "make_sandwich", "pack_grocery", "arrange_cabinet"):
                if "PICK" in caps:
                    for obj in self._get_objects(observation, "object"):
                        candidates.append({
                            "action_id": action_id,
                            "robot": rid,
                            "action": f"PICK {obj}",
                            "type": "PICK",
                            "feasibility": 0.9,
                        })
                        action_id += 1
                if "PLACE" in caps:
                    for target in self._get_reachable_targets(observation, rid):
                        candidates.append({
                            "action_id": action_id,
                            "robot": rid,
                            "action": f"PLACE {target}",
                            "type": "PLACE",
                            "feasibility": 0.9,
                        })
                        action_id += 1
            
            elif self.task_name == "move_rope":
                if "GRIP" in caps:
                    for loc in ["rope_left", "rope_right"]:
                        candidates.append({
                            "action_id": action_id,
                            "robot": rid,
                            "action": f"GRIP {loc}",
                            "type": "GRIP",
                            "feasibility": 0.85,
                        })
                        action_id += 1
                if "LIFT" in caps:
                    for h in ["medium", "high"]:
                        candidates.append({
                            "action_id": action_id,
                            "robot": rid,
                            "action": f"LIFT {h}",
                            "type": "LIFT",
                            "feasibility": 0.85,
                        })
                        action_id += 1
        
        return candidates[:max_actions]
    
    def build_filtered_prompt(
        self,
        task_description: str,
        observation: Dict,
        robot_id: str = None,
    ) -> str:
        """
        构建SayCan风格的受限Prompt
        
        这是创新核心：prompt中包含"你只能从以下动作中选择"
        """
        valid_actions = self.generate_valid_actions(observation, robot_id)
        
        action_list = "\n".join([
            f"{a['action_id']}. [{a['robot']}] {a['action']} (可行性:{a['feasibility']:.0%})"
            for a in valid_actions
        ])
        
        prompt = f"""## 任务
{task_description}

## 当前状态
{self._format_obs(observation)}

## 可用动作（你只能从这些动作中选择，不得创造新动作）
{action_list}

## 请选择最佳动作
输出格式：
<THINK>分析当前状态和最佳动作选择</THINK>
<ACTIONS>
{"action_id": 选中动作的编号}
</ACTIONS>

注意：
- 请考虑每个动作的可行性评分
- 优先选择可行性高的动作
- 如果两个机器人可能冲突，让一个WAIT"""
        
        return prompt
    
    def parse_filtered_response(self, response: str) -> Optional[int]:
        """从SayCan响应中提取action_id"""
        import re
        # 匹配 {"action_id": 数字}
        match = re.search(r'"action_id"\s*:\s*(\d+)', response)
        if match:
            return int(match.group(1))
        # 匹配 action_id: 数字
        match = re.search(r'action_id:\s*(\d+)', response, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None
    
    def select_action(
        self,
        valid_actions: List[Dict],
        observation: Dict,
        llm_client=None,
    ) -> Optional[Dict]:
        """
        从候选动作中选择最佳动作
        
        Args:
            valid_actions: 候选动作列表
            observation: 环境观察
            llm_client: OllamaClient实例（可选，用于LLM选择）
        
        Returns:
            选中的动作字典，或None
        """
        self.filter_count += 1
        
        if not valid_actions:
            return None
        
        # 如果没有LLM客户端，返回可行性最高的动作
        if llm_client is None:
            return max(valid_actions, key=lambda a: a.get("feasibility", 0))
        
        # 用LLM选择
        action_list = "\n".join([
            f"{a['action_id']}. [{a['robot']}] {a['action']}"
            for a in valid_actions[:10]  # 限制候选数
        ])
        
        prompt = f"""从以下可用动作中选择最佳动作。

当前状态: {self._format_obs(observation)}

可用动作:
{action_list}

请只回复选中动作的ID数字，不要其他内容。"""
        
        try:
            response = llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            action_id = self.parse_filtered_response(response)
            if action_id is not None:
                for a in valid_actions:
                    if a["action_id"] == action_id:
                        return a
        except Exception:
            pass
        
        # 回退：选第一个
        return valid_actions[0]
    
    def _get_reachable_targets(self, obs: Dict, robot_id: str) -> List[str]:
        """从observation中提取可达目标"""
        targets = []
        # 通用目标
        for key in ["objects", "cubes", "items", "targets", "positions"]:
            if key in obs:
                val = obs[key]
                if isinstance(val, list):
                    targets.extend([str(v) for v in val])
                elif isinstance(val, dict):
                    targets.extend([str(k) for k in val.keys()])
        if not targets:
            targets = ["table", "bin", "cabinet", "plate"]
        return targets
    
    def _get_objects(self, obs: Dict, obj_type: str = "object") -> List[str]:
        """从observation中提取物体列表"""
        for key in ["cube_states", "objects", "cubes", "items", "ingredients"]:
            if key in obs:
                val = obs[key]
                if isinstance(val, list):
                    return [str(v) for v in val]
                elif isinstance(val, dict):
                    return [str(k) for k in val.keys()]
        return ["unknown_object"]
    
    def _format_obs(self, obs: Dict) -> str:
        """格式化observation为文本"""
        parts = []
        for k, v in obs.items():
            if isinstance(v, dict):
                v = ", ".join(f"{kk}: {vv}" for kk, vv in v.items())
            parts.append(f"{k}: {v}")
        return "\n".join(parts)
    
    def get_stats(self) -> Dict:
        return {
            "task": self.task_name,
            "total_filters": self.filter_count,
            "blocked_invalid": self.blocked_count,
        }
    
    def is_action_valid(self, action_str: str, robot_id: str = None) -> bool:
        """检查一个动作字符串是否合法"""
        self.filter_count += 1
        
        # 提取动作类型（第一个单词）
        parts = action_str.strip().split()
        if not parts:
            return False
        
        action_type = parts[0].upper()
        
        # 检查是否在动作空间中
        valid_types = set()
        for template in self.action_space:
            valid_types.add(template.split()[0])
        
        if action_type not in valid_types:
            self.blocked_count += 1
            return False
        
        # 检查机器人是否有权限执行此动作
        if robot_id and robot_id in self.capabilities:
            if action_type not in self.capabilities[robot_id]:
                self.blocked_count += 1
                return False
        
        return True


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ActionFilter (SayCan风格) 测试")
    print("=" * 60)
    
    # 测试 Sweep Floor
    af = ActionFilter("sweep_floor")
    
    print(f"\n📋 任务: sweep_floor")
    print(f"   动作空间: {af.get_action_templates()}")
    print(f"   rob0能力: {af.get_robot_actions('rob0')}")
    print(f"   rob1能力: {af.get_robot_actions('rob1')}")
    
    # 测试动作合法性
    print("\n🔍 动作合法性检查:")
    tests = [
        ("MOVE blue_cube", "rob0", True),
        ("SWEEP blue_cube", "rob0", True),
        ("DUMP", "rob0", False),        # rob0不能DUMP
        ("DUMP", "rob1", True),          # rob1可以DUMP
        ("FLY moon", "rob0", False),     # 非法动作
        ("WAIT", "rob0", True),
    ]
    for action, rid, expected in tests:
        result = af.is_action_valid(action, rid)
        status = "✅" if result == expected else "❌"
        print(f"   {status} {rid}: {action} -> {'合法' if result else '非法'} (期望{'合法' if expected else '非法'})")
    
    # 测试候选动作生成
    print("\n📊 候选动作生成:")
    test_obs = {
        "cube_states": {"red_cube": "on_table", "blue_cube": "on_table"},
        "robot_positions": {"rob0": "(0.5, 0.3)", "rob1": "(0.2, 0.8)"},
    }
    candidates = af.generate_valid_actions(test_obs, max_actions=10)
    print(f"   生成了 {len(candidates)} 个候选动作:")
    for c in candidates[:5]:
        print(f"   {c['action_id']}. [{c['robot']}] {c['action']} (可行性:{c['feasibility']:.0%})")
    
    # 测试SayCan Prompt
    print("\n📝 SayCan格式Prompt:")
    prompt = af.build_filtered_prompt("清扫地板上的所有方块", test_obs)
    print(f"   {prompt[:300]}...")
    
    print(f"\n📊 统计: {af.get_stats()}")
    print("\n✅ ActionFilter 测试完成！")
