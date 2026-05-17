"""
任务分解器 - TaskDecomposer
=============================
借鉴：LLM-Planner / CoT / Hierarchical Planning
创新点：把长程任务拆成子目标，逐步执行，避免一次性规划崩溃

解决问题：
- 长程任务(8-16步)一次性规划LLM容易"忘掉后面"
- 执行到一半出错了不知道从哪恢复
- 多阶段任务(如Arrange Cabinet:开门→取杯→放杯)需要明确阶段划分

使用方法：
    from task_decomposer import TaskDecomposer
    td = TaskDecomposer()
    subgoals = td.decompose("sweep_floor", observation)
    current = td.get_current_subgoal()
    td.advance()  # 完成当前子目标，进入下一个
"""

from typing import List, Dict, Optional, Tuple, Any
from enum import Enum


class SubgoalStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskDecomposer:
    """任务分解器：长程任务 → 子目标序列"""
    
    # 预定义的子目标分解模板（不依赖LLM，更快更稳定）
    PREDEFINED_SUBGOALS = {
        "sweep_floor": [
            {"id": 0, "name": "定位所有方块", "description": "观察桌面上有哪些方块需要清扫", "criteria": "已识别所有方块位置"},
            {"id": 1, "name": "清扫方块", "description": "逐个将方块扫入簸箕并倒入垃圾桶", "criteria": "所有方块已倒入垃圾桶"},
            {"id": 2, "name": "任务完成", "description": "确认所有方块已清理", "criteria": "桌面无方块"},
        ],
        "sort_cubes": [
            {"id": 0, "name": "观察分类", "description": "观察立方体位置和颜色，确定各立方体目标面板", "criteria": "已确定分类方案"},
            {"id": 1, "name": "就近处理", "description": "各机器人处理自己范围内的立方体", "criteria": "各机器人范围内立方体已处理"},
            {"id": 2, "name": "传递协作", "description": "将需要传递的立方体通过PASS传递给目标机器人", "criteria": "所有立方体在正确位置"},
            {"id": 3, "name": "任务完成", "description": "确认所有立方体已分类到正确面板", "criteria": "所有立方体在目标面板"},
        ],
        "make_sandwich": [
            {"id": 0, "name": "放置底层面包", "description": "rob0将面包底放在盘子上", "criteria": "面包底在盘子中央"},
            {"id": 1, "name": "添加中间配料", "description": "轮流添加肉、蔬菜、奶酪等配料", "criteria": "所有配料按顺序堆叠"},
            {"id": 2, "name": "放置顶层面包", "description": "将面包顶放在最上面", "criteria": "三明治完整堆叠"},
            {"id": 3, "name": "任务完成", "description": "确认三明治制作完毕", "criteria": "三明治堆叠正确"},
        ],
        "pack_grocery": [
            {"id": 0, "name": "观察物品", "description": "识别桌上的杂货物品和箱子位置", "criteria": "已识别所有物品"},
            {"id": 1, "name": "分别打包", "description": "两个机器人分别打包各自范围内的物品", "criteria": "各自范围物品已入箱"},
            {"id": 2, "name": "任务完成", "description": "确认所有物品已打包", "criteria": "所有物品在箱中"},
        ],
        "move_rope": [
            {"id": 0, "name": "就位抓绳", "description": "两个机器人分别移动到绳子两端，抓住绳子", "criteria": "两个机器人各抓住一端"},
            {"id": 1, "name": "同步举起", "description": "两个机器人同时举起绳子到足够高度", "criteria": "绳子高度超过墙壁"},
            {"id": 2, "name": "越过墙壁", "description": "保持高度，两机器人协同移动到墙壁另一侧", "criteria": "绳子越过墙壁"},
            {"id": 3, "name": "放入凹槽", "description": "协同移动到凹槽位置，放入绳子", "criteria": "绳子在凹槽中"},
            {"id": 4, "name": "松开完成", "description": "两个机器人同时松手", "criteria": "绳子在凹槽中且机器人已松手"},
        ],
        "arrange_cabinet": [
            {"id": 0, "name": "打开柜门", "description": "rob0和rob1分别扶住左右柜门并打开", "criteria": "左右柜门均已打开且被扶住"},
            {"id": 1, "name": "取出杯子", "description": "rob2逐个从柜子中取出杯子", "criteria": "所有杯子已取出"},
            {"id": 2, "name": "放置杯垫", "description": "rob2将杯子放到对应颜色杯垫上", "criteria": "所有杯子在正确杯垫上"},
            {"id": 3, "name": "关闭柜门", "description": "rob0和rob1松开柜门", "criteria": "柜门关闭，任务完成"},
        ],
    }
    
    def __init__(self, task_name: str, llm_client=None):
        """
        Args:
            task_name: 任务名称
            llm_client: OllamaClient实例(可选，用于LLM动态分解)
        """
        self.task_name = task_name
        self.llm_client = llm_client
        self.subgoals: List[Dict] = []
        self.current_index: int = 0
        self.use_dynamic = False  # 是否使用LLM动态分解
        
        # 加载预定义子目标
        if task_name in self.PREDEFINED_SUBGOALS:
            self.subgoals = [dict(sg) for sg in self.PREDEFINED_SUBGOALS[task_name]]
            for sg in self.subgoals:
                sg["status"] = SubgoalStatus.PENDING.value
        
        if self.subgoals:
            self.subgoals[0]["status"] = SubgoalStatus.IN_PROGRESS.value
    
    def decompose_with_llm(self, observation: Dict) -> List[Dict]:
        """使用LLM动态分解任务（用于自定义或未见过的任务）"""
        if not self.llm_client:
            return self.subgoals
        
        prompt = f"""请将以下任务分解为3-6个有序子目标。

任务：{self.task_name}
当前观察：{observation}

每个子目标应该是一个清晰、可验证的阶段性目标。
输出格式（JSON）：
{{
  "subgoals": [
    {{"name": "子目标名称", "description": "具体描述", "criteria": "完成标准"}}
  ]
}}

注意：
- 子目标应该从开始到结束排序
- 每个子目标应该是可独立验证的
- 最后一个子目标应该是"任务完成"确认"""

        try:
            response = self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )
            import json, re
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                subgoals = data.get("subgoals", [])
                for i, sg in enumerate(subgoals):
                    sg["id"] = i
                    sg["status"] = SubgoalStatus.PENDING.value
                if subgoals:
                    subgoals[0]["status"] = SubgoalStatus.IN_PROGRESS.value
                self.subgoals = subgoals
                self.use_dynamic = True
        except Exception:
            pass
        
        return self.subgoals
    
    def get_current_subgoal(self) -> Optional[Dict]:
        """获取当前子目标"""
        if not self.subgoals or self.current_index >= len(self.subgoals):
            return None
        return self.subgoals[self.current_index]
    
    def get_next_subgoal(self) -> Optional[Dict]:
        """获取下一个子目标"""
        next_idx = self.current_index + 1
        if next_idx >= len(self.subgoals):
            return None
        return self.subgoals[next_idx]
    
    def check_subgoal_completed(self, observation: Dict) -> bool:
        """
        检查当前子目标是否完成
        
        基于预定义的criteria进行简单检查
        """
        current = self.get_current_subgoal()
        if not current:
            return True
        
        criteria = current.get("criteria", "").lower()
        
        # 简单启发式检查
        checks = []
        
        if "所有方块已倒入垃圾桶" in criteria or "桌面无方块" in criteria:
            cubes_on_table = self._count_cubes_on_table(observation)
            checks.append(cubes_on_table == 0)
        
        elif "所有立方体在目标面板" in criteria or "所有立方体在正确位置" in criteria:
            checks.append(self._check_all_sorted(observation))
        
        elif "三明治完整堆叠" in criteria or "三明治堆叠正确" in criteria:
            checks.append(self._check_sandwich_done(observation))
        
        elif "所有物品在箱中" in criteria or "所有物品已入箱" in criteria:
            checks.append(self._check_all_packed(observation))
        
        elif "绳子在凹槽中" in criteria:
            checks.append(self._check_rope_in_groove(observation))
        
        elif "所有杯子在正确杯垫上" in criteria:
            checks.append(self._check_cups_on_coasters(observation))
        
        # 如果没有匹配的检查规则，返回False（需要手动advance）
        if not checks:
            return False
        
        return all(checks)
    
    def advance(self) -> bool:
        """推进到下一个子目标"""
        current = self.get_current_subgoal()
        if current:
            current["status"] = SubgoalStatus.COMPLETED.value
        
        self.current_index += 1
        
        next_sg = self.get_current_subgoal()
        if next_sg:
            next_sg["status"] = SubgoalStatus.IN_PROGRESS.value
            return True
        return False
    
    def mark_failed(self, reason: str = "") -> None:
        """标记当前子目标失败"""
        current = self.get_current_subgoal()
        if current:
            current["status"] = SubgoalStatus.FAILED.value
            current["fail_reason"] = reason
    
    def is_task_done(self) -> bool:
        """检查整个任务是否完成"""
        return self.current_index >= len(self.subgoals)
    
    def get_progress(self) -> Dict:
        """获取任务进度"""
        total = len(self.subgoals)
        completed = sum(1 for sg in self.subgoals if sg["status"] == SubgoalStatus.COMPLETED.value)
        failed = sum(1 for sg in self.subgoals if sg["status"] == SubgoalStatus.FAILED.value)
        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "current": self.current_index,
            "current_name": self.get_current_subgoal()["name"] if self.get_current_subgoal() else "无",
            "progress_pct": f"{completed/max(1,total)*100:.1f}%",
        }
    
    def build_subgoal_prompt_context(self) -> str:
        """构建包含子目标上下文的Prompt前缀"""
        current = self.get_current_subgoal()
        if not current:
            return ""
        
        lines = [
            f"## 当前子目标: {current['name']}",
            f"描述: {current['description']}",
            f"完成标准: {current.get('criteria', '无')}",
        ]
        
        next_sg = self.get_next_subgoal()
        if next_sg:
            lines.append(f"\n下一个子目标: {next_sg['name']}")
        
        # 显示进度
        progress = self.get_progress()
        lines.append(f"\n任务进度: {progress['completed']}/{progress['total']} 子目标已完成")
        
        return "\n".join(lines)
    
    # ============ 辅助检查方法 ============
    
    def _count_cubes_on_table(self, obs: Dict) -> int:
        """计算桌上还有多少个方块"""
        count = 0
        for key in ["cube_states", "cubes"]:
            if key in obs:
                val = obs[key]
                if isinstance(val, dict):
                    for state in val.values():
                        if "table" in str(state).lower() or "桌面" in str(state):
                            count += 1
                elif isinstance(val, list):
                    count += len(val)
        return count
    
    def _check_all_sorted(self, obs: Dict) -> bool:
        return "all_sorted" in str(obs).lower() or "task_done" in str(obs).lower()
    
    def _check_sandwich_done(self, obs: Dict) -> bool:
        return "sandwich_complete" in str(obs).lower() or "stacked" in str(obs).lower()
    
    def _check_all_packed(self, obs: Dict) -> bool:
        return "all_packed" in str(obs).lower() or "bin_full" in str(obs).lower()
    
    def _check_rope_in_groove(self, obs: Dict) -> bool:
        return "rope_in_groove" in str(obs).lower()
    
    def _check_cups_on_coasters(self, obs: Dict) -> bool:
        return "cups_placed" in str(obs).lower() or "all_coasters" in str(obs).lower()
    
    def get_subgoal_actions(self, subgoal_id: int) -> List[str]:
        """获取某个子目标相关的建议动作模板"""
        if not self.subgoals or subgoal_id >= len(self.subgoals):
            return []
        
        task = self.task_name
        sg_name = self.subgoals[subgoal_id]["name"]
        
        # 启发式映射
        mapping = {
            "sweep_floor": {
                "定位所有方块": ["MOVE", "WAIT"],
                "清扫方块": ["MOVE", "SWEEP", "DUMP", "WAIT"],
                "任务完成": ["WAIT"],
            },
            "sort_cubes": {
                "观察分类": ["MOVE", "WAIT"],
                "就近处理": ["MOVE", "PICK", "PLACE", "WAIT"],
                "传递协作": ["MOVE", "PASS", "PLACE", "WAIT"],
                "任务完成": ["WAIT"],
            },
            "make_sandwich": {
                "放置底层面包": ["MOVE", "PICK", "PLACE", "WAIT"],
                "添加中间配料": ["MOVE", "PICK", "PLACE", "WAIT"],
                "放置顶层面包": ["MOVE", "PICK", "PLACE", "WAIT"],
                "任务完成": ["WAIT"],
            },
            "pack_grocery": {
                "观察物品": ["MOVE", "WAIT"],
                "分别打包": ["MOVE", "PICK", "PLACE", "WAIT"],
                "任务完成": ["WAIT"],
            },
            "move_rope": {
                "就位抓绳": ["MOVE", "GRIP", "WAIT"],
                "同步举起": ["LIFT", "WAIT"],
                "越过墙壁": ["MOVE_PAIR", "WAIT"],
                "放入凹槽": ["MOVE_PAIR", "WAIT"],
                "松开完成": ["RELEASE", "WAIT"],
            },
            "arrange_cabinet": {
                "打开柜门": ["MOVE", "HOLD_DOOR", "WAIT"],
                "取出杯子": ["MOVE", "PICK", "WAIT"],
                "放置杯垫": ["MOVE", "PICK", "PLACE", "WAIT"],
                "关闭柜门": ["RELEASE", "WAIT"],
            },
        }
        
        task_map = mapping.get(task, {})
        return task_map.get(sg_name, ["MOVE", "WAIT"])


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TaskDecomposer 测试")
    print("=" * 60)
    
    for task_name in ["sweep_floor", "make_sandwich", "move_rope", "arrange_cabinet"]:
        print(f"\n{'='*40}")
        print(f"📋 任务: {task_name}")
        td = TaskDecomposer(task_name)
        
        print(f"   子目标数: {len(td.subgoals)}")
        
        # 遍历子目标
        for i in range(len(td.subgoals)):
            current = td.get_current_subgoal()
            if current:
                print(f"   [{i}] {current['name']}: {current['description']}")
                print(f"       标准: {current.get('criteria', 'N/A')}")
                print(f"       建议动作: {td.get_subgoal_actions(i)}")
            
            td.advance()
        
        # 检查进度
        td2 = TaskDecomposer(task_name)
        print(f"\n   初始进度: {td2.get_progress()}")
        td2.advance()
        print(f"   推进后: {td2.get_progress()}")
    
    # 测试Prompt上下文
    print(f"\n{'='*40}")
    print("📝 子目标Prompt上下文:")
    td = TaskDecomposer("sweep_floor")
    ctx = td.build_subgoal_prompt_context()
    print(ctx)
    
    print("\n✅ TaskDecomposer 测试完成！")
