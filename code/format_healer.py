"""
格式修复器 - FormatHealer
================================
借鉴：Toolformer / Structured Output / JSON修复合约
创新点：LLM输出格式不合规时，不直接失败，而是自动二次修复

解决问题：
- LLM有时会输出多余文字夹杂在JSON/动作格式中
- LLM可能忘记输出</ACTIONS>闭合标签
- JSON中可能包含尾随逗号或单引号

使用方法：
    from format_healer import FormatHealer
    healer = FormatHealer(client)
    fixed = healer.heal(raw_output, expected_format="actions")
"""

import re
import json
from typing import Optional, Dict, List, Tuple


class FormatHealer:
    """LLM输出格式自愈器"""
    
    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: OllamaClient实例，用于二次调用LLM修复格式
                       如果为None，则只用规则修复，不用LLM
        """
        self.client = llm_client
        self.heal_count = 0
        self.heal_success_count = 0
    
    def heal(self, raw_output: str, expected_format: str = "actions") -> Tuple[str, bool]:
        """
        修复LLM输出格式
        
        Args:
            raw_output: LLM原始输出
            expected_format: 期望格式
                - "actions": <THINK>...</THINK><ACTIONS>...</ACTIONS>
                - "json": 纯JSON
                - "role_assignment": <ROLES>...</ROLES>
                - "reflection": <REFLECTION>...</REFLECTION><ACTIONS>...</ACTIONS>
        
        Returns:
            (修复后文本, 是否需要二次LLM调用)
        """
        self.heal_count += 1
        
        # 第一层：规则修复（不用LLM，零成本）
        fixed = self._rule_heal(raw_output, expected_format)
        
        # 检查规则修复是否足够
        if self._validate_format(fixed, expected_format):
            self.heal_success_count += 1
            return fixed, False
        
        # 第二层：LLM修复（规则搞不定时）
        if self.client:
            fixed = self._llm_heal(raw_output, expected_format)
            if self._validate_format(fixed, expected_format):
                self.heal_success_count += 1
                return fixed, True
        
        # 实在修复不了，返回原始输出
        return raw_output, False
    
    def _rule_heal(self, text: str, fmt: str) -> str:
        """规则层面修复，不调用LLM"""
        text = text.strip()
        
        if fmt == "actions":
            # 修复1：如果完全没有标签，尝试包裹
            if "<THINK>" not in text and "<ACTIONS>" not in text:
                # 尝试找JSON
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    text = f"<THINK>解析JSON计划</THINK>\n<ACTIONS>\n{json_match.group()}\n</ACTIONS>"
                else:
                    text = f"<THINK>{text[:200]}</THINK>\n<ACTIONS>\nrob0: WAIT\n</ACTIONS>"
            
            # 修复2：补全缺失的闭合标签
            if "<THINK>" in text and "</THINK>" not in text:
                text += "\n</THINK>"
            if "<ACTIONS>" in text and "</ACTIONS>" not in text:
                text += "\n</ACTIONS>"
            
            # 修复3：如果只有ACTIONS没有THINK
            if "<THINK>" not in text and "<ACTIONS>" in text:
                text = "<THINK>快速决策</THINK>\n" + text
        
        elif fmt == "json":
            # 提取JSON
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                text = json_match.group()
            # 修复常见JSON错误
            text = self._fix_json_errors(text)
        
        elif fmt == "role_assignment":
            if "<ROLES>" not in text:
                text = f"<ROLES>\n{text}\n</ROLES>"
            if "</ROLES>" not in text:
                text += "\n</ROLES>"
        
        elif fmt == "reflection":
            if "<REFLECTION>" not in text:
                text = f"<REFLECTION>{text[:300]}</REFLECTION>"
            if "</REFLECTION>" not in text and "<REFLECTION>" in text:
                # 在</ACTIONS>或文末前插入
                if "</ACTIONS>" in text:
                    idx = text.index("</ACTIONS>")
                    text = text[:idx] + "</REFLECTION>\n" + text[idx:]
                else:
                    text += "\n</REFLECTION>"
        
        return text.strip()
    
    def _fix_json_errors(self, json_str: str) -> str:
        """修复常见JSON格式错误"""
        # 去除尾随逗号（在}或]之前）
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        # 单引号转双引号（简单情况）
        # 注意：只在key和value是简单字符串时转换
        json_str = re.sub(r"'(\w+)':", r'"\1":', json_str)
        # 去除JSON前后的非JSON文本
        start = json_str.find('{')
        end = json_str.rfind('}')
        if start >= 0 and end > start:
            json_str = json_str[start:end+1]
        return json_str
    
    def _llm_heal(self, raw: str, fmt: str) -> str:
        """调用LLM修复格式（二次调用）"""
        if not self.client:
            return raw
        
        repair_prompt = f"""你是一个格式修复器。以下LLM输出格式不正确。

期望格式: {fmt}
原始输出:
{raw[:1000]}

请修复为正确格式，只输出修复后的内容，不要解释。"""
        
        try:
            response = self.client.chat(
                messages=[{"role": "user", "content": repair_prompt}],
                temperature=0.0,
                max_tokens=500,
            )
            return response.strip()
        except Exception:
            return raw
    
    def _validate_format(self, text: str, fmt: str) -> bool:
        """验证格式是否正确"""
        text = text.strip()
        
        if fmt == "actions":
            has_think_open = "<THINK>" in text
            has_think_close = "</THINK>" in text
            has_actions_open = "<ACTIONS>" in text
            has_actions_close = "</ACTIONS>" in text
            return has_think_open and has_think_close and has_actions_open and has_actions_close
        
        elif fmt == "json":
            try:
                json.loads(text)
                return True
            except json.JSONDecodeError:
                return False
        
        elif fmt == "role_assignment":
            return "<ROLES>" in text and "</ROLES>" in text
        
        elif fmt == "reflection":
            return "<REFLECTION>" in text and "</REFLECTION>" in text
        
        return True
    
    def get_stats(self) -> Dict:
        """获取修复统计"""
        return {
            "total_heals": self.heal_count,
            "successful_heals": self.heal_success_count,
            "success_rate": f"{self.heal_success_count/max(1,self.heal_count)*100:.1f}%"
        }


# ============================================================
# 便捷函数
# ============================================================

def quick_heal(raw_output: str, expected_format: str = "actions") -> str:
    """快速修复（仅规则，不调LLM）"""
    healer = FormatHealer()
    fixed, _ = healer.heal(raw_output, expected_format)
    return fixed


def extract_json(text: str) -> Optional[Dict]:
    """从任意文本中提取JSON"""
    healer = FormatHealer()
    fixed, _ = healer.heal(text, "json")
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return None


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FormatHealer 测试")
    print("=" * 60)
    
    healer = FormatHealer()
    
    # 测试1：缺失闭合标签
    print("\n测试1: 缺失闭合标签")
    bad = "<THINK>分析状态</THINK>\n<ACTIONS>\nrob0: MOVE table\nrob1: WAIT"
    fixed, used_llm = healer.heal(bad, "actions")
    print(f"  修复后: {fixed[:100]}...")
    print(f"  用LLM: {used_llm}")
    
    # 测试2：完全没有标签
    print("\n测试2: 完全没有标签")
    bad = "机器人应该先移动到桌子旁边，然后扫方块。"
    fixed, used_llm = healer.heal(bad, "actions")
    print(f"  修复后: {fixed[:100]}...")
    print(f"  用LLM: {used_llm}")
    
    # 测试3：JSON修复
    print("\n测试3: JSON修复")
    bad = 'some text {"robot": "rob0", "action": "MOVE", } extra'
    fixed, used_llm = healer.heal(bad, "json")
    print(f"  修复后: {fixed}")
    print(f"  用LLM: {used_llm}")
    
    # 测试4：格式验证
    print("\n测试4: 格式验证")
    good = "<THINK>test</THINK>\n<ACTIONS>\nrob0: WAIT\n</ACTIONS>"
    print(f"  合法: {healer._validate_format(good, 'actions')}")
    bad = "no tags at all"
    print(f"  非法: {healer._validate_format(bad, 'actions')}")
    
    print(f"\n📊 统计: {healer.get_stats()}")
    print("\n✅ FormatHealer 测试完成！")
