"""
候选投票器 - CandidateVoter
=============================
借鉴：Self-Consistency / Best-of-N Sampling / Multi-Agent Debate
创新点：同一状态生成多个候选动作，投票/评分选出最佳

核心思想：
- 单次LLM调用可能输出不稳定（温度>0时）
- 多次采样+投票可以大幅提升稳定性
- 用Critic LLM评分选出最佳候选

三种投票策略：
1. Majority Voting: 选出现频率最高的
2. Scored Voting: LLM Critic评分后选最高分
3. Consensus Voting: 需要多个候选一致才采纳

使用方法：
    from candidate_voter import CandidateVoter
    voter = CandidateVoter(llm_client)
    best_action = voter.vote(prompt, n_candidates=3, strategy="scored")
"""

from typing import Dict, List, Optional, Tuple, Any
from collections import Counter


class CandidateVoter:
    """多候选投票器"""
    
    def __init__(self, llm_client=None, default_temperature: float = 0.6):
        """
        Args:
            llm_client: OllamaClient实例
            default_temperature: 生成候选时的默认温度
        """
        self.llm_client = llm_client
        self.default_temperature = default_temperature
        self.vote_count = 0
        self.consensus_count = 0
    
    def generate_candidates(
        self,
        prompt: str,
        system_prompt: str = None,
        n_candidates: int = 3,
        temperature: float = None,
    ) -> List[str]:
        """
        为同一状态生成多个候选动作
        
        Args:
            prompt: 用户prompt
            system_prompt: 系统prompt
            n_candidates: 候选数量
            temperature: 温度（越高多样性越大）
        
        Returns:
            候选动作列表
        """
        if not self.llm_client:
            return []
        
        temp = temperature if temperature is not None else self.default_temperature
        candidates = []
        
        # 稍微抖动温度以增加多样性
        temps = [max(0.1, temp + i * 0.1 - 0.1) for i in range(n_candidates)]
        
        for i, t in enumerate(temps):
            try:
                response = self.llm_client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    system_prompt=system_prompt,
                    temperature=t,
                    max_tokens=500,
                )
                candidates.append(response.strip())
            except Exception as e:
                print(f"   ⚠️ 候选{i+1}生成失败: {e}")
        
        return candidates
    
    def majority_vote(self, candidates: List[str]) -> Tuple[str, float]:
        """
        多数投票：选出现频率最高的候选
        
        Returns:
            (最佳候选, 置信度)
        """
        if not candidates:
            return "", 0.0
        
        if len(candidates) == 1:
            return candidates[0], 1.0
        
        # 对候选做归一化（去除多余空格、统一大小写）
        normalized = [c.strip().lower() for c in candidates]
        
        # 计数
        counter = Counter(normalized)
        most_common, count = counter.most_common(1)[0]
        
        # 置信度 = 出现次数 / 总候选数
        confidence = count / len(candidates)
        
        # 返回原始格式的候选
        idx = normalized.index(most_common)
        
        self.vote_count += 1
        if confidence >= 0.66:  # 超过2/3一致认为共识
            self.consensus_count += 1
        
        return candidates[idx], confidence
    
    def scored_vote(
        self,
        candidates: List[str],
        context: str = "",
    ) -> Tuple[str, float, List[Dict]]:
        """
        评分投票：LLM Critic对每个候选打分，选最高分
        
        Args:
            candidates: 候选动作列表
            context: 评分上下文（任务描述、状态等）
        
        Returns:
            (最佳候选, 最高分, [所有候选的评分])
        """
        if not candidates:
            return "", 0.0, []
        
        if len(candidates) == 1:
            return candidates[0], 1.0, [{"candidate": candidates[0], "score": 5}]
        
        if not self.llm_client:
            # 没有LLM时用多数投票
            best, conf = self.majority_vote(candidates)
            return best, conf, []
        
        scores = []
        for i, candidate in enumerate(candidates):
            score = self._score_candidate(candidate, context, i)
            scores.append({"candidate": candidate, "score": score})
        
        # 选最高分
        best = max(scores, key=lambda x: x["score"])
        
        self.vote_count += 1
        if best["score"] >= 4:
            self.consensus_count += 1
        
        return best["candidate"], best["score"] / 5.0, scores
    
    def _score_candidate(self, candidate: str, context: str, index: int) -> int:
        """让LLM Critic评分"""
        if not self.llm_client:
            return 3  # 默认中等分
        
        prompt = f"""请对以下机器人动作候选进行评分（1-5分）。

评分标准：
- 5分：动作合理，符合任务逻辑，无碰撞风险
- 4分：动作基本合理，有小瑕疵
- 3分：动作一般，可以执行但不够好
- 2分：动作有较大问题
- 1分：动作完全不合理

{context}

候选动作：
{candidate}

请只回复一个数字（1-5），不要其他内容。"""
        
        try:
            response = self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5,
            )
            # 提取数字
            import re
            match = re.search(r'[1-5]', response)
            if match:
                return int(match.group())
        except Exception:
            pass
        
        return 3
    
    def consensus_vote(
        self,
        candidates: List[str],
        required_agreement: float = 0.66,
    ) -> Tuple[Optional[str], float, bool]:
        """
        共识投票：需要多个候选达成一致才采纳
        
        Args:
            candidates: 候选动作列表
            required_agreement: 所需一致性比例
        
        Returns:
            (动作, 一致性比例, 是否达成共识)
        """
        best, confidence = self.majority_vote(candidates)
        consensus = confidence >= required_agreement
        return best, confidence, consensus
    
    def vote(
        self,
        prompt: str,
        system_prompt: str = None,
        n_candidates: int = 3,
        strategy: str = "scored",
        context: str = "",
        temperature: float = None,
    ) -> Dict[str, Any]:
        """
        统一的投票入口
        
        Args:
            prompt: 用户prompt
            system_prompt: 系统prompt
            n_candidates: 候选数量
            strategy: 投票策略 (majority / scored / consensus)
            context: 评分上下文
            temperature: 候选生成温度
        
        Returns:
            {
                "best_action": str,
                "confidence": float,
                "strategy": str,
                "candidates": list,
                "scores": list (仅scored策略),
                "consensus": bool (仅consensus策略),
            }
        """
        # 生成候选
        candidates = self.generate_candidates(
            prompt, system_prompt, n_candidates, temperature
        )
        
        if not candidates:
            return {
                "best_action": "",
                "confidence": 0.0,
                "strategy": strategy,
                "candidates": [],
                "error": "No candidates generated",
            }
        
        # 投票
        if strategy == "majority":
            best, conf = self.majority_vote(candidates)
            return {
                "best_action": best,
                "confidence": conf,
                "strategy": "majority",
                "candidates": candidates,
            }
        
        elif strategy == "scored":
            best, conf, scores = self.scored_vote(candidates, context)
            return {
                "best_action": best,
                "confidence": conf,
                "strategy": "scored",
                "candidates": candidates,
                "scores": scores,
            }
        
        elif strategy == "consensus":
            best, conf, consensus = self.consensus_vote(candidates)
            return {
                "best_action": best if consensus else None,
                "confidence": conf,
                "strategy": "consensus",
                "candidates": candidates,
                "consensus": consensus,
            }
        
        else:
            # 默认多数投票
            best, conf = self.majority_vote(candidates)
            return {
                "best_action": best,
                "confidence": conf,
                "strategy": "majority",
                "candidates": candidates,
            }
    
    def get_stats(self) -> Dict:
        return {
            "total_votes": self.vote_count,
            "consensus_rate": f"{self.consensus_count/max(1,self.vote_count)*100:.1f}%",
        }


# ============================================================
# 轻量版：不需要LLM的投票
# ============================================================

class LightweightVoter:
    """轻量投票器：不需要额外LLM调用"""
    
    @staticmethod
    def majority(candidates: List[str]) -> str:
        """简单多数投票"""
        if not candidates:
            return ""
        counter = Counter(c.strip().lower() for c in candidates)
        most_common = counter.most_common(1)[0][0]
        # 返回原始格式
        for c in candidates:
            if c.strip().lower() == most_common:
                return c
        return candidates[0]
    
    @staticmethod
    def random_with_temperature(candidates: List[str]) -> str:
        """随机选择（用于探索）"""
        import random
        return random.choice(candidates) if candidates else ""
    
    @staticmethod
    def diversity_check(candidates: List[str], threshold: float = 0.5) -> bool:
        """检查候选是否足够多样化"""
        if len(candidates) < 2:
            return False
        unique = len(set(c.strip().lower() for c in candidates))
        return unique / len(candidates) >= threshold


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CandidateVoter 测试")
    print("=" * 60)
    
    # 测试轻量版
    print("\n📊 轻量投票 - 多数投票:")
    candidates = [
        "rob0: MOVE red_cube\nrob1: WAIT",
        "rob0: MOVE red_cube\nrob1: WAIT",
        "rob0: SWEEP red_cube\nrob1: WAIT",
    ]
    winner = LightweightVoter.majority(candidates)
    print(f"   候选: {[c[:30]+'...' for c in candidates]}")
    print(f"   胜出: {winner[:60]}...")
    
    # 测试多样性
    diverse = LightweightVoter.diversity_check(candidates)
    print(f"   多样性: {diverse}")
    
    # 测试无LLM的投票器
    print("\n📊 CandidateVoter - 多数投票(无LLM):")
    voter = CandidateVoter()  # 无LLM
    result = voter.vote(
        prompt="test",
        n_candidates=1,  # 不实际调用LLM
        strategy="majority",
    )
    print(f"   结果: {result}")
    
    # 测试直接传入候选的投票
    print("\n📊 直接候选投票:")
    result2 = voter.majority_vote(candidates)
    print(f"   最佳: {result2[0][:60]}...")
    print(f"   置信度: {result2[1]:.1%}")
    
    print(f"\n📊 统计: {voter.get_stats()}")
    print("\n✅ CandidateVoter 测试完成！")
    print("\n💡 提示: 完整测试需要Ollama客户端(生成候选+评分)")
