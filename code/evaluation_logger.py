"""
评估日志系统 - EvaluationLogger
=================================
借鉴：AgentBench / HELM评估框架 / RocoBench evaluator.py
创新点：完整记录每次运行，自动生成评估报告

功能：
- 自动记录每次任务的完整执行日志
- 统计各维度指标（成功率、非法动作率、恢复率等）
- 生成JSON/CSV格式报告
- 对比不同版本的性能

使用方法：
    from evaluation_logger import EvaluationLogger
    logger = EvaluationLogger("results/")
    logger.start_task("sweep_floor", "v10")
    logger.log_step(1, "rob0: MOVE red_cube", "success")
    logger.end_task(success=True)
    logger.generate_report()
"""

import json
import os
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict


class EvaluationLogger:
    """评估日志系统"""
    
    def __init__(self, output_dir: str = "results", version: str = "v1"):
        """
        Args:
            output_dir: 结果输出目录
            version: 当前方案版本号
        """
        self.output_dir = output_dir
        self.version = version
        self.current_task: Optional[str] = None
        self.current_run: Optional[Dict] = None
        
        # 累积统计
        self.all_runs: List[Dict] = []
        self.task_stats: Dict[str, List[Dict]] = defaultdict(list)
        
        os.makedirs(output_dir, exist_ok=True)
    
    def start_task(
        self,
        task_name: str,
        version: str = None,
        run_id: int = None,
    ):
        """
        开始记录一个任务
        
        Args:
            task_name: 任务名称
            version: 版本号
            run_id: 运行编号
        """
        self.current_task = task_name
        
        if run_id is None:
            run_id = len(self.task_stats.get(task_name, []))
        
        self.current_run = {
            "task": task_name,
            "version": version or self.version,
            "run_id": run_id,
            "start_time": datetime.now().isoformat(),
            "steps": [],
            "metrics": {
                "total_steps": 0,
                "successful_steps": 0,
                "failed_steps": 0,
                "invalid_actions": 0,
                "format_errors": 0,
                "reflections": 0,
                "recoveries": 0,
                "loops_detected": 0,
                "llm_calls": 0,
                "total_llm_time": 0.0,
            },
            "failure_reasons": [],
            "success": False,
        }
    
    def log_step(
        self,
        step_number: int,
        action: str,
        result: str,
        llm_time: float = 0.0,
        error_message: str = "",
        metadata: Dict = None,
    ):
        """
        记录一步执行
        
        Args:
            step_number: 步骤编号
            action: 执行的动作
            result: 结果 (success/failed/timeout)
            llm_time: LLM调用耗时
            error_message: 错误信息
            metadata: 额外信息
        """
        if not self.current_run:
            return
        
        step = {
            "step": step_number,
            "action": action,
            "result": result,
            "llm_time": llm_time,
            "error": error_message,
            "metadata": metadata or {},
        }
        
        self.current_run["steps"].append(step)
        
        # 更新指标
        m = self.current_run["metrics"]
        m["total_steps"] += 1
        m["llm_calls"] += 1
        m["total_llm_time"] += llm_time
        
        if result == "success":
            m["successful_steps"] += 1
        else:
            m["failed_steps"] += 1
            if error_message:
                self.current_run["failure_reasons"].append({
                    "step": step_number,
                    "action": action,
                    "error": error_message,
                })
    
    def log_metric(self, key: str, value: Any):
        """记录自定义指标"""
        if self.current_run:
            self.current_run["metrics"][key] = value
    
    def end_task(self, success: bool, total_time: float = 0.0):
        """结束当前任务记录"""
        if not self.current_run:
            return
        
        self.current_run["success"] = success
        self.current_run["end_time"] = datetime.now().isoformat()
        self.current_run["total_time_seconds"] = total_time
        self.current_run["metrics"]["success"] = success
        
        # 保存到累积统计
        self.all_runs.append(self.current_run)
        if self.current_task:
            self.task_stats[self.current_task].append(self.current_run)
        
        # 保存到文件
        self._save_run(self.current_run)
        
        self.current_run = None
        self.current_task = None
    
    def _save_run(self, run: Dict):
        """保存单次运行记录"""
        task = run["task"]
        rid = run["run_id"]
        filename = f"{task}_v{run['version']}_run{rid}_{datetime.now().strftime('%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(run, f, ensure_ascii=False, indent=2)
    
    def get_task_stats(self, task_name: str) -> Dict:
        """
        获取某个任务的统计
        
        Returns:
            {
                "total_runs": N,
                "success_rate": "X%",
                "avg_steps": N,
                "avg_llm_time": N,
                ...
            }
        """
        runs = self.task_stats.get(task_name, [])
        if not runs:
            return {"total_runs": 0, "success_rate": "0%"}
        
        successful = sum(1 for r in runs if r.get("success", False))
        total_steps = [r["metrics"]["total_steps"] for r in runs]
        llm_times = [r["metrics"]["total_llm_time"] for r in runs]
        failure_reasons = []
        for r in runs:
            failure_reasons.extend(r.get("failure_reasons", []))
        
        return {
            "total_runs": len(runs),
            "successful_runs": successful,
            "success_rate": f"{successful/max(1,len(runs))*100:.1f}%",
            "avg_steps": f"{sum(total_steps)/max(1,len(total_steps)):.1f}",
            "min_steps": min(total_steps) if total_steps else 0,
            "max_steps": max(total_steps) if total_steps else 0,
            "avg_llm_time": f"{sum(llm_times)/max(1,len(llm_times)):.2f}s",
            "failure_reasons": failure_reasons,
        }
    
    def get_overall_stats(self) -> Dict:
        """获取总体统计"""
        if not self.all_runs:
            return {"total_runs": 0}
        
        successful = sum(1 for r in self.all_runs if r.get("success", False))
        
        tasks = defaultdict(lambda: {"total": 0, "success": 0})
        for r in self.all_runs:
            task = r["task"]
            tasks[task]["total"] += 1
            if r.get("success"):
                tasks[task]["success"] += 1
        
        return {
            "version": self.version,
            "total_runs": len(self.all_runs),
            "total_successful": successful,
            "overall_success_rate": f"{successful/max(1,len(self.all_runs))*100:.1f}%",
            "by_task": dict(tasks),
        }
    
    def generate_report(self, output_format: str = "both") -> str:
        """
        生成评估报告
        
        Args:
            output_format: "json" / "csv" / "both"
        
        Returns:
            报告文本
        """
        lines = [f"# 评估报告 - 版本 {self.version}", f"生成时间: {datetime.now().isoformat()}", ""]
        
        overall = self.get_overall_stats()
        lines.append("## 总体统计")
        lines.append(f"- 总运行次数: {overall['total_runs']}")
        lines.append(f"- 成功次数: {overall['total_successful']}")
        lines.append(f"- 总体成功率: {overall['overall_success_rate']}")
        lines.append("")
        
        lines.append("## 各任务统计")
        lines.append("")
        lines.append("| 任务 | 运行次数 | 成功次数 | 成功率 | 平均步数 | 平均LLM时间 |")
        lines.append("|------|---------|---------|--------|---------|------------|")
        
        for task_name in sorted(self.task_stats.keys()):
            stats = self.get_task_stats(task_name)
            lines.append(
                f"| {task_name} | {stats['total_runs']} | {stats['successful_runs']} | "
                f"{stats['success_rate']} | {stats['avg_steps']} | {stats['avg_llm_time']} |"
            )
        
        lines.append("")
        lines.append("## 常见失败原因")
        for task_name in sorted(self.task_stats.keys()):
            stats = self.get_task_stats(task_name)
            if stats.get("failure_reasons"):
                lines.append(f"\n### {task_name}")
                # 统计失败原因
                error_counts = defaultdict(int)
                for fr in stats["failure_reasons"]:
                    error_msg = fr.get("error", "unknown")[:50]
                    error_counts[error_msg] += 1
                for error, count in sorted(error_counts.items(), key=lambda x: -x[1])[:5]:
                    lines.append(f"- {error}: {count}次")
        
        report = "\n".join(lines)
        
        # 保存文件
        report_path = os.path.join(self.output_dir, f"report_{self.version}.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        if output_format in ("json", "both"):
            json_path = os.path.join(self.output_dir, f"report_{self.version}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "overall": overall,
                    "by_task": {t: self.get_task_stats(t) for t in self.task_stats},
                    "all_runs_summary": [
                        {"task": r["task"], "success": r["success"], "steps": r["metrics"]["total_steps"]}
                        for r in self.all_runs
                    ],
                }, f, ensure_ascii=False, indent=2)
        
        if output_format in ("csv", "both"):
            csv_path = os.path.join(self.output_dir, f"report_{self.version}.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["task", "run_id", "version", "success", "total_steps", "failed_steps", "llm_calls", "total_llm_time"])
                for r in self.all_runs:
                    writer.writerow([
                        r["task"], r["run_id"], r["version"],
                        r["success"], r["metrics"]["total_steps"],
                        r["metrics"]["failed_steps"], r["metrics"]["llm_calls"],
                        f"{r['metrics']['total_llm_time']:.2f}",
                    ])
        
        return report
    
    def compare_versions(self, version_a: str, version_b: str = None) -> Dict:
        """
        比较两个版本的性能
        
        Args:
            version_a: 版本A
            version_b: 版本B（None则比较所有）
        
        Returns:
            比较结果
        """
        runs_a = [r for r in self.all_runs if r["version"] == version_a]
        runs_b = [r for r in self.all_runs if r["version"] == version_b] if version_b else []
        
        def calc_stats(runs):
            if not runs:
                return {"success_rate": "0%", "avg_steps": 0}
            success = sum(1 for r in runs if r["success"])
            avg_steps = sum(r["metrics"]["total_steps"] for r in runs) / len(runs)
            return {"success_rate": f"{success/len(runs)*100:.1f}%", "avg_steps": f"{avg_steps:.1f}"}
        
        result = {
            "version_a": version_a,
            "version_b": version_b or "all_others",
            "a_stats": calc_stats(runs_a),
            "b_stats": calc_stats(runs_b),
        }
        
        return result
    
    def get_best_version(self) -> Optional[str]:
        """找出成功率最高的版本"""
        version_stats = defaultdict(lambda: {"total": 0, "success": 0})
        for r in self.all_runs:
            v = r["version"]
            version_stats[v]["total"] += 1
            if r["success"]:
                version_stats[v]["success"] += 1
        
        if not version_stats:
            return None
        
        best_v = max(
            version_stats.items(),
            key=lambda x: x[1]["success"] / max(1, x[1]["total"])
        )
        
        return best_v[0]


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    import tempfile
    
    tmp_dir = os.path.join(tempfile.gettempdir(), "roco_eval_test")
    
    print("=" * 60)
    print("EvaluationLogger 测试")
    print("=" * 60)
    
    logger = EvaluationLogger(tmp_dir, version="v10")
    
    # 模拟一次成功的sweep_floor
    print("\n📝 模拟 sweep_floor 运行...")
    logger.start_task("sweep_floor", version="v10", run_id=0)
    logger.log_step(1, "rob0: MOVE red_cube", "success", 1.2)
    logger.log_step(2, "rob1: MOVE red_cube", "success", 0.8)
    logger.log_step(3, "rob0: SWEEP red_cube", "success", 0.5)
    logger.log_step(4, "rob1: DUMP", "success", 0.3)
    logger.end_task(success=True, total_time=15.0)
    
    # 模拟一次失败的
    print("📝 模拟 pack_grocery 运行...")
    logger.start_task("pack_grocery", version="v10", run_id=0)
    logger.log_step(1, "rob0: PICK item_1", "success", 1.0)
    logger.log_step(2, "rob1: PICK item_1", "failed", 1.5, "Collision detected")
    logger.log_step(3, "rob1: WAIT", "success", 0.5)
    logger.log_step(4, "rob1: PICK item_2", "failed", 1.0, "Object out of reach")
    logger.log_step(5, "rob0: PLACE bin", "success", 0.8)
    logger.end_task(success=False, total_time=30.0)
    
    # 统计
    print("\n📊 任务统计:")
    for task in ["sweep_floor", "pack_grocery"]:
        stats = logger.get_task_stats(task)
        print(f"   {task}: 成功率={stats['success_rate']}, 平均步数={stats['avg_steps']}")
    
    # 总体统计
    print(f"\n📊 总体统计: {logger.get_overall_stats()}")
    
    # 生成报告
    print("\n📄 生成报告...")
    report = logger.generate_report()
    print(report[:500])
    print(f"\n   完整报告保存在: {tmp_dir}")
    
    # 清理
    import shutil
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    
    print("\n✅ EvaluationLogger 测试完成！")
