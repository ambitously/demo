"""
Ollama LLM 客户端
===================
封装与 Ollama 的通信，支持 OpenAI 兼容接口和原生接口。
你之前在 harness 里调 API 的经验在这里 100% 复用。

使用方法：
    from ollama_client import OllamaClient
    
    client = OllamaClient()
    response = client.chat("你好，请介绍一下你自己")
    print(response)
"""

import json
import time
import os
from datetime import datetime
from typing import Optional, List, Dict, Any

# 尝试导入 OpenAI 库（Ollama 的 OpenAI 兼容接口）
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  openai 库未安装。请运行: pip install openai")

# 尝试导入 config（如果没有，使用默认配置）
try:
    from config import OLLAMA_CONFIG, DEBUG, SAVE_LLM_LOG, LLM_LOG_DIR
except ImportError:
    OLLAMA_CONFIG = {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "model": "qwen2.5:7b",
        "temperature": 0.1,
        "max_tokens": 2048,
        "timeout": 120,
    }
    DEBUG = True
    SAVE_LLM_LOG = False


class OllamaClient:
    """Ollama LLM 客户端"""
    
    def __init__(
        self,
        base_url: str = None,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        timeout: int = None,
    ):
        """
        初始化 Ollama 客户端
        
        Args:
            base_url: Ollama API 地址
            model: 模型名称
            temperature: 温度参数 (0.0~1.0)
            max_tokens: 最大输出 token 数
            timeout: 请求超时时间（秒）
        """
        self.base_url = base_url or OLLAMA_CONFIG["base_url"]
        self.model = model or OLLAMA_CONFIG["model"]
        self.temperature = temperature or OLLAMA_CONFIG["temperature"]
        self.max_tokens = max_tokens or OLLAMA_CONFIG["max_tokens"]
        self.timeout = timeout or OLLAMA_CONFIG["timeout"]
        
        # 初始化客户端
        if OPENAI_AVAILABLE:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=OLLAMA_CONFIG["api_key"],
                timeout=self.timeout,
            )
        else:
            self.client = None
        
        # 统计信息
        self.call_count = 0
        self.total_tokens = 0
        self.call_history = []
        
        if DEBUG:
            print(f"🔧 OllamaClient 初始化完成")
            print(f"   地址: {self.base_url}")
            print(f"   模型: {self.model}")
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        """
        调用 LLM 进行对话
        
        Args:
            messages: 对话消息列表 [{"role": "user", "content": "..."}]
            system_prompt: 系统提示词（可选）
            temperature: 温度参数（可选，覆盖默认值）
            max_tokens: 最大输出 token（可选，覆盖默认值）
        
        Returns:
            LLM 的回复文本
        
        Raises:
            Exception: 调用失败时抛出异常
        """
        # 构造完整消息列表
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            if OPENAI_AVAILABLE and self.client:
                # 使用 OpenAI 兼容接口（推荐方式）
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=full_messages,
                    temperature=temperature or self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                )
                content = response.choices[0].message.content
                
                # 更新统计
                self.call_count += 1
                elapsed = time.time() - start_time
                
                if DEBUG:
                    print(f"📡 LLM 调用 #{self.call_count} | "
                          f"耗时: {elapsed:.1f}s | "
                          f"输入: {len(str(messages))} 字符 | "
                          f"输出: {len(content)} 字符")
                
                # 保存日志
                if SAVE_LLM_LOG:
                    self._save_log(full_messages, content, elapsed)
                
                return content
            else:
                # 使用原生 Ollama HTTP API（回退方案）
                return self._chat_via_http(full_messages, temperature, max_tokens)
                
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"❌ LLM 调用失败 (耗时 {elapsed:.1f}s): {e}"
            print(error_msg)
            
            # 如果失败，记录日志
            if SAVE_LLM_LOG:
                self._save_log(full_messages, error_msg, elapsed, is_error=True)
            
            raise
    
    def _chat_via_http(
        self,
        messages: List[Dict],
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        """使用原生 HTTP 请求调用 Ollama（无需 openai 库）"""
        import urllib.request
        import urllib.error
        
        url = self.base_url.replace("/v1", "") + "/api/chat"
        data = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens,
            }
        }).encode("utf-8")
        
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"}
        )
        
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["message"]["content"]
    
    def chat_with_system(
        self,
        user_message: str,
        system_prompt: str,
        **kwargs,
    ) -> str:
        """
        快捷方法：带系统提示词的单轮对话
        
        Args:
            user_message: 用户消息
            system_prompt: 系统提示词
        
        Returns:
            LLM 的回复文本
        """
        return self.chat(
            messages=[{"role": "user", "content": user_message}],
            system_prompt=system_prompt,
            **kwargs,
        )
    
    def structured_output(
        self,
        user_message: str,
        system_prompt: str,
        output_format: str = "json",
    ) -> Dict[str, Any]:
        """
        获取结构化输出（JSON 格式）
        
        在 system_prompt 中要求 LLM 输出 JSON 格式，
        然后自动解析。
        
        Args:
            user_message: 用户消息
            system_prompt: 系统提示词（应包含 JSON 格式要求）
            output_format: 输出格式（目前只支持 json）
        
        Returns:
            解析后的字典
        """
        response = self.chat_with_system(user_message, system_prompt)
        
        try:
            # 尝试从回复中提取 JSON
            # LLM 可能会在 JSON 前后加一些文字，需要提取
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            elif "{" in response:
                start = response.index("{")
                end = response.rindex("}") + 1
                json_str = response[start:end]
            else:
                json_str = response
            
            return json.loads(json_str.strip())
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON 解析失败: {e}")
            print(f"   原始输出: {response[:500]}")
            return {"raw_output": response, "parse_error": str(e)}
    
    def _save_log(
        self, 
        messages: List[Dict], 
        response: str, 
        elapsed: float,
        is_error: bool = False,
    ):
        """保存 LLM 调用日志"""
        if not LLM_LOG_DIR:
            return
        
        os.makedirs(LLM_LOG_DIR, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(LLM_LOG_DIR, f"llm_call_{timestamp}_{self.call_count}.json")
        
        log_data = {
            "timestamp": timestamp,
            "model": self.model,
            "call_number": self.call_count,
            "elapsed_seconds": elapsed,
            "is_error": is_error,
            "messages": messages,
            "response": response,
        }
        
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
    
    def get_stats(self) -> Dict:
        """获取调用统计"""
        return {
            "model": self.model,
            "total_calls": self.call_count,
            "base_url": self.base_url,
        }
    
    def test_connection(self) -> bool:
        """测试与 Ollama 的连接是否正常"""
        try:
            response = self.chat(
                messages=[{"role": "user", "content": "请回复'连接成功'"}],
                system_prompt="你是一个测试助手。只回复'连接成功'，不要回复其他内容。",
            )
            success = "连接成功" in response
            if success:
                print(f"✅ Ollama 连接测试成功！模型: {self.model}")
            else:
                print(f"⚠️  Ollama 返回了内容，但格式不符: {response[:100]}")
            return success
        except Exception as e:
            print(f"❌ Ollama 连接测试失败: {e}")
            return False


# ==================== 便捷函数 ====================

def create_client(model: str = None) -> OllamaClient:
    """
    快速创建一个 Ollama 客户端
    
    Args:
        model: 模型名称，默认使用 config.py 中的配置
    
    Returns:
        OllamaClient 实例
    """
    return OllamaClient(model=model)


def quick_chat(message: str, model: str = None) -> str:
    """
    快速发送一条消息并获取回复
    
    Args:
        message: 用户消息
        model: 模型名称
    
    Returns:
        LLM 回复
    """
    client = OllamaClient(model=model)
    return client.chat([{"role": "user", "content": message}])


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 50)
    print("Ollama 客户端测试")
    print("=" * 50)
    
    # 创建客户端
    client = OllamaClient()
    
    # 测试连接
    print("\n📡 测试连接...")
    client.test_connection()
    
    # 测试对话
    print("\n💬 测试对话...")
    response = client.chat_with_system(
        user_message="你好！请用一句话介绍你自己。",
        system_prompt="你是一个友好的助手。请用简洁的中文回答。",
    )
    print(f"回复: {response}")
    
    # 测试结构化输出
    print("\n📊 测试结构化输出...")
    response = client.chat_with_system(
        user_message="请输出一个JSON：包含 name 和 age 两个字段。",
        system_prompt="你只输出JSON格式，不要输出任何其他文字。",
    )
    print(f"回复: {response}")
    
    # 打印统计
    print("\n📈 调用统计:")
    stats = client.get_stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")
    
    print("\n✅ 测试完成！")
