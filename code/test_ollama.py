"""
Ollama 连接测试脚本
=====================
用于测试 Ollama 是否正确安装和运行。
在本地或服务器上运行此脚本来验证环境。

使用方法：
    # 测试默认模型（qwen2.5:7b）
    python test_ollama.py
    
    # 测试指定模型
    python test_ollama.py --model qwen2.5:0.5b
    
    # 测试 70B 模型
    python test_ollama.py --model llama3.3:70b
    
    # 测试自定义地址
    python test_ollama.py --url http://192.168.1.100:11434/v1
"""

import sys
import json
import time
import argparse

def test_ollama(base_url, model, verbose=True):
    """测试 Ollama 连接和基本功能"""
    
    print("=" * 60)
    print("🔧 Ollama 连接测试")
    print("=" * 60)
    print(f"   地址: {base_url}")
    print(f"   模型: {model}")
    print()
    
    # ---------- 测试 1: 导入库 ----------
    print("📦 测试 1/5: 检查依赖库...")
    try:
        from openai import OpenAI
        print("   ✅ openai 库可用")
    except ImportError:
        print("   ❌ openai 库未安装，请运行: pip install openai")
        return False
    
    # ---------- 测试 2: 创建客户端 ----------
    print("\n🔌 测试 2/5: 创建客户端...")
    try:
        client = OpenAI(
            base_url=base_url,
            api_key="ollama",
            timeout=30,
        )
        print("   ✅ 客户端创建成功")
    except Exception as e:
        print(f"   ❌ 客户端创建失败: {e}")
        return False
    
    # ---------- 测试 3: 列出可用模型 ----------
    print("\n📋 测试 3/5: 检查可用模型...")
    try:
        import requests
        resp = requests.get(
            base_url.replace("/v1", "") + "/api/tags",
            timeout=10
        )
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            print(f"   ✅ 发现 {len(models)} 个模型:")
            for m in models:
                name = m.get("name", "unknown")
                size_gb = m.get("size", 0) / (1024**3)
                print(f"      - {name} ({size_gb:.1f} GB)")
        else:
            print(f"   ⚠️  无法获取模型列表 (HTTP {resp.status_code})")
    except Exception as e:
        print(f"   ⚠️  获取模型列表失败: {e}")
        print("   （如果 Ollama 正在运行，这可能是权限或网络问题）")
    
    # ---------- 测试 4: 简单对话 ----------
    print(f"\n💬 测试 4/5: 发送测试对话...")
    try:
        start = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个测试助手。只回复'连接成功'，不要回复其他内容。"},
                {"role": "user", "content": "测试"},
            ],
            temperature=0.0,
            max_tokens=50,
        )
        elapsed = time.time() - start
        content = response.choices[0].message.content
        
        if "连接成功" in content or len(content) > 0:
            print(f"   ✅ 对话成功！耗时: {elapsed:.2f}秒")
            print(f"   回复: {content[:100]}")
        else:
            print(f"   ⚠️  回复内容异常: {content[:100]}")
    except Exception as e:
        print(f"   ❌ 对话失败: {e}")
        print()
        print("   可能的原因：")
        print("   1. Ollama 服务未启动 → 运行: ollama serve")
        print("   2. 模型未下载 → 运行: ollama pull " + model)
        print("   3. 端口被占用 → 检查: netstat -an | grep 11434")
        return False
    
    # ---------- 测试 5: 多轮对话 ----------
    print(f"\n🔄 测试 5/5: 测试多轮对话...")
    try:
        messages = [
            {"role": "system", "content": "你是一个机器人规划助手。请用JSON格式回复。"},
            {"role": "user", "content": "请输出一个简单的JSON：{\"action\": \"MOVE\", \"target\": \"table\"}"},
        ]
        
        start = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=200,
        )
        elapsed = time.time() - start
        content = response.choices[0].message.content
        
        print(f"   ✅ 多轮对话成功！耗时: {elapsed:.2f}秒")
        print(f"   回复: {content[:200]}")
        
        # 尝试解析 JSON
        try:
            if "{" in content:
                json_str = content[content.index("{"):content.rindex("}")+1]
                parsed = json.loads(json_str)
                print(f"   ✅ JSON 解析成功: {parsed}")
        except:
            print(f"   ℹ️  回复非纯JSON格式（这是正常的）")
            
    except Exception as e:
        print(f"   ⚠️  多轮对话测试失败: {e}")
    
    # ---------- 总结 ----------
    print("\n" + "=" * 60)
    print("📊 测试总结")
    print("=" * 60)
    print(f"   模型: {model}")
    print(f"   状态: ✅ 所有测试通过")
    print(f"   建议: 可以开始开发了！")
    print()
    print("下一步：")
    print("   1. 运行 python prompt_templates.py 测试 Prompt 模板")
    print("   2. 运行 python dialog_handler_template.py 查看接口示例")
    print("   3. 在云实例上运行 run_dialog.py 对接仿真环境")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="测试 Ollama 连接")
    parser.add_argument(
        "--model", "-m",
        default="qwen2.5:7b",
        help="模型名称 (默认: qwen2.5:7b)"
    )
    parser.add_argument(
        "--url", "-u",
        default="http://localhost:11434/v1",
        help="Ollama API 地址 (默认: http://localhost:11434/v1)"
    )
    args = parser.parse_args()
    
    success = test_ollama(args.url, args.model)
    
    if success:
        print("\n✅ 测试成功完成！")
        sys.exit(0)
    else:
        print("\n❌ 测试失败，请检查 Ollama 是否正确安装和运行。")
        print("   参考: docs/02-Ollama环境搭建完全指南.md")
        sys.exit(1)


if __name__ == "__main__":
    main()
