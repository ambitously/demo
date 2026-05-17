# Ollama环境搭建完全指南 —— 从Windows本地测试到云服务器部署

---

## 一、什么是Ollama（给完全不懂的人）

### 1.1 最简单的比喻

想象你的手机上有"应用商店"，你想用一个APP，就点一下"下载"，等几秒钟，APP就装好了，然后点"打开"就能用。

**Ollama 就是大模型的"应用商店"。**

- 你想用 `qwen2.5:7b` 这个模型 → 敲一行命令 → 等它下载完 → 直接就能对话
- 想换 `llama3.3:70b` → 再敲一行命令 → 自动下载 → 立即可用
- 不想用了 → 敲一行命令删除

整个过程你不需要懂什么"CUDA版本""PyTorch依赖""显卡驱动配置"——Ollama 把这些脏活累活全都帮你搞定了。

### 1.2 Ollama vs 直接调用 OpenAI API

| 对比维度 | Ollama | OpenAI API |
|---------|--------|------------|
| **运行位置** | 你自己的电脑/服务器 | OpenAI 的云端服务器 |
| **费用** | 免费（用你自己的硬件） | 按 token 收费 |
| **网络要求** | 下载模型时需要网络，运行时不需要 | 每次调用都要网络 |
| **隐私** | 数据完全留在本地 | 数据发送到 OpenAI 服务器 |
| **速度** | 取决于你自己的硬件 | 取决于你的套餐和排队情况 |
| **模型选择** | 任意开源模型（qwen、llama等） | 只能用 OpenAI 的模型 |

### 1.3 为什么竞赛要用 Ollama 而不是直接调 API

1. **离线可用**：竞赛环境中网络可能不稳定，Ollama 在本地跑，不依赖外部网络
2. **免费无限制**：API 有调用次数限制和费用，Ollama 跑一次装好想调多少调多少
3. **可控性强**：你可以完全控制模型的参数、超时、并发策略
4. **不需要审核**：API 有时会拒绝某些请求，Ollama 本地运行没有内容审核限制
5. **延迟更低**：本地跑没有网络往返时间，适合需要实时响应的机器人大脑场景

---

## 二、在Windows上安装Ollama（本地测试用）

> **重要说明**：Windows 上装 Ollama 只是为了在你自己的笔记本上先跑通、先学会。实际比赛用云服务器跑大模型，本地的 Windows 版本只是你的"训练场"。

### 2.1 下载与安装

1. 打开浏览器，访问：**https://ollama.com/download/windows**
2. 点击下载按钮，会得到一个 `OllamaSetup.exe` 文件
3. 双击运行，一路点"下一步"，什么都不用改
4. 安装完成后，它会自动在后台运行（任务栏右下角会出现一个小羊驼图标🦙）

> 如果安装时提示需要管理员权限，点"是"即可。

### 2.2 验证安装

打开 PowerShell（按 `Win + R`，输入 `powershell` 回车），运行：

```powershell
ollama --version
```

**预期输出**：类似 `ollama version is 0.5.xx`（版本号可能更新）

如果显示版本号，说明安装成功。如果显示"不是可识别的命令"，关掉 PowerShell 重新打开一次（安装后环境变量需要新窗口才能生效）。

### 2.3 拉取一个小模型测试

```powershell
ollama pull qwen2.5:0.5b
```

**这条命令在做什么？** 从 Ollama 的模型仓库下载 `qwen2.5` 的 `0.5b` 版本（0.5B 参数量，大约只占 400MB，超小超快，专门用来验证环境）。

**预期输出**：
```
pulling manifest
pulling 8f5c... 100% ▕████████████████████▏ 397 MB
pulling 66b9... 100% ▕████████████████████▏   68 B
verifying sha256 digest
writing manifest
success
```

### 2.4 测试对话

```powershell
ollama run qwen2.5:0.5b
```

运行后会进入交互式对话界面，你会看到类似这样的提示：
```
>>> Send a message (/? for help)
```

试着输入：
```
你好，请用一句话介绍你自己
```

**预期输出**：模型会回复一段中文，比如"我是通义千问，由阿里巴巴开发的大语言模型……"

> 输入 `/bye` 可以退出对话。

---

## 三、Ollama的基本使用

这里列出你最常用的命令。**每条命令都在 PowerShell 中运行。**

### 3.1 下载模型

```powershell
ollama pull 模型名:版本
```

示例：
```powershell
ollama pull qwen2.5:7b          # 下载7B版本
ollama pull llama3.3:70b        # 下载70B版本
```

### 3.2 运行模型对话

```powershell
ollama run 模型名:版本
```

这会启动一个交互式终端，你可以直接和模型对话。输入 `/bye` 退出。

### 3.3 查看已下载的模型

```powershell
ollama list
```

**预期输出示例**：
```
NAME                SIZE      MODIFIED
qwen2.5:0.5b        397 MB    2 hours ago
qwen2.5:7b          4.7 GB    1 hour ago
```

### 3.4 删除模型

```powershell
ollama rm 模型名:版本
```

示例：
```powershell
ollama rm qwen2.5:0.5b
```

> 删除后硬盘空间就释放了。想再用的话重新 pull 就行。

### 3.5 启动API服务

```powershell
ollama serve
```

这条命令启动一个 HTTP API 服务，默认监听 `http://localhost:11434`。Python 程序就是通过这个接口调用模型的。

> **注意**：Windows 版安装后会自动在后台启动服务，所以通常你不需要手动执行这条命令。但如果发现 Python 连不上，先确认服务在跑。

### 3.6 查看正在运行的模型

```powershell
ollama ps
```

**预期输出示例**：
```
NAME                SIZE      PROCESSOR    UNTIL
qwen2.5:7b          6.7 GB    100% GPU     4 minutes from now
```

> 如果显示 `no models running`，说明当前没有模型被加载。

---

## 四、常用模型介绍

### 4.1 模型列表

| 模型名 | 参数量 | 大小 | 内存需求 | 适用场景 | 推理速度 |
|--------|--------|------|---------|----------|---------|
| `qwen2.5:0.5b` | 0.5B | ~400MB | 1GB内存 | 测试环境是否正常 | 极快 |
| `qwen2.5:7b` | 7B | ~4.7GB | 8GB内存 | 开发和调试代码 | 快 |
| `qwen2.5:32b` | 32B | ~19GB | 24GB显存 | 中规模验证 | 中等 |
| `llama3.3:70b` | 70B | ~40GB | 40GB显存+ | 正式比赛使用 | 较慢 |
| `deepseek-r1:1.5b` | 1.5B | ~1.1GB | 2GB内存 | 快速实验 | 极快 |

### 4.2 各模型在竞赛中的角色

- **`qwen2.5:0.5b`**：你刚开始学，在笔记本上装 Ollama 后跑这个就行。400MB 大小，10年前的老电脑也能跑。**仅用于确认"Ollama能正常工作"**，不要用它做任何实际任务。

- **`qwen2.5:7b`**：你的主力开发调试模型。在你本机或者普通云服务器上都能跑。写 prompt、调代码逻辑、验证 pipeline 都用它。

- **`llama3.3:70b`**：最终比赛的模型。需要大显存的 GPU 服务器（如 A100、H100 等）。前期不需要碰它——先用 7B 的调通了，最后换 70B 的就行（Ollama 下都是同样的调用方式）。

> **核心思路**：用 7B 做开发 → 用 70B 做最终运行。因为调用接口一模一样，切换只需改一行模型名。

---

## 五、Python调用Ollama（重点）

Ollama 本身是一个服务程序，Python 通过 HTTP 请求和它通信。有两种主流方式：

### 5.1 方式1：官方 Ollama Python 库

#### 安装

```powershell
pip install ollama
```

#### 基础用法

```python
import ollama

# 发送消息，获取回复
response = ollama.chat(
    model='qwen2.5:7b',
    messages=[
        {'role': 'system', 'content': '你是一个机器人大脑，负责控制一个四足机器人。'},
        {'role': 'user', 'content': '前方有障碍物，距离2米，高度30厘米。请给出动作指令。'}
    ]
)

print(response['message']['content'])
# 预期输出类似：{"action": "jump", "target_height_cm": 35, ...}
```

#### 流式输出（逐字输出，像打字机效果）

```python
stream = ollama.chat(
    model='qwen2.5:7b',
    messages=[{'role': 'user', 'content': '写一首关于机器人的诗'}],
    stream=True
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
print()  # 最后换个行
```

### 5.2 方式2：OpenAI 兼容接口（推荐，兼容性更好）

Ollama 提供了一个和 OpenAI API 完全兼容的接口。这意味着：**任何为 OpenAI 写的代码，改两行就能用 Ollama**。

#### 安装

```powershell
pip install openai
```

#### 基础用法

```python
from openai import OpenAI

# 注意 base_url 指向 Ollama，不是 OpenAI 的服务器
# api_key 随便填一个字符串就行，Ollama 不校验
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'  # Ollama 不需要真实 key，但必须填一个非空值
)

response = client.chat.completions.create(
    model='qwen2.5:7b',
    messages=[
        {'role': 'system', 'content': '你是一个机器人大脑，负责控制一个四足机器人。'},
        {'role': 'user', 'content': '前方有障碍物，距离2米，高度30厘米。请给出动作指令。'}
    ]
)

print(response.choices[0].message.content)
# 预期输出和方式1一样
```

#### 流式输出

```python
stream = client.chat.completions.create(
    model='qwen2.5:7b',
    messages=[{'role': 'user', 'content': '写一首关于机器人的诗'}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='', flush=True)
print()
```

### 5.3 两种方式对比

| 对比维度 | 官方 ollama 库 | OpenAI 兼容接口 |
|---------|---------------|----------------|
| **代码量** | 少一点 | 略多一点 |
| **功能** | Ollama 专属功能（如模型管理） | 标准 Chat Completions |
| **兼容性** | 只能用 Ollama | 同一份代码可切 OpenAI / Ollama / 其他 |
| **文档丰富度** | 较少 | 极丰富（OpenAI 文档直接适用） |
| **推荐场景** | 只需模型管理的脚本 | 需要灵活切换后端的项目 |

> **推荐**：竞赛中使用方式2（OpenAI 兼容接口）。因为你写的代码以后要切到真实 OpenAI 测试、或者切到其他兼容接口都很方便，**只改 `base_url` 和 `api_key` 就行**。

### 5.4 完整可运行示例

把下面代码保存为 `test_ollama.py`，然后 `python test_ollama.py` 运行：

```python
from openai import OpenAI

# 配置：比赛时这里改成云服务器的IP和端口即可
OLLAMA_HOST = 'http://localhost:11434/v1'

def ask_robot_brain(observation: str) -> str:
    """给机器人脑子上"看到"的环境信息，让它返回动作指令"""
    client = OpenAI(base_url=OLLAMA_HOST, api_key='ollama')
    
    response = client.chat.completions.create(
        model='qwen2.5:7b',
        messages=[
            {
                'role': 'system',
                'content': (
                    '你是一个四足机器人的控制系统。'
                    '根据接收到的环境观测数据，输出下一步的动作指令。'
                    '指令格式为JSON：{"action": "walk"|"turn"|"jump"|"stop", "params": {...}}'
                )
            },
            {'role': 'user', 'content': observation}
        ],
        temperature=0.1,  # 低温度使输出更确定，适合控制任务
        max_tokens=500
    )
    
    return response.choices[0].message.content


if __name__ == '__main__':
    # 模拟一个环境观测
    obs = '前方1.5米处有15厘米高的障碍物。当前速度为0.5 m/s，方向为正前方。'
    
    result = ask_robot_brain(obs)
    print(f'模型输出: {result}')
```

---

## 六、在云服务器上安装Ollama（Ubuntu 22.04）

> 比赛使用云上的 GPU 服务器跑大模型。这一节针对 Ubuntu 22.04 系统。

### 6.1 一条命令安装

SSH 登录到你的云服务器后，运行：

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**这条命令在做什么？** 从 Ollama 官网下载安装脚本并自动执行。它会自动检测你的系统、安装依赖、配置好一切。

**预期输出**：
```
>>> Installing ollama to /usr/local
>>> Downloading Linux amd64 bundle
>>> ...
>>> The Ollama API is now available at 127.0.0.1:11434.
>>> Install complete. Run "ollama" from the command line.
```

### 6.2 验证安装

```bash
ollama --version
```

### 6.3 拉取模型（和 Windows 上完全一样的命令）

```bash
ollama pull qwen2.5:7b
```

### 6.4 启动API服务

Ollama 安装后在 Ubuntu 上默认不会后台运行，你需要手动启动：

#### 方式A：前台运行（调试时用）

```bash
ollama serve
```

在前台运行，按 `Ctrl+C` 就停了。适合调试时用。

#### 方式B：后台运行（比赛时用）

使用 `nohup` 让服务在后台运行，即使你关闭 SSH 连接也不会停：

```bash
nohup ollama serve > /tmp/ollama.log 2>&1 &
```

**这条命令在做什么？**
- `nohup` — 让命令在你退出登录后继续运行
- `ollama serve` — 启动 Ollama API 服务
- `> /tmp/ollama.log 2>&1` — 把输出和错误信息都写到 `/tmp/ollama.log`
- `&` — 放到后台运行

运行后会输出一个数字（进程ID），比如 `[1] 12345`。

#### 方式C：使用 screen（推荐）

```bash
# 创建一个名为 ollama 的后台会话
screen -S ollama

# 进入 screen 后，启动服务
ollama serve

# 按 Ctrl+A 然后按 D 可以退出 screen（服务继续跑）

# 想重新看一下日志
screen -r ollama
```

### 6.5 检查服务是否正常

```bash
# 看 ollama 进程在不在
ps aux | grep ollama

# 测试 API 是否能通
curl http://localhost:11434/api/tags
```

**预期输出（curl）**：
```json
{"models":[{"name":"qwen2.5:7b","modified_at":"...","size":...}]}
```

---

## 七、修改模型存储路径（竞赛关键）

### 7.1 为什么必须改

Ollama 默认把下载的模型放在：
- **Linux**：`~/.ollama/models/`
- **Windows**：`C:\Users\你的用户名\.ollama\models\`

竞赛环境下，你可能会把自己的代码和环境打包成 Docker 镜像。如果几十 GB 的模型文件也打包进去，镜像会大得离谱，根本无法传输和部署。

**解决方案**：把模型存到一个"外部"路径（如挂载的数据盘），和镜像解耦。

### 7.2 如何修改

Ollama 通过环境变量 `OLLAMA_MODELS` 来指定模型存放路径。

#### 临时生效（当前终端）

```bash
export OLLAMA_MODELS="/data/ollama_models"
```

设置之后，所有 ollama 命令（pull、run、serve）都会用这个新路径。

#### 永久生效（推荐）

编辑 `~/.bashrc` 文件，把设置写入：

```bash
echo 'export OLLAMA_MODELS="/data/ollama_models"' >> ~/.bashrc
source ~/.bashrc
```

**解释**：
- 第一行把环境变量写入 `.bashrc` 配置文件
- 第二行使配置立即生效（不用重新登录）

此后每次登录服务器，这个环境变量都会自动生效。

### 7.3 竞赛中的典型目录结构

```
/data/ollama_models/    ← 模型存在这里（不会被镜像打包）
    └── blobs/          ← 模型的实际数据文件
    └── manifests/      ← 模型的元数据

~/project/              ← 你的代码在这里（会被镜像打包）
    ├── main.py
    ├── requirements.txt
    └── ...
```

### 7.4 验证模型路径已生效

```bash
echo $OLLAMA_MODELS
# 预期输出：/data/ollama_models

ollama pull qwen2.5:0.5b
# 下载后检查新路径下是否有文件
ls /data/ollama_models/
```

---

## 八、完整测试流程

以下是从零到验证成功的一整套流程。**跟着每一步敲命令即可**。

> 假设你已经在云服务器（Ubuntu 22.04）上。先 SSH 登录过去。

### Step 1：安装 Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Step 2：设置模型路径（可选但推荐）

```bash
echo 'export OLLAMA_MODELS="/data/ollama_models"' >> ~/.bashrc
source ~/.bashrc
```

### Step 3：启动 Ollama 服务

```bash
nohup ollama serve > /tmp/ollama.log 2>&1 &
```

### Step 4：拉取模型

```bash
ollama pull qwen2.5:7b
```

这一步需要下载约 4.7GB，等待进度条跑完。

### Step 5：验证模型已下载

```bash
ollama list
```

预期看到 `qwen2.5:7b`。

### Step 6：安装 Python 依赖

```bash
pip install openai
```

### Step 7：编写测试脚本

```bash
cat > test_ollama.py << 'EOF'
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'
)

response = client.chat.completions.create(
    model='qwen2.5:7b',
    messages=[
        {'role': 'system', 'content': '你是一个机器人大脑。你只输出JSON格式的动作指令。'},
        {'role': 'user', 'content': '机器人前方有障碍物，距离1米。请给出动作指令。'}
    ]
)

print('=== 调用成功！===')
print('模型返回:', response.choices[0].message.content)
print('消耗tokens:', response.usage.total_tokens)
EOF
```

### Step 8：运行测试

```bash
python test_ollama.py
```

### Step 9：验证输出

**预期输出**：
```
=== 调用成功！===
模型返回: {"action": "stop", "reason": "前方1米处有障碍物"}  （内容可能不完全一样）
消耗tokens: 85
```

如果看到类似上面这样的输出——**恭喜，你的 Ollama 环境搭建完毕！**

---

## 九、常见问题

### Q1：端口被占用 `bind: address already in use`

**现象**：启动 `ollama serve` 时报错端口 11434 已被占用。

**原因**：已经有一个 Ollama 服务在运行。

**解决**：
```bash
# 先找到占用 11434 端口的进程
lsof -i :11434           # Linux
netstat -ano | findstr 11434   # Windows PowerShell

# 杀掉它（把 PID 换成你查到的数字）
kill -9 <PID>             # Linux
taskkill /PID <PID> /F    # Windows
```

或者，如果你想多个 Ollama 共存，可以改端口：
```bash
export OLLAMA_HOST="0.0.0.0:11435"
ollama serve
```

### Q2：内存/显存不足

**现象**：拉取模型成功，但运行时崩溃或极慢。

**原因**：模型太大，超过了你机器能承受的范围。

**对照表**：

| 模型 | 需要内存 |
|------|---------|
| qwen2.5:0.5b | ~1GB |
| qwen2.5:7b | ~8GB |
| qwen2.5:32b | ~24GB |
| llama3.3:70b | ~40GB |

**解决**：
- 换小一点的模型（先用 7B 开发调试）
- 如果必须跑大模型，使用有足够显存的 GPU 服务器（如 A100 80GB）

> **你的笔记本大概率只能跑 7B 的模型。70B 需要云 GPU 服务器。**

### Q3：下载速度慢

**现象**：`ollama pull` 速度很慢，几 KB/s。

**原因**：Ollama 模型文件托管在境外，国内直连有时会比较慢。

**解决方法1——使用代理**（如果你有）：
```bash
# Linux / macOS
export HTTPS_PROXY=http://127.0.0.1:7890
ollama pull qwen2.5:7b
```

```powershell
# Windows PowerShell
$env:HTTPS_PROXY = "http://127.0.0.1:7890"
ollama pull qwen2.5:7b
```

**解决方法2——使用国内镜像**：

有些高校或机构提供了 Ollama 模型镜像。例如：
```bash
# 设置 OLLAMA_HOST 指向镜像站的 API（如果镜像站提供的话）
# 具体镜像站地址请自行搜索，这里不提供具体链接以防失效
```

**解决方法3——耐心等**：
下载是一次性的，下载完模型就一直在了。建议晚上挂着下载，第二天早上就好了。

### Q4：Windows WSL2 相关

**现象**：你在 WSL2（Windows 上的 Linux 子系统）里装了 Ollama，但 Python 连不上。

**原因**：WSL2 有独立的网络栈，`localhost` 可能不通。

**解决**：
```bash
# 在 WSL2 中，让 Ollama 监听所有网络接口
export OLLAMA_HOST="0.0.0.0:11434"
ollama serve
```

然后在 Python 中用 WSL2 的 IP 连接：
```python
# 用 WSL2 的 IP 而不是 localhost
# 在 PowerShell 中运行 wsl hostname -I 可以查到 IP
client = OpenAI(
    base_url='http://172.xx.xx.xx:11434/v1',  # 换成你的 WSL IP
    api_key='ollama'
)
```

> **建议**：Windows 用户直接用 Windows 版 Ollama（第二节的方法），不要在 WSL2 里装。Windows 版更方便、更稳定。

### Q5：`ollama` 命令找不到

**现象**：终端输入 `ollama` 提示"command not found"。

**解决**（Linux）：
```bash
# 确认安装位置
ls /usr/local/bin/ollama

# 如果文件存在，说明是 PATH 的问题
echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**解决**（Windows）：
关掉当前的 PowerShell / CMD 窗口，重新打开一个新的。安装程序添加了环境变量，但已打开的窗口不会自动刷新。

### Q6：模型加载后响应很慢

**现象**：第一次调用模型要等很久（几十秒），后面就快了。

**原因**：这是正常的。模型第一次使用时要加载到内存/显存，之后的调用都从缓存读取。相当于"冷启动慢、热启动快"。

**优化**：可以在启动服务后提前"预热"一下：
```bash
# 随便发一条请求，让模型加载进内存
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:7b",
  "prompt": "Hi",
  "stream": false
}'
```

### Q7：Docker 里的 Ollama 怎么用

如果比赛环境使用 Docker，不要在 Docker 镜像里装 Ollama。把 Ollama 作为独立容器跑，你的应用容器通过网络调用它：

```bash
# 拉 Ollama 官方镜像
docker run -d --name ollama \
  -p 11434:11434 \
  -v /data/ollama_models:/root/.ollama \
  ollama/ollama
```

然后你的代码里 `base_url='http://localhost:11434/v1'` 或者在 docker-compose 里用 `http://ollama:11434/v1`。

---

> **文档完成。** 如果遇到本文未覆盖的问题，可以在终端中发送 `/help` 或查阅 Ollama 官方文档：https://github.com/ollama/ollama
