# Mem Eval

一个可在本机或公司内网运行的群聊 Memory 评测平台。仓库自带：

- 172 道飞书群聊 Memory 问题
- 18 条 episode 历史轨迹、284 条消息
- BM25 RAG、Mem0、TeamAgent Memory 三种方法
- Top K = 3 / 10 的完整预置结果、Badcase 和 Leaderboard
- 可选的 Reader + LLM Judge 端到端评测

只浏览题库、预置结果和 Leaderboard **不需要 API Key，也不访问公网**。

## 最快启动：Python

要求 Python 3.11。进入仓库根目录：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/run_local.sh
```

上述最小依赖足够浏览全部页面并现场运行 BM25。若还要重新运行 Mem0，再安装：

```bash
python -m pip install -r requirements-full.txt
```

浏览器打开 [http://127.0.0.1:8501](http://127.0.0.1:8501)。

## 公司内网演示

### 方案 A：内网能访问 Python 包镜像

把整个仓库复制到内网电脑，配置公司的 pip 镜像后执行上面的 Python 启动命令。
命令行出现 `Local URL` 后：

- 在启动服务的电脑上打开 `http://127.0.0.1:8501`
- 在同一内网的其他电脑上打开 `http://<服务电脑内网IP>:8501`

服务默认监听 `0.0.0.0:8501`。若其他电脑无法访问，需要在防火墙中放行 TCP 8501。

### 方案 B：内网完全不能访问公网（推荐）

目标是 Intel/AMD Linux 服务器时，可以先在外网电脑下载 GitHub 自动构建的镜像：

```bash
docker pull ghcr.io/gyhthu/mem_eval:latest
docker save ghcr.io/gyhthu/mem_eval:latest -o mem-eval-amd64.tar
```

也可以在一台能联网、且能构建目标服务器架构镜像的机器上自行构建：

```bash
# Intel/AMD 内网服务器
bash scripts/build_offline_image.sh linux/amd64 mem-eval-amd64.tar

# ARM 服务器或 Apple Silicon Mac
bash scripts/build_offline_image.sh linux/arm64 mem-eval-arm64.tar
```

把生成的 `mem-eval-*.tar` 复制进内网，在内网电脑执行：

```bash
bash scripts/run_offline_image.sh mem-eval-amd64.tar
```

若使用 GHCR 镜像，启动前设置镜像名：

```bash
MEM_EVAL_IMAGE=ghcr.io/gyhthu/mem_eval:latest \
  bash scripts/run_offline_image.sh mem-eval-amd64.tar
```

然后打开 `http://127.0.0.1:8501`；其他内网电脑使用
`http://<服务电脑内网IP>:8501`。离线镜像已包含代码、依赖、题库、预置结果和 Mem0
中文 Embedding 模型，启动时不访问 PyPI 或 Hugging Face。

## 现场演示顺序（约 5 分钟）

1. 打开“题库”：展示 172 题、题型筛选、标准答案和 Oracle 证据消息。
2. 打开“Leaderboard”：切换 `Top K = 3` 和 `Top K = 10`，比较三种 Memory 方法。
3. 打开“结果与 Badcase”：选择一条完整结果，展示总体指标、分类指标和具体 Badcase。
4. 打开“启动评测”：取消勾选 Reader + Judge，现场跑一次 BM25（无需模型、几十秒内完成）。
5. 若已接好内网模型，再勾选 Reader + Judge 做少量或完整端到端评测。

## 接入公司内网模型

复制配置模板：

```bash
cp .env.example .env
```

将以下配置改成公司的 OpenAI-compatible Chat Completions 服务：

```text
EVAL_LLM_PROVIDER=openai
EVAL_LLM_BASE_URL=http://内网模型地址/v1
EVAL_LLM_API_KEY=内网令牌
EVAL_READER_MODEL=内网模型名
EVAL_JUDGE_MODEL=内网模型名
```

`.env` 已被 Git 忽略，不要提交密钥。TeamAgent 还需要 OpenAI-compatible Embeddings
接口；默认示例是本机 Ollama 的 `bge-m3`。

## 常用检查

```bash
curl http://127.0.0.1:8501/_stcore/health
lsof -nP -iTCP:8501 -sTCP:LISTEN   # macOS
ss -ltnp | grep 8501               # Linux
```

命令行运行纯检索评测：

```bash
PYTHONPATH=src .venv/bin/python -m eval_platform.runner \
  --method bm25_rag --top-k 3 --run-name "BM25 demo"
```
