<a id="top"></a>

![AI Kit — bộ công cụ phát triển agent](assets/banner.png)

<div align="center">

[English: README.md](README.md)

**Bộ công cụ đã kiểm thử để làm ứng dụng LLM và agent, lắp sẵn thành một app FastAPI + LangGraph chạy được.**

[![CI](https://github.com/awun0105/ai-kit/actions/workflows/ci.yaml/badge.svg)](https://github.com/awun0105/ai-kit/actions/workflows/ci.yaml)
[![Giấy phép](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)

</div>

---

## Mục lục

- [Tổng quan](#tổng-quan)
- [Tính năng chính](#tính-năng-chính)
- [Cấu trúc kho mã nguồn](#cấu-trúc-kho-mã-nguồn)
- [Hướng dẫn bắt đầu](#hướng-dẫn-bắt-đầu)
- [Sử dụng nâng cao](#sử-dụng-nâng-cao)
- [Tài liệu hướng dẫn](#tài-liệu-hướng-dẫn)
- [Lời cảm ơn](#lời-cảm-ơn)
- [Bản quyền](#bản-quyền)

---

## Tổng quan

AI Kit gom những phần mà sản phẩm LLM nào rồi cũng cần: truy xuất tài liệu, lớp bảo vệ, retry
qua nhiều model, đánh giá chất lượng và quy trình phát hành. Mỗi phần nằm trong một package
riêng dưới `app/services/`, có test chạy offline và một trang tài liệu. Phần nào tốn tiền hoặc
làm thay đổi hành vi thì tắt sẵn. Các phần đã được ghép thành một app agent chạy được, nên
project mới có thể bắt đầu từ cả app, hoặc chỉ lấy một phần lắp vào codebase có sẵn.

## Tính năng chính

- **Lõi agent:** agent LangGraph lưu checkpoint trong Postgres, gọi tool, chờ người duyệt, có bộ nhớ dài hạn mem0, phiên JWT, giới hạn tần suất, ngân sách lịch sử và tùy chọn tóm tắt các lượt cũ.
- **RAG:** nạp md, txt, pdf, docx, html; tìm kiếm kết hợp pgvector với BM25 có tách từ tiếng Việt; reranker tùy chọn; câu trả lời có trích dẫn được kiểm tra.
- **Lớp bảo vệ:** chặn prompt injection và dữ liệu cá nhân bằng tiếng Anh lẫn tiếng Việt, LLM rail tùy chọn, duyệt tay các tool rủi ro, danh sách đích được phép gọi ra ngoài, che dữ liệu ngay khi stream.
- **Độ tin cậy:** một lớp retry duy nhất có ngân sách retry, circuit breaker cho từng model, chuỗi fallback qua nhiều model, cache câu trả lời phục vụ bản cũ khi nhà cung cấp gặp sự cố.
- **Tool và agent:** MCP client và server, A2A server và client, nhóm agent nghiên cứu sâu, `ResilientChatModel` dùng được cho mọi agent LangChain.
- **Đánh giá:** golden set với chỉ số truy xuất, judge là một model riêng, kiểm tra tool được gọi, chạy lặp nhiều lần, giả lập người dùng hội thoại nhiều lượt và gate trong CI.
- **Phát hành:** Docker image không chạy bằng root, đẩy lên GHCR sau khi quét Trivy, cảnh báo Prometheus, giới hạn chi phí hằng tháng cho từng người dùng.

## Cấu trúc kho mã nguồn

| Đường dẫn | Nội dung |
|---|---|
| `app/services/` | Bộ công cụ: `rag/`, `guardrails/`, `reliability/`, `evaluation/`, `mcp/`, `a2a/`, `agents/`, `cost_guard.py`, `compaction.py` |
| `app/core/kit.py` | Dựng từng phần của bộ công cụ từ cấu hình; nơi duy nhất đọc cấu hình |
| `app/core/langgraph/` | Graph của agent: kiểm tra đầu vào, chat, tool, kiểm tra đầu ra |
| `app/api/v1/` | Route đăng nhập, chat và stream |
| `scripts/` | `bootstrap`, `ingest`, `eval_rag`, `research` |
| `evals/datasets/` | Tài liệu mẫu, golden set và kịch bản hội thoại |
| `tests/` | Test offline cho từng phần |
| `docs/` | Mỗi phần một trang; `ROADMAP.md` liệt kê tất cả |

## Hướng dẫn bắt đầu

Cần Python 3.13 trở lên, [uv](https://docs.astral.sh/uv/) và Docker.

1. Tạo project và chọn tính năng (RAG, cache câu trả lời, LLM rail, reranker, MCP, A2A, nghiên cứu sâu):

   ```bash
   uvx copier copy --trust gh:awun0105/ai-kit my-app
   cd my-app && git init
   ```

   Lệnh này tạo `.env.development` với JWT secret ngẫu nhiên và bật sẵn các tính năng đã chọn.

2. Cài đặt rồi chạy test offline (không cần API key):

   ```bash
   uv sync        # thêm --extra rerank / a2a / deep cho các tính năng đó
   make test
   ```

3. Điền `OPENAI_API_KEY` vào `.env.development`, rồi chạy API cùng Postgres và pgvector:

   ```bash
   make docker-up
   ```

4. Nạp tài liệu vào kho tri thức rồi mở `http://localhost:8000/docs`:

   ```bash
   make ingest CORPUS=path/to/docs
   ```

<details>
<summary>Gửi request đầu tiên từ dòng lệnh</summary>

```bash
TOKEN=$(curl -s -X POST localhost:8000/api/v1/auth/register -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "password": "Secret123!", "username": "you"}' | jq -r .token.access_token)  # pragma: allowlist secret
SESSION=$(curl -s -X POST localhost:8000/api/v1/auth/session -H "Authorization: Bearer $TOKEN" | jq -r .token.access_token)
curl -X POST localhost:8000/api/v1/chatbot/chat -H "Authorization: Bearer $SESSION" \
  -H "Content-Type: application/json" -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

</details>

Chạy trên máy mà không dùng Docker: xem [docs/getting-started.md](docs/getting-started.md).

## Sử dụng nâng cao

**Lấy một phần sang project khác.** Mỗi package dưới `app/services/` import được độc lập, không
kéo theo cấu hình, database hay danh sách model của app (`tests/test_toolbox_imports.py` kiểm tra
điều này). Copy package cùng các test đi kèm, rồi làm theo hai mục **Copy this module** và **When you
plug it in** trong trang tài liệu tương ứng. `app/core/kit.py` cho thấy app dựng phần đó từ cấu
hình như thế nào.

**Cập nhật cải tiến mới của kit** vào project đã tạo bằng `uvx copier update --trust`.

**Đánh giá trước khi phát hành:** `make eval-rag-offline` kiểm tra truy xuất mà không cần API key
(đây là gate trong CI); `make eval-rag` cho agent thật trả lời golden set và dùng một model làm
judge. Các tùy chọn như `--runs 3` và `--scenarios` có trong [docs/evaluation.md](docs/evaluation.md).

## Tài liệu hướng dẫn

Nên đọc [docs/ROADMAP.md](docs/ROADMAP.md) trước: danh sách các phần, trạng thái và nguồn gốc của
từng phần. Mỗi phần có trang riêng: [RAG](docs/rag.md), [lớp bảo vệ](docs/guardrails.md),
[độ tin cậy](docs/reliability.md), [tool](docs/tools.md), [agent](docs/agents.md),
[đánh giá](docs/evaluation.md), [lịch sử hội thoại](docs/compaction.md),
[phát hành](docs/deployment.md). Cấu hình nằm trong [docs/configuration.md](docs/configuration.md);
quy tắc viết code cho người và coding agent nằm trong [AGENTS.md](AGENTS.md). Tài liệu chi tiết
hiện chỉ có bản tiếng Anh.

## Lời cảm ơn

- [fastapi-langgraph-agent-production-ready-template](https://github.com/wassim249/fastapi-langgraph-agent-production-ready-template) (MIT): app nền.
- Các lab AI20K (VinUniversity): D08, D09, D11, D12, D14 (lời giải của tác giả) và D16, D18, D23, D24, D25 (đề bài).
- [Google ADK](https://github.com/google/adk-python): ý tưởng tóm tắt lịch sử và đánh giá agent, viết lại cho LangGraph.
- LangGraph, LangChain, deepagents, FastMCP, a2a-sdk, MarkItDown, underthesea, bm25s, rerankers, pgvector.

Trang tài liệu của từng phần ghi rõ file và commit gốc.

## Bản quyền

Giấy phép MIT, kế thừa từ template gốc: xem [LICENSE](LICENSE). Để báo lỗi bảo mật, xem
[SECURITY.md](SECURITY.md).

---

<div align="center">

[Về đầu trang](#top)

</div>
