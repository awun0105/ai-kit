# Harness cho agent trong app (in-app agent harness)

Ngày: 2026-10-27. Trạng thái: bản nháp, chờ duyệt.

## 1. Mục đích

Khi làm một app (viết bằng bất kỳ ngôn ngữ nào) và muốn agent **thao tác ngay trong app**, cần có một
quy trình chuẩn để agent làm đúng: biết ngữ cảnh, chỉ sửa trên bản nháp, có người duyệt thao tác nguy
hiểm, tự kiểm tra kết quả, và hoàn tác được. fframes Studio đã có quy trình như vậy, nhưng viết riêng
cho app đó bằng Rust.

AI Kit sẽ chứa quy trình đó ở dạng **dùng lại được**:

- **Spec (đặc tả) và JSON schema** không phụ thuộc ngôn ngữ.
- **Harness mẫu bằng Python**, chạy được ngay, dùng làm sidecar (tiến trình chạy kèm app) hoặc làm
  bản gốc để port.
- **Conformance test suite (bộ test kiểm tra tuân thủ)** chạy qua giao thức: bản port nào qua được bộ
  test là đúng flow.
- **Hướng dẫn port**: phần nào dịch sang ngôn ngữ của app, phần nào giữ ở Python và gọi qua MCP.

Cách dùng mà người dùng muốn: khi làm app mới, vào kit lấy spec và bản mẫu, rồi port sang ngôn ngữ
của app, hoặc chạy luôn bản Python làm sidecar nếu app chấp nhận được.

## 2. Mục tiêu, ngoài phạm vi, điều không được xảy ra

**Mục tiêu**

1. Một app chỉ cần viết **adapter mỏng** (khai báo thao tác, trả trạng thái, áp dụng và hoàn tác thay
   đổi) là có đủ quy trình agent.
2. Cùng một app dùng được nhiều loại agent: agent của kit, và agent bên ngoài qua ACP (Agent Client
   Protocol: Claude Code, Codex, Gemini CLI, Pi...).
3. Mọi thay đổi của agent đi qua **draft → validate → (duyệt) → revision**, hoàn tác được.
4. Quyền theo từng task: tool đọc dùng tự do, tool ghi chỉ chạm vào draft của task, thao tác nguy hiểm
   phải có người duyệt.
5. Bộ test kiểm tra tuân thủ chạy được với adapter viết bằng ngôn ngữ bất kỳ.
6. Agent của kit chạy được như một ACP agent, để app nào làm ACP client cũng nhúng được.

**Ngoài phạm vi (đợt này)**

- Port sẵn sang C++, Rust hay ngôn ngữ khác. Chỉ port khi có app cụ thể cần.
- Kênh riêng cho Claude Code (stream-json) và Codex (app-server) như fframes có. Đợt này đi qua ACP,
  dùng adapter có sẵn (`claude-code-acp`, `codex-acp`).
- Giao diện người dùng (khung chat, xem diff). Harness chỉ cung cấp sự kiện và dữ liệu để app tự vẽ.
- Bộ offline (SQLite, embedding trong process). Đó là dự án riêng, làm sau.
- Sandbox (cô lập) tiến trình agent ở mức hệ điều hành như fframes làm (job object, process group).
  Ghi rõ trong spec là chưa có.

**Điều không được xảy ra (dù về kỹ thuật vẫn chạy)**

- Agent ghi thẳng vào dữ liệu thật của app mà không qua draft.
- Thao tác nguy hiểm chạy mà không có người duyệt, kể cả khi agent tự nhận là "đã được phép".
- Spec bị gắn với LangGraph, Python hay một agent cụ thể, khiến không port được.
- Quyền của một task còn hiệu lực sau khi task kết thúc.

## 3. Khái niệm

| Khái niệm | Nghĩa |
|---|---|
| **App** | Ứng dụng chủ, viết bằng bất kỳ ngôn ngữ nào |
| **Adapter** | Phần mỏng app tự viết, nói chuyện với harness qua JSON-RPC |
| **Harness** | Phần điều phối: task, quyền, draft, kết nối agent, ghi log, eval |
| **Agent** | "Não": agent của kit (chạy trong harness) hoặc agent bên ngoài qua ACP |
| **Operation (thao tác)** | Một việc app làm được, có tên, JSON schema tham số và cấp độ `read`, `write` hoặc `dangerous` |
| **Task** | Một yêu cầu của người dùng giao cho agent, từ lúc giao đến lúc chấp nhận, từ chối hoặc hủy |
| **Draft** | Bản nháp tách khỏi dữ liệu thật; mọi thao tác `write` của task chỉ áp dụng lên draft |
| **Revision** | Một trạng thái đã được chấp nhận; hoàn tác là quay về revision trước |
| **Capability** | Token quyền dùng tool, gắn với đúng một task, có thời hạn, bị thu hồi khi task kết thúc |

## 4. Kiến trúc

```
 App (C++, Rust, Python, JS...)        Harness (bản mẫu Python, hoặc bản port)         Agent
 ┌───────────────────────┐ JSON-RPC  ┌──────────────────────────────────┐  ACP    ┌──────────────┐
 │ adapter:              │ ◄───────► │ task state machine               │ ◄─────► │ Claude Code, │
 │  describe, context,   │ stdio     │ permission policy + capability   │         │ Codex, Pi... │
 │  draft.*, op.call,    │ hoặc      │ tool broker (MCP cho agent)      │  MCP    └──────────────┘
 │  validate, commit,    │ socket    │ agent drivers: in-process | ACP  │ ◄─────► agent ngoài gọi tool
 │  undo                 │           │ transcript, redaction, eval      │         ┌──────────────┐
 └───────────────────────┘           │                                  │ ◄─────► │ agent của kit│
        ▲  sự kiện cho UI            └──────────────────────────────────┘ in-proc └──────────────┘
        └────────────────────────────── task.*, permission.*, message.* events
```

Có hai cách triển khai, cùng một spec:

- **Sidecar**: app khởi động harness Python thành tiến trình riêng. App chỉ viết adapter.
- **Port**: app viết lại harness core bằng ngôn ngữ của mình theo spec, rồi chạy bộ test kiểm tra.

## 5. Các hợp đồng (phần không phụ thuộc ngôn ngữ)

Mọi hợp đồng là JSON-RPC 2.0, mỗi bản tin một dòng JSON. Tên method dưới đây là đề xuất; builder
được đổi nếu có lý do, miễn là spec, schema và bộ test khớp nhau.

### 5.1 Adapter protocol (harness gọi app)

| Method | Việc |
|---|---|
| `app.describe` | Trả danh sách operation (tên, mô tả, JSON schema tham số, cấp độ) và phiên bản protocol |
| `app.context` | Trạng thái hiện tại có giới hạn kích thước, kèm revision hiện tại và thứ người dùng đang chọn |
| `draft.open(base_revision)` | Tạo draft từ một revision; trả `draft_id` |
| `op.call(draft_id, op, args)` | Chạy một operation. `read` có thể không cần draft; `write` chỉ chạy trên draft |
| `draft.validate(draft_id)` | Trả `ValidationReport`: đạt hay không, danh sách lỗi có cấu trúc |
| `draft.diff(draft_id)` | Thay đổi so với base, ở dạng app tự định nghĩa nhưng có phần tóm tắt dạng chữ |
| `draft.commit(draft_id, base_revision)` | Áp dụng draft thành revision mới; trả `conflict` nếu base đã đổi |
| `draft.discard(draft_id)` | Bỏ draft |
| `revision.undo(revision)` | Tạo draft đảo ngược một revision, đi qua validate như mọi draft khác |

App tự quyết định draft là gì: bản sao thư mục, một transaction chưa commit, một nhánh trong bộ nhớ.
Spec chỉ yêu cầu draft không ảnh hưởng dữ liệu thật trước `draft.commit`.

### 5.2 Tool surface cho agent

Harness biến các operation thành tool cho agent:

- Agent của kit: tool LangChain, đi qua tool node có guardrails như hiện nay.
- Agent bên ngoài: MCP server của harness (stdio hoặc socket cục bộ), mỗi task một capability.

Tool luôn gắn với task đang chạy: agent không chọn được `draft_id`; harness tự điền. Tool `write` không
thấy dữ liệu thật. Có thêm hai tool do harness cung cấp: `task_context` (mục tiêu, phạm vi, revision gốc)
và `validate_draft` (để agent tự kiểm tra trước khi kết thúc).

### 5.3 Agent driver

Interface chung cho mọi loại agent, rút từ `AgentDriver` của fframes:

- `start(session_config)`, `restore(session_id)` nếu agent hỗ trợ
- `prompt(text, attachments)` trả số lượt; mỗi lượt kết thúc bằng đúng một sự kiện `turn.finished`
- `cancel_turn()`, `reply_permission(id, choice)`, `shutdown()`
- `capabilities`: khôi phục session, nhận ảnh hay file, dừng giữa lượt
- Sự kiện: `message.delta`, `tool.call`, `tool.update`, `permission.requested`, `permission.closed`,
  `plan`, `turn.finished(stop_reason)`, `failure`

Hai bản cài đặt trong đợt này: agent của kit chạy trong process, và ACP client.

### 5.4 Task state machine

Rút gọn từ máy trạng thái của fframes:

```
ContextReady → Running ⇄ WaitingPermission
Running → Validating                      (agent báo xong lượt)
Validating → CandidateReady               (validate đạt)
Validating → RepairNeeded → Running       (tối đa 1 lần tự sửa, cấu hình được)
RepairNeeded → Failed                     (hết lượt sửa)
CandidateReady → Accepted                 (theo review policy)
CandidateReady → Rejected | Conflict      (người dùng từ chối | base đã đổi)
mọi trạng thái đang chạy → Cancelled
```

- **Review policy**: `auto` (validate đạt thì commit, vẫn hoàn tác được) hoặc `manual` (người dùng
  bấm chấp nhận, có thể chỉ chấp nhận một phần nếu app hỗ trợ).
- **Một writer**: mỗi app hoặc project chỉ một task ghi tại một thời điểm.

### 5.5 Permission policy

- Ba mức: `ask` (hỏi mọi thao tác `write`), `allow_in_draft` (thao tác `write` trên draft tự cho phép,
  `dangerous` vẫn hỏi), `deny_writes`.
- `dangerous` **luôn** hỏi người dùng, không policy nào tắt được.
- Yêu cầu xin quyền của agent ACP và `interrupt()` của agent kit đổ về cùng một sự kiện
  `permission.requested`. Trả lời trễ, trùng hoặc sai id bị từ chối.
- Capability: tạo khi task bắt đầu, có thời hạn, thu hồi khi task kết thúc dù kết thúc kiểu gì.

### 5.6 Transcript và redaction

Mỗi session lưu một file JSONL sự kiện. Văn bản đi qua bộ redaction của guardrails trước khi vào
transcript, log hay sự kiện gửi UI. Không lưu giá trị API key; chỉ lưu tên biến môi trường đăng nhập.

## 6. Bản mẫu Python trong kit

- Toolbox component mới `app/services/harness/`, theo luật của kit: nhận cấu hình qua constructor,
  không import `app.core.config`, có doc với mục **When you plug it in** và **Copy this module**,
  có test offline.
- Phụ thuộc: SDK MCP (đã có qua FastMCP), `agent-client-protocol` (SDK Python chính thức của ACP,
  repo `agentclientprotocol/python-sdk`) làm extra `acp`. Không nhầm với `acp-sdk` của IBM.
- Harness core (state machine, permission, capability, broker logic) **không import LangChain**, để
  khởi động nhanh và dễ port. Driver cho agent của kit là phần duy nhất dùng LangGraph.
- **ACP adapter cho agent của kit**: chạy agent của kit như một ACP agent (`python -m ...`), dùng lại
  `interrupt()` và approval tools sẵn có cho bước xin quyền.

## 7. Bộ test kiểm tra tuân thủ

- Chạy bằng pytest, nói chuyện với adapter và harness **chỉ qua giao thức**, nên dùng được cho adapter
  viết bằng ngôn ngữ bất kỳ (truyền lệnh khởi động adapter vào).
- Có **app mẫu** viết bằng Python (một app ghi chú hoặc danh sách việc đơn giản) với adapter đầy đủ, để
  bộ test có đối tượng chạy và để làm ví dụ viết adapter.
- Có **agent ACP giả lập** (scripted) để test không cần model thật, giống cách fframes test workflow.
- Các nhóm test tối thiểu: tool `write` không chạm dữ liệu thật trước commit; `dangerous` luôn hỏi;
  trả lời quyền trễ hoặc trùng bị từ chối; validate lỗi dẫn tới đúng một lần sửa; conflict khi base
  đổi; hoàn tác đưa app về revision trước; capability hết hiệu lực sau khi task kết thúc; hủy task ở
  mọi trạng thái đang chạy.

## 8. Eval cho task trong app

Dùng lại evaluation của kit: một bộ task mẫu (mục tiêu, trạng thái đầu, trạng thái mong đợi), chạy
qua harness với agent thật hoặc giả lập, chấm bằng trajectory (tool được gọi), trạng thái app sau
cùng (đọc qua `app.context`) và `pass_rate` khi chạy lặp.

## 9. Hướng dẫn port

Một trang doc nói rõ:

- Phần cần port: state machine, permission policy, capability, broker, driver ACP.
- Phần nên giữ ở Python và gọi qua MCP: RAG, guardrails dùng LLM, eval.
- Cách chạy bộ test kiểm tra với bản port.
- Khi nào chọn sidecar, khi nào port (đóng gói, khởi động, RAM; hiệu năng xử lý không phải lý do).

## 10. Ràng buộc

- Theo luật toolbox trong `AGENTS.md`; `tests/test_toolbox_imports.py` thêm `app.services.harness`.
- Test chạy offline; tính năng gọi model thật hoặc agent thật thì tắt mặc định.
- Ghi nguồn: thiết kế rút từ fframes Studio (`/home/lqaq/PROJECT/fframes-desktop-awun0105`, commit
  `a8dc7985`): `desktop/crates/studio-harness/src/driver/api.rs`, `events.rs`,
  `desktop/crates/studio-engine/src/agent_tools/broker.rs`, `backend/writer_gate.rs`,
  `docs/desktop/architecture.md` mục 4, 5 và 9. Viết lại, không copy code.

## 11. Đã quyết định

- Kit cung cấp spec, bộ test kiểm tra và harness mẫu bằng Python; port chỉ làm khi một app cụ thể cần
  (người dùng quyết định).
- Lấy fframes Studio làm mẫu (người dùng quyết định).
- Sidecar Python là một lựa chọn hợp lệ cho app (người dùng chấp nhận).

**Quyết định của người viết tài liệu (cần người dùng xem lại):**

- Đợt này chỉ có driver in-process và ACP; không làm kênh riêng cho Claude Code và Codex.
- Một writer cho mỗi app hoặc project; tối đa một lần tự sửa.
- App mẫu là app ghi chú hoặc danh sách việc viết bằng Python.
- Harness core không phụ thuộc LangChain.

## 12. Để builder quyết định

- Tên method cuối cùng, stdio hay socket cục bộ cho adapter, định dạng chi tiết của diff và transcript.
- Cấu trúc thư mục bên trong `app/services/harness/`.
- Thứ tự commit, miễn mỗi bước có test.
