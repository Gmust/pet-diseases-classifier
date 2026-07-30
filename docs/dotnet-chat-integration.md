# `/chat` Integration Guide — .NET Backend

Integration reference for the **unified conversational endpoint**. For the full
endpoint catalogue (`/predict`, `/wellness`, `/health`, enums, auth, errors) see
`api-reference.md` — this document is the deep dive on `/chat`.

## The two things to understand first

**1. The microservice is stateless. Your backend owns all session state.**
The classifier has no memory. On every turn **you** send the recent message
history plus the rolling `symptomSummary` it returned last turn, and you persist
the new `symptomSummary` it returns. That string is the conversation's memory —
store it, replay it, don't edit it.

**2. One endpoint, self-routing.** `/chat` decides under the hood whether the
message is general care or a health concern and returns a **`mode`**. You branch
on `mode`; you never call a different endpoint and the answer never tells the user to.

| `mode` | Meaning | What you get |
|---|---|---|
| `general` | General-care question (diet, grooming, behaviour…). | `answer` + `relatedTopics`. `prediction` is **null**. |
| `health` | Symptom/health concern. | `answer` + populated `prediction`. `symptomSummary` updated. |
| `emergency` | A red-flag phrase was detected (collapse, not breathing, seizure…). | Emergency `answer` + first-aid `homeAdvice`, `prediction.urgency = EMERGENCY`. |

## Turn sequence

```
User sends a message
      │
      ▼
[.NET backend]
  1. Load session: message history + last symptomSummary (from your DB)
  2. Append the new user message
  3. (optional) Trim history to the last ~6–10 messages (cost control)
  4. POST /chat { sessionId, messages, symptomSummary, petType }
      │
      ▼
[AI microservice]  — stateless
  • red-flag safety check (always)
  • local symptom classifier
  • ONE Gemini call → decides general vs health AND writes the answer
      │
      ▼
[.NET backend]
  5. Append assistant `answer` to history
  6. Persist `symptomSummary` from the response  ← critical (the memory)
  7. Branch on `mode` to render
```

Step 6 matters most: if you don't save the returned `symptomSummary`, the
conversation loses its memory. (On `general` turns the summary is returned
unchanged — still safe to persist.)

## How the memory actually works (read this — it prevents the common mistake)

There are **two layers of context**, and neither of them is the prediction:

1. **`messages[]` — the conversational memory.** The literal transcript of the
   chat, including the assistant's own previous answers (role `"assistant"`). This
   is what lets the reply flow naturally ("you mentioned a runny nose earlier…").
2. **`symptomSummary` — the medical memory.** A compact, rolling distillation of
   every symptom mentioned so far. This is what survives even when you trim old
   messages, and it's what the classifier runs on.

**You do NOT store or resend the prediction. The prediction is recomputed from
scratch on every turn.** This is the key idea, and it's deliberate:

```
prediction = classifier(symptomSummary + newestUserMessage)
```

Because the summary already holds every symptom so far, the classifier always
predicts on the **full picture** — it doesn't need last turn's result, it re-derives
a fresh one. That's why a prediction can *change* between turns (e.g. "Ear
Conditions" on turn 1 → "Respiratory Conditions" on turn 2 as detail accumulates):
the recomputed prediction is simply more informed. **Locking to an old prediction
would be a bug, not a feature.**

The rule of thumb: **persist the *source* (the symptom summary + the transcript),
recompute the *derived value* (the prediction) each turn.** A prediction is derived
state — storing it just invites staleness. Keep the symptoms; the prediction is
always correct and current.

So per turn your backend persists exactly two things and replays them:
- the **message history** (append the user message, then append the assistant's
  `answer` as a `"assistant"` message), and
- the latest **`symptomSummary`** (overwrite it with the response value).

You never send predictions back. *(Optional: if you want the model to explicitly
reconcile a shift — "earlier this looked like an ear issue, but now…" — you can
store last turn's `predictedCondition` and pass it as a hint. Not required, and not
part of the base contract.)*

## Endpoint contract

```
POST {AI_BASE_URL}/chat
Content-Type: application/json
X-API-Key: {shared secret}     # required only if API_KEY is set on the service
```

Set `HttpClient.Timeout` to **60s** (warm calls are 1–3s; a cold Lambda can take ~30s).

### Request (camelCase)

| Field | Type | Required | Notes |
|---|---|---|---|
| `messages` | array | **yes** | Oldest-first. Last entry must be the new user message. Max 50. |
| `messages[].role` | string | yes | `"user"` or `"assistant"`. |
| `messages[].content` | string | yes | 1–4000 chars. |
| `symptomSummary` | string | no | Returned last turn. Omit/`null` on turn 1. |
| `sessionId` | string | no | Opaque; logs/telemetry only. |
| `petType` | string | no | `dog,cat,rabbit,hamster,guinea_pig,bird,fish,turtle`. |

### Response (camelCase)

| Field | Type | Notes |
|---|---|---|
| `mode` | string | `general` \| `health` \| `emergency` — **branch on this**. |
| `answer` | string | Reply to show the user. Never references internal endpoints. |
| `symptomSummary` | string | **Persist and replay next turn.** |
| `prediction` | object \| **null** | Null when `mode=general`. |
| `relatedTopics` | string[] | Tags when `mode=general`; empty otherwise. |
| `needsClarification` | bool | True when the reply asks a follow-up. |
| `disclaimer` | string | Always present — display it. |
| `prediction.predictedCondition` | string | One of 16 classes. Can change between turns. |
| `prediction.confidence` | number | 0–1. |
| `prediction.topK` | array | `[{condition, confidence}]`. |
| `prediction.urgency` | string | `MONITOR\|CONSULT_SOON\|URGENT\|EMERGENCY`. |
| `prediction.specialist` / `diseaseCategory` / `homeAdvice` | mixed | See `api-reference.md`. |

## Persistence model

```csharp
public class ChatSession
{
    public Guid Id { get; set; }
    public string UserId { get; set; } = default!;
    public string? PetType { get; set; }
    public string? SymptomSummary { get; set; }   // overwrite every turn with the response value
    public DateTimeOffset CreatedAt { get; set; }
    public DateTimeOffset UpdatedAt { get; set; }
    public List<ChatMessageEntity> Messages { get; set; } = new();
}

public class ChatMessageEntity
{
    public long Id { get; set; }
    public Guid SessionId { get; set; }
    public string Role { get; set; } = default!;   // "user" | "assistant"
    public string Content { get; set; } = default!;
    public DateTimeOffset CreatedAt { get; set; }
}
```

Store the full history, but send only the last ~6–10 messages — the rolling
`symptomSummary` already carries the older medical context, which keeps each
Gemini call small and cheap.

## DTOs

```csharp
using System.Text.Json.Serialization;

public record ChatTurnRequest
{
    [JsonPropertyName("sessionId")]      public string? SessionId { get; init; }
    [JsonPropertyName("petType")]        public string? PetType { get; init; }
    [JsonPropertyName("symptomSummary")] public string? SymptomSummary { get; init; }
    [JsonPropertyName("messages")]       public required List<ChatMessageDto> Messages { get; init; }
}

public record ChatMessageDto
{
    [JsonPropertyName("role")]    public required string Role { get; init; }
    [JsonPropertyName("content")] public required string Content { get; init; }
}

public record ChatTurnResponse
{
    [JsonPropertyName("mode")]               public string Mode { get; init; } = "health";   // general|health|emergency
    [JsonPropertyName("answer")]             public string Answer { get; init; } = "";
    [JsonPropertyName("symptomSummary")]     public string SymptomSummary { get; init; } = "";
    [JsonPropertyName("prediction")]         public ChatPredictionDto? Prediction { get; init; }  // null when general
    [JsonPropertyName("relatedTopics")]      public List<string> RelatedTopics { get; init; } = new();
    [JsonPropertyName("needsClarification")] public bool NeedsClarification { get; init; }
    [JsonPropertyName("disclaimer")]         public string Disclaimer { get; init; } = "";
}

public record ChatPredictionDto
{
    [JsonPropertyName("predictedCondition")] public string PredictedCondition { get; init; } = "";
    [JsonPropertyName("confidence")]         public double Confidence { get; init; }
    [JsonPropertyName("topK")]               public List<TopKItemDto> TopK { get; init; } = new();
    [JsonPropertyName("urgency")]            public string Urgency { get; init; } = "";
    [JsonPropertyName("specialist")]         public string Specialist { get; init; } = "";
    [JsonPropertyName("diseaseCategory")]    public string DiseaseCategory { get; init; } = "";
    [JsonPropertyName("homeAdvice")]         public List<string> HomeAdvice { get; init; } = new();
}

public record TopKItemDto
{
    [JsonPropertyName("condition")]  public string Condition { get; init; } = "";
    [JsonPropertyName("confidence")] public double Confidence { get; init; }
}
```

## Typed client + orchestration

```csharp
builder.Services.AddHttpClient<PetAiClient>(c =>
{
    c.BaseAddress = new Uri(builder.Configuration["PetAi:BaseUrl"]!);   // e.g. https://…/Prod/
    c.Timeout = TimeSpan.FromSeconds(60);
    c.DefaultRequestHeaders.Add("X-API-Key", builder.Configuration["PetAi:ApiKey"]);
});
```

```csharp
public async Task<ChatTurnResponse> HandleUserMessageAsync(Guid sessionId, string userText, CancellationToken ct)
{
    var session = await _db.ChatSessions.Include(s => s.Messages)
        .FirstAsync(s => s.Id == sessionId, ct);

    session.Messages.Add(new ChatMessageEntity {
        SessionId = sessionId, Role = "user", Content = userText, CreatedAt = DateTimeOffset.UtcNow });

    var recent = session.Messages.OrderBy(m => m.CreatedAt).TakeLast(8)
        .Select(m => new ChatMessageDto { Role = m.Role, Content = m.Content }).ToList();

    var result = await _petAi.ChatAsync(new ChatTurnRequest {
        SessionId = sessionId.ToString(), PetType = session.PetType,
        SymptomSummary = session.SymptomSummary, Messages = recent }, ct);

    session.Messages.Add(new ChatMessageEntity {
        SessionId = sessionId, Role = "assistant", Content = result.Answer, CreatedAt = DateTimeOffset.UtcNow });
    session.SymptomSummary = result.SymptomSummary;   // ← persist the memory every turn
    session.UpdatedAt = DateTimeOffset.UtcNow;
    await _db.SaveChangesAsync(ct);

    return result;
}
```

## Rendering by `mode`

```csharp
switch (result.Mode)
{
    case "emergency":
        // Show answer prominently + a "contact emergency vet" CTA. prediction.urgency == EMERGENCY.
        break;
    case "health":
        // Show answer + prediction card (condition, urgency, homeAdvice).
        // If needsClarification, it's asking a follow-up — let the user reply.
        break;
    case "general":
        // Show answer + relatedTopics chips. prediction is null — don't render a condition card.
        break;
}
```

## Failure handling

The service degrades gracefully (if Gemini is down it still returns a valid
response via local fallback, defaulting to the `health` path), so a 5xx is usually
transport/timeout.

- Timeout/5xx: keep the user message, show a soft retry, and **do not** overwrite
  `SymptomSummary` (so the next attempt resumes from the last good state).
- `400`: malformed — usually the last `messages` entry wasn't a non-empty `user` message.
- `403`: missing/wrong `X-API-Key` (only when auth is enabled).

## Notes

- Always display `disclaimer`. This is a triage aid, not a diagnosis.
- `prediction.predictedCondition` can change turn to turn as symptoms accumulate — render the current one; don't lock it.
- Without a `GEMINI_API_KEY` on the server, general-care answering isn't available and the service defaults to the `health` path — set the key in the deploy environment.

---

## Implementation prompt (paste into your AI coding assistant)

> Copy everything in the block below into Cursor/Copilot/Claude to scaffold the
> full backend flow. It encodes the rules above so the agent can't get the memory
> mechanism wrong.

```text
Implement a stateful pet-care chat flow in our ASP.NET Core backend that talks to
an external, STATELESS AI microservice at POST {AI_BASE_URL}/chat. Our backend owns
all conversation state; the microservice stores nothing.

## The contract with the microservice
Request JSON (camelCase):
  { "sessionId": string?, "petType": string?, "symptomSummary": string?,
    "messages": [ { "role": "user"|"assistant", "content": string } ] }
  - messages are oldest-first; the LAST entry must be the new user message.
  - Auth: header "X-API-Key: <secret>" (from config PetAi:ApiKey), only if configured.

Response JSON (camelCase):
  { "mode": "general"|"health"|"emergency", "answer": string,
    "symptomSummary": string, "needsClarification": bool, "disclaimer": string,
    "relatedTopics": string[], "prediction": null | {
      "predictedCondition": string, "confidence": number,
      "topK": [ { "condition": string, "confidence": number } ],
      "urgency": "MONITOR"|"CONSULT_SOON"|"URGENT"|"EMERGENCY",
      "specialist": string, "diseaseCategory": string, "homeAdvice": string[] } }

## Core rules (do not violate)
1. STATE = two things only: the message history, and a rolling `symptomSummary` string.
2. Persist the source, recompute the derived value. NEVER store or resend the
   `prediction` — the microservice recomputes it every turn from symptomSummary +
   the new message. The prediction may change between turns; that is correct.
3. Every turn: append the user's message; call /chat; append the response `answer`
   as an "assistant" message; OVERWRITE the stored symptomSummary with the response
   value (even on `mode=general`, where it's unchanged).
4. Send only the last ~8 messages to /chat (cost control), but ALWAYS send the
   latest symptomSummary — it carries the older context that the trimmed messages drop.
5. On timeout/5xx: keep the user's message, surface a soft retry, and DO NOT
   overwrite symptomSummary (so the next attempt resumes from the last good state).

## Deliverables
- EF Core entities: ChatSession { Id, UserId, PetType?, SymptomSummary?, CreatedAt,
  UpdatedAt, ICollection<ChatMessage> } and ChatMessage { Id, SessionId, Role,
  Content, CreatedAt }. Add a migration.
- DTOs matching the camelCase contract above (System.Text.Json). Enum PetType
  serialized with JsonNamingPolicy.SnakeCaseLower so GuineaPig -> "guinea_pig".
- A typed HttpClient `PetAiClient` (BaseAddress = PetAi:BaseUrl, Timeout 60s,
  default header X-API-Key = PetAi:ApiKey) with `Task<ChatTurnResponse> ChatAsync(
  ChatTurnRequest, CancellationToken)` that throws on non-success.
- A `ChatService.HandleUserMessageAsync(Guid sessionId, string userText, CancellationToken)`
  that implements the per-turn flow and the Core rules above, saving to the DB.
- A minimal API / controller endpoint `POST /api/sessions/{sessionId}/messages`
  that takes { text }, calls ChatService, and returns the ChatTurnResponse.
- Map `mode` to the client response: general -> answer + relatedTopics (no condition
  card); health -> answer + prediction; emergency -> answer + prediction with an
  "urgent, contact an emergency vet" flag. Always include disclaimer.
- Unit tests: (a) assistant answer is appended and symptomSummary overwritten each
  turn; (b) on a simulated 5xx, symptomSummary is NOT overwritten; (c) only the last
  ~8 messages are sent but symptomSummary is always included.

Use the existing DTO and orchestration snippets in docs/dotnet-chat-integration.md
as the reference implementation; extend rather than diverge from them.
```

That prompt hands the agent the contract, the non-negotiable memory rules, and a
concrete deliverables list — so what it builds matches this doc instead of
reinventing the flow.
