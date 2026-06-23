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
