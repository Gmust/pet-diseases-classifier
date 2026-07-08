# Privacy and Data Handling

Symptom text and chat content may contain personal or sensitive data. The service
processes requests in memory and stores no conversation state. Structured logs
must omit request bodies, symptom summaries, prompts, provider responses, API
keys, and raw session identifiers. Lambda logs retain operational metadata for
30 days.

Only licensed, provenance-recorded, de-identified records may enter dataset
builds. Never commit production payloads, raw provider exports, model/data
binaries, `.env`, or credentials. Restrict artifact access to release operators,
retain releases according to their manifest, and protect the current rollback
version. Handle access/deletion requests in the system that collected the data;
this stateless service is not the system of record.
