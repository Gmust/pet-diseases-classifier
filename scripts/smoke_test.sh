#!/usr/bin/env bash
# Smoke test for the Pet Care AI Microservice.
#
# Usage:
#   BASE=http://localhost:8000 API_KEY=your_secret bash scripts/smoke_test.sh
#
# BASE defaults to http://localhost:8000. API_KEY is optional (omit if the
# server was started without API_KEY set).

set -uo pipefail

BASE="${BASE:-http://localhost:8000}"
API_KEY="${API_KEY:-}"

HDR=(-H 'Content-Type: application/json')
[ -n "$API_KEY" ] && HDR+=(-H "X-API-Key: $API_KEY")

hr() { printf '\n\033[1;36m===== %s =====\033[0m\n' "$1"; }

# Run a request; print the pretty JSON body, then the HTTP status on its own line.
req() { # method path [json]
  local method="$1" path="$2" data="${3:-}" resp code body
  if [ -n "$data" ]; then
    resp=$(curl -s -w $'\n%{http_code}' -X "$method" "${HDR[@]}" -d "$data" "$BASE$path")
  else
    resp=$(curl -s -w $'\n%{http_code}' -X "$method" "$BASE$path")
  fi
  code="${resp##*$'\n'}"     # last line = status code
  body="${resp%$'\n'*}"      # everything before it = response body
  if [ -n "$body" ]; then
    printf '%s\n' "$body" | python3 -m json.tool 2>/dev/null || printf '%s\n' "$body"
  fi
  printf 'HTTP %s\n' "$code"
}

hr "1. health (no auth)"
req GET /health

hr "2. predict — clear case (expect Digestive Issues, high confidence)"
req POST /predict '{"text":"my dog has been vomiting and wont eat for 2 days"}'

hr "3. predict — RED FLAG (expect urgency EMERGENCY + first-aid homeAdvice)"
req POST /predict '{"text":"my dog collapsed and is not breathing"}'

hr "4. predict — input too long (expect HTTP 422)"
req POST /predict "{\"text\":\"$(head -c 5000 < /dev/zero | tr '\0' 'x')\"}"

hr "5. chat — GENERAL question (expect mode=general, prediction=null, relatedTopics)"
req POST /chat '{"messages":[{"role":"user","content":"how often should I brush a Persian cat?"}],"petType":"cat"}'

hr "6. chat — HEALTH, turn 1 (expect mode=health, returns symptomSummary)"
req POST /chat '{"messages":[{"role":"user","content":"my cat is sneezing"}],"petType":"cat"}'

hr "7. chat — HEALTH, turn 2 (expect Respiratory as detail accumulates)"
req POST /chat '{
  "symptomSummary":"The cat is sneezing.",
  "petType":"cat",
  "messages":[
    {"role":"user","content":"my cat is sneezing"},
    {"role":"assistant","content":"Does your cat have a runny nose or watery eyes?"},
    {"role":"user","content":"yes, runny nose and watery eyes, and a bit of coughing"}
  ]}'

hr "8. chat — EMERGENCY (expect mode=emergency, urgency EMERGENCY, first-aid advice)"
req POST /chat '{"messages":[{"role":"user","content":"he had a seizure and collapsed"}]}'

hr "9. chat — vague input (expect needsClarification=true)"
req POST /chat '{"messages":[{"role":"user","content":"he seems a bit off today"}]}'

hr "10. chat — no user message (expect HTTP 400)"
req POST /chat '{"messages":[{"role":"assistant","content":"hello"}]}'

hr "11. wellness — full payload (expect score + short narrative + breakdown)"
req POST /wellness '{
  "pet": {"species":"dog","breed":"Labrador","ageMonths":36,"weightKg":28.5},
  "activity": {"avgStepsPerDay":8000,"avgActiveMinutesPerDay":60,"avgSleepHoursPerDay":12,"daysTracked":7},
  "feeding": {"avgMealsPerDay":2,"consistencyDays":7},
  "preventiveCare": {"recentVetVisit":true,"vaccinationsUpToDate":true},
  "previousScore": 82
}'

hr "12. wellness — partial baseline data (expect a score, no error)"
req POST /wellness '{"pet":{"species":"cat","weightKg":4.2}}'

hr "13. wellness — species only (expect HTTP 422 insufficient data)"
req POST /wellness '{"pet":{"species":"cat"}}'

hr "14. /ask is removed (expect HTTP 404)"
req POST /ask '{"question":"anything"}'

printf '\n\033[1;32mDone.\033[0m\n'
