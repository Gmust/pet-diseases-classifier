# Smart Pet Care App — Contributors

This file lists the human contributors to the Smart Pet Care App project, based on the objective
history of each repository. Automated accounts and bots are not listed.

Notes on how this list is built:

- Contributors are listed per repository, based on that repository's own history. A contributor
  is not added to a repository they do not appear in.
- Multiple Git author identities are grouped under one person only where the repository history
  itself supports it (for example, the same email address used with different display names).
- Contributed areas describe the parts of the repository a contributor's commits touched. They
  are descriptive only.
- **No contribution percentages, rankings or ownership shares are recorded here**, and none
  should be derived from this file.
- Names are shown as they appear in the repository history. Contributors may correct their own
  entry.

Authorship and internal project usage rights are described in
[`PROJECT_AGREEMENT.md`](PROJECT_AGREEMENT.md). Being listed here is not acceptance of that
Agreement; acceptance is recorded separately in the Agreement's Acceptance table.

---

## Frontend / Mobile Application — `Gmust/smart-pet-care-app`

### Contributor

GitHub aliases:

* `Gmust`
* `illiadolbnia-code`

Git author identities in history:

* `Illia Dolbnia <illiadolbnia@gmail.com>`
* `Illia Dolbnia <95234452+Gmust@users.noreply.github.com>`

Contributed areas (per repository history):

* home, pets, assistant, reminders, auth and profile modules
* shared UI components and design-system layer (`src/common`, `src/shadecn/ui`)
* app navigation and tab structure (`src/app`)
* activity module
* internationalization setup and shared locales
* generated API client and `docs/openapi.json` integration
* project configuration, tooling and repository documentation

### Contributor

GitHub aliases:

* `Hihihikka`

Git author identities in history:

* `Anastasia Leonova <hihikka22@gmail.com>`
* `Hihihikka <hihikka22@gmail.com>`

Contributed areas (per repository history):

* care module (components, queries, API layer, schemas, hooks, locales)
* health module (components and queries)
* pets module components and pages
* profile module components
* shared UI components and design-system layer contributions
* project icons (grooming, general)

---

## Main Backend API — `VolodymyrBiletskyi/smart-pet-care-api`

### Contributor

GitHub aliases:

* `VolodymyrBiletskyi`

Git author identities in history:

* `VolodymyrBiletskyi <vladimirbilez@gmail.com>`
* `Vova <vladimirbilez@gmail.com>`

Contributed areas (per repository history):

* reminder, auth, user, activity, nutrition, health and notification modules
* data configuration, EF Core migrations and model snapshot
* application startup and infrastructure wiring (`Program.cs`)
* tests for activity and nutrition modules
* backend setup documentation

### Contributor

GitHub aliases:

* `Kushlak`

Git author identities in history:

* `Kushlak <62809394+Kushlak@users.noreply.github.com>`
* `Kushlak <kkushlak@edu.cdv.pl>` (co-author trailer)

Contributed areas (per repository history):

* pet, chat, pet weight history and feeding modules
* classifier and Cloudinary infrastructure integrations
* tests for chat, pet weight history and pet modules
* related data configuration and migrations

---

## AI Microservice — `Gmust/pet-diseases-classifier`

### Contributor

GitHub aliases:

* `Gmust`
* `illiadolbnia-code`

Git author identities in history:

* `Illia Dolbnia <illiadolbnia@gmail.com>`
* `Illia Dolbnia <95234452+Gmust@users.noreply.github.com>`

Contributed areas (per repository history):

* ML pipeline and model code (`app/ml`)
* services, use cases, schemas and FastAPI application entrypoint
* wellness scoring and Q&A endpoints
* AWS Lambda deployment configuration and runbooks
* repository documentation, contribution and security policies

---

## External project materials

Design (Figma), project management (Jira) and other external materials are part of the Project
as described in [`PROJECT_COMPONENTS.md`](PROJECT_COMPONENTS.md). Contributors to those
materials are not fully reflected by Git history; the team may extend this file once the
relevant records (Figma version history, Jira history) are reviewed.

---

## Initial concept

Initial concept contribution: [to be confirmed by contributors]

This field is intentionally left unassigned. Recognition of the originator of the initial idea
is recorded here only if the Contributors state it explicitly; it is not inferred from Git
history, repository ownership or commit order. As described in Section 7 of
[`PROJECT_AGREEMENT.md`](PROJECT_AGREEMENT.md), such recognition does not transfer ownership of
other Contributors' work and does not create exclusive ownership of the Project.
