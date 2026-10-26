# Internal documents

Nan Fung's own knowledge for the agent: investment policies, investment-committee notes, checklists and deal debriefs. `lom setup` loads every `.md` file here (except this README) into the same store as the news, as the `internal` source: chunked along its headings, embedded, and searchable with `search_documents` and `source_types: ["internal"]`.

**The six bundled files are mock examples**, written for the proof of concept. They are about fictional buildings and companies, every title starts with `[MOCK]`, and they are stored as labelled test items. They are not Nan Fung records.

## Format

```
---
title: Investment committee note: 1 Example Street
kind: Investment committee note
date: 2024-03-14
mock: false                          # true marks a labelled test item
url: https://intranet.example/doc    # optional: where the analyst can open the original
---
## A heading
Text...
```

## Before loading real documents

Real documents would be sent as context to the model providers behind OpenRouter. Pin approved providers (or self-host the models), add a login with per-document access, and get compliance sign-off first.
