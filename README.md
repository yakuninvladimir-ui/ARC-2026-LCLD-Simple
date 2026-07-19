# ARC-2026 LCLD Simple

> A research repository for the ARC Prize 2026 / ARC-AGI-3 interactive competition containing multiple generations of the LCLD agent architecture, reproducible notebook builders, competition wrappers, engineering specifications, and regression tests.

---

## Overview

**ARC-2026 LCLD Simple** is the primary development repository for the LCLD family of ARC-AGI-3 interactive agents.

Unlike repositories that expose only the latest implementation, this project intentionally preserves multiple architectural generations. Each version represents a distinct design point, allowing ideas to evolve without losing reproducibility or historical context.

At the time of this snapshot the repository contains two maintained implementations:

| Version | Architecture | Status |
|----------|--------------|--------|
| **V8.3** | Compact Verified Hypothesis Agent | Stable |
| **V9.0** | Reverse-Semantic Trajectory Agent | Current research generation |

Both implementations share the same overall competition philosophy:

- deterministic execution outside model inference;
- official-transition driven reasoning;
- object-centric perception;
- bounded interaction with Qwen;
- reproducible notebook generation;
- offline competition execution;
- strict verification of every executed action.

The repository intentionally excludes model weights, ARC environments, generated notebooks, Kaggle datasets, local traces, vLLM wheels, and llama.cpp binaries.

---

# Design Goals

The project was built around several long-term goals.

## Deterministic execution

Everything except language-model generation should be deterministic and replayable.

Given identical observations and identical model output, the agent should produce identical behaviour.

---

## Official-transition authority

The environment—not the language model—is the source of truth.

Every action is evaluated only after the official next observation arrives.

Predicted state is never treated as fact.

---

## Object-centric reasoning

The agent reasons about:

- persistent objects;
- relations;
- geometry;
- action effects;
- semantic hypotheses.

It is **not** a raw-pixel policy.

---

## Competition reproducibility

The repository is designed so that a competition notebook can be reproduced directly from source.

Notebook generation, packaging and validation are part of the repository rather than external scripts.

---

## Architecture evolution

Older generations are preserved because architectural evolution is itself valuable documentation.

Many ideas introduced in V8 remain useful references even after the implementation moved to V9.

---

# Repository Philosophy

The repository deliberately separates four concerns:

1. architecture;
2. implementation;
3. competition packaging;
4. documentation.

Specifications describe *what* the system should do.

Runtime packages implement those specifications.

Notebook builders package the implementation for Kaggle.

Tests ensure that implementation and specifications remain aligned.

---

# Implemented Architectures

## Version 8.3

**ARC-AGI-3 Compact Verified Hypothesis Agent**

V8.3 represents the final compact hypothesis-driven architecture before the transition to trajectory-based semantic planning.

Core characteristics include:

- deterministic ARGALite perception;
- persistent object tracking;
- stable relation graph;
- grounded hypothesis bank;
- fixed verification contracts;
- bounded Qwen proposals;
- official-transition verification;
- replayable execution loop.

The architecture intentionally avoids:

- unrestricted symbolic planners;
- general-purpose DSL compilation;
- autonomous action execution by the language model;
- forward simulation.

Instead, every proposed hypothesis must be grounded by deterministic verification before it can become accepted knowledge.

Documentation:

```
ARCHITECTURAL_SPECIFICATION_V8_3.md
ENGINEERING_SPECIFICATION_V8_3.md
CHANGELOG_V8_3.md
VALIDATION_V8_3.txt
```

---

## Version 9

**ARC-AGI-3 Reverse-Semantic Trajectory Agent**

V9 evolves the architecture toward complete semantic trajectories rather than isolated local hypotheses.

Major additions include:

- reverse-semantic reasoning;
- trajectory-level verification;
- multimodal Qwen packets;
- reset-safe semantic rebinding;
- deterministic component graph;
- frame media generation;
- richer action research;
- semantic invariant extraction;
- trajectory comparison across level resets.

Although considerably more sophisticated than V8, the architecture preserves the same fundamental authority hierarchy:

```
official observation
        ↓
deterministic perception
        ↓
grounded semantic binding
        ↓
verification
        ↓
memory update
```

The language model proposes explanations.

The runtime decides whether those explanations survive contact with the environment.

Documentation:

```
ARCHITECTURAL_SPECIFICATION_V9.md
ENGINEERING_SPECIFICATION_V9.md
```

---

# Architecture Evolution

The repository documents the evolution of the LCLD agent family.

```
V6.x
  │
  ├── compact object reasoning
  │
  ▼
V8.3
  │
  ├── grounded hypothesis verification
  ├── fixed contracts
  ├── deterministic memory
  │
  ▼
V9
  │
  ├── reverse-semantic trajectories
  ├── semantic rebinding
  ├── multimodal packets
  ├── component graph
  └── trajectory verification
```

Each generation remains available as an implementation reference.

---

# High-Level Runtime Pipeline

The current agent pipeline can be summarized as:

```text
official observation
        │
        ▼
world normalization
        │
        ▼
ARGALite perception
        │
        ▼
persistent objects
        │
        ▼
relations
        │
        ▼
component graph
        │
        ▼
action research
        │
        ▼
Qwen proposal
        │
        ▼
deterministic binding
        │
        ▼
verification
        │
        ▼
official transition ingestion
        │
        ▼
memory update
```

Every executed action passes through this pipeline.

---

# Repository Layout

```
.
├── v8_agent/
│   ├── runtime implementation
│   ├── verification
│   ├── memory
│   ├── policy
│   ├── Qwen integration
│   └── notebook runtime
│
├── v9_agent/
│   ├── reverse semantics
│   ├── trajectory verification
│   ├── frame media
│   ├── component graph
│   ├── action research
│   ├── policy
│   ├── memory
│   └── competition runtime
│
├── notebook_wrapper/
│   ├── Kaggle wrapper
│   ├── competition bootstrap
│   └── runtime packaging
│
├── tests/
│   ├── unit tests
│   ├── integration tests
│   ├── regression tests
│   └── notebook validation
│
├── build_notebook.py
├── build_notebook_v9.py
│
├── ARCHITECTURAL_SPECIFICATION_*.md
├── ENGINEERING_SPECIFICATION_*.md
├── CHANGELOG_*.md
├── VALIDATION_*.txt
└── README.md
```

---

# Major Repository Components

## Runtime packages

The runtime packages contain the complete agent implementations.

```
v8_agent/
v9_agent/
```

Each package is independently buildable and contains its own runtime logic.

The two implementations intentionally coexist so that architectural evolution remains transparent.

---

## Specifications

Every maintained architecture has two primary specification documents.

- Architectural Specification
- Engineering Specification

The architectural document explains concepts and design decisions.

The engineering document defines implementation contracts, module responsibilities, interfaces and behavioural requirements.

These documents are treated as normative references for development.

---

## Notebook builders

The repository generates Kaggle notebooks directly from source.

Separate builders exist for different generations where required.

The generated notebooks are **derived artifacts** and are intentionally excluded from version control.

---
# Competition Pipeline

The repository contains everything required to reproduce the competition notebook used for ARC Prize submissions.

The production pipeline consists of four major stages.

```text
source code
      │
      ▼
notebook builder
      │
      ▼
generated Kaggle notebook
      │
      ▼
competition wrapper
      │
      ▼
ARC gateway
```

Only generated notebooks are submitted.

The repository itself remains the canonical development source.

---

## Competition Runtime

The competition runtime follows a strict execution contract.

```text
initialize session
        │
        ▼
initial RESET
        │
        ▼
official observation
        │
        ▼
agent decision
        │
        ▼
environment step
        │
        ▼
official transition
        │
        ▼
verification
        │
        ▼
memory update
        │
        ▼
repeat
```

The runtime never treats predicted observations as factual.

Every decision is evaluated only after the official environment response has
been received.

---

## Action Authority

The architecture intentionally separates responsibilities.

| Component | Responsibility |
|-----------|----------------|
| Environment | Ground truth |
| Perception | Deterministic scene parsing |
| Qwen | Proposal generation |
| Binder | Ground semantic proposals |
| Verifier | Decide progress |
| Memory | Store evidence |
| Policy | Choose next action |

The language model never becomes the source of truth.

---

# Competition Profile

The repository targets the ARC Prize 2026 interactive competition.

Typical production configuration includes:

- NVIDIA RTX6000 inference target;
- Qwen 3.6 FP8 served through vLLM;
- structured JSON output;
- deterministic execution outside model inference;
- offline notebook execution;
- bounded model calls;
- bounded interaction budgets;
- replayable transition processing.

Exact runtime parameters may evolve independently of repository structure.

---

## Competition Constraints

The runtime is designed around the competition contract.

Typical limits include:

- bounded actions per game;
- bounded attempts per level;
- bounded Qwen calls;
- bounded coordinate research;
- bounded wall-clock execution.

The repository keeps these limits configurable while maintaining deterministic
behaviour.

---

# Notebook Generation

The repository contains notebook builders rather than storing generated
competition notebooks.

Typical workflow:

```bash
python build_notebook.py
```

or

```bash
python build_notebook_v9.py
```

The builders package:

- runtime sources;
- competition wrapper;
- metadata;
- configuration;
- validation helpers.

Generated notebooks remain derived artifacts and are intentionally excluded from
version control.

---

## Generated Artifacts

Typical generated output includes:

```text
notebooks/
    arc-prize-2026-lcld-qwen.ipynb
    kernel-metadata.json
```

Additional generated files may appear depending on the selected builder.

---

# Testing

The repository contains an extensive regression suite.

Testing focuses on deterministic behaviour rather than model quality.

Typical categories include:

- parser tests;
- perception tests;
- verification tests;
- memory tests;
- packet generation;
- notebook generation;
- competition wrapper;
- regression tests.

Run the complete suite:

```bash
python -m pytest tests -q
```

---

## Validation Philosophy

Tests are intended to guarantee that:

- architectural invariants remain intact;
- engineering contracts remain satisfied;
- notebook generation remains reproducible;
- deterministic execution does not regress.

Passing tests should not be interpreted as competition success.

Only real ARC gateway execution can validate competition behaviour.

---

# Specifications

The repository treats specifications as first-class development artifacts.

Each maintained architecture contains two complementary documents.

## Architectural Specification

Defines:

- concepts;
- authority hierarchy;
- runtime model;
- semantic contracts;
- lifecycle;
- reasoning philosophy.

---

## Engineering Specification

Defines:

- module responsibilities;
- interfaces;
- configuration;
- data contracts;
- implementation requirements;
- testing expectations.

---

# Runtime Design Principles

The implementation follows several engineering principles.

## Deterministic before intelligent

Whenever deterministic reasoning is possible it is preferred over model
inference.

---

## Official observations only

The runtime never commits imagined state.

Only environment observations become evidence.

---

## Ground everything

Semantic hypotheses must always be grounded to:

- current objects;
- current relations;
- current geometry;
- measurable effects.

Ungrounded proposals remain proposals.

---

## Verification over confidence

Language-model confidence is not evidence.

Observed transitions are evidence.

---

## Preserve failures

Negative results are valuable information.

The agent stores failed attempts, rejected hypotheses and irrelevant actions in
order to reduce repeated mistakes.

---

## Replayability

The repository aims to make every decision reproducible.

Given:

- identical observations;
- identical configuration;
- identical model output,

the runtime should reproduce identical behaviour.

---

# Source Code Organization

The repository intentionally separates implementation layers.

```
Perception
        │
        ▼
Memory
        │
        ▼
Policy
        │
        ▼
Verification
        │
        ▼
Competition wrapper
```

Each layer has a narrowly defined responsibility.

---

# Qwen Integration

The repository treats Qwen as a bounded semantic advisor rather than an
autonomous controller.

Depending on the architecture generation, the model may produce:

- semantic hypotheses;
- complete semantic trajectories;
- coordinate plans;
- bounded reasoning.

The runtime validates every proposal before execution.

Invalid identifiers, unavailable actions or unsupported coordinates are rejected
before reaching the environment.

---

# Configuration

Runtime configuration is centralized.

Typical configuration groups include:

- model backend;
- runtime limits;
- competition limits;
- verification thresholds;
- packet generation;
- exploration budgets;
- prompt compaction;
- logging.

The repository supports deterministic defaults while allowing controlled
competition overrides.

---

# Documentation

Repository documentation is intentionally extensive.

Primary documents include:

```text
README.md

ARCHITECTURAL_SPECIFICATION_*.md

ENGINEERING_SPECIFICATION_*.md

CHANGELOG_*.md

VALIDATION_*.txt

COMPETITION_WRAPPER_AUDIT_*.md
```

Architectural documents explain *why* the system works.

Engineering documents explain *how* it is implemented.

---

# Repository Conventions

The project follows several conventions.

- Specifications are treated as normative.
- Runtime behaviour should remain deterministic.
- Generated artifacts are not committed.
- Historical architecture versions remain available.
- Compatibility layers are explicitly documented.
- Public APIs should remain stable whenever practical.

---

# Contributing

Contributions should preserve the architectural philosophy of the repository.

Changes are expected to:

- maintain deterministic behaviour;
- preserve replayability;
- keep specifications synchronized with implementation;
- include regression tests when behaviour changes;
- avoid introducing parallel inactive implementations.

Large architectural changes should update both the architectural and engineering
specifications.

---

# License

Repository source code and documentation are distributed under the terms of the
MIT License.

Generated notebooks may additionally contain attribution notices required by the
competition environment.

External assets—including language models, ARC environments, Kaggle datasets,
vLLM distributions and other third-party components—remain subject to their
respective licenses and are not redistributed by this repository.

---

# Acknowledgements

This repository was developed as part of ongoing research into interactive
reasoning systems for the ARC Prize 2026 / ARC-AGI-3 competition.

The project explores deterministic perception, grounded semantic reasoning,
verification-driven planning and reproducible competition execution while
preserving the complete architectural evolution of the LCLD agent family.
