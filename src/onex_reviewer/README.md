# ONEX Opus Nightly Agents - Baseline & Daily Reviewer

Deterministic, diff-driven Opus 4.1 agents that run on `main`. Goal: prevent drift while migrating away from archived code.

## Overview

The ONEX Reviewer system provides automated code review against ONEX architectural standards. It operates in two modes:

1. **Baseline Review**: One-time comprehensive review of entire non-archived codebase
2. **Nightly Review**: Incremental review of changes since last successful run

## Quick Start

### Run a Baseline Review

```bash
# Review entire codebase
./scripts/onex_reviewer/run_review.sh baseline
```

### Run a Nightly Review

```bash
# Review incremental changes
./scripts/onex_reviewer/run_review.sh nightly
```

### Manual Testing

```bash
# Run test suite
python3 test_onex_reviewer.py
```

## Architecture

```
src/onex_reviewer/
├── agents/           # Review agents (baseline, nightly)
├── rules/            # ONEX rule definitions and engine
├── models/           # Data models (findings, inputs, outputs)
├── config/           # Policy configuration
├── run_baseline.py   # Baseline orchestrator
└── run_nightly.py    # Nightly orchestrator

scripts/onex_reviewer/
├── baseline_producer.sh  # Creates sharded baseline inputs
├── nightly_producer.sh    # Creates incremental diff inputs
└── run_review.sh          # Convenience wrapper script
```

## Rule Categories

### A. Naming Rules
- `ONEX.NAMING.PROTOCOL_001`: Protocol classes must start with "Protocol"
- `ONEX.NAMING.MODEL_001`: Model classes must start with "Model"
- `ONEX.NAMING.ENUM_001`: Enum classes must start with "Enum"
- `ONEX.NAMING.NODE_001`: Node classes must start with "Node"

### B. Boundary Rules
- `ONEX.BOUNDARY.FORBIDDEN_IMPORT_001`: Detects forbidden cross-boundary imports

### C. SPI Purity Rules
- `ONEX.SPI.RUNTIMECHECKABLE_001`: Protocol classes need @runtime_checkable
- `ONEX.SPI.FORBIDDEN_LIB_001`: No os/pathlib/sqlite3/requests in SPI

### D. Typing Hygiene
- `ONEX.TYPE.UNANNOTATED_DEF_001`: Functions must have type annotations
- `ONEX.TYPE.ANY_001`: Avoid Any type in non-test code
- `ONEX.TYPE.OPTIONAL_ASSERT_001`: Don't assert Optional immediately

### E. Waiver Hygiene
- `ONEX.WAIVER.MALFORMED_001`: Waivers need reason= and expires=
- `ONEX.WAIVER.EXPIRED_001`: Expired waivers are errors

## Output Format

### NDJSON Findings
One JSON object per line with fields:
- `ruleset_version`: Version of rules applied
- `rule_id`: Unique rule identifier
- `severity`: "error" or "warning"
- `repo`: Repository name
- `path`: File path
- `line`: Line number
- `message`: Violation description
- `evidence`: Supporting evidence
- `suggested_fix`: Recommended fix
- `fingerprint`: Unique violation fingerprint

### Markdown Summary
Structured summary with:
- Executive summary with risk score (0-100)
- Top violations
- Waiver issues
- Next actions
- Coverage notes

## Configuration

### Policy Configuration

Edit `src/onex_reviewer/config/policy.yaml` to configure:
- Repository-specific forbidden imports
- Custom boundary rules

### Exclusion Patterns

In producer scripts, configure:
- `EXCLUDES_REGEX`: Directories to skip (archive, deprecated, etc.)
- `INCLUDE_EXT`: File extensions to review

## Operational Notes

- **Baseline**: Shards large diffs for processing (200KB per shard)
- **Nightly**: Maintains `.onex_nightly_prev` marker for incremental reviews
- **Exit codes**: Non-zero on critical errors (severity="error")
- **Output locations**:
  - Baseline: `.onex_baseline/<repo>/<timestamp>/`
  - Nightly: `.onex_nightly/<repo>/<timestamp>/`

## Development

### Adding New Rules

1. Add rule definition in `src/onex_reviewer/rules/definitions.py`
2. Implement check logic in `src/onex_reviewer/rules/engine.py`
3. Update tests in `test_onex_reviewer.py`

### Rule ID Format

`ONEX.<CATEGORY>.<SPECIFIC>_<NUMBER>`

Examples:
- `ONEX.NAMING.PROTOCOL_001`
- `ONEX.BOUNDARY.FORBIDDEN_IMPORT_001`

## Integration

### CI/CD Pipeline

```yaml
# Example GitHub Actions workflow
- name: Run ONEX Nightly Review
  run: |
    ./scripts/onex_reviewer/run_review.sh nightly
  schedule:
    - cron: '0 2 * * *'  # 22:00 America/New_York
```

### Waiver Format

```python
# onex:ignore RULE_ID reason=Temporary_rename_in_flight expires=2025-10-15
```

## Limitations

- Diff size capped for performance (configurable in producer scripts)
- Partial coverage disclosure when truncated
- Deterministic regex-based checks only
- No external repository crawling