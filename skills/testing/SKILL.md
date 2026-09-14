---
name: testing
description: Design and implement focused regression tests for a code change
triggers:
  - add tests
  - test coverage
auto_activate: false
---
Inspect the relevant implementation and existing test style before editing.
Prefer deterministic tests that cover behavior and important failure paths.
Avoid asserting private implementation details unless there is no stable public
interface. Run the smallest relevant test selection first, then the complete
suite. Report the commands run and any remaining coverage gaps.
