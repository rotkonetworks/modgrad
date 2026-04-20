//! Canonical prompts shared across the modgrad stack.
//!
//! Each module hosts one family of prompts tied to a published method
//! or a stable internal convention. Strings only — no parsers, no
//! validators, no retry logic. When a second caller appears that
//! needs the parse/validate/retry layer, this crate should be wrapped
//! by `modgrad-skills` rather than grown in place.
//!
//! Design rules (enforce on every addition):
//!
//! 1. **Provenance.** Every non-trivial prompt has a doc-comment
//!    pointing to the paper/spec it comes from, with listing numbers
//!    or page refs so the source stays traceable after the upstream
//!    changes.
//! 2. **No silent drift.** Prompts are `&'static str` constants. If
//!    we want to patch a prompt, we bump a new `_V2` constant and
//!    leave the old one for reproducibility of past experiments.
//! 3. **Render helpers are thin.** Anything beyond straightforward
//!    string formatting belongs in a downstream skills crate. The
//!    helpers here only insert argument strings into canonical
//!    templates — they do not construct messages, they do not call
//!    any LLM, they do not validate inputs beyond basic bounds.

pub mod ssot;
