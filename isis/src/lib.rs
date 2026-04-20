//! isis — library surface shared between the `isis` CLI and the `isisd`
//! daemon binary. Today this is just the `daemon` module; as the runtime
//! grows, per-subcommand logic migrates here so both binaries stay thin.

pub mod daemon;
