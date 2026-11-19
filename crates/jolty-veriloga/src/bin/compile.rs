//! CLI binary that compiles every model in [`jolty_veriloga::MODELS`] from Verilog-A into its
//! Rust kernel file.
//!
//! Passing two arguments compiles a single source file into the given output path, resolving
//! non-standard includes from the source directory. An optional third argument overrides the
//! source path written into the generated file header.

use std::path::{Path, PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let models: Vec<(PathBuf, PathBuf, String)> = match arguments.as_slice() {
        [] => jolty_veriloga::MODELS
            .iter()
            .map(|(source, kernel)| {
                (
                    root.join(source),
                    root.join(kernel),
                    format!("crates/jolty-veriloga/{source}"),
                )
            })
            .collect(),
        [source, kernel] => vec![(PathBuf::from(source), PathBuf::from(kernel), source.clone())],
        [source, kernel, named] => {
            vec![(PathBuf::from(source), PathBuf::from(kernel), named.clone())]
        }
        _ => {
            return Err(
                "the compiler takes no arguments, or a source and a kernel, and optionally the name its header gives the source"
                    .into(),
            );
        }
    };
    for (source, kernel, named) in models {
        let text = std::fs::read_to_string(&source)?;
        let directory = source.parent().map_or_default(Path::to_path_buf);
        let files = |name: &str| std::fs::read_to_string(directory.join(name)).ok();
        let compiled = jolty_veriloga::compile(&text, &named, &files)?;
        std::fs::write(kernel, compiled)?;
    }
    Ok(())
}
