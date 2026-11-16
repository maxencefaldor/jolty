//! Writes each kernel of [`jolty_veriloga::MODELS`] from its Verilog-A source, or with
//! two arguments, a source and a kernel, the kernel of that source, the files it
//! includes read beside it, and with a third the name its header gives the source, such
//! as the address a third party publishes it at.

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
        let beside = source.parent().map_or_default(Path::to_path_buf);
        let files = |name: &str| std::fs::read_to_string(beside.join(name)).ok();
        let compiled = jolty_veriloga::compile(&text, &named, &files)?;
        std::fs::write(kernel, compiled)?;
    }
    Ok(())
}
