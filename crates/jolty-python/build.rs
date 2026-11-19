// Copyright 2026 Maxence Faldor
//! Bundle the offline guide pages and example scripts into `OUT_DIR/docs.json`.
//!
//! `maturin` includes `docs.json` from `OUT_DIR` into `jolty/assets/` so wheels and
//! editable installs carry the guide and examples offline without duplicating their
//! sources in version control.

use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};

fn emit_rerun(path: &Path) -> io::Result<()> {
    let mut stdout = io::stdout().lock();
    writeln!(stdout, "cargo::rerun-if-changed={}", path.display())
}

fn io_error(action: &str, path: &Path, error: &io::Error) -> io::Error {
    io::Error::new(
        error.kind(),
        format!("{action} `{}`: {error}", path.display()),
    )
}

fn collect_tree(
    root: &Path,
    directory: &Path,
    extension: &str,
    entries: &mut BTreeMap<String, String>,
) -> io::Result<()> {
    emit_rerun(directory)?;
    let mut paths: Vec<PathBuf> = fs::read_dir(directory)
        .map_err(|error| io_error("reading directory", directory, &error))?
        .map(|entry| {
            entry
                .map(|item| item.path())
                .map_err(|error| io_error("reading directory entry in", directory, &error))
        })
        .collect::<io::Result<_>>()?;
    paths.sort();
    for path in paths {
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if file_name.starts_with('.') {
            continue;
        }
        let file_type = fs::symlink_metadata(&path)
            .map_err(|error| io_error("inspecting", &path, &error))?
            .file_type();
        if file_type.is_symlink() {
            continue;
        }
        if file_type.is_dir() {
            if file_name == "__pycache__" {
                continue;
            }
            collect_tree(root, &path, extension, entries)?;
            continue;
        }
        if !file_type.is_file() {
            continue;
        }
        if path.extension().and_then(|suffix| suffix.to_str()) != Some(extension) {
            continue;
        }
        emit_rerun(&path)?;
        let relative = path.strip_prefix(root).map_err(|error| {
            io::Error::other(format!(
                "stripping `{}` from `{}`: {error}",
                root.display(),
                path.display()
            ))
        })?;
        let without_extension = relative.with_extension("");
        let slug = without_extension
            .iter()
            .map(|part| part.to_string_lossy())
            .collect::<Vec<_>>()
            .join("/");
        let text = fs::read_to_string(&path)
            .map_err(|error| io_error("reading file", &path, &error))?
            .replace("\r\n", "\n");
        entries.insert(slug, text);
    }
    Ok(())
}

fn write_json_string(buffer: &mut String, value: &str) {
    buffer.push('"');
    for character in value.chars() {
        match character {
            '"' => buffer.push_str("\\\""),
            '\\' => buffer.push_str("\\\\"),
            '\u{0008}' => buffer.push_str("\\b"),
            '\u{000C}' => buffer.push_str("\\f"),
            '\n' => buffer.push_str("\\n"),
            '\r' => buffer.push_str("\\r"),
            '\t' => buffer.push_str("\\t"),
            code if (code as u32) < 0x20 => {
                let _ = write!(buffer, "\\u{:04x}", code as u32);
            }
            other => buffer.push(other),
        }
    }
    buffer.push('"');
}

fn write_json_map(buffer: &mut String, entries: &BTreeMap<String, String>) {
    buffer.push('{');
    for (index, (key, value)) in entries.iter().enumerate() {
        if index > 0 {
            buffer.push(',');
        }
        write_json_string(buffer, key);
        buffer.push(':');
        write_json_string(buffer, value);
    }
    buffer.push('}');
}

fn main() -> Result<(), Box<dyn Error>> {
    let manifest_directory =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, "CARGO_MANIFEST_DIR is not set")
        })?);
    let repository_root = manifest_directory
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "workspace root could not be resolved from CARGO_MANIFEST_DIR `{}`",
                    manifest_directory.display()
                ),
            )
        })?;
    let out_directory = PathBuf::from(
        env::var_os("OUT_DIR")
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "OUT_DIR is not set"))?,
    );

    let guide_directory = repository_root.join("docs").join("guide");
    let mut guide = BTreeMap::new();
    collect_tree(&guide_directory, &guide_directory, "md", &mut guide)?;

    let examples_directory = repository_root.join("examples");
    let mut examples = BTreeMap::new();
    collect_tree(&examples_directory, &examples_directory, "py", &mut examples)?;

    let mut payload = String::from("{\"examples\":");
    write_json_map(&mut payload, &examples);
    payload.push_str(",\"guide\":");
    write_json_map(&mut payload, &guide);
    payload.push_str("}\n");

    let output_path = out_directory.join("docs.json");
    fs::write(&output_path, payload)
        .map_err(|error| io_error("writing file", &output_path, &error))?;
    Ok(())
}
