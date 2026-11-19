//! Duplicate-declaration, alias, discipline, reactive-factor, and initial-event checks
//! for Verilog-A module resolution.

use std::collections::BTreeSet;

use crate::Error;
use crate::ast::{self, Type};

use super::{Branch, Module, Statement, apart, identifier};

/// Rejects duplicate `localparam` names in `local_parameters` and any name for which
/// `collision` returns a conflict phrase.
fn check_local_params(
    local_parameters: &[ast::LocalParameter],
    collision: impl Fn(&str) -> Option<&'static str>,
) -> Result<(), Error> {
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for local in local_parameters {
        if !seen.insert(local.name.as_str()) {
            return Err(Error::at(
                local.line,
                format!("the localparam {} is declared more than once", local.name),
            ));
        }
        if let Some(conflict) = collision(&local.name) {
            return Err(Error::at(local.line, format!("{} {conflict}", local.name)));
        }
    }
    Ok(())
}

/// Rejects duplicate `localparam` names and collisions with a `parameter`, variable, or
/// `aliasparam` in module scope.
pub(super) fn check_module_duplicates(module: &ast::Module) -> Result<(), Error> {
    check_local_params(&module.local_parameters, |name| {
        if module.parameters.iter().any(|item| item.name == name) {
            Some("is declared as both a parameter and a localparam")
        } else if module.variables.iter().any(|(item, _)| item == name) {
            Some("is declared as both a localparam and a variable")
        } else if module.aliases.iter().any(|(item, _)| item == name) {
            Some("is declared as both a localparam and an aliasparam")
        } else {
            None
        }
    })
}

/// Rejects duplicate `localparam` names and collisions with variables in the same named block.
pub(super) fn check_block_duplicates(
    local_parameters: &[ast::LocalParameter],
    variables: &[(String, Type)],
) -> Result<(), Error> {
    check_local_params(local_parameters, |name| {
        variables
            .iter()
            .any(|(variable, _)| variable == name)
            .then_some("is declared as both a localparam and a variable")
    })
}

/// Rejects duplicate `localparam` names and collisions with inputs, variables, or the
/// result name of an analog function.
pub(super) fn check_function_duplicates(function: &ast::Function) -> Result<(), Error> {
    check_local_params(&function.local_parameters, |name| {
        (name == function.name
            || function.inputs.iter().any(|input| input == name)
            || function.variables.iter().any(|(item, _)| item == name))
        .then_some("is declared as both a localparam and a variable")
    })
}

/// Checks that every `aliasparam` targets a declared `parameter`, not a `localparam` or
/// an undeclared name.
pub(super) fn check_aliases(module: &ast::Module) -> Result<(), Error> {
    for (alias, parameter) in &module.aliases {
        if module.parameters.iter().any(|item| item.name == *parameter) {
            continue;
        }
        if module
            .local_parameters
            .iter()
            .any(|local| local.name == *parameter)
        {
            return Err(Error::at(
                0,
                format!("the alias {alias} targets localparam {parameter}, not a parameter"),
            ));
        }
        return Err(Error::at(
            0,
            format!("the alias {alias} names no parameter, {parameter}"),
        ));
    }
    Ok(())
}

/// The nodes of `module`, its ports first, and whether each is thermal.
pub(super) fn nodes_of(module: &ast::Module) -> Result<(Vec<String>, Vec<bool>), Error> {
    let mut nodes: Vec<String> = module.ports.clone();
    for (node, _) in &module.nodes {
        if !nodes.contains(node) {
            nodes.push(node.clone());
        }
    }
    for port in &module.ports {
        if !module.nodes.iter().any(|(node, _)| node == port) {
            return Err(Error::at(0, format!("the port {port} has no discipline")));
        }
    }
    let thermal = nodes
        .iter()
        .map(|node| {
            module
                .nodes
                .iter()
                .any(|(name, discipline)| name == node && discipline == "thermal")
        })
        .collect();
    Ok((nodes, thermal))
}

/// The index of the node `name` among `nodes`.
pub(super) fn node_index(nodes: &[String], name: &str, line: usize) -> Result<usize, Error> {
    nodes
        .iter()
        .position(|node| node == name)
        .ok_or_else(|| Error::at(line, format!("{name} is not a node")))
}

/// `branches`, each by its Rust name.
pub(super) fn named_apart(branches: Vec<Branch>) -> Vec<Branch> {
    let names: Vec<String> = branches.iter().map(|branch| branch.name.clone()).collect();
    branches
        .into_iter()
        .zip(rust_names(&names))
        .map(|(branch, name)| Branch { name, ..branch })
        .collect()
}

/// The Rust names of `names`, in snake case, each apart from those before it.
pub(super) fn rust_names(names: &[String]) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    for name in names {
        let unique = apart(&identifier(name), |taken| {
            found.iter().any(|existing| existing == taken)
        });
        found.push(unique);
    }
    found
}

/// Checks that no variable assigned under an initial event depends on the probes.
pub(super) fn check_initial_variables(
    resolved: &Module,
    initial_variables: &BTreeSet<usize>,
    origins: &[usize],
) -> Result<(), Error> {
    let initial = (0..resolved.variables.len()).find(|&variable| {
        resolved.dynamic[variable] && initial_variables.contains(&origins[variable])
    });
    if let Some(variable) = initial {
        return Err(Error::at(
            0,
            format!(
                "{}, which code under an initial event assigns, depends on the probes, which is not compiled yet",
                resolved.variables[variable].0
            ),
        ));
    }
    Ok(())
}

/// Checks that every expression factored from outside `ddt(...)` (`k * ddt(x)` folded
/// into `ddt(k * x)`) is static, so every reactive contribution conserves charge or flux.
pub(super) fn check_reactive_factors(module: &Module) -> Result<(), Error> {
    let mut failure = None;
    Statement::visit_stmts(&module.analog, &mut |statement| {
        if failure.is_none()
            && let Statement::Contribute {
                reactive_factors, ..
            } = statement
            && let Some((_, line)) = reactive_factors
                .iter()
                .find(|(factor, _)| module.is_dynamic(factor))
        {
            failure = Some(Error::at(
                *line,
                "ddt scaled by a dynamic expression is not charge-conserving and is not compiled yet"
                    .into(),
            ));
        }
    });
    failure.map_or(Ok(()), Err)
}
