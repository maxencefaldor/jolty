//! Emission of kernel declarations, parameter/prepared/shape/memory structs, limiting
//! functions, `interface`, `layout`, `prepare`, `warnings`, and parameter range checks.

use std::collections::{BTreeMap, BTreeSet};

use jolty_models::Nature;

use crate::Error;
use crate::ast::Type;
use crate::ir::{Branch, Expr, GROUND, Limiter, Module, Parameter, Stmt, constant};

use super::expr::Emitter;
use super::facts::{Facts, warns};
use super::liveness::{live, prune, reads};
use super::stamp::print;
use super::types::{
    ATOM, Activity, CodeWriter, Kind, Mode, Names, Place, Prepared, Rust, Site, literal, rust_type,
    too_long, zero,
};

impl Facts<'_> {
    pub(super) fn header(
        &self,
        (source, notice): (&str, &str),
        names: &Names,
        out: &mut CodeWriter,
    ) {
        let name = &self.module.name;
        out.line(format!(
            "//! The `{name}` kernel, compiled by jolty-veriloga from `{source}`.\n\
             //!\n\
             //! Written by `cargo run -p jolty-veriloga --bin compile`, never by hand. Its source says:\n\
             //!"
        ));
        for line in std::iter::once("```text")
            .chain(notice.lines())
            .chain(std::iter::once("```"))
        {
            if line.is_empty() {
                out.line("//!");
            } else {
                out.line(format!("//! {line}"));
            }
        }
        out.blank();
        out.line(
            "#![cfg_attr(\n    \
                 not(test),\n    \
                 deny(\n        \
                     clippy::indexing_slicing,\n        \
                     clippy::panic,\n        \
                     reason = \"a kernel never panics, and reads fixed arrays through `get`\"\n    \
                 )\n\
             )]",
        );
        out.blank();
        out.line(format!(
            "/// The `{name}` model, compiled from its Verilog-A source.\n\
             #[derive(Debug, Clone, Copy)]\n\
             pub struct {};",
            names.kernel
        ));
        out.blank();
    }

    pub(super) fn parameters(&self, names: &Names, out: &mut CodeWriter) {
        let derive_default = !names.parameters_generic
            && self.module.parameters.iter().all(|parameter| {
                parameter.optional
                    || matches!(constant(&parameter.default), Some(value) if value == 0.0)
            });
        let derives = if derive_default {
            "Debug, Clone, Default, Deserialize, Serialize, JsonSchema, Lift"
        } else {
            "Debug, Clone, Deserialize, Serialize, JsonSchema, Lift"
        };
        let serde_attribute = if self.module.parameters.is_empty() {
            "#[serde(deny_unknown_fields)]"
        } else {
            "#[serde(default, deny_unknown_fields)]"
        };
        let type_parameters = if names.parameters_generic {
            "<R: Real = f64>"
        } else {
            ""
        };
        out.line(format!(
            "/// The parameters of a [`{}`], with its Verilog-A source's defaults.\n\
             #[derive({derives})]\n\
             {serde_attribute}",
            names.kernel
        ));
        if names.parameters_generic {
            out.line("#[schemars(bound = \"R: Serialize + JsonSchema\")]");
        }
        if self.module.parameters.is_empty() {
            out.line(format!(
                "pub struct {}{type_parameters} {{}}",
                names.parameters
            ));
            out.blank();
            return;
        }
        let mut defaults = Vec::with_capacity(self.module.parameters.len());
        out.block(
            format!("pub struct {}{type_parameters} {{", names.parameters),
            "}",
            |out| {
                for parameter in &self.module.parameters {
                    defaults.push(write_parameter_field(parameter, out));
                }
            },
        );
        out.blank();
        if !derive_default {
            let mut body = CodeWriter::with_depth(2);
            body.block("Self {", "}", |body| {
                for entry in &defaults {
                    body.line(format!("{entry},"));
                }
            });
            let body = body.finish();
            let impl_header = if names.parameters_generic {
                format!("impl<R: Real> Default for {}<R> {{", names.parameters)
            } else {
                format!("impl Default for {} {{", names.parameters)
            };
            out.block(impl_header, "}", |out| {
                out.raw(&too_long(&body, "    "));
                out.line("fn default() -> Self {");
                out.raw(&body);
                out.line("}");
            });
            out.blank();
        }
    }

    pub(super) fn interface(&self, names: &Names, out: &mut CodeWriter) {
        let ports = self.module.ports;
        let mut edges: Vec<(usize, usize, bool)> = Vec::new();
        for (index, branch) in self.module.branches.iter().enumerate() {
            let collapses = self.flags.contains_key(&(index, Site::Collapse));
            let stamps = self.carries_flow(index) || self.holds_potential(index);
            let conducts =
                collapses || self.holds_potential(index) || (stamps && self.conducts(index));
            if (stamps || collapses) && branch.minus != GROUND {
                edges.push((branch.plus, branch.minus, conducts));
            }
        }
        let mut branches = Vec::new();
        for from in 0..ports {
            for to in from + 1..ports {
                let kind = [true, false]
                    .into_iter()
                    .find(|&conducting| Self::joined(&edges, (from, to), ports, conducting))
                    .map(|conducting| if conducting { "Conductive" } else { "Capacitive" });
                if let Some(kind) = kind {
                    branches.push(format!("Branch::new({from}, {to}, BranchKind::{kind})"));
                }
            }
        }
        let terminals: Vec<String> = self.module.labels[..ports]
            .iter()
            .map(|name| format!("{name:?}"))
            .collect();
        let thermal: Vec<String> = (0..ports)
            .filter(|&port| self.module.natures[port] == Nature::Temperature)
            .map(|port| port.to_string())
            .collect();
        let interface = format!(
            "Interface::new(&[{}], &[{}])",
            terminals.join(", "),
            branches.join(", ")
        );
        out.block(
            format!(
                "fn interface<R: Real>(_: &{}) -> Interface {{",
                names.parameters_type("R")
            ),
            "}",
            |out| {
                if thermal.is_empty() {
                    out.line(interface);
                } else {
                    out.block("Interface {", "}", |out| {
                        out.line(format!("thermal: vec![{}],", thermal.join(", ")));
                        out.line(format!("..{interface}"));
                    });
                }
            },
        );
        out.blank();
    }

    /// The kernel's output variables: their names, and how they are read from its memory.
    pub(super) fn outputs_fns(&self, names: &Names, out: &mut CodeWriter) {
        let module = self.module;
        if module.outputs.is_empty() {
            return;
        }
        let outputs: Vec<&str> = module
            .outputs
            .iter()
            .map(|&output| module.variables[output].0.as_str())
            .collect();
        let quoted: Vec<String> = outputs.iter().map(|name| format!("{name:?}")).collect();
        let kept: Vec<String> = outputs.iter().map(|name| format!("memory.{name}")).collect();
        out.line(format!(
            "fn variable_names() -> &'static [&'static str] {{\n    \
                 &[{}]\n\
             }}\n\n\
             fn variables(memory: &{}, values: &mut [f64]) {{\n    \
                 for (value, kept) in values.iter_mut().zip([{}]) {{\n        \
                     *value = kept;\n    \
                 }}\n\
             }}",
            quoted.join(", "),
            names.memory,
            kept.join(", ")
        ));
        out.blank();
    }

    pub(super) fn prepared_struct(&self, prepared: &Prepared, names: &Names, out: &mut CodeWriter) {
        let truths = prepared
            .fields
            .iter()
            .filter(|(_, kind, _)| *kind == Kind::Bool)
            .count();
        let type_parameters = if names.prepared_generic {
            "<R: Real>"
        } else {
            ""
        };
        out.line(format!(
            "/// A [`{}`]'s numbers at its temperature: what depends on no probe.\n\
             #[derive(Debug, Clone, Copy)]",
            names.kernel
        ));
        if prepared.fields.is_empty() && self.conditions.is_empty() {
            out.line(format!(
                "pub struct {}{type_parameters} {{}}",
                names.prepared
            ));
            out.blank();
            return;
        }
        if truths > 3 {
            out.line(
                "#[expect(\n    \
                     clippy::struct_excessive_bools,\n    \
                     reason = \"the source's conditions on its parameters, each prepared once\"\n\
                 )]",
            );
        }
        out.block(
            format!("pub struct {}{type_parameters} {{", names.prepared),
            "}",
            |out| {
                for (name, kind, _) in &prepared.fields {
                    out.line(format!("{name}: {},", rust_type(kind)));
                }
                if !self.conditions.is_empty() {
                    out.line(format!(
                        "/// The conditions that shape the kernel.\n\
                         conditions: [bool; {}],",
                        self.conditions.len()
                    ));
                }
            },
        );
        out.blank();
    }

    pub(super) fn shape_struct(
        &self,
        entries: &BTreeMap<(Place, Place), BTreeSet<(usize, Site)>>,
        names: &Names,
        out: &mut CodeWriter,
    ) {
        out.line(format!(
            "/// Where a [`{}`]'s contributions go among its local unknowns, and the conditions\n\
             /// that chose them.\n\
             #[derive(Debug, Clone, Copy, PartialEq, Eq)]",
            names.kernel
        ));
        out.block(format!("pub struct {} {{", names.shape), "}", |out| {
            for node in &self.module.nodes[self.module.ports..] {
                out.line(format!("{node}: usize,"));
            }
            for (index, branch) in self.module.branches.iter().enumerate() {
                if self.holds_potential(index) {
                    out.line(format!("{}: usize,", branch.name));
                }
            }
            for key in entries.keys() {
                out.line(format!("{}: usize,", self.entry_name(*key)));
            }
            if !self.conditions.is_empty() {
                out.line(format!("conditions: [bool; {}],", self.conditions.len()));
            }
            if self.flag_count() > 0 {
                out.line(format!("active: [bool; {}],", self.flag_count()));
            }
        });
        out.blank();
    }

    pub(super) fn memory_struct(&self, names: &Names, out: &mut CodeWriter) {
        if !self.remembers() {
            return;
        }
        let mut parts = Vec::new();
        if !self.module.limits.is_empty() {
            parts.push("each limited voltage, which the next is limited from");
        }
        if !self.module.outputs.is_empty() {
            parts.push("each output variable");
        }
        if !self.warned.is_empty() {
            parts.push("each warning that depends on the probes, once found");
        }
        out.line(format!(
            "/// What a [`{}`] remembers of its last evaluation: {}.\n\
             #[derive(Debug, Clone, Copy, Default)]",
            names.kernel,
            parts.join("; ")
        ));
        out.block(format!("pub struct {} {{", names.memory), "}", |out| {
            for index in 0..self.module.limits.len() {
                out.line(format!("limited{index}: f64,"));
            }
            for &output in &self.module.outputs {
                out.line(format!("{}: f64,", self.module.variables[output].0));
            }
            if !self.warned.is_empty() {
                out.line(format!(
                    "/// Whether an evaluation found each warning that depends on the probes, as\n\
                     /// [`{}::warned`] names them.\n\
                     warned: [bool; {}],",
                    names.kernel,
                    self.warned.len()
                ));
            }
            let values: usize = self.warned.iter().map(|(_, count)| count).sum();
            if values > 0 {
                out.line(format!(
                    "/// The values each of those warnings formats, as the last evaluation that\n\
                     /// found it read them.\n\
                     warned_values: [f64; {values}],"
                ));
            }
        });
        out.blank();
    }

    /// The analog functions that limit probes, each a function of plain floats.
    pub(super) fn functions(&self, out: &mut CodeWriter) -> Result<(), Error> {
        let used: BTreeSet<usize> = self
            .module
            .limits
            .iter()
            .filter_map(|limit| match limit.function {
                Limiter::Function(function) => Some(function),
                Limiter::Junction | Limiter::Gate => None,
            })
            .collect();
        for &index in &used {
            let function = &self.module.functions[index];
            let mut prepared = Prepared::default();
            let mut entries = BTreeMap::new();
            let mut emitter = Emitter::new(self, Mode::Function, &mut prepared, &mut entries);
            emitter.function = Some(index);
            emitter.live = vec![true; function.variables.len()];
            let mut statements = Vec::new();
            emitter.statements(&function.body, &mut statements)?;
            let locals: Vec<(String, Kind)> = function.variables[function.inputs..]
                .iter()
                .map(|(name, kind)| (name.clone(), Kind::from_type(*kind, true)))
                .collect();
            let mut body = CodeWriter::with_depth(1);
            write_locals_struct(&mut body, &locals);
            print(&statements, &mut body);
            body.line(format!("v.{}", function.name));
            let body = body.finish();
            let inputs: Vec<String> = function.variables[..function.inputs]
                .iter()
                .map(|(name, kind)| {
                    format!(
                        "{name}: {}",
                        if *kind == Type::Integer { "i32" } else { "f64" }
                    )
                })
                .collect();
            out.line(format!(
                "/// The Verilog-A analog function `{}`, which limits a voltage's step.",
                function.name
            ));
            out.raw(&too_long(&body, ""));
            out.line(format!("fn {}({}) -> f64 {{", function.name, inputs.join(", ")));
            out.raw(&body);
            out.line("}");
            out.blank();
        }
        Ok(())
    }

    pub(super) fn layout(
        &self,
        entries: &BTreeMap<(Place, Place), BTreeSet<(usize, Site)>>,
        names: &Names,
        out: &mut CodeWriter,
    ) {
        let module = self.module;
        let mut body = CodeWriter::with_depth(2);
        let flags = self.flag_count();
        if flags > 0 {
            body.line(format!("let mut active = [false; {flags}];"));
            let mut walked = Vec::new();
            self.walk_layout(&module.analog, &mut walked, &mut BTreeSet::new());
            print(&walked, &mut body);
        }
        self.write_switch_checks(&mut body);
        self.write_internal_nodes(&mut body);
        for (index, branch) in module.branches.iter().enumerate() {
            if self.holds_potential(index) {
                let claim = format!("layout.internal({:?}, Nature::Current)", branch.name);
                let code = match self.flag(index, Site::Potential).index() {
                    Some(flag) => format!("if active[{flag}] {{ {claim} }} else {{ NOWHERE }}"),
                    None => claim,
                };
                body.line(format!("let {} = {code};", branch.name));
            }
        }
        let mut fields: Vec<String> = module.nodes[module.ports..].to_vec();
        fields.extend(
            module
                .branches
                .iter()
                .enumerate()
                .filter(|(index, _)| self.holds_potential(*index))
                .map(|(_, branch)| branch.name.clone()),
        );
        for (key, users) in entries {
            fields.push(format!(
                "{}: {}",
                self.entry_name(*key),
                self.entry_claim(*key, users)
            ));
        }
        if !self.conditions.is_empty() {
            fields.push("conditions: prepared.conditions".to_owned());
        }
        if flags > 0 {
            fields.push("active".to_owned());
        }
        body.struct_literal(&format!("Ok({}", names.shape), &fields, ")");
        let body = body.finish();
        let prepared = if self.conditions.is_empty() {
            "_"
        } else {
            "prepared"
        };
        out.raw(&too_long(&body, "    "));
        out.line(format!(
            "fn layout<R: Real>(\n    \
                 _: &{},\n    \
                 {prepared}: &{},\n    \
                 layout: &mut Layout,\n\
             ) -> Result<{}, ParameterError> {{",
            names.parameters_type("R"),
            names.prepared_type("R"),
            names.shape
        ));
        out.raw(&body);
        out.line("}");
        out.blank();
    }

    /// Writes the layout's claims of the internal nodes, each a new unknown unless a short
    /// the shape makes joins it to the ground, a port or an earlier node.
    fn write_internal_nodes(&self, body: &mut CodeWriter) {
        let module = self.module;
        let claim = |node: usize| {
            format!(
                "layout.internal({:?}, Nature::{:?})",
                module.labels[node], module.natures[node]
            )
        };
        let shorts = self.shorts();
        if shorts.is_empty() {
            for node in module.ports..module.nodes.len() {
                body.line(format!("let {} = {};", module.nodes[node], claim(node)));
            }
            return;
        }
        let place = |node: usize| if node == GROUND { 0 } else { node + 1 };
        let count = module.nodes.len() + 1;
        let places: Vec<String> = (0..count).map(|place| place.to_string()).collect();
        body.line(format!(
            "// The leader of each node's group, the ground first, then the nodes in order.\n\
             let leader = |joined: &[usize; {count}], mut place: usize| {{\n    \
                 while let Some(&next) = joined.get(place)\n        \
                     && next != place\n    \
                 {{\n        \
                     place = next;\n    \
                 }}\n    \
                 place\n\
             }};\n\
             let mut joined = [{}];",
            places.join(", ")
        ));
        for (branch, flag) in shorts {
            let Branch { plus, minus, .. } = module.branches[branch];
            let join = format!(
                "let (one, other) = (leader(&joined, {}), leader(&joined, {})); if let Some(slot) = joined.get_mut(one.max(other)) {{ *slot = one.min(other); }}",
                place(plus),
                place(minus)
            );
            match flag {
                Some(flag) => body.line(format!("if active[{flag}] {{ {join} }}")),
                None => body.line(format!("{{ {join} }}")),
            }
        }
        let mut locals = vec!["NOWHERE".to_owned()];
        locals.extend((0..module.ports).map(|port| port.to_string()));
        locals.resize(count, "NOWHERE".to_owned());
        body.line(format!("let mut local = [{}];", locals.join(", ")));
        for node in module.ports..module.nodes.len() {
            let at = place(node);
            body.line(format!(
                "local[{at}] = match leader(&joined, {at}) {{ {at} => {}, leader => local.get(leader).copied().unwrap_or(NOWHERE) }};",
                claim(node)
            ));
        }
        let mut names: Vec<String> = vec!["_".to_owned(); module.ports + 1];
        names.extend(module.nodes[module.ports..].iter().cloned());
        body.line(format!("let [{}] = local;", names.join(", ")));
    }

    /// The layout's claim of the entry at `key`, where one of `users` is active and each
    /// branch current and each node it reads exists.
    fn entry_claim(&self, key: (Place, Place), users: &BTreeSet<(usize, Site)>) -> String {
        let active = |(branch, site): (usize, Site)| {
            self.flag(branch, site)
                .index()
                .map(|index| format!("active[{index}]"))
        };
        let claim = format!(
            "layout.entry({}, {})",
            self.place(key.0, ""),
            self.place(key.1, "")
        );
        let users: Option<Vec<String>> = users.iter().map(|&user| active(user)).collect();
        let currents: Vec<String> = [key.0, key.1]
            .into_iter()
            .filter_map(|place| match place {
                Place::Current(branch) => active((branch, Site::Potential)),
                Place::Node(node) if self.grounded(node) => {
                    Some(format!("{} != NOWHERE", self.node(node, "")))
                }
                Place::Node(_) => None,
            })
            .collect();
        let mut conditions: Vec<String> = Vec::new();
        if let Some(users) = users {
            let users =
                if users.len() > 1 && currents.iter().any(|current| !users.contains(current)) {
                    format!("({})", users.join(" || "))
                } else {
                    users.join(" || ")
                };
            conditions.push(users);
        }
        for current in currents {
            if !conditions.contains(&current) {
                conditions.push(current);
            }
        }
        match conditions.as_slice() {
            [] => claim,
            [only] if only.ends_with(" != NOWHERE") => format!(
                "if {} == NOWHERE {{ NOWHERE }} else {{ {claim} }}",
                only.trim_end_matches(" != NOWHERE")
            ),
            _ => format!(
                "if {} {{ {claim} }} else {{ NOWHERE }}",
                conditions.join(" && ")
            ),
        }
    }

    /// Writes the layout's checks that no branch is both a flow and a potential in one shape.
    fn write_switch_checks(&self, body: &mut CodeWriter) {
        for (index, branch) in self.module.branches.iter().enumerate() {
            let flags: Vec<String> = [Site::Flow, Site::Potential]
                .map(|site| self.flag(index, site))
                .into_iter()
                .filter_map(Activity::index)
                .map(|flag| format!("active[{flag}]"))
                .collect();
            if self.carries_flow(index) && self.holds_potential(index) && !flags.is_empty() {
                body.block(format!("if {} {{", flags.join(" && ")), "}", |body| {
                    body.line(format!(
                        "return Err(ParameterError(\"the branch {} both carries a flow and holds a potential\".to_owned()));",
                        branch.name
                    ));
                });
            }
        }
    }

    /// The layout's walk of `statements`: each flag set once where the shape's conditions
    /// make its branch active.
    fn walk_layout(&self, statements: &[Stmt], out: &mut Vec<Rust>, active: &mut BTreeSet<usize>) {
        for statement in statements {
            match statement {
                Stmt::Contribute {
                    branch,
                    potential,
                    resistive,
                    reactive,
                    ..
                } if resistive.is_some() || reactive.is_some() => {
                    let site =
                        self.site(*branch, *potential, resistive.as_ref(), reactive.as_ref());
                    if let Activity::Flag(index) = self.flag(*branch, site)
                        && active.insert(index)
                    {
                        out.push(Rust::Line(format!("active[{index}] = true;")));
                    }
                }
                Stmt::If(id, _, then, otherwise) => {
                    if let Some(index) = self.shaped.get(id) {
                        let (mut inner_then, mut inner_otherwise) = (Vec::new(), Vec::new());
                        let mut then_active = active.clone();
                        let mut otherwise_active = active.clone();
                        self.walk_layout(then, &mut inner_then, &mut then_active);
                        self.walk_layout(otherwise, &mut inner_otherwise, &mut otherwise_active);
                        if !(inner_then.is_empty() && inner_otherwise.is_empty()) {
                            out.push(Rust::If(
                                (format!("prepared.conditions[{index}]"), ATOM),
                                inner_then,
                                inner_otherwise,
                            ));
                        }
                    } else {
                        self.walk_layout(then, out, active);
                        self.walk_layout(otherwise, out, active);
                    }
                }
                Stmt::While(_, body) => self.walk_layout(body, out, active),
                Stmt::Assign(..) | Stmt::Contribute { .. } | Stmt::Fail(..) | Stmt::Warn(..) => {}
            }
        }
    }

    /// Writes the lines that check each parameter against its ranges.
    fn write_parameter_checks(
        &self,
        emitter: &mut Emitter<'_, '_>,
        out: &mut CodeWriter,
    ) -> Result<(), Error> {
        for (index, parameter) in self.module.parameters.iter().enumerate() {
            for (exclude, lower, upper) in &parameter.ranges {
                let mut check_body = Vec::new();
                let mut bound = |bound: &Option<(Expr, bool)>| -> Result<Option<Bound>, Error> {
                    let Some((expr, included)) = bound else {
                        return Ok(None);
                    };
                    let constant = constant(expr);
                    let code = match constant {
                        Some(value) => literal(value),
                        None => Emitter::value_of(&emitter.expression(expr, &mut check_body)?).code,
                    };
                    Ok(Some(Bound {
                        code,
                        constant,
                        included: *included,
                    }))
                };
                let (lower, upper) = (bound(lower)?, bound(upper)?);
                let value = emitter.expression(&Expr::Parameter(index), &mut check_body)?;
                print(&check_body, out);
                out.line(check(
                    (&value.code, &parameter.external),
                    parameter.kind,
                    *exclude,
                    (lower.as_ref(), upper.as_ref()),
                ));
            }
        }
        Ok(())
    }

    /// The inherent `impl` holding `warnings` and `warned`, when the module defines any.
    pub(super) fn warnings_impl(
        &self,
        prepared: &Prepared,
        names: &Names,
    ) -> Result<String, Error> {
        let mut inherent = CodeWriter::with_depth(1);
        let mut has_any = false;
        if self.module.analog.iter().any(warns) {
            self.prepare(Mode::Warnings, prepared, names, &mut inherent)?;
            has_any = true;
        }
        if !self.warned.is_empty() {
            if has_any {
                inherent.blank();
            }
            let mut messages = Vec::new();
            let mut offset = 0;
            for (message, count) in &self.warned {
                let values: Vec<String> = (offset..offset + count)
                    .map(|index| format!("memory.warned_values[{index}]"))
                    .collect();
                offset += count;
                messages.push(format_message(message, &values));
            }
            inherent.line(
                "/// The message of each warning that depends on the probes that an evaluation\n\
                 /// of an instance, whose memory is `memory`, has found since the memory was made,\n\
                 /// with the values the last evaluation that found it read.",
            );
            inherent.block(
                format!(
                    "pub fn warned(memory: &{}) -> impl Iterator<Item = String> {{",
                    names.memory
                ),
                "}",
                |inherent| {
                    inherent.line(format!(
                        "[{}].into_iter().zip(memory.warned).filter_map(|(message, found)| found.then_some(message))",
                        messages.join(", ")
                    ));
                },
            );
            has_any = true;
        }
        if !has_any {
            return Ok(String::new());
        }
        let body = inherent.finish();
        Ok(format!(
            "impl {} {{\n{}\n}}\n",
            names.kernel,
            body.trim_end_matches('\n')
        ))
    }

    /// The fields of the prepared numbers, as `prepare` writes them at its end, in
    /// [`Mode::Prepare`]: each of `fields`, and the conditions that shape the kernel.
    fn prepared_values(
        &self,
        emitter: &mut Emitter<'_, '_>,
        (mode, fields): (Mode, &[(String, Kind, Expr)]),
        body: &mut Vec<Rust>,
    ) -> Result<Vec<String>, Error> {
        let mut values = Vec::new();
        for (name, kind, expr) in fields {
            let operand = emitter.expression(expr, body)?;
            let operand = emitter.convert(operand, kind, body)?;
            values.push(if operand.code == *name {
                name.clone()
            } else {
                format!("{name}: {}", operand.code)
            });
        }
        if mode == Mode::Prepare && !self.conditions.is_empty() {
            let mut conditions = Vec::with_capacity(self.conditions.len());
            for condition in &self.conditions {
                let operand = emitter.expression(condition, body)?;
                conditions.push(emitter.truth(operand).0);
            }
            let joined = conditions.join(", ");
            if conditions.len() > 2 || joined.len() > 72 {
                let mut block = CodeWriter::new();
                block.block("conditions: [", "", |block| {
                    for condition in conditions {
                        block.line(format!("{condition},"));
                    }
                });
                block.raw("]");
                values.push(block.finish());
            } else {
                values.push(format!("conditions: [{joined}]"));
            }
        }
        Ok(values)
    }

    /// Writes `prepare`, in [`Mode::Prepare`], or in [`Mode::Warnings`] the kernel's
    /// function `warnings`, which runs preparation again for what its `$warning`s state.
    pub(super) fn prepare(
        &self,
        mode: Mode,
        prepared: &Prepared,
        names: &Names,
        out: &mut CodeWriter,
    ) -> Result<(), Error> {
        let module = self.module;
        let mut entries = BTreeMap::new();
        let mut scratch = Prepared::default();
        let mut emitter = Emitter::new(self, mode, &mut scratch, &mut entries);
        let mut roots = BTreeSet::new();
        let fields: &[(String, Kind, Expr)] = if mode == Mode::Prepare {
            &prepared.fields
        } else {
            &[]
        };
        for (_, _, expr) in fields {
            reads(expr, module, &mut roots);
        }
        if mode == Mode::Prepare {
            for condition in &self.conditions {
                reads(condition, module, &mut roots);
            }
        }
        emitter.analog = prune(&module.analog, module, mode, &mut roots.clone());
        emitter.live = live(module, &emitter.analog, mode, &roots);
        let mut statements = Vec::new();
        let analog = emitter.analog.clone();
        emitter.statements(&analog, &mut statements)?;
        let values = self.prepared_values(&mut emitter, (mode, fields), &mut statements)?;
        let mut checks = CodeWriter::with_depth(2);
        self.write_parameter_checks(&mut emitter, &mut checks)?;
        let (constant_fields, constant_body) = emit_local_parameters(&mut emitter)?;
        let locals = static_locals(module, &emitter.live);
        let mut body = CodeWriter::with_depth(2);
        write_struct_def(&mut body, "Constants", &constant_fields);
        write_struct_def(&mut body, "Locals", &locals);
        write_struct_init(&mut body, "Constants", "c", &constant_fields);
        print(&constant_body, &mut body);
        body.raw(&checks.finish());
        write_struct_init(&mut body, "Locals", "v", &locals);
        if mode == Mode::Warnings {
            body.line("let mut found = Vec::new();");
        }
        print(&statements, &mut body);
        if mode == Mode::Prepare {
            body.struct_literal(&format!("Ok({}", names.prepared), &values, ")");
        } else {
            body.line("Ok(found)");
        }
        write_prepare(
            mode,
            names,
            &body.finish(),
            !emitter.parameters_read.is_empty(),
            emitter.uses_temperature,
            out,
        );
        Ok(())
    }
}

/// Writes one struct field declaration for `parameter` into `out` and returns its
/// `Default` initializer entry.
fn write_parameter_field(parameter: &Parameter, out: &mut CodeWriter) -> String {
    let mut documentation = parameter
        .description
        .as_deref()
        .map_or_else(|| format!("The parameter `{}`", parameter.external), prose);
    if let Some(units) = parameter.units.as_deref().filter(|units| !units.is_empty()) {
        documentation = format!("{documentation}, in {}", prose(units));
    }
    let default = match constant(&parameter.default) {
        Some(value) => format!("default {value}"),
        None => "by default, as its source computes it from the parameters before it".to_owned(),
    };
    out.line(format!("/// {documentation}; {default}."));
    if parameter.name != parameter.external {
        out.line(format!("#[serde(rename = {:?})]", parameter.external));
    }
    for alias in &parameter.aliases {
        out.line(format!("#[serde(alias = {alias:?})]"));
    }
    let number = if parameter.kind == Type::Integer {
        "i32"
    } else {
        "R"
    };
    let kind = if parameter.optional {
        format!("Option<{number}>")
    } else {
        number.to_owned()
    };
    out.line(format!("pub {}: {kind},", parameter.name));
    let value = match (
        parameter.optional,
        parameter.kind,
        constant(&parameter.default),
    ) {
        (true, _, _) | (false, _, None) => "None".to_owned(),
        (false, Type::Integer, Some(value)) => format!("{value}"),
        (false, Type::Real, Some(value)) => format!("R::constant({})", literal(value)),
    };
    format!("{}: {value}", parameter.name)
}

/// The static variables kept for preparation, each with its emitted kind.
fn static_locals(module: &Module, live: &[bool]) -> Vec<(String, Kind)> {
    module
        .variables
        .iter()
        .enumerate()
        .filter(|(index, _)| live[*index] && !module.dynamic[*index])
        .map(|(_, (name, kind))| (name.clone(), Kind::from_type(*kind, false)))
        .collect()
}

/// Writes `prepare` or `warnings` with its signature, doc comment, and pre-indented `body`.
fn write_prepare(
    mode: Mode,
    names: &Names,
    body: &str,
    uses_parameters: bool,
    uses_temperature: bool,
    out: &mut CodeWriter,
) {
    let temperature = if uses_temperature { "temperature" } else { "_" };
    let parameters = if uses_parameters { "parameters" } else { "_" };
    let (name, result) = if mode == Mode::Prepare {
        ("fn prepare", names.prepared_type("R"))
    } else {
        out.line(
            "/// The facts the `$warning`s of the Verilog-A source, and their like, state of an\n\
             /// instance with `parameters` at `temperature`, in kelvins, in the order it states\n\
             /// them.\n\
             ///\n\
             /// # Errors\n\
             ///\n\
             /// Returns [`ParameterError`] when a parameter is out of range.",
        );
        ("pub fn warnings", "Vec<String>".to_owned())
    };
    out.raw(&too_long(body, "    "));
    out.line(format!(
        "{name}<R: Real>(\n    \
             {parameters}: &{},\n    \
             {temperature}: R,\n\
         ) -> Result<{result}, ParameterError> {{",
        names.parameters_type("R")
    ));
    out.raw(body);
    out.line("}");
    if mode == Mode::Prepare {
        out.blank();
    }
}

/// Field descriptors and initialization statements emitted for live `localparam` constants.
type EmittedLocals = (Vec<(String, Kind)>, Vec<Rust>);

/// Emits the field descriptors and initialization statements for every non-constant
/// `localparam` read during preparation, in ascending declaration order.
fn emit_local_parameters(emitter: &mut Emitter<'_, '_>) -> Result<EmittedLocals, Error> {
    let count = emitter.facts.module.local_parameters.len();
    let mut entries: Vec<(String, Kind, Vec<Rust>)> = Vec::new();
    for index in (0..count).rev() {
        if !emitter.locals_read.contains(&index) {
            continue;
        }
        let local = &emitter.facts.module.local_parameters[index];
        let (name, kind, value) = (
            local.name.clone(),
            Kind::from_type(local.kind, false),
            local.value.clone(),
        );
        let mut statements = Vec::new();
        let operand = emitter.expression(&value, &mut statements)?;
        let operand = emitter.convert(operand, &kind, &mut statements)?;
        statements.push(Rust::Line(format!("c.{name} = {};", operand.code)));
        entries.push((name, kind, statements));
    }
    entries.reverse();
    let mut fields = Vec::with_capacity(entries.len());
    let mut body = Vec::new();
    for (name, kind, statements) in entries {
        fields.push((name, kind));
        body.extend(statements);
    }
    Ok((fields, body))
}

/// Writes the `struct {struct_name}` definition for `fields`, if non-empty.
fn write_struct_def(out: &mut CodeWriter, struct_name: &str, fields: &[(String, Kind)]) {
    if fields.is_empty() {
        return;
    }
    let generic = fields
        .iter()
        .any(|(_, kind)| matches!(kind, Kind::Real | Kind::Dual(_)));
    let parameters = if generic { "<R: Real>" } else { "" };
    out.block(format!("struct {struct_name}{parameters} {{"), "}", |out| {
        for (name, kind) in fields {
            out.line(format!("{name}: {},", rust_type(kind)));
        }
    });
}

/// Writes `let mut {binding} = {struct_name} { ... };` initializing `fields` to zero.
fn write_struct_init(
    out: &mut CodeWriter,
    struct_name: &str,
    binding: &str,
    fields: &[(String, Kind)],
) {
    if fields.is_empty() {
        return;
    }
    let values: Vec<String> = fields
        .iter()
        .map(|(name, kind)| format!("{name}: {}", zero(kind)))
        .collect();
    out.struct_literal(&format!("let mut {binding} = {struct_name}"), &values, ";");
}

/// Writes both the `Locals` struct definition and its zero-initialized `let mut v` binding.
pub(super) fn write_locals_struct(out: &mut CodeWriter, locals: &[(String, Kind)]) {
    write_struct_def(out, "Locals", locals);
    write_struct_init(out, "Locals", "v", locals);
}

/// A bound of a parameter's range, as the check reads it: its code, its value when it is
/// a constant, and whether it is included.
struct Bound {
    code: String,
    constant: Option<f64>,
    included: bool,
}

/// The check that the parameter `name`, named `external` in its source, of `kind`, lies
/// in the range of `lower` and `upper`, or, when `exclude`, does not.
fn check(
    (name, external): (&str, &str),
    kind: Type,
    exclude: bool,
    (lower, upper): (Option<&Bound>, Option<&Bound>),
) -> String {
    let (value, shown) = if kind == Type::Integer {
        (format!("f64::from({name})"), name.to_owned())
    } else {
        let reading = format!("{name}.value()");
        (reading.clone(), reading)
    };
    if !exclude
        && upper.is_none()
        && kind == Type::Real
        && lower.is_some_and(|low| low.constant == Some(0.0))
    {
        let helper = if lower.is_some_and(|low| low.included) {
            "non_negative"
        } else {
            "positive"
        };
        return format!("ParameterError::{helper}({external:?}, {name})?;");
    }
    let contains = match (lower, upper) {
        (Some(low), Some(high)) if low.included && !high.included => {
            format!("({}..{}).contains(&{value})", low.code, high.code)
        }
        (Some(low), Some(high)) if low.included => {
            format!("({}..={}).contains(&{value})", low.code, high.code)
        }
        (Some(low), None) if low.included => format!("({}..).contains(&{value})", low.code),
        (None, Some(high)) if !high.included => format!("(..{}).contains(&{value})", high.code),
        (None, Some(high)) => format!("(..={}).contains(&{value})", high.code),
        (Some(low), high) => {
            let above = format!(
                "matches!({value}.partial_cmp(&{}), Some(Ordering::Greater))",
                low.code
            );
            match high {
                Some(high) if high.included => format!("{above} && {value} <= {}", high.code),
                Some(high) => format!("{above} && {value} < {}", high.code),
                None => above,
            }
        }
        (None, None) => "true".to_owned(),
    };
    let mut arguments = Vec::new();
    let mut words = |bound: &Bound, low: bool| {
        let text = bound.constant.map_or_else(
            || {
                arguments.push(bound.code.clone());
                "{:?}".to_owned()
            },
            |value| format!("{value}"),
        );
        match (low, bound.included) {
            (true, true) => format!("at least {text}"),
            (true, false) => format!("above {text}"),
            (false, true) => format!("at most {text}"),
            (false, false) => format!("below {text}"),
        }
    };
    let mut described = Vec::new();
    if let Some(low) = lower {
        described.push(words(low, true));
    }
    if let Some(high) = upper {
        described.push(words(high, false));
    }
    arguments.push(shown);
    let (condition, verb) = if exclude {
        (contains, "must not be")
    } else if contains.contains("&&") {
        (format!("!({contains})"), "must be")
    } else {
        (format!("!{contains}"), "must be")
    };
    format!(
        "if {condition} {{\n    return Err(ParameterError(format!(\"{external} {verb} {}, not {{:?}}\", {})));\n}}",
        described.join(" and "),
        arguments.join(", ")
    )
}

/// The code that makes the message of a task, `message` with each of its `%g`s and
/// their like the value that code of `values` at its place makes, and each value past
/// them after it, as Verilog's `$display` prints an argument no format takes.
pub(super) fn format_message(message: &str, values: &[String]) -> String {
    let mut format = message.replace('{', "{{").replace('}', "}}");
    let mut taken = 0;
    for placeholder in ["%g", "%e", "%f", "%d", "%r"] {
        taken += format.matches(placeholder).count();
        format = format.replace(placeholder, "{:?}");
    }
    for _ in taken..values.len() {
        format.push_str("{:?}");
    }
    if values.is_empty() {
        format!("{format:?}.to_owned()")
    } else {
        format!(
            "format!(\"{}\", {})",
            format.replace('"', "\\\""),
            values.join(", ")
        )
    }
}

/// `text`, the source's prose, as documentation writes it: each word that reads as code,
/// with an underscore or a capital after a small letter, in backticks, as clippy's
/// `doc_markdown` asks.
fn prose(text: &str) -> String {
    text.split(' ')
        .map(|word| {
            let core = word
                .trim_matches(|character: char| !character.is_alphanumeric() && character != '_');
            let camel = core
                .chars()
                .zip(core.chars().skip(1))
                .any(|(before, after)| before.is_lowercase() && after.is_uppercase());
            if core.is_empty() || !(core.contains('_') || camel) {
                word.to_owned()
            } else {
                word.replacen(core, &format!("`{core}`"), 1)
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}
