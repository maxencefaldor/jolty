//! The emitter: a resolved module written as a plain Rust kernel, the same form a person
//! writes, generic over jolty's number types.
//!
//! Its parameters become a `Parameters<R>` struct with their defaults; what depends on no
//! probe is computed in `prepare`, whose numbers the evaluation reads; the branches its
//! parameters switch, collapse or leave out are decided in `layout`, which claims the
//! internal unknowns and the entries of the shape they give; and `evaluate` runs the
//! analog block on jolty's dual numbers, each variable carrying the derivatives of the
//! probes it depends on and no others, then stamps each branch once.

mod decl;
mod expr;
mod facts;
mod liveness;
mod stamp;
mod types;

use std::collections::BTreeMap;

use crate::Error;
use crate::ast::Type;
use crate::ir::Module;

use expr::Emitter;
use facts::{Facts, makes_noise};
use types::{CodeWriter, Kind, MethodBody, Mode, Names, Prepared, mentions, too_long};

/// The kernel of `module`, compiled from `source`, as the text of a Rust module.
///
/// # Errors
///
/// Returns an [`Error`] for a construct the emitter does not compile yet.
pub(crate) fn emit(module: &Module, (source, header): (&str, &str)) -> Result<String, Error> {
    let facts = Facts::new(module)?;
    let mut prepared = Prepared::default();
    let mut entries = BTreeMap::new();
    let evaluate_body =
        Emitter::new(&facts, Mode::Evaluate, &mut prepared, &mut entries).evaluate_body()?;
    let noise_body = if module.analog.iter().any(makes_noise) {
        Some(Emitter::new(&facts, Mode::Noise, &mut prepared, &mut entries).noise_body()?)
    } else {
        None
    };
    let linear_body =
        Emitter::new(&facts, Mode::Evaluate, &mut prepared, &mut entries).linear_body();
    let parameters_generic = module
        .parameters
        .iter()
        .any(|parameter| parameter.kind == Type::Real);
    let prepared_generic = prepared
        .fields
        .iter()
        .any(|(_, kind, _)| matches!(kind, Kind::Real | Kind::Dual(_)));
    let names = Names::from_module(&module.name, parameters_generic, prepared_generic);

    let mut out = CodeWriter::new();
    facts.header((source, header), &names, &mut out);
    facts.parameters(&names, &mut out);
    let warnings = facts.warnings_impl(&prepared, &names)?;
    facts.prepared_struct(&prepared, &names, &mut out);
    facts.shape_struct(&entries, &names, &mut out);
    facts.memory_struct(&names, &mut out);
    facts.functions(&mut out)?;

    let memory = if facts.remembers() {
        names.memory.as_str()
    } else {
        "()"
    };
    let mut prepare_error = None;
    out.block(format!("impl Kernel for {} {{", names.kernel), "}", |out| {
        out.line(format!(
            "const NAME: &'static str = {:?};\n\
             type Parameters<R: Real> = {};\n\
             type Shape = {};\n\
             type Prepared<R: Real> = {};\n\
             type Memory = {memory};",
            module.name,
            names.parameters_type("R"),
            names.shape,
            names.prepared_type("R")
        ));
        out.blank();
        facts.interface(&names, out);
        facts.outputs_fns(&names, out);
        if let Some(linear) = linear_body.as_deref() {
            write_linear(linear, &names, out);
        }
        if let Err(error) = facts.prepare(Mode::Prepare, &prepared, &names, out) {
            prepare_error = Some(error);
            return;
        }
        facts.layout(&entries, &names, out);
        write_kernel_method(
            "evaluate",
            &evaluate_body,
            &names,
            &[
                evaluate_body.memory.as_str(),
                "at: &Local<'_>",
                "stamp: &mut Stamp<'_, R>",
            ],
            out,
        );
        if let Some(noise) = noise_body.as_ref() {
            out.blank();
            let at_parameter = format!("{}: &Local<'_>", noise.at);
            write_kernel_method(
                "noise",
                noise,
                &names,
                &[at_parameter.as_str(), "sources: &mut Vec<NoiseSource>"],
                out,
            );
        }
    });
    if let Some(error) = prepare_error {
        return Err(error);
    }
    if !warnings.is_empty() {
        out.blank();
        out.raw(&warnings);
    }
    Ok(imports(&out.finish()))
}

fn write_linear(body: &str, names: &Names, out: &mut CodeWriter) {
    let shape = if mentions(body, "shape") {
        "shape"
    } else {
        "_"
    };
    let prepared = if mentions(body, "prepared") {
        "prepared"
    } else {
        "_"
    };
    out.block(
        format!(
            "fn linear({shape}: &{}, {prepared}: &{}) -> bool {{",
            names.shape,
            names.prepared_type("f64")
        ),
        "}",
        |out| out.line(body),
    );
    out.blank();
}

fn write_kernel_method(
    name: &str,
    body: &MethodBody,
    names: &Names,
    extra_parameters: &[&str],
    out: &mut CodeWriter,
) {
    out.raw(&too_long(&body.body, "    "));
    out.block(format!("fn {name}<R: Real>("), ") {", |out| {
        out.line(format!("{}: &{},", body.shape, names.shape));
        out.line(format!("{}: &{},", body.prepared, names.prepared_type("R")));
        for parameter in extra_parameters {
            out.line(format!("{parameter},"));
        }
    });
    out.raw(&body.body);
    out.line("}");
}

/// `text` with the imports its code uses, after its module documentation.
fn imports(text: &str) -> String {
    const CANDIDATES: [&str; 19] = [
        "BOLTZMANN",
        "Branch",
        "BranchKind",
        "Dual",
        "ELEMENTARY_CHARGE",
        "Interface",
        "Kernel",
        "Layout",
        "Lift",
        "Local",
        "Nature",
        "NoiseSource",
        "ParameterError",
        "Real",
        "SMALL_RESISTANCE",
        "Stamp",
        "UnknownIndex",
        "limit_gate",
        "limit_junction",
    ];
    let end = ")]\n\n";
    let (documentation, rest) = text.split_at(text.find(end).map_or(0, |at| at + end.len()));
    let used: Vec<&str> = CANDIDATES
        .into_iter()
        .filter(|name| mentions(rest, name))
        .collect();
    let mut out = String::with_capacity(text.len() + 256);
    out.push_str(documentation);
    if mentions(rest, "Ordering") {
        out.push_str("use std::cmp::Ordering;\n\n");
    }
    let single = format!("use jolty_models::{{{}}};\n", used.join(", "));
    if single.len() <= 100 {
        out.push_str(&single);
    } else {
        out.push_str("use jolty_models::{\n");
        let mut line = String::from("    ");
        for item in used {
            if line.len() > 4 && line.len() + item.len() + 2 > 100 {
                out.push_str(&line);
                out.push('\n');
                line = String::from("    ");
            }
            if line.len() > 4 {
                line.push(' ');
            }
            line.push_str(item);
            line.push(',');
        }
        if line.len() > 4 {
            out.push_str(&line);
            out.push('\n');
        }
        out.push_str("};\n");
    }
    out.push_str("use schemars::JsonSchema;\nuse serde::{Deserialize, Serialize};\n\n");
    if mentions(rest, "NOWHERE") {
        out.push_str(
            "/// A local index no instance has: a write to it goes nowhere, and a read of it reads\n\
             /// zero, as `Stamp` and `Local` promise.\n\
             const NOWHERE: usize = usize::MAX;\n\n",
        );
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::compile;
    use crate::ir::resolve;
    use crate::lexer::tokens;
    use crate::parser::parse;

    use super::liveness::mark;
    use super::types::Mode;

    /// Returns the body of `pub struct {name}` up to its closing brace.
    fn struct_body<'a>(code: &'a str, header: &str) -> &'a str {
        let start = code
            .find(header)
            .unwrap_or_else(|| panic!("missing {header} in:\n{code}"));
        let rest = &code[start..];
        let end = rest
            .find("\n}\n")
            .unwrap_or_else(|| panic!("unterminated {header} in:\n{rest}"));
        &rest[..end]
    }

    #[test]
    fn integer_real_product_folds_without_prepared_field() {
        let source = "\
`include \"disciplines.vams\"
module m(a, b);
    inout a, b;
    electrical a, b;
    parameter real g = 1.0;
    analog begin
        I(a, b) <+ V(a, b) * g;
        I(a, b) <+ white_noise(2 * 1.602176634e-19 * abs(V(a, b)), \"shot\");
    end
endmodule
";
        let code = compile(source, "m.va", &|_| None).expect("compile");
        let prepared = struct_body(&code, "pub struct PreparedM<R: Real> {");
        assert!(
            !prepared.contains(": f64,"),
            "expected folded constant instead of hoisted f64 field:\n{prepared}"
        );
        assert!(
            code.contains("3.204_353_268e-19"),
            "expected folded constant inline in:\n{code}"
        );
    }

    #[test]
    fn negations_and_signed_products_reuse_one_prepared_field() {
        let source = "\
`include \"disciplines.vams\"
module m(a, b);
    inout a, b;
    electrical a, b;
    parameter real g = 1.0;
    real s;
    analog begin
        s = g * 2.0;
        if (V(a, b) < -3 * s)
            I(a, b) <+ -s * V(a, b);
        else if (V(a, b) >= -(s * 3))
            I(a, b) <+ (s * -3) * V(a, b);
        else
            I(a, b) <+ (s * 3) * V(a, b);
    end
endmodule
";
        let code = compile(source, "m.va", &|_| None).expect("compile");
        let prepared = struct_body(&code, "pub struct PreparedM<R: Real> {");
        let real_fields = prepared
            .lines()
            .filter(|line| line.trim_end().ends_with(": R,"))
            .count();
        assert_eq!(
            real_fields, 2,
            "expected only `s` and one shared `s * 3.0` field:\n{prepared}"
        );
        assert!(!code.contains("-v.s"), "unexpected negated field:\n{code}");
        assert!(
            !code.contains("v.s * -3.0"),
            "unexpected negatively-signed product field:\n{code}"
        );
        assert!(
            !code.contains("-(v.s * 3.0)"),
            "unexpected negated product field:\n{code}"
        );
    }

    #[test]
    fn deduplicates_shaped_conditions_and_active_flags_and_inverts_empty_if() {
        let source = "\
`include \"disciplines.vams\"
module m(a, b);
    inout a, b;
    electrical a, b, n;
    parameter integer mode = 0;
    analog begin
        if (mode != 0)
            V(a, n) <+ 0;
        else
            I(a, n) <+ V(a, n) / 10.0;
        if (mode != 0) begin
            I(n, b) <+ V(n, b) / 100.0;
            I(n, b) <+ ddt(1e-12 * V(n, b));
        end else begin
            I(n, b) <+ V(n, b) / 200.0;
        end
    end
endmodule
";
        let code = compile(source, "m.va", &|_| None).expect("compile");
        assert!(
            code.contains("conditions: [bool; 1],"),
            "identical shaped conditions must share one slot:\n{code}"
        );
        assert!(
            !code.contains("{\n        } else"),
            "empty if-then branch must be inverted:\n{code}"
        );
        assert!(
            code.contains("if !shape.conditions[0] {"),
            "expected inverted condition in evaluate:\n{code}"
        );
        let lines: Vec<&str> = code.lines().map(str::trim).collect();
        for pair in lines.windows(2) {
            assert!(
                !(pair[0].starts_with("active[") && pair[0] == pair[1]),
                "duplicate consecutive active flag assignment `{}`:\n{code}",
                pair[0]
            );
        }
    }

    #[test]
    fn diode_model_folds_constant_deduplicates_conditions_and_inverts_empty_if() {
        let source = include_str!("../../models/diode.va");
        let code = compile(source, "crates/jolty-veriloga/models/diode.va", &|_| None)
            .expect("diode compiles");
        let prepared = struct_body(&code, "pub struct PreparedDiode<R: Real> {");
        assert!(
            prepared.contains("conditions: [bool; 5],"),
            "expected 5 deduplicated conditions in diode:\n{prepared}"
        );
        assert!(
            !prepared.contains(": f64,"),
            "expected 2 * P_Q folded without f64 prepared field:\n{prepared}"
        );
        assert!(
            !code.contains("{\n        } else"),
            "expected no empty if blocks in diode"
        );
        let lines: Vec<&str> = code.lines().map(str::trim).collect();
        for pair in lines.windows(2) {
            assert!(
                !(pair[0].starts_with("active[") && pair[0] == pair[1]),
                "duplicate consecutive active flag assignment `{}` in diode",
                pair[0]
            );
        }
        assert!(
            code.contains(".embed::<2>("),
            "expected Dual::embed widening in diode"
        );
    }

    #[test]
    fn resistor_linear_factors_common_disjunct_and_omits_value_on_f64() {
        let source = include_str!("../../models/resistor.va");
        let code = compile(source, "crates/jolty-veriloga/models/resistor.va", &|_| {
            None
        })
        .expect("resistor compiles");
        let linear_start = code.find("fn linear(").expect("linear method");
        let linear_end = linear_start + code[linear_start..].find("\n    }\n").expect("end");
        let linear = &code[linear_start..linear_end];
        assert!(
            !linear.contains(".value()"),
            "linear() receives Prepared<f64> and must not call .value():\n{linear}"
        );
        assert!(
            !linear.contains("if shape.conditions[0]"),
            "common disjunct in linear() must be factored out of if/else:\n{linear}"
        );
        assert!(
            linear.contains("(shape.conditions[0] && prepared.r == 0.0)"),
            "expected factored linear() condition:\n{linear}"
        );
        assert!(
            code.contains(".embed::<2>("),
            "expected Dual::embed widening in resistor"
        );
    }

    #[test]
    fn mark_visits_noise_exponent_variables() {
        let source = "\
`include \"disciplines.vams\"
module m(a, b);
    inout a, b;
    electrical a, b;
    parameter real p = 1.0;
    real e;
    analog begin
        e = p + 0.5;
        I(a, b) <+ V(a, b) / 100.0;
        I(a, b) <+ flicker_noise(1e-20, e, \"flicker\");
    end
endmodule
";
        let stream = tokens(source, &|_| None).expect("tokens");
        let parsed = parse(&stream).expect("parse");
        let module = resolve(&parsed).expect("resolve");
        let e_index = module
            .variables
            .iter()
            .position(|(name, _)| name == "e")
            .expect("variable e");
        let mut found = BTreeSet::new();
        mark(
            &module.analog,
            &module,
            Mode::Noise,
            &vec![false; module.variables.len()],
            &mut found,
        );
        assert!(
            found.contains(&e_index),
            "mark in Mode::Noise must visit variables in flicker_noise exponent"
        );
    }
}
