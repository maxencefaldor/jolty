//! Compilation and runtime verification for Verilog-A `localparam` constants and
//! static-scaled `ddt(...)` reactive contributions.

#![cfg(test)]

use jolty_models::testing::{Bench, Rig, assert_lift_matches_serde_with_optionals};
use jolty_models::{Differentiable, Model, REFERENCE_TEMPERATURE};
use jolty_veriloga::compile;

/// Compiles `source` as `"test.va"` with no external includes.
fn compile_va(source: &str) -> String {
    compile(source, "test.va", &|_| None)
        .unwrap_or_else(|error| panic!("expected module to compile, got: {error}"))
}

/// Verifies that pure constant `localparam`s fold directly into expressions without emitting a
/// `Constants` struct or exposing fields in `Parameters`.
#[test]
fn constant_folded_localparams_omit_constants_struct_and_parameters() {
    let source = r#"
`include "disciplines.vams"
module folded(a, b);
    inout a, b;
    electrical a, b;
    (* desc = "Fixed weights" *)
    localparam real alpha = 1.5, beta = sqrt(4.0) + ln(exp(1.0));
    localparam integer count = 4;
    localparam untyped = 3;
    parameter real resistance = alpha + beta from (0:inf);

    analog function real gain;
        input x;
        localparam real factor = 0.5;
        real x;
        gain = x * factor;
    endfunction

    analog function real clamp_v;
        input vnew, vold;
        localparam real step_max = 0.5;
        real vnew, vold;
        clamp_v = vnew - vold > step_max ? vold + step_max : vnew;
    endfunction

    real limited;
    analog begin : core
        localparam real inner = alpha + count + untyped;
        limited = $limit(V(a, b), clamp_v);
        I(a, b) <+ gain(limited) * inner / resistance;
    end
endmodule
"#;
    let code = compile_va(source);
    assert!(
        !code.contains("struct Constants"),
        "unexpected Constants:\n{code}"
    );
    assert!(
        code.contains("pub resistance: R,"),
        "missing resistance:\n{code}"
    );
    for hidden in [
        "alpha", "beta", "count", "untyped", "factor", "step_max", "inner",
    ] {
        assert!(
            !code.contains(&format!("pub {hidden}:")),
            "localparam {hidden} leaked into Parameters:\n{code}"
        );
    }
    assert!(
        code.contains("fn clamp_v("),
        "missing clamp_v function:\n{code}"
    );
}

/// Verifies that parameter- and temperature-dependent `localparam`s emit into
/// `struct Constants<R: Real>` in declaration order while omitting unused entries.
#[test]
fn parameter_dependent_localparams_emit_constants_struct_with_derivatives() {
    let source = r#"
`include "disciplines.vams"
module derived(a, b);
    inout a, b;
    electrical a, b;
    parameter real r = 100.0 from (0:inf);
    parameter real opt_p = 1.0 from (0:inf);
    localparam real scale = 2.0 * r;
    localparam real chained = scale + 1.0;
    localparam real temp_gain = $temperature / 300.0;
    localparam integer has_opt = $param_given(opt_p);
    localparam real dead_unused = r * 99.0;

    real g;
    analog begin
        g = (has_opt ? opt_p : 1.0) * temp_gain / chained;
        I(a, b) <+ V(a, b) * g;
    end
endmodule
"#;
    let code = compile_va(source);
    assert!(code.contains("pub r: R,"), "{code}");
    assert!(code.contains("pub opt_p: Option<R>,"), "{code}");
    for hidden in ["scale", "chained", "temp_gain", "has_opt", "dead_unused"] {
        assert!(
            !code.contains(&format!("pub {hidden}:")),
            "localparam {hidden} leaked into Parameters:\n{code}"
        );
    }
    assert!(
        !code.contains("dead_unused"),
        "unused localparam emitted:\n{code}"
    );
    assert!(code.contains("struct Constants<R: Real> {"), "{code}");
    let constants_pos = code.find("struct Constants<R: Real> {").expect("Constants");
    let locals_pos = code.find("struct Locals<R: Real> {").expect("Locals");
    let init_pos = code.find("let mut c = Constants {").expect("c init");
    assert!(
        constants_pos < locals_pos && locals_pos < init_pos,
        "struct definitions must precede statements:\n{code}"
    );
    let scale_pos = code.find("c.scale = parameters.r * 2.0;").expect("c.scale");
    let chained_pos = code.find("c.chained = c.scale + 1.0;").expect("c.chained");
    assert!(scale_pos < chained_pos, "wrong declaration order:\n{code}");
    assert!(code.contains("parameters.opt_p.is_some()"), "{code}");
    assert!(
        code.contains("temperature: R,"),
        "temperature binding lost:\n{code}"
    );
}

/// Verifies that parameter defaults and ranges can reference `localparam`s, keeping folded
/// defaults non-optional and dependent defaults optional (`Option<R>`).
#[test]
fn parameter_defaults_and_ranges_use_localparams() {
    let source = r#"
`include "disciplines.vams"
module ranged(a, b);
    inout a, b;
    electrical a, b;
    localparam real base = 4.0;
    parameter real root = sqrt(base) from (0:inf);
    localparam real doubled = 2.0 * root;
    parameter real width = doubled from (0:upper];
    localparam real upper = doubled * 3.0;

    analog I(a, b) <+ V(a, b) / width;
endmodule
"#;
    let code = compile_va(source);
    assert!(
        code.contains("pub root: R,"),
        "constant-folded default is non-optional:\n{code}"
    );
    assert!(
        code.contains("pub width: Option<R>,"),
        "localparam-dependent default is optional:\n{code}"
    );
    assert!(!code.contains("pub base:"), "{code}");
    assert!(!code.contains("pub doubled:"), "{code}");
    assert!(!code.contains("pub upper:"), "{code}");
    let doubled_pos = code.find("c.doubled = ").expect("c.doubled");
    let upper_pos = code.find("c.upper = ").expect("c.upper");
    let check_pos = code
        .find("c.upper.value()")
        .expect("range check using c.upper");
    assert!(doubled_pos < upper_pos && upper_pos < check_pos, "{code}");
    assert!(
        code.contains("parameters.width.unwrap_or(c.doubled)"),
        "{code}"
    );
}

/// Verifies `localparam` usage in shaped conditionals, `$warning` checks, and `@(initial_step)`
/// blocks without leaking `prepare`-only constants into `warnings`.
#[test]
fn localparams_in_shaped_conditions_warnings_and_initial_events() {
    let source = r#"
`include "disciplines.vams"
module shaped_warn(a, b);
    inout a, b;
    electrical a, b;
    parameter integer mode = 1 from [0:1];
    parameter real base = 10.0 from (0:inf);
    localparam integer active = mode > 0;
    localparam real warn_limit = 2.0 * base;
    localparam real eval_only = 5.0 * base;

    real g;
    analog begin
        @(initial_step) begin
            g = 1.0 / eval_only;
        end
        if (warn_limit > 15.0)
            $warning("limit %g exceeded", warn_limit);
        if (active)
            I(a, b) <+ V(a, b) * g;
    end
endmodule
"#;
    let code = compile_va(source);
    assert!(code.contains("pub fn warnings<R: Real>("), "{code}");
    let warnings_start = code.find("pub fn warnings<R: Real>(").expect("warnings");
    let warnings_fn = &code[warnings_start..];
    assert!(
        warnings_fn.contains("warn_limit"),
        "warnings missing warn_limit:\n{warnings_fn}"
    );
    assert!(
        !warnings_fn.contains("eval_only"),
        "prepare-only localparam leaked into warnings:\n{warnings_fn}"
    );
    assert!(code.contains("conditions: [bool; 1],"), "{code}");
}

/// Verifies that `localparam` names such as `temperature`, `parameters`, or `conditions` never
/// collide with generated bindings or prepared struct fields.
#[test]
fn localparam_names_do_not_collide_with_prepare_bindings_or_prepared_fields() {
    let temp_named = r#"
`include "disciplines.vams"
module temp_name(a, b);
    inout a, b;
    electrical a, b;
    parameter real p = 2.0 from (0:inf);
    localparam real temperature = p * 2.0;
    analog I(a, b) <+ V(a, b) * temperature;
endmodule
"#;
    let code = compile_va(temp_named);
    // `$temperature` is never read, so `prepare` binds `_: R` even though a localparam is named `temperature`.
    assert!(
        code.contains("parameters: &TempNameParameters<R>,\n        _: R,"),
        "expected unused temperature parameter:\n{code}"
    );

    let params_named = r#"
`include "disciplines.vams"
module param_name(a, b);
    inout a, b;
    electrical a, b;
    localparam real parameters = $temperature * 2.0;
    analog I(a, b) <+ V(a, b) / parameters;
endmodule
"#;
    let code = compile_va(params_named);
    // No `parameter` is read, so `prepare` binds `_: &ParamNameParameters`.
    assert!(
        code.contains("_: &ParamNameParameters,\n        temperature: R,"),
        "expected unused parameters binding:\n{code}"
    );

    let field_collision = r#"
`include "disciplines.vams"
module collide(a, b);
    inout a, b;
    electrical a, b;
    parameter integer en = 1 from [0:1];
    parameter real fooBar = 2.0 from (0:inf);
    localparam real foo_bar = fooBar * 3.0;
    localparam real conditions = fooBar * 4.0;
    analog begin
        if (en)
            I(a, b) <+ V(a, b) * fooBar + (V(a, b) * V(a, b)) * foo_bar + ddt(V(a, b) * conditions);
    end
endmodule
"#;
    let code = compile_va(field_collision);
    let prepared_start = code
        .find("pub struct PreparedCollide<R: Real>")
        .expect("PreparedCollide");
    let prepared_end = prepared_start + code[prepared_start..].find("}\n").expect("closing brace");
    let prepared_def = &code[prepared_start..prepared_end];
    assert!(
        prepared_def.contains("conditions: [bool; 1],"),
        "{prepared_def}"
    );
    assert!(prepared_def.contains("conditions2: R,"), "{prepared_def}");
    assert!(prepared_def.contains("foo_bar: R,"), "{prepared_def}");
    assert!(prepared_def.contains("foo_bar2: R,"), "{prepared_def}");
}

/// Verifies that zero-parameter `localparam`-only modules emit non-generic parameter and
/// prepared structs and derive `Default`.
#[test]
fn zero_parameter_localparam_module_emits_non_generic_structs() {
    let fixed_cap = r#"
`include "disciplines.vams"
module fixed_cap(a, b);
    inout a, b;
    electrical a, b;
    localparam real C = 1.0e-12;
    analog I(a, b) <+ C * ddt(V(a, b));
endmodule
"#;
    let code = compile_va(fixed_cap);
    assert!(
        code.contains("#[derive(Debug, Clone, Default, Deserialize, Serialize, JsonSchema, Lift)]"),
        "{code}"
    );
    assert!(code.contains("#[serde(deny_unknown_fields)]"), "{code}");
    assert!(
        !code.contains("#[serde(default, deny_unknown_fields)]"),
        "{code}"
    );
    assert!(code.contains("pub struct FixedCapParameters {}"), "{code}");
    assert!(
        !code.contains("impl Default for FixedCapParameters"),
        "{code}"
    );
    assert!(!code.contains("impl<R: Real> Default"), "{code}");
    assert!(code.contains("pub struct PreparedFixedCap {}"), "{code}");
    assert!(
        code.contains("type Parameters<R: Real> = FixedCapParameters;"),
        "{code}"
    );
    assert!(
        code.contains("type Prepared<R: Real> = PreparedFixedCap;"),
        "{code}"
    );
    assert!(
        code.contains("fn linear(_: &FixedCapShape, _: &PreparedFixedCap) -> bool {\n        true"),
        "{code}"
    );
}

/// Verifies that integer-only modules emit non-generic structs, derive `Default` when
/// zero-initialized, and use `i32::wrapping_*` for literal left operands.
#[test]
fn integer_only_parameter_modules_handle_default_and_wrapping_arithmetic() {
    let zero_int = r#"
`include "disciplines.vams"
module zero_int(a, b);
    inout a, b;
    electrical a, b;
    parameter integer flag = 0 from [0:1];
    localparam integer active = flag > 0;
    analog begin
        if (active)
            I(a, b) <+ V(a, b);
    end
endmodule
"#;
    let code = compile_va(zero_int);
    assert!(
        code.contains("#[derive(Debug, Clone, Default, Deserialize, Serialize, JsonSchema, Lift)]"),
        "{code}"
    );
    assert!(
        code.contains("#[serde(default, deny_unknown_fields)]"),
        "{code}"
    );
    assert!(code.contains("pub struct ZeroIntParameters {"), "{code}");
    assert!(
        !code.contains("impl Default for ZeroIntParameters"),
        "{code}"
    );

    let non_zero_int = r#"
`include "disciplines.vams"
module non_zero_int(a, b);
    inout a, b;
    electrical a, b;
    parameter integer n = 2 from [1:10];
    localparam integer doubled = 2 * n;
    analog I(a, b) <+ V(a, b) * doubled;
endmodule
"#;
    let code = compile_va(non_zero_int);
    assert!(code.contains("pub struct NonZeroIntParameters {"), "{code}");
    assert!(
        code.contains("impl Default for NonZeroIntParameters {"),
        "{code}"
    );
    assert!(code.contains("struct Constants {\n"), "{code}");
    assert!(
        code.contains("i32::wrapping_mul(2, parameters.n)"),
        "{code}"
    );
    assert!(code.contains("pub struct PreparedNonZeroInt {"), "{code}");
}

/// Verifies that a two-terminal MOM capacitor (IHP SG13G2 `cap_cmomf` pattern) and a pure
/// inductor compile as linear kernels with DC-conducting inductor interfaces.
#[test]
fn two_terminal_mom_capacitor_and_pure_inductor_compile_linear() {
    let two_terminal = r#"
`include "disciplines.vams"
`include "constants.vams"
module mom_cap_2t(p, n);
    inout p, n;
    electrical p, n;
    branch (p, n) br_cmain;

    parameter real length = 10.0e-6 from [1.0e-6:inf);
    parameter real width = 5.0e-6 from [1.0e-6:inf);
    parameter integer layers = 3 from [1:5];

    localparam real GLEAK = 1.0e-12;
    localparam real AREACAP_M1 = 0.372;
    localparam real AREACAP_MN = 0.305;

    real c_main, area_um2, n_eff;
    analog begin
        area_um2 = (length * 1.0e6) * (width * 1.0e6);
        n_eff = floor(layers);
        c_main = (AREACAP_M1 + (n_eff - 1.0) * AREACAP_MN) * area_um2 * 1.0e-15 * $mfactor;
        I(br_cmain) <+ c_main * ddt(V(br_cmain));
        I(br_cmain) <+ GLEAK * V(br_cmain);
    end
endmodule
"#;
    let code = compile_va(two_terminal);
    assert!(!code.contains("struct Constants"), "{code}");
    assert!(!code.contains("pub gleak:"), "{code}");
    assert!(
        code.contains(
            "fn linear(_: &MomCap2tShape, _: &PreparedMomCap2t<f64>) -> bool {\n        true\n    }"
        ),
        "expected always-linear 2T MOM cap:\n{code}"
    );

    let pure_inductor = r#"
`include "disciplines.vams"
module pure_inductor(a, b);
    inout a, b;
    electrical a, b;
    parameter real inductance = 1.0e-9 from (0:inf);
    analog V(a, b) <+ inductance * ddt(I(a, b));
endmodule
"#;
    let code = compile_va(pure_inductor);
    assert!(
        code.contains("Branch::new(0, 1, BranchKind::Conductive)"),
        "pure inductor conducts at DC:\n{code}"
    );
    assert!(
        code.contains(
            "fn linear(_: &PureInductorShape, _: &PreparedPureInductor<f64>) -> bool {\n        true\n    }"
        ),
        "{code}"
    );
}

/// Verifies that a three-terminal RF MOM capacitor (IHP SG13G2 `cap_cmomi` pattern) compiles
/// cleanly as a linear kernel.
#[test]
fn three_terminal_rf_mom_capacitor_compiles_linear() {
    let three_terminal_rf = r#"
`include "disciplines.vams"
`include "constants.vams"
module mom_cap_rf(p, n, sub);
    inout p, n, sub;
    electrical p, n, sub;
    electrical n_skin, n_mid, n_ser, n_sub;

    branch (p, n_skin) br_lskin, br_rskin;
    branch (n_skin, n_mid) br_lcore;
    branch (n_mid, n_ser) br_cmain;
    branch (n_ser, n) br_rser;
    branch (p, n_sub) br_cox;
    branch (n_sub, sub) br_rsub, br_csub;
    branch (p, sub) br_lkp;
    branch (n, sub) br_lkm;

    parameter real w = 10.0e-6 from (0:inf);
    parameter real l = 10.0e-6 from (0:inf);
    parameter integer mmin = 1 from [1:3];

    localparam real GLEAK = 1.0e-12;
    localparam real CFEED_SLOPE = 0.1625;
    localparam real CFEED_END = 0.0916;
    localparam real CFEED2_SLOPE = 0.152;

    real c_main, l_skin, l_core, r_skin, r_ser, c_ox, r_sub, c_sub, scale_len;
    analog begin
        scale_len = sqrt(w * l) * `M_TWO_PI * 1.0e6;
        c_main = (CFEED_END + (mmin == 1 ? CFEED_SLOPE : (mmin == 2 ? CFEED2_SLOPE : 0.1)) * scale_len) * 1.0e-15;
        l_skin = (1.0 + ln(1.0 + scale_len)) * 1.0e-11;
        l_core = 2.0e-11;
        r_skin = 50.0;
        r_ser = 0.5;
        c_ox = 5.0e-15;
        r_sub = 1.0e3;
        c_sub = 2.0e-15;

        V(br_lskin) <+ l_skin * ddt(I(br_lskin));
        I(br_rskin) <+ V(br_rskin) / r_skin;
        V(br_lcore) <+ l_core * ddt(I(br_lcore));
        I(br_cmain) <+ c_main * ddt(V(br_cmain));
        I(br_rser) <+ V(br_rser) / r_ser;
        I(br_cox) <+ c_ox * ddt(V(br_cox));
        I(br_rsub) <+ V(br_rsub) / r_sub;
        I(br_csub) <+ c_sub * ddt(V(br_csub));
        I(br_lkp) <+ GLEAK * V(br_lkp);
        I(br_lkm) <+ GLEAK * V(br_lkm);
    end
endmodule
"#;
    let code = compile_va(three_terminal_rf);
    assert!(!code.contains("struct Constants"), "{code}");
    assert!(
        code.contains(
            "fn linear(_: &MomCapRfShape, _: &PreparedMomCapRf<f64>) -> bool {\n        true\n    }"
        ),
        "{code}"
    );
}

/// Inline kernel matching the compiler's output for a parallel RC with a parameter-derived
/// `localparam real scale = 2.0 * c;` and optional `parameter real c_eff = scale;`.
mod localparam_rc {
    use jolty_models::{
        Branch, BranchKind, Dual, Interface, Kernel, Layout, Lift, Local, ParameterError, Real,
        Stamp,
    };
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Copy)]
    pub(super) struct LocalparamRc;

    #[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift)]
    #[serde(default, deny_unknown_fields)]
    #[schemars(bound = "R: Serialize + JsonSchema")]
    pub(super) struct LocalparamRcParameters<R: Real = f64> {
        pub(super) g: R,
        pub(super) c: R,
        pub(super) c_eff: Option<R>,
    }

    impl<R: Real> Default for LocalparamRcParameters<R> {
        fn default() -> Self {
            Self {
                g: R::constant(1e-3),
                c: R::constant(1e-12),
                c_eff: None,
            }
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub(super) struct PreparedLocalparamRc<R: Real> {
        g: R,
        c_eff: R,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(super) struct LocalparamRcShape {
        a_by_a: usize,
        a_by_b: usize,
        b_by_a: usize,
        b_by_b: usize,
    }

    impl Kernel for LocalparamRc {
        const NAME: &'static str = "localparam_rc";
        type Parameters<R: Real> = LocalparamRcParameters<R>;
        type Shape = LocalparamRcShape;
        type Prepared<R: Real> = PreparedLocalparamRc<R>;
        type Memory = ();

        fn interface<R: Real>(_: &LocalparamRcParameters<R>) -> Interface {
            Interface::new(&["a", "b"], &[Branch::new(0, 1, BranchKind::Conductive)])
        }

        fn linear(_: &LocalparamRcShape, _: &PreparedLocalparamRc<f64>) -> bool {
            true
        }

        fn prepare<R: Real>(
            parameters: &LocalparamRcParameters<R>,
            _: R,
        ) -> Result<PreparedLocalparamRc<R>, ParameterError> {
            struct Constants<R: Real> {
                scale: R,
            }
            let mut c = Constants {
                scale: R::constant(0.0),
            };
            c.scale = parameters.c * 2.0;
            ParameterError::positive("g", parameters.g)?;
            ParameterError::positive("c", parameters.c)?;
            ParameterError::positive("c_eff", parameters.c_eff.unwrap_or(c.scale))?;
            Ok(PreparedLocalparamRc {
                g: parameters.g,
                c_eff: parameters.c_eff.unwrap_or(c.scale),
            })
        }

        fn layout<R: Real>(
            _: &LocalparamRcParameters<R>,
            _: &PreparedLocalparamRc<R>,
            layout: &mut Layout,
        ) -> Result<LocalparamRcShape, ParameterError> {
            Ok(LocalparamRcShape {
                a_by_a: layout.entry(0, 0),
                a_by_b: layout.entry(0, 1),
                b_by_a: layout.entry(1, 0),
                b_by_b: layout.entry(1, 1),
            })
        }

        fn evaluate<R: Real>(
            shape: &LocalparamRcShape,
            prepared: &PreparedLocalparamRc<R>,
            (): &mut (),
            at: &Local<'_>,
            stamp: &mut Stamp<'_, R>,
        ) {
            let voltage = Dual {
                value: at.unknown::<R>(0) - at.unknown::<R>(1),
                gradient: [R::constant(1.0)],
            };
            let current = voltage.scale(prepared.g);
            let charge = voltage.scale(prepared.c_eff);
            stamp.add_residual(0, current.value);
            stamp.add_residual(1, -current.value);
            let [conductance] = current.gradient;
            stamp.add_conductance(shape.a_by_a, conductance);
            stamp.add_conductance(shape.a_by_b, -conductance);
            stamp.add_conductance(shape.b_by_a, -conductance);
            stamp.add_conductance(shape.b_by_b, conductance);
            stamp.add_charge(0, charge.value);
            stamp.add_charge(1, -charge.value);
            let [capacitance] = charge.gradient;
            stamp.add_capacitance(shape.a_by_a, capacitance);
            stamp.add_capacitance(shape.a_by_b, -capacitance);
            stamp.add_capacitance(shape.b_by_a, -capacitance);
            stamp.add_capacitance(shape.b_by_b, capacitance);
        }
    }
}

/// Inline kernel matching the compiler's non-generic output for a zero-parameter
/// `localparam`-only capacitor.
mod fixed_cap_kernel {
    use jolty_models::{
        Branch, BranchKind, Dual, Interface, Kernel, Layout, Lift, Local, ParameterError, Real,
        Stamp,
    };
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Copy)]
    pub(super) struct FixedCap;

    #[derive(Debug, Clone, Default, Deserialize, Serialize, JsonSchema, Lift)]
    #[serde(deny_unknown_fields)]
    pub(super) struct FixedCapParameters {}

    #[derive(Debug, Clone, Copy)]
    pub(super) struct PreparedFixedCap {}

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(super) struct FixedCapShape {
        a_by_a: usize,
        a_by_b: usize,
        b_by_a: usize,
        b_by_b: usize,
    }

    impl Kernel for FixedCap {
        const NAME: &'static str = "fixed_cap";
        type Parameters<R: Real> = FixedCapParameters;
        type Shape = FixedCapShape;
        type Prepared<R: Real> = PreparedFixedCap;
        type Memory = ();

        fn interface<R: Real>(_: &FixedCapParameters) -> Interface {
            Interface::new(&["a", "b"], &[Branch::new(0, 1, BranchKind::Capacitive)])
        }

        fn linear(_: &FixedCapShape, _: &PreparedFixedCap) -> bool {
            true
        }

        fn prepare<R: Real>(
            _: &FixedCapParameters,
            _: R,
        ) -> Result<PreparedFixedCap, ParameterError> {
            Ok(PreparedFixedCap {})
        }

        fn layout<R: Real>(
            _: &FixedCapParameters,
            _: &PreparedFixedCap,
            layout: &mut Layout,
        ) -> Result<FixedCapShape, ParameterError> {
            Ok(FixedCapShape {
                a_by_a: layout.entry(0, 0),
                a_by_b: layout.entry(0, 1),
                b_by_a: layout.entry(1, 0),
                b_by_b: layout.entry(1, 1),
            })
        }

        fn evaluate<R: Real>(
            shape: &FixedCapShape,
            _: &PreparedFixedCap,
            (): &mut (),
            at: &Local<'_>,
            stamp: &mut Stamp<'_, R>,
        ) {
            let voltage = Dual {
                value: at.unknown::<R>(0) - at.unknown::<R>(1),
                gradient: [R::constant(1.0)],
            };
            let charge = 1.0e-12 * voltage;
            stamp.add_charge(0, charge.value);
            stamp.add_charge(1, -charge.value);
            let [capacitance] = charge.gradient;
            stamp.add_capacitance(shape.a_by_a, capacitance);
            stamp.add_capacitance(shape.a_by_b, -capacitance);
            stamp.add_capacitance(shape.b_by_a, -capacitance);
            stamp.add_capacitance(shape.b_by_b, capacitance);
        }
    }
}

/// Verifies that exact derivatives propagate through prepared `localparam`s while schema,
/// `serde`, and `Lift` exclude `localparam` names.
#[test]
fn runtime_derivatives_schema_and_lift_through_localparam() {
    let source = r#"
`include "disciplines.vams"
module localparam_rc(a, b);
    inout a, b;
    electrical a, b;
    parameter real g = 1e-3 from (0:inf);
    parameter real c = 1e-12 from (0:inf);
    localparam real scale = 2.0 * c;
    parameter real c_eff = scale from (0:inf);
    analog begin
        I(a, b) <+ g * V(a, b) + c_eff * ddt(V(a, b));
    end
endmodule
"#;
    let compiled = compile_va(source);
    assert!(
        compiled.contains("struct Constants<R: Real> {"),
        "{compiled}"
    );
    assert!(
        compiled.contains("c.scale = parameters.c * 2.0;"),
        "{compiled}"
    );
    assert!(
        compiled.contains("parameters.c_eff.unwrap_or(c.scale)"),
        "{compiled}"
    );
    assert!(!compiled.contains("pub scale:"), "{compiled}");

    // Serde rejects `localparam` names because of `deny_unknown_fields`.
    assert!(
        serde_json::from_str::<localparam_rc::LocalparamRcParameters<f64>>(r#"{"scale": 1e-12}"#)
            .is_err()
    );

    let model = Differentiable::<localparam_rc::LocalparamRc>::new();
    let schema = model.schema().to_value();
    let properties = schema["properties"]
        .as_object()
        .expect("schema properties object");
    assert!(properties.contains_key("g"));
    assert!(properties.contains_key("c"));
    assert!(properties.contains_key("c_eff"));
    assert!(!properties.contains_key("scale"));

    // Default `c_eff` inherits `localparam scale = 2.0 * c`.
    let mut bench = Bench::at::<localparam_rc::LocalparamRc>(
        r#"{"g": 2e-3, "c": 3e-12}"#,
        REFERENCE_TEMPERATURE,
    );
    let evaluation = bench.evaluate(&[0.5, 0.0]);
    assert!((evaluation.residual[0] - 1e-3).abs() < 1e-15);
    assert!((evaluation.charge[0] - 3e-12).abs() < 1e-24);
    assert!((evaluation.capacitance[0][0] - 6e-12).abs() < 1e-24);

    // Exact forward-mode derivative dQ/dc propagates through `c.scale = parameters.c * 2.0`.
    let rig = Rig { model: &model };
    let base = serde_json::json!({"g": 2e-3, "c": 3e-12});
    let object = base.as_object().cloned().expect("object");
    let at = [0.5, 0.1];
    let exact_c = rig.exact(&object, "c", (&at, Some(0.0))).expect("twin");
    let [dq_dc] = exact_c[1][0].derivatives;
    // Q = (2 * c) * (0.5 - 0.1) = 0.8 * c => dQ/dc = 0.8.
    assert!(
        (dq_dc - 0.8).abs() < 1e-12,
        "expected dQ/dc = 0.8, got {dq_dc}"
    );

    // When `c_eff` is explicitly overridden, dQ/dc is zero and dQ/dc_eff = V(a, b) = 0.4.
    let overridden = serde_json::json!({"g": 2e-3, "c": 3e-12, "c_eff": 5e-12});
    let overridden_object = overridden.as_object().cloned().expect("object");
    let exact_override_c = rig
        .exact(&overridden_object, "c", (&at, Some(0.0)))
        .expect("twin");
    assert_eq!(exact_override_c[1][0].derivatives, [0.0]);
    let exact_override_ceff = rig
        .exact(&overridden_object, "c_eff", (&at, Some(0.0)))
        .expect("twin");
    let [dq_dceff] = exact_override_ceff[1][0].derivatives;
    assert!(
        (dq_dceff - 0.4).abs() < 1e-12,
        "expected dQ/dc_eff = 0.4, got {dq_dceff}"
    );

    assert_eq!(
        assert_lift_matches_serde_with_optionals(&model, "{}", &[]),
        1
    );
    assert_eq!(
        assert_lift_matches_serde_with_optionals(
            &model,
            r#"{"g": 2e-3, "c": 3e-12, "c_eff": 5e-12}"#,
            &[],
        ),
        0
    );
}

/// Verifies runtime evaluation, `serde`, and `Lift` behavior for a non-generic zero-parameter
/// `localparam`-only kernel.
#[test]
fn runtime_non_generic_zero_parameter_kernel() {
    assert!(serde_json::from_str::<fixed_cap_kernel::FixedCapParameters>("{}").is_ok());
    assert!(
        serde_json::from_str::<fixed_cap_kernel::FixedCapParameters>(r#"{"C": 1e-12}"#).is_err()
    );
    let mut bench = Bench::at::<fixed_cap_kernel::FixedCap>("{}", REFERENCE_TEMPERATURE);
    let evaluation = bench.evaluate(&[2.0, 0.5]);
    assert!((evaluation.charge[0] - 1.5e-12).abs() < 1e-24);
    assert!((evaluation.capacitance[0][0] - 1.0e-12).abs() < 1e-24);

    let model = Differentiable::<fixed_cap_kernel::FixedCap>::new();
    assert_eq!(
        assert_lift_matches_serde_with_optionals(&model, "{}", &[]),
        0
    );
}

/// Verifies that nested ternary chains flatten into `else if` chains without redundant `let`
/// bindings.
#[test]
fn nested_ternary_chains_flatten_to_else_if_without_let_and_return() {
    let source = r#"
`include "disciplines.vams"
module tiered_cap(a, b);
    inout a, b;
    electrical a, b;
    parameter integer nlay = 3 from [1:6];
    real c;
    analog begin
        c = (nlay <= 2) ? 0.55e-15 : ((nlay == 3) ? 0.82e-15 : ((nlay == 4) ? 1.09e-15 : 1.36e-15));
        I(a, b) <+ c * ddt(V(a, b));
    end
endmodule
"#;
    let compiled = compile_va(source);
    assert!(
        compiled.contains("} else if parameters.nlay == 3 {"),
        "{compiled}"
    );
    assert!(
        !compiled.contains("let t0000 = if") && !compiled.contains("let t0001 = if"),
        "expected flattened else-if chain without nested let-and-return:\n{compiled}"
    );
}
