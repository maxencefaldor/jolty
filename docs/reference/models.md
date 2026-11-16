# Models

Each model a device can be, as the engine defines it, and every parameter a device of it takes: its default, which a device that leaves it out takes, and its description, both from the engine's own parameter types.
A parameter within another is named by its path, its keys joined by dots and `*` standing for each entry of a list, such as `waveform.pulse.rise` or `windings.*.inductance`; of the parameters of one shape, such as `waveform.dc` and `waveform.pulse`, a device gives one.
Values are in SI base units, as the circuit format holds them, while the functions of `jolty.devices` that place each model take [quantities with units](../guide/units.md).

## `behavioral_current_source`

| Parameter | Default | Description |
| --- | --- | --- |
| `expression` | required | What the source's output follows: an expression of the voltages of the terminals it reads, of the currents through sensing ports of its own, and of time. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |

## `behavioral_voltage_source`

| Parameter | Default | Description |
| --- | --- | --- |
| `expression` | required | What the source's output follows: an expression of the voltages of the terminals it reads, of the currents through sensing ports of its own, and of time. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |

## `bipolar_transistor`

| Parameter | Default | Description |
| --- | --- | --- |
| `base_collector_capacitance` | `0` | The zero-bias base-collector depletion capacitance, `CJC`, in farads; default 0. |
| `base_collector_grading` | `0.33` | The base-collector grading coefficient, `MJC`, below 1; default 0.33. |
| `base_collector_leakage_current` | `0` | The base-collector leakage saturation current, `ISC`, in amperes; default 0. |
| `base_collector_leakage_emission_coefficient` | `2` | The base-collector leakage emission coefficient, `NC`; default 2. |
| `base_collector_potential` | `0.75` | The base-collector built-in potential, `VJC`, in volts; default 0.75. |
| `base_emitter_capacitance` | `0` | The zero-bias base-emitter depletion capacitance, `CJE`, in farads; default 0. |
| `base_emitter_grading` | `0.33` | The base-emitter grading coefficient, `MJE`, below 1; default 0.33. |
| `base_emitter_leakage_current` | `0` | The base-emitter leakage saturation current, `ISE`, in amperes; default 0. |
| `base_emitter_leakage_emission_coefficient` | `1.5` | The base-emitter leakage emission coefficient, `NE`; default 1.5. |
| `base_emitter_potential` | `0.75` | The base-emitter built-in potential, `VJE`, in volts; default 0.75. |
| `base_resistance` | `0` | The zero-bias base resistance, `RB`, in ohms; default 0. |
| `base_resistance_current` |  | The base current, `IRB`, at which the base resistance falls halfway to its minimum, in amperes; without it, the base resistance falls with the base charge. |
| `beta_temperature_exponent` | `0` | The exponent of the temperature ratio that scales the betas, `XTB`; default 0. |
| `collector_resistance` | `0` | The collector resistance, `RC`, in ohms; default 0. |
| `depletion_coefficient` | `0.5` | The fraction of each junction's potential above which its depletion capacitance is extended linearly, `FC`, below 1; default 0.5. |
| `emitter_resistance` | `0` | The emitter resistance, `RE`, in ohms; default 0. |
| `energy_gap` | `1.11` | The energy gap that scales the saturation currents with temperature, `EG`, in volts; default 1.11, silicon's. |
| `excess_phase` | `0` | The excess phase at the frequency `1 / (2π · TF)`, `PTF`, in degrees, which delays the forward transport current by `PTF · π / 180 · TF`; default 0. |
| `flicker_coefficient` | `0` | The flicker noise coefficient of the base current, `KF`; default 0, no flicker noise. |
| `flicker_exponent` | `1` | The power of the base current in the flicker noise, `AF`; default 1. |
| `forward_beta` | `100` | The ideal maximum forward current gain, `BF`; default 100. |
| `forward_early_voltage` |  | The forward Early voltage, `VAF`, in volts; infinite by default. |
| `forward_emission_coefficient` | `1` | The forward emission coefficient, `NF`; default 1. |
| `forward_knee_current` |  | The forward knee current, `IKF`, where high injection sets in, in amperes; infinite by default. |
| `forward_transit_time` | `0` | The ideal forward transit time, `TF`, in seconds; default 0. |
| `internal_base_fraction` | `1` | The fraction of the base-collector capacitance on the internal base, `XCJC`, the rest on the external base; default 1. |
| `min_base_resistance` |  | The base resistance at high current, `RBM`, in ohms; the base resistance by default. |
| `nominal_temperature` | `300.15` | The temperature the parameters were measured at, in kelvins; default 27 °C. |
| `polarity` | `npn` | The doping order; default `npn`. One of `npn`, `pnp`. |
| `reverse_beta` | `1` | The ideal maximum reverse current gain, `BR`; default 1. |
| `reverse_early_voltage` |  | The reverse Early voltage, `VAR`, in volts; infinite by default. |
| `reverse_emission_coefficient` | `1` | The reverse emission coefficient, `NR`; default 1. |
| `reverse_knee_current` |  | The reverse knee current, `IKR`, in amperes; infinite by default. |
| `reverse_transit_time` | `0` | The ideal reverse transit time, `TR`, in seconds; default 0. |
| `saturation_current` | `1e-16` | The transport saturation current, `IS`, in amperes; default 1e-16. |
| `saturation_current_temperature_exponent` | `3` | The exponent of the temperature ratio in the saturation currents' scaling, `XTI`; default 3. |
| `substrate` |  | How the transistor is built, if it has a substrate, as ngspice's `SUBS` of 1 and −1 builds it; without it, the transistor has none. The parameters below, of the junction to the substrate, need one. One of `vertical`, `lateral`. |
| `substrate_capacitance` |  | The substrate junction's zero-bias depletion capacitance, `CJS`, in farads; default 0. |
| `substrate_emission_coefficient` |  | The substrate junction's emission coefficient, `NS`; default 1. |
| `substrate_grading` |  | The substrate junction's grading coefficient, `MJS`, below 1; default 0. |
| `substrate_potential` |  | The substrate junction's built-in potential, `VJS`, in volts; default 0.75. |
| `substrate_saturation_current` |  | The substrate junction's saturation current, `ISS`, in amperes; without it, the junction carries the minimum conductance only. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |
| `thermal` | `false` | Whether the device heats by the power it dissipates, through a terminal `thermal` joined to a thermal net, whose temperature above the ambient it runs at. |
| `transit_time_bias_coefficient` | `0` | How much the forward transit time grows with bias, `XTF`; default 0. |
| `transit_time_current` | `0` | The current, `ITF`, above which the forward transit time grows, in amperes; default 0. |
| `transit_time_voltage` |  | The base-collector voltage, `VTF`, that sets how the forward transit time grows with it, in volts; infinite by default. |

## `bsim3`

| Parameter | Default | Description |
| --- | --- | --- |
| `a0` | `1` | The bulk charge effect of a short channel, `a0`; default 1. |
| `a1` | `0` | The first coefficient of the non-saturation effect, `a1`, per volt; default 0. |
| `a2` | `1` | The second coefficient of the non-saturation effect, `a2`; default 1. |
| `acde` | `1` | The exponential coefficient of the charge layer's thickness, `capmod` 3's, `acde`, in meters per volt; default 1. |
| `af` | `1` | The flicker noise's current exponent, `noimod` 1, 4 and 5's, `af`; default 1. |
| `ags` | `0` | The gate bias's effect on the bulk charge, `ags`, per volt; default 0. |
| `alpha0` | `0` | The first coefficient of the impact ionization's substrate current, `alpha0`, in meters per volt; default 0. |
| `alpha1` | `0` | The length scaling of the impact ionization's substrate current, `alpha1`, per volt; default 0. |
| `at` | `33000` | The temperature coefficient of `vsat`, `at`, in meters per second; default 33000. |
| `b0` | `0` | The narrow-width effect on the bulk charge, `b0`, in meters; default 0. |
| `b1` | `0` | The width offset of the narrow-width effect on the bulk charge, `b1`, in meters; default 0. |
| `beta0` | `30` | The second coefficient of the impact ionization's substrate current, `beta0`, in volts; default 30. |
| `capacitance_model` | `3` | The charge model, `capmod`, 0 to 3; default 3. |
| `cdsc` | `0.00024` | The coupling capacitance between the channel and the source and drain, `cdsc`, in farads per square meter; default 0.00024. |
| `cdscb` | `0` | The body bias's effect on `cdsc`, `cdscb`, in farads per volt-square-meter; default 0. |
| `cdscd` | `0` | The drain bias's effect on `cdsc`, `cdscd`, in farads per volt-square-meter; default 0. |
| `cf` |  | The fringing capacitance per width, `cf`, in farads per meter; default 2 · εox / π · ln(1 + 0.4 µm / `tox`). |
| `cgbo` |  | The gate-bulk overlap capacitance per length, `cgbo`, in farads per meter; default 2 · `dwc` · Cox. |
| `cgdl` | `0` | The bias-dependent overlap capacitance between the gate and the lightly doped drain, `cgdl`, in farads per meter; default 0. |
| `cgdo` |  | The gate-drain overlap capacitance per width, beyond the bias-dependent one, `cgdo`, in farads per meter; default `dlc` · Cox − `cgdl` with a positive `dlc`, else 0.6 · `xj` · Cox. |
| `cgsl` | `0` | The bias-dependent overlap capacitance between the gate and the lightly doped source, `cgsl`, in farads per meter; default 0. |
| `cgso` |  | The gate-source overlap capacitance per width, beyond the bias-dependent one, `cgso`, in farads per meter; default `dlc` · Cox − `cgsl` with a positive `dlc`, else 0.6 · `xj` · Cox. |
| `channel` | `n` | The channel; default `n`. One of `n`, `p`. |
| `charge_partition` | `0` | The share of the channel's charge the drain takes in saturation, `xpart`: 0.4 below 0.5, 0.5 at 0.5 and none above it, and no channel charge at all below 0; default 0. |
| `check_parameters` | `false` | Whether the size-dependent parameters are checked and clamped where they lie out of range, `paramchk = 1`; default false. |
| `cit` | `0` | The interface traps' capacitance, `cit`, in farads per square meter; default 0. |
| `cj` | `0.0005` | The junctions' zero-bias bottom capacitance per area, `cj`, in farads per square meter; default 0.0005. |
| `cjsw` | `5e-10` | The junctions' zero-bias sidewall capacitance per length along the field oxide, `cjsw`, in farads per meter; default 5e-10. |
| `cjswg` |  | The junctions' zero-bias sidewall capacitance per length along the gate, `cjswg`, in farads per meter; default `cjsw`. |
| `ckappa` | `0.6` | The coefficient of the bias-dependent overlap capacitances, `ckappa`, in volts; default 0.6. |
| `clc` | `1e-07` | The first coefficient of the short-channel effect on the charges, `clc`, in meters; default 1e-7. |
| `cle` | `0.6` | The second coefficient of the short-channel effect on the charges, `cle`; default 0.6. |
| `delta` | `0.01` | The smoothing of the effective drain-source voltage at saturation, `delta`, in volts; default 0.01. |
| `dlc` |  | The length each side of the channel loses in the charges, `dlc`, in meters; default `lint`. |
| `drain_area` | `0` | The drain diffusion's area, `ad`, in square meters; default 0. |
| `drain_perimeter` | `0` | The drain diffusion's perimeter, `pd`, in meters; default 0. |
| `drain_squares` | `1` | The drain diffusion's length in squares, `nrd`; default 1. |
| `drout` | `0.56` | The length dependence of the drain-induced barrier lowering on the output resistance, `drout`; default 0.56. |
| `dsub` |  | The drain-induced barrier lowering's coefficient below the threshold, `dsub`; default `drout`. |
| `dvt0` | `2.2` | The first coefficient of the short-channel effect, `dvt0`; default 2.2. |
| `dvt0w` | `0` | The first coefficient of the narrow-width effect on a short channel, `dvt0w`; default 0. |
| `dvt1` | `0.53` | The second coefficient of the short-channel effect, `dvt1`; default 0.53. |
| `dvt1w` | `5300000` | The second coefficient of the narrow-width effect on a short channel, `dvt1w`, per meter; default 5300000. |
| `dvt2` | `-0.032` | The body effect on the short-channel effect, `dvt2`, per volt; default -0.032. |
| `dvt2w` | `-0.032` | The body effect on the narrow-width effect on a short channel, `dvt2w`, per volt; default -0.032. |
| `dwb` | `0` | The body bias's narrowing of the channel, `dwb`, in meters per square-root volt; default 0. |
| `dwc` |  | The width each side of the channel loses in the charges, `dwc`, in meters; default `wint`. |
| `dwg` | `0` | The gate bias's narrowing of the channel, `dwg`, in meters per volt; default 0. |
| `ef` | `1` | The flicker noise's frequency exponent, `ef`; default 1. |
| `em` | `41000000` | The saturation field in the flicker noise, `em`, in volts per meter; default 41000000. |
| `eta0` | `0.08` | The drain-induced barrier lowering below the threshold, `eta0`; default 0.08. |
| `etab` | `-0.07` | The body effect on the drain-induced barrier lowering below the threshold, `etab`, per volt; default -0.07. |
| `gamma1` |  | The body effect near the surface, `gamma1`, in square-root volts; default derived from `nch`. |
| `gamma2` |  | The body effect in the bulk, `gamma2`, in square-root volts; default derived from `nsub`. |
| `ijth` | `0.1` | The junction current above which the diodes' exponential continues linearly, zero for none, `ijth`, in amperes; default 0.1. |
| `js` | `0.0001` | The junctions' saturation current per area, `js`, in amperes per square meter; default 0.0001. |
| `jsw` | `0` | The junctions' saturation current per length of perimeter, `jsw`, in amperes per meter; default 0. |
| `k1` |  | The first-order body effect, `k1`, in square-root volts; default derived from `gamma1` and `gamma2`, or 0.53 √V where only `k2` is given. |
| `k2` |  | The second-order body effect, `k2`; default derived from `gamma1` and `gamma2`, or -0.0186 where only `k1` is given. |
| `k3` | `80` | The narrow-width coefficient, `k3`; default 80. |
| `k3b` | `0` | The body effect on `k3`, `k3b`, per volt; default 0. |
| `keta` | `-0.047` | The body effect on the bulk charge, `keta`, per volt; default -0.047. |
| `kf` | `0` | The flicker noise's coefficient, `noimod` 1, 4 and 5's, `kf`; default 0. |
| `kt1` | `-0.11` | The threshold's temperature coefficient, `kt1`, in volts; default -0.11. |
| `kt1l` | `0` | The channel length's effect on `kt1`, `kt1l`, in volt-meters; default 0. |
| `kt2` | `0.022` | The body effect on `kt1`, `kt2`; default 0.022. |
| `la0` | `0` | The length dependence of `a0`, `la0`: its coefficient of the reciprocal of the effective length, in the unit of `a0` times meters; default 0. |
| `la1` | `0` | The length dependence of `a1`, `la1`: its coefficient of the reciprocal of the effective length, in the unit of `a1` times meters; default 0. |
| `la2` | `0` | The length dependence of `a2`, `la2`: its coefficient of the reciprocal of the effective length, in the unit of `a2` times meters; default 0. |
| `lacde` | `0` | The length dependence of `acde`, `lacde`: its coefficient of the reciprocal of the effective length, in the unit of `acde` times meters; default 0. |
| `lags` | `0` | The length dependence of `ags`, `lags`: its coefficient of the reciprocal of the effective length, in the unit of `ags` times meters; default 0. |
| `lalpha0` | `0` | The length dependence of `alpha0`, `lalpha0`: its coefficient of the reciprocal of the effective length, in the unit of `alpha0` times meters; default 0. |
| `lalpha1` | `0` | The length dependence of `alpha1`, `lalpha1`: its coefficient of the reciprocal of the effective length, in the unit of `alpha1` times meters; default 0. |
| `lat` | `0` | The length dependence of `at`, `lat`: its coefficient of the reciprocal of the effective length, in the unit of `at` times meters; default 0. |
| `lb0` | `0` | The length dependence of `b0`, `lb0`: its coefficient of the reciprocal of the effective length, in the unit of `b0` times meters; default 0. |
| `lb1` | `0` | The length dependence of `b1`, `lb1`: its coefficient of the reciprocal of the effective length, in the unit of `b1` times meters; default 0. |
| `lbeta0` | `0` | The length dependence of `beta0`, `lbeta0`: its coefficient of the reciprocal of the effective length, in the unit of `beta0` times meters; default 0. |
| `lcdsc` | `0` | The length dependence of `cdsc`, `lcdsc`: its coefficient of the reciprocal of the effective length, in the unit of `cdsc` times meters; default 0. |
| `lcdscb` | `0` | The length dependence of `cdscb`, `lcdscb`: its coefficient of the reciprocal of the effective length, in the unit of `cdscb` times meters; default 0. |
| `lcdscd` | `0` | The length dependence of `cdscd`, `lcdscd`: its coefficient of the reciprocal of the effective length, in the unit of `cdscd` times meters; default 0. |
| `lcf` | `0` | The length dependence of `cf`, `lcf`: its coefficient of the reciprocal of the effective length, in the unit of `cf` times meters; default 0. |
| `lcgdl` | `0` | The length dependence of `cgdl`, `lcgdl`: its coefficient of the reciprocal of the effective length, in the unit of `cgdl` times meters; default 0. |
| `lcgsl` | `0` | The length dependence of `cgsl`, `lcgsl`: its coefficient of the reciprocal of the effective length, in the unit of `cgsl` times meters; default 0. |
| `lcit` | `0` | The length dependence of `cit`, `lcit`: its coefficient of the reciprocal of the effective length, in the unit of `cit` times meters; default 0. |
| `lckappa` | `0` | The length dependence of `ckappa`, `lckappa`: its coefficient of the reciprocal of the effective length, in the unit of `ckappa` times meters; default 0. |
| `lclc` | `0` | The length dependence of `clc`, `lclc`: its coefficient of the reciprocal of the effective length, in the unit of `clc` times meters; default 0. |
| `lcle` | `0` | The length dependence of `cle`, `lcle`: its coefficient of the reciprocal of the effective length, in the unit of `cle` times meters; default 0. |
| `ldelta` | `0` | The length dependence of `delta`, `ldelta`: its coefficient of the reciprocal of the effective length, in the unit of `delta` times meters; default 0. |
| `ldrout` | `0` | The length dependence of `drout`, `ldrout`: its coefficient of the reciprocal of the effective length, in the unit of `drout` times meters; default 0. |
| `ldsub` | `0` | The length dependence of `dsub`, `ldsub`: its coefficient of the reciprocal of the effective length, in the unit of `dsub` times meters; default 0. |
| `ldvt0` | `0` | The length dependence of `dvt0`, `ldvt0`: its coefficient of the reciprocal of the effective length, in the unit of `dvt0` times meters; default 0. |
| `ldvt0w` | `0` | The length dependence of `dvt0w`, `ldvt0w`: its coefficient of the reciprocal of the effective length, in the unit of `dvt0w` times meters; default 0. |
| `ldvt1` | `0` | The length dependence of `dvt1`, `ldvt1`: its coefficient of the reciprocal of the effective length, in the unit of `dvt1` times meters; default 0. |
| `ldvt1w` | `0` | The length dependence of `dvt1w`, `ldvt1w`: its coefficient of the reciprocal of the effective length, in the unit of `dvt1w` times meters; default 0. |
| `ldvt2` | `0` | The length dependence of `dvt2`, `ldvt2`: its coefficient of the reciprocal of the effective length, in the unit of `dvt2` times meters; default 0. |
| `ldvt2w` | `0` | The length dependence of `dvt2w`, `ldvt2w`: its coefficient of the reciprocal of the effective length, in the unit of `dvt2w` times meters; default 0. |
| `ldwb` | `0` | The length dependence of `dwb`, `ldwb`: its coefficient of the reciprocal of the effective length, in the unit of `dwb` times meters; default 0. |
| `ldwg` | `0` | The length dependence of `dwg`, `ldwg`: its coefficient of the reciprocal of the effective length, in the unit of `dwg` times meters; default 0. |
| `length` | `5e-06` | The channel's drawn length, `l`, in meters; default 5 µm. |
| `leta0` | `0` | The length dependence of `eta0`, `leta0`: its coefficient of the reciprocal of the effective length, in the unit of `eta0` times meters; default 0. |
| `letab` | `0` | The length dependence of `etab`, `letab`: its coefficient of the reciprocal of the effective length, in the unit of `etab` times meters; default 0. |
| `lgamma1` | `0` | The length dependence of `gamma1`, `lgamma1`: its coefficient of the reciprocal of the effective length, in the unit of `gamma1` times meters; default 0. |
| `lgamma2` | `0` | The length dependence of `gamma2`, `lgamma2`: its coefficient of the reciprocal of the effective length, in the unit of `gamma2` times meters; default 0. |
| `lint` | `0` | The length each side of the channel loses, which shortens it, `lint`, in meters; default 0. |
| `lintnoi` | `0` | The length each side of the channel loses in the flicker noise, beyond `lint`, `lintnoi`, in meters; default 0. |
| `lk1` | `0` | The length dependence of `k1`, `lk1`: its coefficient of the reciprocal of the effective length, in the unit of `k1` times meters; default 0. |
| `lk2` | `0` | The length dependence of `k2`, `lk2`: its coefficient of the reciprocal of the effective length, in the unit of `k2` times meters; default 0. |
| `lk3` | `0` | The length dependence of `k3`, `lk3`: its coefficient of the reciprocal of the effective length, in the unit of `k3` times meters; default 0. |
| `lk3b` | `0` | The length dependence of `k3b`, `lk3b`: its coefficient of the reciprocal of the effective length, in the unit of `k3b` times meters; default 0. |
| `lketa` | `0` | The length dependence of `keta`, `lketa`: its coefficient of the reciprocal of the effective length, in the unit of `keta` times meters; default 0. |
| `lkt1` | `0` | The length dependence of `kt1`, `lkt1`: its coefficient of the reciprocal of the effective length, in the unit of `kt1` times meters; default 0. |
| `lkt1l` | `0` | The length dependence of `kt1l`, `lkt1l`: its coefficient of the reciprocal of the effective length, in the unit of `kt1l` times meters; default 0. |
| `lkt2` | `0` | The length dependence of `kt2`, `lkt2`: its coefficient of the reciprocal of the effective length, in the unit of `kt2` times meters; default 0. |
| `ll` | `0` | The length dependence of the length lost, `ll`, in meters to the power 1 + `lln`; default 0. |
| `llc` |  | The length dependence of the length the charges lose, `llc`, in meters to the power 1 + `lln`; default `ll`. |
| `lln` | `1` | The length exponent of `ll`, `lln`; default 1. |
| `lmoin` | `0` | The length dependence of `moin`, `lmoin`: its coefficient of the reciprocal of the effective length, in the unit of `moin` times meters; default 0. |
| `lnch` | `0` | The length dependence of `nch`, `lnch`: its coefficient of the reciprocal of the effective length, in the unit of `nch` times meters; default 0. |
| `lnfactor` | `0` | The length dependence of `nfactor`, `lnfactor`: its coefficient of the reciprocal of the effective length, in the unit of `nfactor` times meters; default 0. |
| `lngate` | `0` | The length dependence of `ngate`, `lngate`: its coefficient of the reciprocal of the effective length, in the unit of `ngate` times meters; default 0. |
| `lnlx` | `0` | The length dependence of `nlx`, `lnlx`: its coefficient of the reciprocal of the effective length, in the unit of `nlx` times meters; default 0. |
| `lnoff` | `0` | The length dependence of `noff`, `lnoff`: its coefficient of the reciprocal of the effective length, in the unit of `noff` times meters; default 0. |
| `lnsub` | `0` | The length dependence of `nsub`, `lnsub`: its coefficient of the reciprocal of the effective length, in the unit of `nsub` times meters; default 0. |
| `lpclm` | `0` | The length dependence of `pclm`, `lpclm`: its coefficient of the reciprocal of the effective length, in the unit of `pclm` times meters; default 0. |
| `lpdiblc1` | `0` | The length dependence of `pdiblc1`, `lpdiblc1`: its coefficient of the reciprocal of the effective length, in the unit of `pdiblc1` times meters; default 0. |
| `lpdiblc2` | `0` | The length dependence of `pdiblc2`, `lpdiblc2`: its coefficient of the reciprocal of the effective length, in the unit of `pdiblc2` times meters; default 0. |
| `lpdiblcb` | `0` | The length dependence of `pdiblcb`, `lpdiblcb`: its coefficient of the reciprocal of the effective length, in the unit of `pdiblcb` times meters; default 0. |
| `lprt` | `0` | The length dependence of `prt`, `lprt`: its coefficient of the reciprocal of the effective length, in the unit of `prt` times meters; default 0. |
| `lprwb` | `0` | The length dependence of `prwb`, `lprwb`: its coefficient of the reciprocal of the effective length, in the unit of `prwb` times meters; default 0. |
| `lprwg` | `0` | The length dependence of `prwg`, `lprwg`: its coefficient of the reciprocal of the effective length, in the unit of `prwg` times meters; default 0. |
| `lpscbe1` | `0` | The length dependence of `pscbe1`, `lpscbe1`: its coefficient of the reciprocal of the effective length, in the unit of `pscbe1` times meters; default 0. |
| `lpscbe2` | `0` | The length dependence of `pscbe2`, `lpscbe2`: its coefficient of the reciprocal of the effective length, in the unit of `pscbe2` times meters; default 0. |
| `lpvag` | `0` | The length dependence of `pvag`, `lpvag`: its coefficient of the reciprocal of the effective length, in the unit of `pvag` times meters; default 0. |
| `lrdsw` | `0` | The length dependence of `rdsw`, `lrdsw`: its coefficient of the reciprocal of the effective length, in the unit of `rdsw` times meters; default 0. |
| `lu0` | `0` | The length dependence of `u0`, `lu0`: its coefficient of the reciprocal of the effective length, in the unit of `u0` times meters; default 0. |
| `lua` | `0` | The length dependence of `ua`, `lua`: its coefficient of the reciprocal of the effective length, in the unit of `ua` times meters; default 0. |
| `lua1` | `0` | The length dependence of `ua1`, `lua1`: its coefficient of the reciprocal of the effective length, in the unit of `ua1` times meters; default 0. |
| `lub` | `0` | The length dependence of `ub`, `lub`: its coefficient of the reciprocal of the effective length, in the unit of `ub` times meters; default 0. |
| `lub1` | `0` | The length dependence of `ub1`, `lub1`: its coefficient of the reciprocal of the effective length, in the unit of `ub1` times meters; default 0. |
| `luc` | `0` | The length dependence of `uc`, `luc`: its coefficient of the reciprocal of the effective length, in the unit of `uc` times meters; default 0. |
| `luc1` | `0` | The length dependence of `uc1`, `luc1`: its coefficient of the reciprocal of the effective length, in the unit of `uc1` times meters; default 0. |
| `lute` | `0` | The length dependence of `ute`, `lute`: its coefficient of the reciprocal of the effective length, in the unit of `ute` times meters; default 0. |
| `lvbm` | `0` | The length dependence of `vbm`, `lvbm`: its coefficient of the reciprocal of the effective length, in the unit of `vbm` times meters; default 0. |
| `lvbx` | `0` | The length dependence of `vbx`, `lvbx`: its coefficient of the reciprocal of the effective length, in the unit of `vbx` times meters; default 0. |
| `lvfb` | `0` | The length dependence of `vfb`, `lvfb`: its coefficient of the reciprocal of the effective length, in the unit of `vfb` times meters; default 0. |
| `lvfbcv` | `0` | The length dependence of `vfbcv`, `lvfbcv`: its coefficient of the reciprocal of the effective length, in the unit of `vfbcv` times meters; default 0. |
| `lvoff` | `0` | The length dependence of `voff`, `lvoff`: its coefficient of the reciprocal of the effective length, in the unit of `voff` times meters; default 0. |
| `lvoffcv` | `0` | The length dependence of `voffcv`, `lvoffcv`: its coefficient of the reciprocal of the effective length, in the unit of `voffcv` times meters; default 0. |
| `lvsat` | `0` | The length dependence of `vsat`, `lvsat`: its coefficient of the reciprocal of the effective length, in the unit of `vsat` times meters; default 0. |
| `lvth0` | `0` | The length dependence of `vth0`, `lvth0`: its coefficient of the reciprocal of the effective length, in the unit of `vth0` times meters; default 0. |
| `lw` | `0` | The width dependence of the length lost, `lw`, in meters times meters to the power `lwn`; default 0. |
| `lw0` | `0` | The length dependence of `w0`, `lw0`: its coefficient of the reciprocal of the effective length, in the unit of `w0` times meters; default 0. |
| `lwc` |  | The width dependence of the length the charges lose, `lwc`, in meters times meters to the power `lwn`; default `lw`. |
| `lwl` | `0` | The area dependence of the length lost, `lwl`, in meters times meters to the power `lln` + `lwn`; default 0. |
| `lwlc` |  | The area dependence of the length the charges lose, `lwlc`, in meters times meters to the power `lln` + `lwn`; default `lwl`. |
| `lwn` | `1` | The width exponent of `lw`, `lwn`; default 1. |
| `lwr` | `0` | The length dependence of `wr`, `lwr`: its coefficient of the reciprocal of the effective length, in the unit of `wr` times meters; default 0. |
| `lxj` | `0` | The length dependence of `xj`, `lxj`: its coefficient of the reciprocal of the effective length, in the unit of `xj` times meters; default 0. |
| `lxt` | `0` | The length dependence of `xt`, `lxt`: its coefficient of the reciprocal of the effective length, in the unit of `xt` times meters; default 0. |
| `mj` | `0.5` | The bottom capacitance's grading coefficient, `mj`; default 0.5. |
| `mjsw` | `0.33` | The field-oxide sidewall's grading coefficient, `mjsw`; default 0.33. |
| `mjswg` |  | The gate sidewall's grading coefficient, `mjswg`; default `mjsw`. |
| `mobility_model` | `1` | The mobility model, `mobmod`, 1, 2 or 3; default 1. |
| `mobility_multiplier` | `1` | A factor of the low-field mobility, `mulu0`; default 1. |
| `moin` | `15` | The gate bias's effect on the surface potential, `capmod` 3's, `moin`; default 15. |
| `multiplier` | `1` | The number of identical transistors in parallel, `m`; default 1. |
| `nch` |  | The channel's doping concentration near the surface, `nch`, per cubic meter; default 1.7e23 per cubic meter, or derived from `gamma1` where it alone is given. |
| `nfactor` | `1` | The subthreshold swing's coefficient, `nfactor`; default 1. |
| `ngate` | `0` | The poly gate's doping concentration, zero for no depletion of the gate, `ngate`, per cubic meter; default 0. |
| `nj` | `1` | The junctions' emission coefficient, `nj`; default 1. |
| `nlx` | `1.74e-07` | The lateral non-uniform doping along the channel, `nlx`, in meters; default 1.74e-7. |
| `noff` | `1` | The subthreshold swing's coefficient in the charges, `capmod` 1 to 3's, `noff`; default 1. |
| `noia` |  | The first oxide trap density coefficient of the flicker noise, `noimod` 2, 3 and 6's, `noia`, in the manual's unit; default 1e20 for an n-channel, 9.9e18 for a p-channel. |
| `noib` |  | The second oxide trap density coefficient of the flicker noise, `noib`, in the manual's unit; default 5e4 for an n-channel, 2.4e3 for a p-channel. |
| `noic` |  | The third oxide trap density coefficient of the flicker noise, `noic`, in the manual's unit; default -1.4e-12 for an n-channel, 1.4e-12 for a p-channel. |
| `noise_model` | `1` | The noise model, `noimod`, 1 to 6; default 1. |
| `nominal_temperature` | `300.15` | The temperature the parameters were measured at, `tnom`, in kelvins; default 27 °C. |
| `nsub` | `6e+22` | The substrate's doping concentration, `nsub`, per cubic meter; default 6e22. |
| `pa0` | `0` | The area dependence of `a0`, `pa0`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `a0` times square meters; default 0. |
| `pa1` | `0` | The area dependence of `a1`, `pa1`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `a1` times square meters; default 0. |
| `pa2` | `0` | The area dependence of `a2`, `pa2`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `a2` times square meters; default 0. |
| `pacde` | `0` | The area dependence of `acde`, `pacde`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `acde` times square meters; default 0. |
| `pags` | `0` | The area dependence of `ags`, `pags`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `ags` times square meters; default 0. |
| `palpha0` | `0` | The area dependence of `alpha0`, `palpha0`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `alpha0` times square meters; default 0. |
| `palpha1` | `0` | The area dependence of `alpha1`, `palpha1`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `alpha1` times square meters; default 0. |
| `pat` | `0` | The area dependence of `at`, `pat`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `at` times square meters; default 0. |
| `pb` | `1` | The bottom junction's built-in potential, at least 0.1 V, `pb`, in volts; default 1. |
| `pb0` | `0` | The area dependence of `b0`, `pb0`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `b0` times square meters; default 0. |
| `pb1` | `0` | The area dependence of `b1`, `pb1`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `b1` times square meters; default 0. |
| `pbeta0` | `0` | The area dependence of `beta0`, `pbeta0`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `beta0` times square meters; default 0. |
| `pbsw` | `1` | The field-oxide sidewall's built-in potential, at least 0.1 V, `pbsw`, in volts; default 1. |
| `pbswg` |  | The gate sidewall's built-in potential, at least 0.1 V, `pbswg`, in volts; default `pbsw`. |
| `pcdsc` | `0` | The area dependence of `cdsc`, `pcdsc`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `cdsc` times square meters; default 0. |
| `pcdscb` | `0` | The area dependence of `cdscb`, `pcdscb`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `cdscb` times square meters; default 0. |
| `pcdscd` | `0` | The area dependence of `cdscd`, `pcdscd`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `cdscd` times square meters; default 0. |
| `pcf` | `0` | The area dependence of `cf`, `pcf`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `cf` times square meters; default 0. |
| `pcgdl` | `0` | The area dependence of `cgdl`, `pcgdl`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `cgdl` times square meters; default 0. |
| `pcgsl` | `0` | The area dependence of `cgsl`, `pcgsl`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `cgsl` times square meters; default 0. |
| `pcit` | `0` | The area dependence of `cit`, `pcit`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `cit` times square meters; default 0. |
| `pckappa` | `0` | The area dependence of `ckappa`, `pckappa`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `ckappa` times square meters; default 0. |
| `pclc` | `0` | The area dependence of `clc`, `pclc`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `clc` times square meters; default 0. |
| `pcle` | `0` | The area dependence of `cle`, `pcle`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `cle` times square meters; default 0. |
| `pclm` | `1.3` | The channel-length modulation's coefficient, `pclm`; default 1.3. |
| `pdelta` | `0` | The area dependence of `delta`, `pdelta`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `delta` times square meters; default 0. |
| `pdiblc1` | `0.39` | The first coefficient of the drain-induced barrier lowering on the output resistance, `pdiblc1`; default 0.39. |
| `pdiblc2` | `0.0086` | The second coefficient of the drain-induced barrier lowering on the output resistance, `pdiblc2`; default 0.0086. |
| `pdiblcb` | `0` | The body effect on the drain-induced barrier lowering on the output resistance, `pdiblcb`, per volt; default 0. |
| `pdrout` | `0` | The area dependence of `drout`, `pdrout`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `drout` times square meters; default 0. |
| `pdsub` | `0` | The area dependence of `dsub`, `pdsub`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `dsub` times square meters; default 0. |
| `pdvt0` | `0` | The area dependence of `dvt0`, `pdvt0`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `dvt0` times square meters; default 0. |
| `pdvt0w` | `0` | The area dependence of `dvt0w`, `pdvt0w`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `dvt0w` times square meters; default 0. |
| `pdvt1` | `0` | The area dependence of `dvt1`, `pdvt1`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `dvt1` times square meters; default 0. |
| `pdvt1w` | `0` | The area dependence of `dvt1w`, `pdvt1w`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `dvt1w` times square meters; default 0. |
| `pdvt2` | `0` | The area dependence of `dvt2`, `pdvt2`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `dvt2` times square meters; default 0. |
| `pdvt2w` | `0` | The area dependence of `dvt2w`, `pdvt2w`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `dvt2w` times square meters; default 0. |
| `pdwb` | `0` | The area dependence of `dwb`, `pdwb`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `dwb` times square meters; default 0. |
| `pdwg` | `0` | The area dependence of `dwg`, `pdwg`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `dwg` times square meters; default 0. |
| `peta0` | `0` | The area dependence of `eta0`, `peta0`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `eta0` times square meters; default 0. |
| `petab` | `0` | The area dependence of `etab`, `petab`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `etab` times square meters; default 0. |
| `pgamma1` | `0` | The area dependence of `gamma1`, `pgamma1`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `gamma1` times square meters; default 0. |
| `pgamma2` | `0` | The area dependence of `gamma2`, `pgamma2`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `gamma2` times square meters; default 0. |
| `pk1` | `0` | The area dependence of `k1`, `pk1`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `k1` times square meters; default 0. |
| `pk2` | `0` | The area dependence of `k2`, `pk2`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `k2` times square meters; default 0. |
| `pk3` | `0` | The area dependence of `k3`, `pk3`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `k3` times square meters; default 0. |
| `pk3b` | `0` | The area dependence of `k3b`, `pk3b`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `k3b` times square meters; default 0. |
| `pketa` | `0` | The area dependence of `keta`, `pketa`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `keta` times square meters; default 0. |
| `pkt1` | `0` | The area dependence of `kt1`, `pkt1`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `kt1` times square meters; default 0. |
| `pkt1l` | `0` | The area dependence of `kt1l`, `pkt1l`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `kt1l` times square meters; default 0. |
| `pkt2` | `0` | The area dependence of `kt2`, `pkt2`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `kt2` times square meters; default 0. |
| `pmoin` | `0` | The area dependence of `moin`, `pmoin`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `moin` times square meters; default 0. |
| `pnch` | `0` | The area dependence of `nch`, `pnch`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `nch` times square meters; default 0. |
| `pnfactor` | `0` | The area dependence of `nfactor`, `pnfactor`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `nfactor` times square meters; default 0. |
| `pngate` | `0` | The area dependence of `ngate`, `pngate`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `ngate` times square meters; default 0. |
| `pnlx` | `0` | The area dependence of `nlx`, `pnlx`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `nlx` times square meters; default 0. |
| `pnoff` | `0` | The area dependence of `noff`, `pnoff`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `noff` times square meters; default 0. |
| `pnsub` | `0` | The area dependence of `nsub`, `pnsub`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `nsub` times square meters; default 0. |
| `ppclm` | `0` | The area dependence of `pclm`, `ppclm`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `pclm` times square meters; default 0. |
| `ppdiblc1` | `0` | The area dependence of `pdiblc1`, `ppdiblc1`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `pdiblc1` times square meters; default 0. |
| `ppdiblc2` | `0` | The area dependence of `pdiblc2`, `ppdiblc2`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `pdiblc2` times square meters; default 0. |
| `ppdiblcb` | `0` | The area dependence of `pdiblcb`, `ppdiblcb`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `pdiblcb` times square meters; default 0. |
| `pprt` | `0` | The area dependence of `prt`, `pprt`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `prt` times square meters; default 0. |
| `pprwb` | `0` | The area dependence of `prwb`, `pprwb`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `prwb` times square meters; default 0. |
| `pprwg` | `0` | The area dependence of `prwg`, `pprwg`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `prwg` times square meters; default 0. |
| `ppscbe1` | `0` | The area dependence of `pscbe1`, `ppscbe1`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `pscbe1` times square meters; default 0. |
| `ppscbe2` | `0` | The area dependence of `pscbe2`, `ppscbe2`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `pscbe2` times square meters; default 0. |
| `ppvag` | `0` | The area dependence of `pvag`, `ppvag`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `pvag` times square meters; default 0. |
| `prdsw` | `0` | The area dependence of `rdsw`, `prdsw`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `rdsw` times square meters; default 0. |
| `prt` | `0` | The temperature coefficient of `rdsw`, as `rdsw` is, `prt`, in ohms; default 0. |
| `prwb` | `0` | The body effect on `rdsw`, `prwb`, per square-root volt; default 0. |
| `prwg` | `0` | The gate bias's effect on `rdsw`, `prwg`, per volt; default 0. |
| `pscbe1` | `424000000` | The first coefficient of the substrate current's body effect on the output resistance, `pscbe1`, in volts per meter; default 424000000. |
| `pscbe2` | `1e-05` | The second coefficient of the substrate current's body effect on the output resistance, `pscbe2`, in meters per volt; default 1e-5. |
| `pu0` | `0` | The area dependence of `u0`, `pu0`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `u0` times square meters; default 0. |
| `pua` | `0` | The area dependence of `ua`, `pua`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `ua` times square meters; default 0. |
| `pua1` | `0` | The area dependence of `ua1`, `pua1`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `ua1` times square meters; default 0. |
| `pub` | `0` | The area dependence of `ub`, `pub`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `ub` times square meters; default 0. |
| `pub1` | `0` | The area dependence of `ub1`, `pub1`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `ub1` times square meters; default 0. |
| `puc` | `0` | The area dependence of `uc`, `puc`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `uc` times square meters; default 0. |
| `puc1` | `0` | The area dependence of `uc1`, `puc1`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `uc1` times square meters; default 0. |
| `pute` | `0` | The area dependence of `ute`, `pute`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `ute` times square meters; default 0. |
| `pvag` | `0` | The gate bias's effect on the Early voltage, `pvag`; default 0. |
| `pvbm` | `0` | The area dependence of `vbm`, `pvbm`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `vbm` times square meters; default 0. |
| `pvbx` | `0` | The area dependence of `vbx`, `pvbx`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `vbx` times square meters; default 0. |
| `pvfb` | `0` | The area dependence of `vfb`, `pvfb`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `vfb` times square meters; default 0. |
| `pvfbcv` | `0` | The area dependence of `vfbcv`, `pvfbcv`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `vfbcv` times square meters; default 0. |
| `pvoff` | `0` | The area dependence of `voff`, `pvoff`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `voff` times square meters; default 0. |
| `pvoffcv` | `0` | The area dependence of `voffcv`, `pvoffcv`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `voffcv` times square meters; default 0. |
| `pvsat` | `0` | The area dependence of `vsat`, `pvsat`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `vsat` times square meters; default 0. |
| `pvth0` | `0` | The area dependence of `vth0`, `pvth0`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `vth0` times square meters; default 0. |
| `pw0` | `0` | The area dependence of `w0`, `pw0`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `w0` times square meters; default 0. |
| `pwr` | `0` | The area dependence of `wr`, `pwr`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `wr` times square meters; default 0. |
| `pxj` | `0` | The area dependence of `xj`, `pxj`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `xj` times square meters; default 0. |
| `pxt` | `0` | The area dependence of `xt`, `pxt`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `xt` times square meters; default 0. |
| `rdsw` | `0` | The source and drain resistance of a channel 1 µm wide, which the drain current folds in, `RDSW` in ohms times micrometers to the power `wr`, `rdsw`, in ohms; default 0. |
| `rsh` | `0` | The source and drain diffusions' sheet resistance, `rsh`, in ohms per square; default 0. |
| `source_area` | `0` | The source diffusion's area, `as`, in square meters; default 0. |
| `source_perimeter` | `0` | The source diffusion's perimeter, `ps`, in meters; default 0. |
| `source_squares` | `1` | The source diffusion's length in squares, `nrs`; default 1. |
| `tcj` | `0` | The temperature coefficient of `cj`, `tcj`, per kelvin; default 0. |
| `tcjsw` | `0` | The temperature coefficient of `cjsw`, `tcjsw`, per kelvin; default 0. |
| `tcjswg` | `0` | The temperature coefficient of `cjswg`, `tcjswg`, per kelvin; default 0. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |
| `thermal` | `false` | Whether the device heats by the power it dissipates, through a terminal `thermal` joined to a thermal net, whose temperature above the ambient it runs at. |
| `threshold_adjustment` | `0` | A shift of the zero-bias threshold, `delvto`, in volts; default 0. |
| `tox` | `1.5e-08` | The gate oxide's thickness, `tox`, in meters; default 1.5e-8. |
| `toxm` |  | The oxide thickness the parameters were extracted at, which scales `k1` and `k2`, `toxm`, in meters; default `tox`. |
| `tpb` | `0` | The temperature coefficient of `pb`, `tpb`, in volts per kelvin; default 0. |
| `tpbsw` | `0` | The temperature coefficient of `pbsw`, `tpbsw`, in volts per kelvin; default 0. |
| `tpbswg` | `0` | The temperature coefficient of `pbswg`, `tpbswg`, in volts per kelvin; default 0. |
| `u0` |  | The low-field mobility at the nominal temperature, `u0`, in square meters per volt-second; default 0.067 m²/(V·s) for an n-channel, 0.025 m²/(V·s) for a p-channel. |
| `ua` | `2.25e-09` | The first-order degradation of the mobility by the gate field, `ua`, in meters per volt; default 2.25e-9. |
| `ua1` | `4.31e-09` | The temperature coefficient of `ua`, `ua1`, in meters per volt; default 4.31e-9. |
| `ub` | `5.87e-19` | The second-order degradation of the mobility by the gate field, `ub`, in square meters per square volt; default 5.87e-19. |
| `ub1` | `-7.61e-18` | The temperature coefficient of `ub`, `ub1`, in square meters per square volt; default -7.61e-18. |
| `uc` |  | The body effect on the mobility's degradation, `uc`, in meters per square volt, or per volt at `mobmod` 3; default -4.65e-11 m/V², or -0.0465 per volt at `mobmod` 3. |
| `uc1` |  | The temperature coefficient of `uc`, `uc1`, in meters per square volt, or per volt at `mobmod` 3; default -5.6e-11 m/V², or -0.056 per volt at `mobmod` 3. |
| `ute` | `-1.5` | The mobility's temperature exponent, `ute`; default -1.5. |
| `vbm` | `-3` | The most negative bulk-source voltage the threshold's body effect reads, `vbm`, in volts; default -3. |
| `vbx` |  | The bulk-source voltage at which the depletion reaches `xt`, `vbx`, in volts; default derived from `nch` and `xt`. |
| `vfb` |  | The flat-band voltage, `vfb`, in volts; default derived from `vth0`, or -1 V. |
| `vfbcv` | `-1` | The flat-band voltage of `capmod` 0, `vfbcv`, in volts; default -1. |
| `voff` | `-0.08` | The offset of the threshold below it, in the drain current's smoothing, `voff`, in volts; default -0.08. |
| `voffcv` | `0` | The offset of the threshold in the charges, `capmod` 1 to 3's, `voffcv`, in volts; default 0. |
| `vsat` | `80000` | The carriers' saturation velocity at the nominal temperature, `vsat`, in meters per second; default 80000. |
| `vth0` |  | The zero-bias threshold voltage of a long, wide channel, negative for a p-channel, `vth0`, in volts; default derived from `vfb`, `phi` and `k1`. |
| `w0` | `2.5e-06` | The width offset of the narrow-width effect, `w0`, in meters; default 2.5e-6. |
| `wa0` | `0` | The width dependence of `a0`, `wa0`: its coefficient of the reciprocal of the effective width, in the unit of `a0` times meters; default 0. |
| `wa1` | `0` | The width dependence of `a1`, `wa1`: its coefficient of the reciprocal of the effective width, in the unit of `a1` times meters; default 0. |
| `wa2` | `0` | The width dependence of `a2`, `wa2`: its coefficient of the reciprocal of the effective width, in the unit of `a2` times meters; default 0. |
| `wacde` | `0` | The width dependence of `acde`, `wacde`: its coefficient of the reciprocal of the effective width, in the unit of `acde` times meters; default 0. |
| `wags` | `0` | The width dependence of `ags`, `wags`: its coefficient of the reciprocal of the effective width, in the unit of `ags` times meters; default 0. |
| `walpha0` | `0` | The width dependence of `alpha0`, `walpha0`: its coefficient of the reciprocal of the effective width, in the unit of `alpha0` times meters; default 0. |
| `walpha1` | `0` | The width dependence of `alpha1`, `walpha1`: its coefficient of the reciprocal of the effective width, in the unit of `alpha1` times meters; default 0. |
| `wat` | `0` | The width dependence of `at`, `wat`: its coefficient of the reciprocal of the effective width, in the unit of `at` times meters; default 0. |
| `wb0` | `0` | The width dependence of `b0`, `wb0`: its coefficient of the reciprocal of the effective width, in the unit of `b0` times meters; default 0. |
| `wb1` | `0` | The width dependence of `b1`, `wb1`: its coefficient of the reciprocal of the effective width, in the unit of `b1` times meters; default 0. |
| `wbeta0` | `0` | The width dependence of `beta0`, `wbeta0`: its coefficient of the reciprocal of the effective width, in the unit of `beta0` times meters; default 0. |
| `wcdsc` | `0` | The width dependence of `cdsc`, `wcdsc`: its coefficient of the reciprocal of the effective width, in the unit of `cdsc` times meters; default 0. |
| `wcdscb` | `0` | The width dependence of `cdscb`, `wcdscb`: its coefficient of the reciprocal of the effective width, in the unit of `cdscb` times meters; default 0. |
| `wcdscd` | `0` | The width dependence of `cdscd`, `wcdscd`: its coefficient of the reciprocal of the effective width, in the unit of `cdscd` times meters; default 0. |
| `wcf` | `0` | The width dependence of `cf`, `wcf`: its coefficient of the reciprocal of the effective width, in the unit of `cf` times meters; default 0. |
| `wcgdl` | `0` | The width dependence of `cgdl`, `wcgdl`: its coefficient of the reciprocal of the effective width, in the unit of `cgdl` times meters; default 0. |
| `wcgsl` | `0` | The width dependence of `cgsl`, `wcgsl`: its coefficient of the reciprocal of the effective width, in the unit of `cgsl` times meters; default 0. |
| `wcit` | `0` | The width dependence of `cit`, `wcit`: its coefficient of the reciprocal of the effective width, in the unit of `cit` times meters; default 0. |
| `wckappa` | `0` | The width dependence of `ckappa`, `wckappa`: its coefficient of the reciprocal of the effective width, in the unit of `ckappa` times meters; default 0. |
| `wclc` | `0` | The width dependence of `clc`, `wclc`: its coefficient of the reciprocal of the effective width, in the unit of `clc` times meters; default 0. |
| `wcle` | `0` | The width dependence of `cle`, `wcle`: its coefficient of the reciprocal of the effective width, in the unit of `cle` times meters; default 0. |
| `wdelta` | `0` | The width dependence of `delta`, `wdelta`: its coefficient of the reciprocal of the effective width, in the unit of `delta` times meters; default 0. |
| `wdrout` | `0` | The width dependence of `drout`, `wdrout`: its coefficient of the reciprocal of the effective width, in the unit of `drout` times meters; default 0. |
| `wdsub` | `0` | The width dependence of `dsub`, `wdsub`: its coefficient of the reciprocal of the effective width, in the unit of `dsub` times meters; default 0. |
| `wdvt0` | `0` | The width dependence of `dvt0`, `wdvt0`: its coefficient of the reciprocal of the effective width, in the unit of `dvt0` times meters; default 0. |
| `wdvt0w` | `0` | The width dependence of `dvt0w`, `wdvt0w`: its coefficient of the reciprocal of the effective width, in the unit of `dvt0w` times meters; default 0. |
| `wdvt1` | `0` | The width dependence of `dvt1`, `wdvt1`: its coefficient of the reciprocal of the effective width, in the unit of `dvt1` times meters; default 0. |
| `wdvt1w` | `0` | The width dependence of `dvt1w`, `wdvt1w`: its coefficient of the reciprocal of the effective width, in the unit of `dvt1w` times meters; default 0. |
| `wdvt2` | `0` | The width dependence of `dvt2`, `wdvt2`: its coefficient of the reciprocal of the effective width, in the unit of `dvt2` times meters; default 0. |
| `wdvt2w` | `0` | The width dependence of `dvt2w`, `wdvt2w`: its coefficient of the reciprocal of the effective width, in the unit of `dvt2w` times meters; default 0. |
| `wdwb` | `0` | The width dependence of `dwb`, `wdwb`: its coefficient of the reciprocal of the effective width, in the unit of `dwb` times meters; default 0. |
| `wdwg` | `0` | The width dependence of `dwg`, `wdwg`: its coefficient of the reciprocal of the effective width, in the unit of `dwg` times meters; default 0. |
| `weta0` | `0` | The width dependence of `eta0`, `weta0`: its coefficient of the reciprocal of the effective width, in the unit of `eta0` times meters; default 0. |
| `wetab` | `0` | The width dependence of `etab`, `wetab`: its coefficient of the reciprocal of the effective width, in the unit of `etab` times meters; default 0. |
| `wgamma1` | `0` | The width dependence of `gamma1`, `wgamma1`: its coefficient of the reciprocal of the effective width, in the unit of `gamma1` times meters; default 0. |
| `wgamma2` | `0` | The width dependence of `gamma2`, `wgamma2`: its coefficient of the reciprocal of the effective width, in the unit of `gamma2` times meters; default 0. |
| `width` | `5e-06` | The channel's drawn width, `w`, in meters; default 5 µm. |
| `wint` | `0` | The width each side of the channel loses, which narrows it, `wint`, in meters; default 0. |
| `wk1` | `0` | The width dependence of `k1`, `wk1`: its coefficient of the reciprocal of the effective width, in the unit of `k1` times meters; default 0. |
| `wk2` | `0` | The width dependence of `k2`, `wk2`: its coefficient of the reciprocal of the effective width, in the unit of `k2` times meters; default 0. |
| `wk3` | `0` | The width dependence of `k3`, `wk3`: its coefficient of the reciprocal of the effective width, in the unit of `k3` times meters; default 0. |
| `wk3b` | `0` | The width dependence of `k3b`, `wk3b`: its coefficient of the reciprocal of the effective width, in the unit of `k3b` times meters; default 0. |
| `wketa` | `0` | The width dependence of `keta`, `wketa`: its coefficient of the reciprocal of the effective width, in the unit of `keta` times meters; default 0. |
| `wkt1` | `0` | The width dependence of `kt1`, `wkt1`: its coefficient of the reciprocal of the effective width, in the unit of `kt1` times meters; default 0. |
| `wkt1l` | `0` | The width dependence of `kt1l`, `wkt1l`: its coefficient of the reciprocal of the effective width, in the unit of `kt1l` times meters; default 0. |
| `wkt2` | `0` | The width dependence of `kt2`, `wkt2`: its coefficient of the reciprocal of the effective width, in the unit of `kt2` times meters; default 0. |
| `wl` | `0` | The length dependence of the width lost, `wl`, in meters times meters to the power `wln`; default 0. |
| `wlc` |  | The length dependence of the width the charges lose, `wlc`, in meters times meters to the power `wln`; default `wl`. |
| `wln` | `1` | The length exponent of `wl`, `wln`; default 1. |
| `wmoin` | `0` | The width dependence of `moin`, `wmoin`: its coefficient of the reciprocal of the effective width, in the unit of `moin` times meters; default 0. |
| `wnch` | `0` | The width dependence of `nch`, `wnch`: its coefficient of the reciprocal of the effective width, in the unit of `nch` times meters; default 0. |
| `wnfactor` | `0` | The width dependence of `nfactor`, `wnfactor`: its coefficient of the reciprocal of the effective width, in the unit of `nfactor` times meters; default 0. |
| `wngate` | `0` | The width dependence of `ngate`, `wngate`: its coefficient of the reciprocal of the effective width, in the unit of `ngate` times meters; default 0. |
| `wnlx` | `0` | The width dependence of `nlx`, `wnlx`: its coefficient of the reciprocal of the effective width, in the unit of `nlx` times meters; default 0. |
| `wnoff` | `0` | The width dependence of `noff`, `wnoff`: its coefficient of the reciprocal of the effective width, in the unit of `noff` times meters; default 0. |
| `wnsub` | `0` | The width dependence of `nsub`, `wnsub`: its coefficient of the reciprocal of the effective width, in the unit of `nsub` times meters; default 0. |
| `wpclm` | `0` | The width dependence of `pclm`, `wpclm`: its coefficient of the reciprocal of the effective width, in the unit of `pclm` times meters; default 0. |
| `wpdiblc1` | `0` | The width dependence of `pdiblc1`, `wpdiblc1`: its coefficient of the reciprocal of the effective width, in the unit of `pdiblc1` times meters; default 0. |
| `wpdiblc2` | `0` | The width dependence of `pdiblc2`, `wpdiblc2`: its coefficient of the reciprocal of the effective width, in the unit of `pdiblc2` times meters; default 0. |
| `wpdiblcb` | `0` | The width dependence of `pdiblcb`, `wpdiblcb`: its coefficient of the reciprocal of the effective width, in the unit of `pdiblcb` times meters; default 0. |
| `wprt` | `0` | The width dependence of `prt`, `wprt`: its coefficient of the reciprocal of the effective width, in the unit of `prt` times meters; default 0. |
| `wprwb` | `0` | The width dependence of `prwb`, `wprwb`: its coefficient of the reciprocal of the effective width, in the unit of `prwb` times meters; default 0. |
| `wprwg` | `0` | The width dependence of `prwg`, `wprwg`: its coefficient of the reciprocal of the effective width, in the unit of `prwg` times meters; default 0. |
| `wpscbe1` | `0` | The width dependence of `pscbe1`, `wpscbe1`: its coefficient of the reciprocal of the effective width, in the unit of `pscbe1` times meters; default 0. |
| `wpscbe2` | `0` | The width dependence of `pscbe2`, `wpscbe2`: its coefficient of the reciprocal of the effective width, in the unit of `pscbe2` times meters; default 0. |
| `wpvag` | `0` | The width dependence of `pvag`, `wpvag`: its coefficient of the reciprocal of the effective width, in the unit of `pvag` times meters; default 0. |
| `wr` | `1` | The width exponent of `rdsw`, `wr`; default 1. |
| `wrdsw` | `0` | The width dependence of `rdsw`, `wrdsw`: its coefficient of the reciprocal of the effective width, in the unit of `rdsw` times meters; default 0. |
| `wu0` | `0` | The width dependence of `u0`, `wu0`: its coefficient of the reciprocal of the effective width, in the unit of `u0` times meters; default 0. |
| `wua` | `0` | The width dependence of `ua`, `wua`: its coefficient of the reciprocal of the effective width, in the unit of `ua` times meters; default 0. |
| `wua1` | `0` | The width dependence of `ua1`, `wua1`: its coefficient of the reciprocal of the effective width, in the unit of `ua1` times meters; default 0. |
| `wub` | `0` | The width dependence of `ub`, `wub`: its coefficient of the reciprocal of the effective width, in the unit of `ub` times meters; default 0. |
| `wub1` | `0` | The width dependence of `ub1`, `wub1`: its coefficient of the reciprocal of the effective width, in the unit of `ub1` times meters; default 0. |
| `wuc` | `0` | The width dependence of `uc`, `wuc`: its coefficient of the reciprocal of the effective width, in the unit of `uc` times meters; default 0. |
| `wuc1` | `0` | The width dependence of `uc1`, `wuc1`: its coefficient of the reciprocal of the effective width, in the unit of `uc1` times meters; default 0. |
| `wute` | `0` | The width dependence of `ute`, `wute`: its coefficient of the reciprocal of the effective width, in the unit of `ute` times meters; default 0. |
| `wvbm` | `0` | The width dependence of `vbm`, `wvbm`: its coefficient of the reciprocal of the effective width, in the unit of `vbm` times meters; default 0. |
| `wvbx` | `0` | The width dependence of `vbx`, `wvbx`: its coefficient of the reciprocal of the effective width, in the unit of `vbx` times meters; default 0. |
| `wvfb` | `0` | The width dependence of `vfb`, `wvfb`: its coefficient of the reciprocal of the effective width, in the unit of `vfb` times meters; default 0. |
| `wvfbcv` | `0` | The width dependence of `vfbcv`, `wvfbcv`: its coefficient of the reciprocal of the effective width, in the unit of `vfbcv` times meters; default 0. |
| `wvoff` | `0` | The width dependence of `voff`, `wvoff`: its coefficient of the reciprocal of the effective width, in the unit of `voff` times meters; default 0. |
| `wvoffcv` | `0` | The width dependence of `voffcv`, `wvoffcv`: its coefficient of the reciprocal of the effective width, in the unit of `voffcv` times meters; default 0. |
| `wvsat` | `0` | The width dependence of `vsat`, `wvsat`: its coefficient of the reciprocal of the effective width, in the unit of `vsat` times meters; default 0. |
| `wvth0` | `0` | The width dependence of `vth0`, `wvth0`: its coefficient of the reciprocal of the effective width, in the unit of `vth0` times meters; default 0. |
| `ww` | `0` | The width dependence of the width lost, `ww`, in meters to the power 1 + `wwn`; default 0. |
| `ww0` | `0` | The width dependence of `w0`, `ww0`: its coefficient of the reciprocal of the effective width, in the unit of `w0` times meters; default 0. |
| `wwc` |  | The width dependence of the width the charges lose, `wwc`, in meters to the power 1 + `wwn`; default `ww`. |
| `wwl` | `0` | The area dependence of the width lost, `wwl`, in meters times meters to the power `wln` + `wwn`; default 0. |
| `wwlc` |  | The area dependence of the width the charges lose, `wwlc`, in meters times meters to the power `wln` + `wwn`; default `wwl`. |
| `wwn` | `1` | The width exponent of `ww`, `wwn`; default 1. |
| `wwr` | `0` | The width dependence of `wr`, `wwr`: its coefficient of the reciprocal of the effective width, in the unit of `wr` times meters; default 0. |
| `wxj` | `0` | The width dependence of `xj`, `wxj`: its coefficient of the reciprocal of the effective width, in the unit of `xj` times meters; default 0. |
| `wxt` | `0` | The width dependence of `xt`, `wxt`: its coefficient of the reciprocal of the effective width, in the unit of `xt` times meters; default 0. |
| `xj` | `1.5e-07` | The source and drain junctions' depth, `xj`, in meters; default 1.5e-7. |
| `xl` | `0` | The length the mask and etching add, `xl`, in meters; default 0. |
| `xt` | `1.55e-07` | The depth of the doping near the surface, `xt`, in meters; default 1.55e-7. |
| `xti` | `3` | The junctions' saturation current temperature exponent, `xti`; default 3. |
| `xw` | `0` | The width the mask and etching add, `xw`, in meters; default 0. |

## `bsim4`

| Parameter | Default | Description |
| --- | --- | --- |
| `a0` | `1` | The non-uniform depletion width effect coefficient, `a0`; default 1. |
| `a1` | `0` | The non-saturation effect coefficient, `a1`, in per volt; default 0. |
| `a2` | `1` | The non-saturation effect coefficient, `a2`; default 1. |
| `acde` | `1` | The exponential coefficient for finite charge thickness, `acde`, in meters per volt; default 1. |
| `af` | `1` | The flicker noise exponent, `af`; default 1. |
| `agidl` | `0` | The pre-exponential constant for GIDL, `agidl`, in siemens; default 0. |
| `ags` | `0` | The gate bias coefficient of Abulk, `ags`, in per volt; default 0. |
| `aigbacc` | `0.0136` | The parameter for Igb, `aigbacc`, in the manual's units; default 0.0136. |
| `aigbinv` | `0.0111` | The parameter for Igb, `aigbinv`, in the manual's units; default 0.0111. |
| `aigc` |  | The parameter for Igc, `aigc`, in the manual's units; default 1.36e-2 for an n-channel and 9.8e-3 for a p-channel. |
| `aigsd` |  | The parameter for Igs,d, `aigsd`, in the manual's units; default 1.36e-2 for an n-channel and 9.8e-3 for a p-channel. |
| `alpha0` | `0` | The first coefficient of the impact ionization's substrate current, `alpha0`, in meters per volt; default 0. |
| `alpha1` | `0` | The channel-length dependence of the substrate current, `alpha1`, in per volt; default 0. |
| `at` | `33000` | The temperature coefficient of vsat, `at`, in meters per second; default 33000. |
| `b0` | `0` | The first narrow-width coefficient of Abulk, `b0`, in meters; default 0. |
| `b1` | `0` | The second narrow-width coefficient of Abulk, `b1`, in meters; default 0. |
| `beta0` | `0` | The exponential coefficient of the substrate current, `beta0`, in volts; default 0. |
| `bgidl` | `2300000000` | The exponential constant for GIDL, `bgidl`, in volts per meter; default 2.3e+09. |
| `bigbacc` | `0.00171` | The parameter for Igb, `bigbacc`, in the manual's units; default 0.00171. |
| `bigbinv` | `0.000949` | The parameter for Igb, `bigbinv`, in the manual's units; default 0.000949. |
| `bigc` |  | The parameter for Igc, `bigc`, in the manual's units; default 1.71e-3 for an n-channel and 7.59e-4 for a p-channel. |
| `bigsd` |  | The parameter for Igs,d, `bigsd`, in the manual's units; default 1.71e-3 for an n-channel and 7.59e-4 for a p-channel. |
| `body_resistance_model` | `0` | The body's resistance network, `rbodymod`, 0 or 1; default 0. |
| `bvd` |  | The drain diode breakdown voltage, `bvd`, in volts; default `bvs`. |
| `bvs` | `10` | The source diode breakdown voltage, `bvs`, in volts; default 10. |
| `cdsc` | `0.00024` | The coupling capacitance of the drain and the source to the channel, `cdsc`, in farads per square meter; default 0.00024. |
| `cdscb` | `0` | The body-bias dependence of `cdsc`, `cdscb`, in farads per volt square meter; default 0. |
| `cdscd` | `0` | The drain-bias dependence of `cdsc`, `cdscd`, in farads per volt square meter; default 0. |
| `cf` |  | The fringe capacitance parameter, `cf`, in farads per meter; default `2 · epsrox · ε0 / π · ln(1 + 0.4 µm / toxe)`, of `toxe` where it is given, else of 3 nm. |
| `cgbo` |  | The gate-bulk overlap capacitance per length, `cgbo`, in farads per meter; default twice `dwc` times the oxide capacitance. |
| `cgdl` | `0` | The gate-drain overlap capacitance per width of the lightly doped extension, `cgdl`, in farads per meter; default 0. |
| `cgdo` |  | The gate-drain overlap capacitance per width, `cgdo`, in farads per meter; default derived from `dlc` and `cgdl`, or from `xj`. |
| `cgidl` | `0.5` | The parameter for body-bias dependence of GIDL, `cgidl`, in cubic volts; default 0.5. |
| `cgsl` | `0` | The gate-source overlap capacitance per width of the lightly doped extension, `cgsl`, in farads per meter; default 0. |
| `cgso` |  | The gate-source overlap capacitance per width, `cgso`, in farads per meter; default derived from `dlc` and `cgsl`, or from `xj`. |
| `channel` | `n` | The channel; default `n`. One of `n`, `p`. |
| `charge_partition` | `0` | The share of the channel's charge the drain takes in saturation, `xpart`: 0.4 below 0.5, 0.5 at 0.5 and none above it; default 0. |
| `check_parameters` | `true` | Whether the size-dependent parameters are checked and clamped where they lie out of range, `paramchk = 1`; default true. |
| `cigbacc` | `0.075` | The parameter for Igb, `cigbacc`, in per volt; default 0.075. |
| `cigbinv` | `0.006` | The parameter for Igb, `cigbinv`, in per volt; default 0.006. |
| `cigc` |  | The parameter for Igc, `cigc`, in per volt; default 0.075 per volt for an n-channel and 0.03 per volt for a p-channel. |
| `cigsd` |  | The parameter for Igs,d, `cigsd`, in per volt; default 0.075 per volt for an n-channel and 0.03 per volt for a p-channel. |
| `cit` | `0` | The interface state capacitance, `cit`, in farads per square meter; default 0. |
| `cjd` |  | The drain bottom junction capacitance per unit area, `cjd`, in farads per square meter; default `cjs`. |
| `cjs` | `0.0005` | The source bottom junction capacitance per unit area, `cjs`, in farads per square meter; default 0.0005. |
| `cjswd` |  | The drain sidewall junction capacitance per unit periphery, `cjswd`, in farads per meter; default `cjsws`. |
| `cjswgd` |  | The drain (gate side) sidewall junction capacitance per unit width, `cjswgd`, in farads per meter; default `cjswgs`. |
| `cjswgs` |  | The source (gate side) sidewall junction capacitance per unit width, `cjswgs`, in farads per meter; default `cjsws`. |
| `cjsws` | `5e-10` | The source sidewall junction capacitance per unit periphery, `cjsws`, in farads per meter; default 5e-10. |
| `ckappad` |  | The coefficient of the bias dependence of the gate-drain overlap, `ckappad`, in volts; default `ckappas`. |
| `ckappas` | `0.6` | The coefficient of the bias dependence of the gate-source overlap, `ckappas`, in volts; default 0.6. |
| `clc` | `1e-07` | The vdsat parameter for C-V model, `clc`, in meters; default 1e-07. |
| `cle` | `0.6` | The vdsat parameter for C-V model, `cle`; default 0.6. |
| `delta` | `0.01` | The effective Vds parameter, `delta`, in volts; default 0.01. |
| `diffusion_edge_a` | `0` | The distance from the gate to the edge of the diffusion on one side, `sa`, in meters, which sets the stress of the length of diffusion where it and `sb` are positive; default 0. |
| `diffusion_edge_b` | `0` | The distance from the gate to the edge of the diffusion on the other side, `sb`, in meters; default 0. |
| `dlc` |  | The length lost on each side of the channel in the charges, `dlc`, in meters; default `lint`. |
| `dlcig` |  | The length lost on each side of the channel in the gate tunneling currents, `dlcig`, in meters; default `lint`. |
| `dmcg` | `0` | The distance of Mid-Contact to Gate edge, `dmcg`, in meters; default 0. |
| `dmcgt` | `0` | The distance of Mid-Contact to Gate edge in Test structures, `dmcgt`, in meters; default 0. |
| `dmci` |  | The distance of Mid-Contact to Isolation, `dmci`, in meters; default `dmcg`. |
| `dmdg` | `0` | The distance of Mid-Diffusion to Gate edge, `dmdg`, in meters; default 0. |
| `drain_area` |  | The drain diffusion's area, `ad`, in square meters; default derived from the fingers and `dmcg`, `dmci` and `dmcgt`. |
| `drain_perimeter` |  | The drain diffusion's perimeter, `pd`, in meters; default derived from the fingers and `dmcg`, `dmci` and `dmcgt`. |
| `drain_source_resistance_model` | `0` | The source and drain resistances, `rdsmod`, 0 or 1; default 0. |
| `drain_squares` |  | The drain diffusion's length in squares, `nrd`, behind which `rsh` sets a resistance where it is positive; default none, and no resistance. |
| `drout` | `0.56` | The DIBL coefficient of output resistance, `drout`; default 0.56. |
| `dsub` |  | The DIBL coefficient in the subthreshold region, `dsub`; default `drout`. |
| `dtox` | `0` | The electrical less the physical thickness of the gate oxide, `dtox`, in meters; default 0. |
| `dvt0` | `2.2` | The first coefficient of the short-channel effect, `dvt0`; default 2.2. |
| `dvt0w` | `0` | The first coefficient of the narrow-width effect, `dvt0w`, in per meter; default 0. |
| `dvt1` | `0.53` | The second coefficient of the short-channel effect, `dvt1`; default 0.53. |
| `dvt1w` | `5300000` | The second coefficient of the narrow-width effect, `dvt1w`, in per meter; default 5.3e+06. |
| `dvt2` | `-0.032` | The body-bias coefficient of the short-channel effect, `dvt2`, in per volt; default -0.032. |
| `dvt2w` | `-0.032` | The body-bias coefficient of the narrow-width effect, `dvt2w`, in per volt; default -0.032. |
| `dvtp0` | `0` | The first parameter for Vth shift due to pocket, `dvtp0`, in meters; default 0. |
| `dvtp1` | `0` | The second parameter for Vth shift due to pocket, `dvtp1`, in per volt; default 0. |
| `dwb` | `0` | The body-bias dependence of the width lost, `dwb`, in meters per square-root volt; default 0. |
| `dwc` |  | The width lost on each side of the channel in the charges, `dwc`, in meters; default `wint`. |
| `dwg` | `0` | The gate-bias dependence of the width lost, `dwg`, in meters per volt; default 0. |
| `dwj` |  | The width lost on each side of the source and drain junctions, `dwj`, in meters; default `dwc`. |
| `ef` | `1` | The flicker noise frequency exponent, `ef`; default 1. |
| `egidl` | `0.8` | The band bending of the gate-induced drain leakage, `egidl`, in volts; default 0.8. |
| `eigbinv` | `1.1` | The parameter for the Si bandgap for Igbinv, `eigbinv`, in volts; default 1.1. |
| `em` | `41000000` | The flicker noise parameter, `em`, in volts per meter; default 4.1e+07. |
| `epsrox` | `3.9` | The dielectric constant of the gate oxide relative to vacuum, `epsrox`; default 3.9. |
| `eta0` | `0.08` | The drain-induced barrier lowering in the subthreshold region, `eta0`; default 0.08. |
| `etab` | `-0.07` | The body-bias coefficient of `eta0`, `etab`, in per volt; default -0.07. |
| `eu` |  | The mobility exponent, `eu`; default 1.67 for an n-channel and 1 for a p-channel. |
| `finger_spacing` |  | The distance between neighbouring fingers, `sd`, in meters; default twice `dmcg`. |
| `fingers` | `1` | The number of fingers the channel's width is split into, `nf`; default 1. |
| `flicker_noise_model` | `1` | The flicker noise model, `fnoimod`, 0 or 1; default 1. |
| `fprout` | `0` | The rout degradation coefficient for pocket devices, `fprout`, in volts per square-root meter; default 0. |
| `gamma1` |  | The body effect near the surface, `gamma1`, in square-root volts; default derived from `ndep`. |
| `gamma2` |  | The body effect in the bulk, `gamma2`, in square-root volts; default derived from `nsub`. |
| `gate_resistance_model` | `0` | The gate's resistance network, `rgatemod`, 0 to 3; default 0. |
| `gbmin` | `1e-12` | The least conductance of each body resistance, `gbmin`, in siemens; default 1e-12. |
| `ijthdfwd` |  | The forward drain diode forward limiting current, `ijthdfwd`, in amperes; default `ijthsfwd`. |
| `ijthdrev` |  | The reverse drain diode forward limiting current, `ijthdrev`, in amperes; default `ijthsrev`. |
| `ijthsfwd` | `0.1` | The forward source diode forward limiting current, `ijthsfwd`, in amperes; default 0.1. |
| `ijthsrev` | `0.1` | The reverse source diode forward limiting current, `ijthsrev`, in amperes; default 0.1. |
| `jsd` |  | The bottom drain junction reverse saturation current density, `jsd`, in amperes per square meter; default `jss`. |
| `jss` | `0.0001` | The bottom source junction reverse saturation current density, `jss`, in amperes per square meter; default 0.0001. |
| `jswd` |  | The isolation edge sidewall drain junction reverse saturation current density, `jswd`, in amperes per meter; default `jsws`. |
| `jswgd` |  | The gate edge drain junction reverse saturation current density, `jswgd`, in amperes per meter; default `jswgs`. |
| `jswgs` | `0` | The gate edge source junction reverse saturation current density, `jswgs`, in amperes per meter; default 0. |
| `jsws` | `0` | The isolation edge sidewall source junction reverse saturation current density, `jsws`, in amperes per meter; default 0. |
| `jtsd` |  | The drain bottom trap-assisted saturation current density, `jtsd`, in amperes per square meter; default `jtss`. |
| `jtss` | `0` | The source bottom trap-assisted saturation current density, `jtss`, in amperes per square meter; default 0. |
| `jtsswd` |  | The drain STI sidewall trap-assisted saturation current density, `jtsswd`, in amperes per meter; default `jtssws`. |
| `jtsswgd` |  | The drain gate-edge sidewall trap-assisted saturation current density, `jtsswgd`, in amperes per meter; default `jtsswgs`. |
| `jtsswgs` | `0` | The source gate-edge sidewall trap-assisted saturation current density, `jtsswgs`, in amperes per meter; default 0. |
| `jtssws` | `0` | The source STI sidewall trap-assisted saturation current density, `jtssws`, in amperes per meter; default 0. |
| `k1` |  | The first-order body effect, `k1`, in square-root volts; default derived from `gamma1` and `gamma2`, or 0.53 √V where only `k2` is given. |
| `k2` |  | The second-order body effect, `k2`; default derived from `gamma1` and `gamma2`, or -0.0186 where only `k1` is given. |
| `k2we` | `0` | The K2 shift factor for well proximity effect, `k2we`; default 0. |
| `k3` | `80` | The narrow width effect coefficient, `k3`; default 80. |
| `k3b` | `0` | The body effect coefficient of k3, `k3b`, in per volt; default 0. |
| `keta` | `-0.047` | The body-bias coefficient of the non-uniform depletion width effect, `keta`, in per volt; default -0.047. |
| `kf` | `0` | The flicker noise coefficient, `kf`, in the manual's units; default 0. |
| `kt1` | `-0.11` | The temperature coefficient of the threshold, `kt1`, in volts; default -0.11. |
| `kt1l` | `0` | The channel-length dependence of the threshold's temperature coefficient, `kt1l`, in volt meters; default 0. |
| `kt2` | `0.022` | The body-coefficient of kt1, `kt2`; default 0.022. |
| `ku0` | `0` | The mobility degradation/enhancement coefficient for LOD, `ku0`, in meters; default 0. |
| `ku0we` | `0` | The mobility degradation factor for well proximity effect, `ku0we`; default 0. |
| `kvsat` | `0` | The saturation velocity degradation/enhancement parameter for LOD, `kvsat`; default 0. |
| `kvth0` | `0` | The threshold degradation/enhancement parameter for LOD, `kvth0`, in volt meters; default 0. |
| `kvth0we` | `0` | The threshold shift factor for well proximity effect, `kvth0we`, in volts; default 0. |
| `la0` | `0` | The length dependence of `a0`, `la0`: its coefficient of the reciprocal of the effective length, in the unit of `a0` times meters; default 0. |
| `la1` | `0` | The length dependence of `a1`, `la1`: its coefficient of the reciprocal of the effective length, in the unit of `a1` times meters; default 0. |
| `la2` | `0` | The length dependence of `a2`, `la2`: its coefficient of the reciprocal of the effective length, in the unit of `a2` times meters; default 0. |
| `lacde` | `0` | The length dependence of `acde`, `lacde`: its coefficient of the reciprocal of the effective length, in the unit of `acde` times meters; default 0. |
| `lagidl` | `0` | The length dependence of `agidl`, `lagidl`: its coefficient of the reciprocal of the effective length, in the unit of `agidl` times meters; default 0. |
| `lags` | `0` | The length dependence of `ags`, `lags`: its coefficient of the reciprocal of the effective length, in the unit of `ags` times meters; default 0. |
| `laigbacc` | `0` | The length dependence of `aigbacc`, `laigbacc`: its coefficient of the reciprocal of the effective length, in the unit of `aigbacc` times meters; default 0. |
| `laigbinv` | `0` | The length dependence of `aigbinv`, `laigbinv`: its coefficient of the reciprocal of the effective length, in the unit of `aigbinv` times meters; default 0. |
| `laigc` | `0` | The length dependence of `aigc`, `laigc`: its coefficient of the reciprocal of the effective length, in the unit of `aigc` times meters; default 0. |
| `laigsd` | `0` | The length dependence of `aigsd`, `laigsd`: its coefficient of the reciprocal of the effective length, in the unit of `aigsd` times meters; default 0. |
| `lalpha0` | `0` | The length dependence of `alpha0`, `lalpha0`: its coefficient of the reciprocal of the effective length, in the unit of `alpha0` times meters; default 0. |
| `lalpha1` | `0` | The length dependence of `alpha1`, `lalpha1`: its coefficient of the reciprocal of the effective length, in the unit of `alpha1` times meters; default 0. |
| `lambda` |  | The velocity overshoot parameter, `lambda`; default 0, and no velocity overshoot where it is not given. |
| `lat` | `0` | The length dependence of `at`, `lat`: its coefficient of the reciprocal of the effective length, in the unit of `at` times meters; default 0. |
| `lb0` | `0` | The length dependence of `b0`, `lb0`: its coefficient of the reciprocal of the effective length, in the unit of `b0` times meters; default 0. |
| `lb1` | `0` | The length dependence of `b1`, `lb1`: its coefficient of the reciprocal of the effective length, in the unit of `b1` times meters; default 0. |
| `lbeta0` | `0` | The length dependence of `beta0`, `lbeta0`: its coefficient of the reciprocal of the effective length, in the unit of `beta0` times meters; default 0. |
| `lbgidl` | `0` | The length dependence of `bgidl`, `lbgidl`: its coefficient of the reciprocal of the effective length, in the unit of `bgidl` times meters; default 0. |
| `lbigbacc` | `0` | The length dependence of `bigbacc`, `lbigbacc`: its coefficient of the reciprocal of the effective length, in the unit of `bigbacc` times meters; default 0. |
| `lbigbinv` | `0` | The length dependence of `bigbinv`, `lbigbinv`: its coefficient of the reciprocal of the effective length, in the unit of `bigbinv` times meters; default 0. |
| `lbigc` | `0` | The length dependence of `bigc`, `lbigc`: its coefficient of the reciprocal of the effective length, in the unit of `bigc` times meters; default 0. |
| `lbigsd` | `0` | The length dependence of `bigsd`, `lbigsd`: its coefficient of the reciprocal of the effective length, in the unit of `bigsd` times meters; default 0. |
| `lc` | `5e-09` | The backscattering mean free path, `lc`, in meters; default 5e-09. |
| `lcdsc` | `0` | The length dependence of `cdsc`, `lcdsc`: its coefficient of the reciprocal of the effective length, in the unit of `cdsc` times meters; default 0. |
| `lcdscb` | `0` | The length dependence of `cdscb`, `lcdscb`: its coefficient of the reciprocal of the effective length, in the unit of `cdscb` times meters; default 0. |
| `lcdscd` | `0` | The length dependence of `cdscd`, `lcdscd`: its coefficient of the reciprocal of the effective length, in the unit of `cdscd` times meters; default 0. |
| `lcf` | `0` | The length dependence of `cf`, `lcf`: its coefficient of the reciprocal of the effective length, in the unit of `cf` times meters; default 0. |
| `lcgdl` | `0` | The length dependence of `cgdl`, `lcgdl`: its coefficient of the reciprocal of the effective length, in the unit of `cgdl` times meters; default 0. |
| `lcgidl` | `0` | The length dependence of `cgidl`, `lcgidl`: its coefficient of the reciprocal of the effective length, in the unit of `cgidl` times meters; default 0. |
| `lcgsl` | `0` | The length dependence of `cgsl`, `lcgsl`: its coefficient of the reciprocal of the effective length, in the unit of `cgsl` times meters; default 0. |
| `lcigbacc` | `0` | The length dependence of `cigbacc`, `lcigbacc`: its coefficient of the reciprocal of the effective length, in the unit of `cigbacc` times meters; default 0. |
| `lcigbinv` | `0` | The length dependence of `cigbinv`, `lcigbinv`: its coefficient of the reciprocal of the effective length, in the unit of `cigbinv` times meters; default 0. |
| `lcigc` | `0` | The length dependence of `cigc`, `lcigc`: its coefficient of the reciprocal of the effective length, in the unit of `cigc` times meters; default 0. |
| `lcigsd` | `0` | The length dependence of `cigsd`, `lcigsd`: its coefficient of the reciprocal of the effective length, in the unit of `cigsd` times meters; default 0. |
| `lcit` | `0` | The length dependence of `cit`, `lcit`: its coefficient of the reciprocal of the effective length, in the unit of `cit` times meters; default 0. |
| `lckappad` | `0` | The length dependence of `ckappad`, `lckappad`: its coefficient of the reciprocal of the effective length, in the unit of `ckappad` times meters; default 0. |
| `lckappas` | `0` | The length dependence of `ckappas`, `lckappas`: its coefficient of the reciprocal of the effective length, in the unit of `ckappas` times meters; default 0. |
| `lclc` | `0` | The length dependence of `clc`, `lclc`: its coefficient of the reciprocal of the effective length, in the unit of `clc` times meters; default 0. |
| `lcle` | `0` | The length dependence of `cle`, `lcle`: its coefficient of the reciprocal of the effective length, in the unit of `cle` times meters; default 0. |
| `ldelta` | `0` | The length dependence of `delta`, `ldelta`: its coefficient of the reciprocal of the effective length, in the unit of `delta` times meters; default 0. |
| `ldrout` | `0` | The length dependence of `drout`, `ldrout`: its coefficient of the reciprocal of the effective length, in the unit of `drout` times meters; default 0. |
| `ldsub` | `0` | The length dependence of `dsub`, `ldsub`: its coefficient of the reciprocal of the effective length, in the unit of `dsub` times meters; default 0. |
| `ldvt0` | `0` | The length dependence of `dvt0`, `ldvt0`: its coefficient of the reciprocal of the effective length, in the unit of `dvt0` times meters; default 0. |
| `ldvt0w` | `0` | The length dependence of `dvt0w`, `ldvt0w`: its coefficient of the reciprocal of the effective length, in the unit of `dvt0w` times meters; default 0. |
| `ldvt1` | `0` | The length dependence of `dvt1`, `ldvt1`: its coefficient of the reciprocal of the effective length, in the unit of `dvt1` times meters; default 0. |
| `ldvt1w` | `0` | The length dependence of `dvt1w`, `ldvt1w`: its coefficient of the reciprocal of the effective length, in the unit of `dvt1w` times meters; default 0. |
| `ldvt2` | `0` | The length dependence of `dvt2`, `ldvt2`: its coefficient of the reciprocal of the effective length, in the unit of `dvt2` times meters; default 0. |
| `ldvt2w` | `0` | The length dependence of `dvt2w`, `ldvt2w`: its coefficient of the reciprocal of the effective length, in the unit of `dvt2w` times meters; default 0. |
| `ldvtp0` | `0` | The length dependence of `dvtp0`, `ldvtp0`: its coefficient of the reciprocal of the effective length, in the unit of `dvtp0` times meters; default 0. |
| `ldvtp1` | `0` | The length dependence of `dvtp1`, `ldvtp1`: its coefficient of the reciprocal of the effective length, in the unit of `dvtp1` times meters; default 0. |
| `ldwb` | `0` | The length dependence of `dwb`, `ldwb`: its coefficient of the reciprocal of the effective length, in the unit of `dwb` times meters; default 0. |
| `ldwg` | `0` | The length dependence of `dwg`, `ldwg`: its coefficient of the reciprocal of the effective length, in the unit of `dwg` times meters; default 0. |
| `legidl` | `0` | The length dependence of `egidl`, `legidl`: its coefficient of the reciprocal of the effective length, in the unit of `egidl` times meters; default 0. |
| `leigbinv` | `0` | The length dependence of `eigbinv`, `leigbinv`: its coefficient of the reciprocal of the effective length, in the unit of `eigbinv` times meters; default 0. |
| `length` | `5e-06` | The channel's drawn length, `l`, in meters; default 5 µm. |
| `leta0` | `0` | The length dependence of `eta0`, `leta0`: its coefficient of the reciprocal of the effective length, in the unit of `eta0` times meters; default 0. |
| `letab` | `0` | The length dependence of `etab`, `letab`: its coefficient of the reciprocal of the effective length, in the unit of `etab` times meters; default 0. |
| `leu` | `0` | The length dependence of `eu`, `leu`: its coefficient of the reciprocal of the effective length, in the unit of `eu` times meters; default 0. |
| `lfprout` | `0` | The length dependence of `fprout`, `lfprout`: its coefficient of the reciprocal of the effective length, in the unit of `fprout` times meters; default 0. |
| `lgamma1` | `0` | The length dependence of `gamma1`, `lgamma1`: its coefficient of the reciprocal of the effective length, in the unit of `gamma1` times meters; default 0. |
| `lgamma2` | `0` | The length dependence of `gamma2`, `lgamma2`: its coefficient of the reciprocal of the effective length, in the unit of `gamma2` times meters; default 0. |
| `lint` | `0` | The length lost on each side of the channel, `lint`, in meters; default 0. |
| `lintnoi` | `0` | The length each side of the channel loses in the flicker noise, beyond `lint`, `lintnoi`, in meters; default 0. |
| `lk1` | `0` | The length dependence of `k1`, `lk1`: its coefficient of the reciprocal of the effective length, in the unit of `k1` times meters; default 0. |
| `lk2` | `0` | The length dependence of `k2`, `lk2`: its coefficient of the reciprocal of the effective length, in the unit of `k2` times meters; default 0. |
| `lk2we` | `0` | The length dependence of `k2we`, `lk2we`: its coefficient of the reciprocal of the effective length, in the unit of `k2we` times meters; default 0. |
| `lk3` | `0` | The length dependence of `k3`, `lk3`: its coefficient of the reciprocal of the effective length, in the unit of `k3` times meters; default 0. |
| `lk3b` | `0` | The length dependence of `k3b`, `lk3b`: its coefficient of the reciprocal of the effective length, in the unit of `k3b` times meters; default 0. |
| `lketa` | `0` | The length dependence of `keta`, `lketa`: its coefficient of the reciprocal of the effective length, in the unit of `keta` times meters; default 0. |
| `lkt1` | `0` | The length dependence of `kt1`, `lkt1`: its coefficient of the reciprocal of the effective length, in the unit of `kt1` times meters; default 0. |
| `lkt1l` | `0` | The length dependence of `kt1l`, `lkt1l`: its coefficient of the reciprocal of the effective length, in the unit of `kt1l` times meters; default 0. |
| `lkt2` | `0` | The length dependence of `kt2`, `lkt2`: its coefficient of the reciprocal of the effective length, in the unit of `kt2` times meters; default 0. |
| `lku0` | `0` | The length dependence of ku0, `lku0`, in the manual's units; default 0. |
| `lku0we` | `0` | The length dependence of `ku0we`, `lku0we`: its coefficient of the reciprocal of the effective length, in the unit of `ku0we` times meters; default 0. |
| `lkvth0` | `0` | The length dependence of kvth0, `lkvth0`, in the manual's units; default 0. |
| `lkvth0we` | `0` | The length dependence of `kvth0we`, `lkvth0we`: its coefficient of the reciprocal of the effective length, in the unit of `kvth0we` times meters; default 0. |
| `ll` | `0` | The coefficient of the length dependence of the length lost, `ll`, in the manual's units; default 0. |
| `llambda` | `0` | The length dependence of `lambda`, `llambda`: its coefficient of the reciprocal of the effective length, in the unit of `lambda` times meters; default 0. |
| `llc` |  | The coefficient of the length dependence of the length lost in the charges, `llc`, in the manual's units; default `ll`. |
| `lln` | `1` | The exponent of the length dependence of the length lost, `lln`; default 1. |
| `llodku0` | `0` | The length parameter for u0 LOD effect, `llodku0`; default 0. |
| `llodvth` | `0` | The length parameter for vth LOD effect, `llodvth`; default 0. |
| `llp` | `0` | The length dependence of `lp`, `llp`: its coefficient of the reciprocal of the effective length, in the unit of `lp` times meters; default 0. |
| `llpe0` | `0` | The length dependence of `lpe0`, `llpe0`: its coefficient of the reciprocal of the effective length, in the unit of `lpe0` times meters; default 0. |
| `llpeb` | `0` | The length dependence of `lpeb`, `llpeb`: its coefficient of the reciprocal of the effective length, in the unit of `lpeb` times meters; default 0. |
| `lminv` | `0` | The length dependence of `minv`, `lminv`: its coefficient of the reciprocal of the effective length, in the unit of `minv` times meters; default 0. |
| `lmoin` | `0` | The length dependence of `moin`, `lmoin`: its coefficient of the reciprocal of the effective length, in the unit of `moin` times meters; default 0. |
| `lndep` | `0` | The length dependence of `ndep`, `lndep`: its coefficient of the reciprocal of the effective length, in the unit of `ndep` times meters; default 0. |
| `lnfactor` | `0` | The length dependence of `nfactor`, `lnfactor`: its coefficient of the reciprocal of the effective length, in the unit of `nfactor` times meters; default 0. |
| `lngate` | `0` | The length dependence of `ngate`, `lngate`: its coefficient of the reciprocal of the effective length, in the unit of `ngate` times meters; default 0. |
| `lnigbacc` | `0` | The length dependence of `nigbacc`, `lnigbacc`: its coefficient of the reciprocal of the effective length, in the unit of `nigbacc` times meters; default 0. |
| `lnigbinv` | `0` | The length dependence of `nigbinv`, `lnigbinv`: its coefficient of the reciprocal of the effective length, in the unit of `nigbinv` times meters; default 0. |
| `lnigc` | `0` | The length dependence of `nigc`, `lnigc`: its coefficient of the reciprocal of the effective length, in the unit of `nigc` times meters; default 0. |
| `lnoff` | `0` | The length dependence of `noff`, `lnoff`: its coefficient of the reciprocal of the effective length, in the unit of `noff` times meters; default 0. |
| `lnsd` | `0` | The length dependence of `nsd`, `lnsd`: its coefficient of the reciprocal of the effective length, in the unit of `nsd` times meters; default 0. |
| `lnsub` | `0` | The length dependence of `nsub`, `lnsub`: its coefficient of the reciprocal of the effective length, in the unit of `nsub` times meters; default 0. |
| `lntox` | `0` | The length dependence of `ntox`, `lntox`: its coefficient of the reciprocal of the effective length, in the unit of `ntox` times meters; default 0. |
| `lodeta0` | `1` | The eta0 shift modification factor for stress effect, `lodeta0`; default 1. |
| `lodk2` | `1` | The K2 shift modification factor for stress effect, `lodk2`; default 1. |
| `lp` | `1e-08` | The channel length exponential factor of mobility, `lp`, in meters; default 1e-08. |
| `lpclm` | `0` | The length dependence of `pclm`, `lpclm`: its coefficient of the reciprocal of the effective length, in the unit of `pclm` times meters; default 0. |
| `lpdiblc1` | `0` | The length dependence of `pdiblc1`, `lpdiblc1`: its coefficient of the reciprocal of the effective length, in the unit of `pdiblc1` times meters; default 0. |
| `lpdiblc2` | `0` | The length dependence of `pdiblc2`, `lpdiblc2`: its coefficient of the reciprocal of the effective length, in the unit of `pdiblc2` times meters; default 0. |
| `lpdiblcb` | `0` | The length dependence of `pdiblcb`, `lpdiblcb`: its coefficient of the reciprocal of the effective length, in the unit of `pdiblcb` times meters; default 0. |
| `lpdits` | `0` | The length dependence of `pdits`, `lpdits`: its coefficient of the reciprocal of the effective length, in the unit of `pdits` times meters; default 0. |
| `lpditsd` | `0` | The length dependence of `pditsd`, `lpditsd`: its coefficient of the reciprocal of the effective length, in the unit of `pditsd` times meters; default 0. |
| `lpe0` | `1.74e-07` | The equivalent length of pocket region at zero bias, `lpe0`, in meters; default 1.74e-07. |
| `lpeb` | `0` | The equivalent length of pocket region accounting for body bias, `lpeb`, in meters; default 0. |
| `lphin` | `0` | The length dependence of `phin`, `lphin`: its coefficient of the reciprocal of the effective length, in the unit of `phin` times meters; default 0. |
| `lpigcd` | `0` | The length dependence of `pigcd`, `lpigcd`: its coefficient of the reciprocal of the effective length, in the unit of `pigcd` times meters; default 0. |
| `lpoxedge` | `0` | The length dependence of `poxedge`, `lpoxedge`: its coefficient of the reciprocal of the effective length, in the unit of `poxedge` times meters; default 0. |
| `lprt` | `0` | The length dependence of `prt`, `lprt`: its coefficient of the reciprocal of the effective length, in the unit of `prt` times meters; default 0. |
| `lprwb` | `0` | The length dependence of `prwb`, `lprwb`: its coefficient of the reciprocal of the effective length, in the unit of `prwb` times meters; default 0. |
| `lprwg` | `0` | The length dependence of `prwg`, `lprwg`: its coefficient of the reciprocal of the effective length, in the unit of `prwg` times meters; default 0. |
| `lpscbe1` | `0` | The length dependence of `pscbe1`, `lpscbe1`: its coefficient of the reciprocal of the effective length, in the unit of `pscbe1` times meters; default 0. |
| `lpscbe2` | `0` | The length dependence of `pscbe2`, `lpscbe2`: its coefficient of the reciprocal of the effective length, in the unit of `pscbe2` times meters; default 0. |
| `lpvag` | `0` | The length dependence of `pvag`, `lpvag`: its coefficient of the reciprocal of the effective length, in the unit of `pvag` times meters; default 0. |
| `lrdsw` | `0` | The length dependence of `rdsw`, `lrdsw`: its coefficient of the reciprocal of the effective length, in the unit of `rdsw` times meters; default 0. |
| `lrdw` | `0` | The length dependence of `rdw`, `lrdw`: its coefficient of the reciprocal of the effective length, in the unit of `rdw` times meters; default 0. |
| `lrsw` | `0` | The length dependence of `rsw`, `lrsw`: its coefficient of the reciprocal of the effective length, in the unit of `rsw` times meters; default 0. |
| `ltvfbsdoff` | `0` | The length dependence of `tvfbsdoff`, `ltvfbsdoff`: its coefficient of the reciprocal of the effective length, in the unit of `tvfbsdoff` times meters; default 0. |
| `ltvoff` | `0` | The length dependence of `tvoff`, `ltvoff`: its coefficient of the reciprocal of the effective length, in the unit of `tvoff` times meters; default 0. |
| `lu0` | `0` | The length dependence of `u0`, `lu0`: its coefficient of the reciprocal of the effective length, in the unit of `u0` times meters; default 0. |
| `lua` | `0` | The length dependence of `ua`, `lua`: its coefficient of the reciprocal of the effective length, in the unit of `ua` times meters; default 0. |
| `lua1` | `0` | The length dependence of `ua1`, `lua1`: its coefficient of the reciprocal of the effective length, in the unit of `ua1` times meters; default 0. |
| `lub` | `0` | The length dependence of `ub`, `lub`: its coefficient of the reciprocal of the effective length, in the unit of `ub` times meters; default 0. |
| `lub1` | `0` | The length dependence of `ub1`, `lub1`: its coefficient of the reciprocal of the effective length, in the unit of `ub1` times meters; default 0. |
| `luc` | `0` | The length dependence of `uc`, `luc`: its coefficient of the reciprocal of the effective length, in the unit of `uc` times meters; default 0. |
| `luc1` | `0` | The length dependence of `uc1`, `luc1`: its coefficient of the reciprocal of the effective length, in the unit of `uc1` times meters; default 0. |
| `lud` | `0` | The length dependence of `ud`, `lud`: its coefficient of the reciprocal of the effective length, in the unit of `ud` times meters; default 0. |
| `lud1` | `0` | The length dependence of `ud1`, `lud1`: its coefficient of the reciprocal of the effective length, in the unit of `ud1` times meters; default 0. |
| `lup` | `0` | The length dependence of `up`, `lup`: its coefficient of the reciprocal of the effective length, in the unit of `up` times meters; default 0. |
| `lute` | `0` | The length dependence of `ute`, `lute`: its coefficient of the reciprocal of the effective length, in the unit of `ute` times meters; default 0. |
| `lvbm` | `0` | The length dependence of `vbm`, `lvbm`: its coefficient of the reciprocal of the effective length, in the unit of `vbm` times meters; default 0. |
| `lvbx` | `0` | The length dependence of `vbx`, `lvbx`: its coefficient of the reciprocal of the effective length, in the unit of `vbx` times meters; default 0. |
| `lvfb` | `0` | The length dependence of `vfb`, `lvfb`: its coefficient of the reciprocal of the effective length, in the unit of `vfb` times meters; default 0. |
| `lvfbcv` | `0` | The length dependence of `vfbcv`, `lvfbcv`: its coefficient of the reciprocal of the effective length, in the unit of `vfbcv` times meters; default 0. |
| `lvfbsdoff` | `0` | The length dependence of `vfbsdoff`, `lvfbsdoff`: its coefficient of the reciprocal of the effective length, in the unit of `vfbsdoff` times meters; default 0. |
| `lvoff` | `0` | The length dependence of `voff`, `lvoff`: its coefficient of the reciprocal of the effective length, in the unit of `voff` times meters; default 0. |
| `lvoffcv` | `0` | The length dependence of `voffcv`, `lvoffcv`: its coefficient of the reciprocal of the effective length, in the unit of `voffcv` times meters; default 0. |
| `lvsat` | `0` | The length dependence of `vsat`, `lvsat`: its coefficient of the reciprocal of the effective length, in the unit of `vsat` times meters; default 0. |
| `lvth0` | `0` | The length dependence of `vth0`, `lvth0`: its coefficient of the reciprocal of the effective length, in the unit of `vth0` times meters; default 0. |
| `lvtl` | `0` | The length dependence of `vtl`, `lvtl`: its coefficient of the reciprocal of the effective length, in the unit of `vtl` times meters; default 0. |
| `lw` | `0` | The coefficient of the width dependence of the length lost, `lw`, in the manual's units; default 0. |
| `lw0` | `0` | The length dependence of `w0`, `lw0`: its coefficient of the reciprocal of the effective length, in the unit of `w0` times meters; default 0. |
| `lwc` |  | The coefficient of the width dependence of the length lost in the charges, `lwc`, in the manual's units; default `lw`. |
| `lwl` | `0` | The coefficient of the area dependence of the length lost, `lwl`, in the manual's units; default 0. |
| `lwlc` |  | The coefficient of the area dependence of the length lost in the charges, `lwlc`, in the manual's units; default `lwl`. |
| `lwn` | `1` | The exponent of the width dependence of the length lost, `lwn`; default 1. |
| `lwr` | `0` | The length dependence of `wr`, `lwr`: its coefficient of the reciprocal of the effective length, in the unit of `wr` times meters; default 0. |
| `lxj` | `0` | The length dependence of `xj`, `lxj`: its coefficient of the reciprocal of the effective length, in the unit of `xj` times meters; default 0. |
| `lxn` | `0` | The length dependence of `xn`, `lxn`: its coefficient of the reciprocal of the effective length, in the unit of `xn` times meters; default 0. |
| `lxrcrg1` | `0` | The length dependence of `xrcrg1`, `lxrcrg1`: its coefficient of the reciprocal of the effective length, in the unit of `xrcrg1` times meters; default 0. |
| `lxrcrg2` | `0` | The length dependence of `xrcrg2`, `lxrcrg2`: its coefficient of the reciprocal of the effective length, in the unit of `xrcrg2` times meters; default 0. |
| `lxt` | `0` | The length dependence of `xt`, `lxt`: its coefficient of the reciprocal of the effective length, in the unit of `xt` times meters; default 0. |
| `minv` | `0` | The fitting parameter of moderate inversion in Vgsteff, `minv`; default 0. |
| `mjd` |  | The drain bottom junction capacitance grading coefficient, `mjd`; default `mjs`. |
| `mjs` | `0.5` | The source bottom junction capacitance grading coefficient, `mjs`; default 0.5. |
| `mjswd` |  | The drain sidewall junction capacitance grading coefficient, `mjswd`; default `mjsws`. |
| `mjswgd` |  | The drain (gate side) sidewall junction capacitance grading coefficient, `mjswgd`; default `mjswgs`. |
| `mjswgs` |  | The source (gate side) sidewall junction capacitance grading coefficient, `mjswgs`; default `mjsws`. |
| `mjsws` | `0.33` | The source sidewall junction capacitance grading coefficient, `mjsws`; default 0.33. |
| `mobility_multiplier` | `1` | A factor of the low-field mobility, `mulu0`; default 1. |
| `moin` | `15` | The coefficient for gate-bias dependent surface potential, `moin`; default 15. |
| `multiplier` | `1` | The number of identical transistors in parallel, `m`; default 1. |
| `ndep` |  | The channel doping concentration at the depletion edge, `ndep`, in per cubic meter; default 1.7e23 per cubic meter, or derived from `gamma1` where it alone is given. |
| `nfactor` | `1` | The subthreshold swing coefficient, `nfactor`; default 1. |
| `ngate` | `0` | The poly-gate doping concentration, `ngate`, in per cubic meter; default 0. |
| `ngcon` | `1` | The number of gate contacts, `ngcon`; default 1. |
| `nigbacc` | `1` | The parameter for Igbacc slope, `nigbacc`; default 1. |
| `nigbinv` | `3` | The parameter for Igbinv slope, `nigbinv`; default 3. |
| `nigc` | `1` | The parameter for Igc slope, `nigc`; default 1. |
| `njd` |  | The drain junction emission coefficient, `njd`; default `njs`. |
| `njs` | `1` | The source junction emission coefficient, `njs`; default 1. |
| `njts` | `20` | The non-ideality factor for bottom junction, `njts`; default 20. |
| `njtssw` | `20` | The non-ideality factor for STI sidewall junction, `njtssw`; default 20. |
| `njtsswg` | `20` | The non-ideality factor for gate-edge sidewall junction, `njtsswg`; default 20. |
| `noff` | `1` | The C-V turn-on/off parameter, `noff`; default 1. |
| `noia` |  | The flicker noise parameter, `noia`, in the manual's units; default 6.25e41 for an n-channel and 6.188e40 for a p-channel. |
| `noib` |  | The flicker noise parameter, `noib`, in the manual's units; default 3.125e26 for an n-channel and 1.5e25 for a p-channel. |
| `noic` | `8750000000` | The flicker noise parameter, `noic`, in the manual's units; default 8.75e+09. |
| `nominal_temperature` | `300.15` | The temperature the parameters were measured at, `tnom`, in kelvins; default 27 °C. |
| `nsd` | `1e+26` | The source and drain doping concentration, `nsd`, in per cubic meter; default 1e+26. |
| `nsub` | `6e+22` | The substrate doping concentration, `nsub`, in per cubic meter; default 6e+22. |
| `ntnoi` | `1` | The thermal noise parameter, `ntnoi`; default 1. |
| `ntox` | `1` | The exponent for Tox ratio, `ntox`; default 1. |
| `pa0` | `0` | The area dependence of `a0`, `pa0`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `a0` times square meters; default 0. |
| `pa1` | `0` | The area dependence of `a1`, `pa1`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `a1` times square meters; default 0. |
| `pa2` | `0` | The area dependence of `a2`, `pa2`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `a2` times square meters; default 0. |
| `pacde` | `0` | The area dependence of `acde`, `pacde`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `acde` times square meters; default 0. |
| `pagidl` | `0` | The area dependence of `agidl`, `pagidl`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `agidl` times square meters; default 0. |
| `pags` | `0` | The area dependence of `ags`, `pags`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `ags` times square meters; default 0. |
| `paigbacc` | `0` | The area dependence of `aigbacc`, `paigbacc`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `aigbacc` times square meters; default 0. |
| `paigbinv` | `0` | The area dependence of `aigbinv`, `paigbinv`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `aigbinv` times square meters; default 0. |
| `paigc` | `0` | The area dependence of `aigc`, `paigc`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `aigc` times square meters; default 0. |
| `paigsd` | `0` | The area dependence of `aigsd`, `paigsd`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `aigsd` times square meters; default 0. |
| `palpha0` | `0` | The area dependence of `alpha0`, `palpha0`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `alpha0` times square meters; default 0. |
| `palpha1` | `0` | The area dependence of `alpha1`, `palpha1`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `alpha1` times square meters; default 0. |
| `pat` | `0` | The area dependence of `at`, `pat`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `at` times square meters; default 0. |
| `pb0` | `0` | The area dependence of `b0`, `pb0`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `b0` times square meters; default 0. |
| `pb1` | `0` | The area dependence of `b1`, `pb1`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `b1` times square meters; default 0. |
| `pbd` |  | The drain junction built-in potential, `pbd`, in volts; default `pbs`. |
| `pbeta0` | `0` | The area dependence of `beta0`, `pbeta0`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `beta0` times square meters; default 0. |
| `pbgidl` | `0` | The area dependence of `bgidl`, `pbgidl`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `bgidl` times square meters; default 0. |
| `pbigbacc` | `0` | The area dependence of `bigbacc`, `pbigbacc`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `bigbacc` times square meters; default 0. |
| `pbigbinv` | `0` | The area dependence of `bigbinv`, `pbigbinv`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `bigbinv` times square meters; default 0. |
| `pbigc` | `0` | The area dependence of `bigc`, `pbigc`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `bigc` times square meters; default 0. |
| `pbigsd` | `0` | The area dependence of `bigsd`, `pbigsd`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `bigsd` times square meters; default 0. |
| `pbs` | `1` | The source junction built-in potential, `pbs`, in volts; default 1. |
| `pbswd` |  | The drain sidewall junction capacitance built in potential, `pbswd`, in volts; default `pbsws`. |
| `pbswgd` |  | The drain (gate side) sidewall junction capacitance built in potential, `pbswgd`, in volts; default `pbswgs`. |
| `pbswgs` |  | The source (gate side) sidewall junction capacitance built in potential, `pbswgs`, in volts; default `pbsws`. |
| `pbsws` | `1` | The source sidewall junction capacitance built in potential, `pbsws`, in volts; default 1. |
| `pcdsc` | `0` | The area dependence of `cdsc`, `pcdsc`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `cdsc` times square meters; default 0. |
| `pcdscb` | `0` | The area dependence of `cdscb`, `pcdscb`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `cdscb` times square meters; default 0. |
| `pcdscd` | `0` | The area dependence of `cdscd`, `pcdscd`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `cdscd` times square meters; default 0. |
| `pcf` | `0` | The area dependence of `cf`, `pcf`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `cf` times square meters; default 0. |
| `pcgdl` | `0` | The area dependence of `cgdl`, `pcgdl`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `cgdl` times square meters; default 0. |
| `pcgidl` | `0` | The area dependence of `cgidl`, `pcgidl`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `cgidl` times square meters; default 0. |
| `pcgsl` | `0` | The area dependence of `cgsl`, `pcgsl`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `cgsl` times square meters; default 0. |
| `pcigbacc` | `0` | The area dependence of `cigbacc`, `pcigbacc`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `cigbacc` times square meters; default 0. |
| `pcigbinv` | `0` | The area dependence of `cigbinv`, `pcigbinv`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `cigbinv` times square meters; default 0. |
| `pcigc` | `0` | The area dependence of `cigc`, `pcigc`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `cigc` times square meters; default 0. |
| `pcigsd` | `0` | The area dependence of `cigsd`, `pcigsd`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `cigsd` times square meters; default 0. |
| `pcit` | `0` | The area dependence of `cit`, `pcit`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `cit` times square meters; default 0. |
| `pckappad` | `0` | The area dependence of `ckappad`, `pckappad`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `ckappad` times square meters; default 0. |
| `pckappas` | `0` | The area dependence of `ckappas`, `pckappas`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `ckappas` times square meters; default 0. |
| `pclc` | `0` | The area dependence of `clc`, `pclc`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `clc` times square meters; default 0. |
| `pcle` | `0` | The area dependence of `cle`, `pcle`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `cle` times square meters; default 0. |
| `pclm` | `1.3` | The channel length modulation Coefficient, `pclm`; default 1.3. |
| `pdelta` | `0` | The area dependence of `delta`, `pdelta`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `delta` times square meters; default 0. |
| `pdiblc1` | `0.39` | The first coefficient of the drain-induced barrier lowering of the output resistance, `pdiblc1`; default 0.39. |
| `pdiblc2` | `0.0086` | The second coefficient of the drain-induced barrier lowering of the output resistance, `pdiblc2`; default 0.0086. |
| `pdiblcb` | `0` | The body-effect on drain-induced barrier lowering, `pdiblcb`, in per volt; default 0. |
| `pdits` | `0` | The coefficient for drain-induced Vth shifts, `pdits`, in per volt; default 0. |
| `pditsd` | `0` | The vds dependence of drain-induced Vth shifts, `pditsd`, in per volt; default 0. |
| `pditsl` | `0` | The length dependence of drain-induced Vth shifts, `pditsl`, in per meter; default 0. |
| `pdrout` | `0` | The area dependence of `drout`, `pdrout`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `drout` times square meters; default 0. |
| `pdsub` | `0` | The area dependence of `dsub`, `pdsub`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `dsub` times square meters; default 0. |
| `pdvt0` | `0` | The area dependence of `dvt0`, `pdvt0`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `dvt0` times square meters; default 0. |
| `pdvt0w` | `0` | The area dependence of `dvt0w`, `pdvt0w`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `dvt0w` times square meters; default 0. |
| `pdvt1` | `0` | The area dependence of `dvt1`, `pdvt1`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `dvt1` times square meters; default 0. |
| `pdvt1w` | `0` | The area dependence of `dvt1w`, `pdvt1w`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `dvt1w` times square meters; default 0. |
| `pdvt2` | `0` | The area dependence of `dvt2`, `pdvt2`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `dvt2` times square meters; default 0. |
| `pdvt2w` | `0` | The area dependence of `dvt2w`, `pdvt2w`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `dvt2w` times square meters; default 0. |
| `pdvtp0` | `0` | The area dependence of `dvtp0`, `pdvtp0`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `dvtp0` times square meters; default 0. |
| `pdvtp1` | `0` | The area dependence of `dvtp1`, `pdvtp1`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `dvtp1` times square meters; default 0. |
| `pdwb` | `0` | The area dependence of `dwb`, `pdwb`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `dwb` times square meters; default 0. |
| `pdwg` | `0` | The area dependence of `dwg`, `pdwg`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `dwg` times square meters; default 0. |
| `pegidl` | `0` | The area dependence of `egidl`, `pegidl`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `egidl` times square meters; default 0. |
| `peigbinv` | `0` | The area dependence of `eigbinv`, `peigbinv`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `eigbinv` times square meters; default 0. |
| `perimeter_model` | `1` | What the diffusions' perimeters `ps` and `pd` include, `permod`, 0 or 1; default 1. |
| `peta0` | `0` | The area dependence of `eta0`, `peta0`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `eta0` times square meters; default 0. |
| `petab` | `0` | The area dependence of `etab`, `petab`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `etab` times square meters; default 0. |
| `peu` | `0` | The area dependence of `eu`, `peu`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `eu` times square meters; default 0. |
| `pfprout` | `0` | The area dependence of `fprout`, `pfprout`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `fprout` times square meters; default 0. |
| `pgamma1` | `0` | The area dependence of `gamma1`, `pgamma1`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `gamma1` times square meters; default 0. |
| `pgamma2` | `0` | The area dependence of `gamma2`, `pgamma2`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `gamma2` times square meters; default 0. |
| `phin` | `0` | The adjusting parameter for surface potential due to non-uniform vertical doping, `phin`, in volts; default 0. |
| `pigcd` | `1` | The parameter for Igc partition, `pigcd`; default 1. |
| `pk1` | `0` | The area dependence of `k1`, `pk1`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `k1` times square meters; default 0. |
| `pk2` | `0` | The area dependence of `k2`, `pk2`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `k2` times square meters; default 0. |
| `pk2we` | `0` | The area dependence of `k2we`, `pk2we`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `k2we` times square meters; default 0. |
| `pk3` | `0` | The area dependence of `k3`, `pk3`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `k3` times square meters; default 0. |
| `pk3b` | `0` | The area dependence of `k3b`, `pk3b`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `k3b` times square meters; default 0. |
| `pketa` | `0` | The area dependence of `keta`, `pketa`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `keta` times square meters; default 0. |
| `pkt1` | `0` | The area dependence of `kt1`, `pkt1`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `kt1` times square meters; default 0. |
| `pkt1l` | `0` | The area dependence of `kt1l`, `pkt1l`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `kt1l` times square meters; default 0. |
| `pkt2` | `0` | The area dependence of `kt2`, `pkt2`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `kt2` times square meters; default 0. |
| `pku0` | `0` | The cross-term dependence of ku0, `pku0`, in the manual's units; default 0. |
| `pku0we` | `0` | The area dependence of `ku0we`, `pku0we`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `ku0we` times square meters; default 0. |
| `pkvth0` | `0` | The cross-term dependence of kvth0, `pkvth0`, in the manual's units; default 0. |
| `pkvth0we` | `0` | The area dependence of `kvth0we`, `pkvth0we`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `kvth0we` times square meters; default 0. |
| `plambda` | `0` | The area dependence of `lambda`, `plambda`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `lambda` times square meters; default 0. |
| `plp` | `0` | The area dependence of `lp`, `plp`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `lp` times square meters; default 0. |
| `plpe0` | `0` | The area dependence of `lpe0`, `plpe0`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `lpe0` times square meters; default 0. |
| `plpeb` | `0` | The area dependence of `lpeb`, `plpeb`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `lpeb` times square meters; default 0. |
| `pminv` | `0` | The area dependence of `minv`, `pminv`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `minv` times square meters; default 0. |
| `pmoin` | `0` | The area dependence of `moin`, `pmoin`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `moin` times square meters; default 0. |
| `pndep` | `0` | The area dependence of `ndep`, `pndep`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `ndep` times square meters; default 0. |
| `pnfactor` | `0` | The area dependence of `nfactor`, `pnfactor`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `nfactor` times square meters; default 0. |
| `pngate` | `0` | The area dependence of `ngate`, `pngate`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `ngate` times square meters; default 0. |
| `pnigbacc` | `0` | The area dependence of `nigbacc`, `pnigbacc`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `nigbacc` times square meters; default 0. |
| `pnigbinv` | `0` | The area dependence of `nigbinv`, `pnigbinv`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `nigbinv` times square meters; default 0. |
| `pnigc` | `0` | The area dependence of `nigc`, `pnigc`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `nigc` times square meters; default 0. |
| `pnoff` | `0` | The area dependence of `noff`, `pnoff`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `noff` times square meters; default 0. |
| `pnsd` | `0` | The area dependence of `nsd`, `pnsd`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `nsd` times square meters; default 0. |
| `pnsub` | `0` | The area dependence of `nsub`, `pnsub`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `nsub` times square meters; default 0. |
| `pntox` | `0` | The area dependence of `ntox`, `pntox`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `ntox` times square meters; default 0. |
| `poxedge` | `1` | The factor for the gate edge Tox, `poxedge`; default 1. |
| `ppclm` | `0` | The area dependence of `pclm`, `ppclm`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `pclm` times square meters; default 0. |
| `ppdiblc1` | `0` | The area dependence of `pdiblc1`, `ppdiblc1`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `pdiblc1` times square meters; default 0. |
| `ppdiblc2` | `0` | The area dependence of `pdiblc2`, `ppdiblc2`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `pdiblc2` times square meters; default 0. |
| `ppdiblcb` | `0` | The area dependence of `pdiblcb`, `ppdiblcb`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `pdiblcb` times square meters; default 0. |
| `ppdits` | `0` | The area dependence of `pdits`, `ppdits`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `pdits` times square meters; default 0. |
| `ppditsd` | `0` | The area dependence of `pditsd`, `ppditsd`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `pditsd` times square meters; default 0. |
| `pphin` | `0` | The area dependence of `phin`, `pphin`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `phin` times square meters; default 0. |
| `ppigcd` | `0` | The area dependence of `pigcd`, `ppigcd`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `pigcd` times square meters; default 0. |
| `ppoxedge` | `0` | The area dependence of `poxedge`, `ppoxedge`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `poxedge` times square meters; default 0. |
| `pprt` | `0` | The area dependence of `prt`, `pprt`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `prt` times square meters; default 0. |
| `pprwb` | `0` | The area dependence of `prwb`, `pprwb`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `prwb` times square meters; default 0. |
| `pprwg` | `0` | The area dependence of `prwg`, `pprwg`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `prwg` times square meters; default 0. |
| `ppscbe1` | `0` | The area dependence of `pscbe1`, `ppscbe1`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `pscbe1` times square meters; default 0. |
| `ppscbe2` | `0` | The area dependence of `pscbe2`, `ppscbe2`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `pscbe2` times square meters; default 0. |
| `ppvag` | `0` | The area dependence of `pvag`, `ppvag`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `pvag` times square meters; default 0. |
| `prdsw` | `0` | The area dependence of `rdsw`, `prdsw`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `rdsw` times square meters; default 0. |
| `prdw` | `0` | The area dependence of `rdw`, `prdw`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `rdw` times square meters; default 0. |
| `prsw` | `0` | The area dependence of `rsw`, `prsw`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `rsw` times square meters; default 0. |
| `prt` | `0` | The temperature coefficient of parasitic resistance, `prt`, in ohms; default 0. |
| `prwb` | `0` | The body-effect on parasitic resistance, `prwb`, in per square-root volt; default 0. |
| `prwg` | `1` | The gate-bias effect on parasitic resistance, `prwg`, in per volt; default 1. |
| `pscbe1` | `424000000` | The exponential coefficient of the substrate current's effect on the output resistance, `pscbe1`, in volts per meter; default 4.24e+08. |
| `pscbe2` | `1e-05` | The coefficient of the substrate current's effect on the output resistance, `pscbe2`, in meters per volt; default 1e-05. |
| `ptvfbsdoff` | `0` | The area dependence of `tvfbsdoff`, `ptvfbsdoff`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `tvfbsdoff` times square meters; default 0. |
| `ptvoff` | `0` | The area dependence of `tvoff`, `ptvoff`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `tvoff` times square meters; default 0. |
| `pu0` | `0` | The area dependence of `u0`, `pu0`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `u0` times square meters; default 0. |
| `pua` | `0` | The area dependence of `ua`, `pua`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `ua` times square meters; default 0. |
| `pua1` | `0` | The area dependence of `ua1`, `pua1`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `ua1` times square meters; default 0. |
| `pub` | `0` | The area dependence of `ub`, `pub`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `ub` times square meters; default 0. |
| `pub1` | `0` | The area dependence of `ub1`, `pub1`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `ub1` times square meters; default 0. |
| `puc` | `0` | The area dependence of `uc`, `puc`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `uc` times square meters; default 0. |
| `puc1` | `0` | The area dependence of `uc1`, `puc1`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `uc1` times square meters; default 0. |
| `pud` | `0` | The area dependence of `ud`, `pud`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `ud` times square meters; default 0. |
| `pud1` | `0` | The area dependence of `ud1`, `pud1`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `ud1` times square meters; default 0. |
| `pup` | `0` | The area dependence of `up`, `pup`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `up` times square meters; default 0. |
| `pute` | `0` | The area dependence of `ute`, `pute`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `ute` times square meters; default 0. |
| `pvag` | `0` | The gate dependence of output resistance parameter, `pvag`; default 0. |
| `pvbm` | `0` | The area dependence of `vbm`, `pvbm`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `vbm` times square meters; default 0. |
| `pvbx` | `0` | The area dependence of `vbx`, `pvbx`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `vbx` times square meters; default 0. |
| `pvfb` | `0` | The area dependence of `vfb`, `pvfb`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `vfb` times square meters; default 0. |
| `pvfbcv` | `0` | The area dependence of `vfbcv`, `pvfbcv`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `vfbcv` times square meters; default 0. |
| `pvfbsdoff` | `0` | The area dependence of `vfbsdoff`, `pvfbsdoff`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `vfbsdoff` times square meters; default 0. |
| `pvoff` | `0` | The area dependence of `voff`, `pvoff`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `voff` times square meters; default 0. |
| `pvoffcv` | `0` | The area dependence of `voffcv`, `pvoffcv`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `voffcv` times square meters; default 0. |
| `pvsat` | `0` | The area dependence of `vsat`, `pvsat`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `vsat` times square meters; default 0. |
| `pvth0` | `0` | The area dependence of `vth0`, `pvth0`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `vth0` times square meters; default 0. |
| `pvtl` | `0` | The area dependence of `vtl`, `pvtl`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `vtl` times square meters; default 0. |
| `pw0` | `0` | The area dependence of `w0`, `pw0`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `w0` times square meters; default 0. |
| `pwr` | `0` | The area dependence of `wr`, `pwr`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `wr` times square meters; default 0. |
| `pxj` | `0` | The area dependence of `xj`, `pxj`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `xj` times square meters; default 0. |
| `pxn` | `0` | The area dependence of `xn`, `pxn`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `xn` times square meters; default 0. |
| `pxrcrg1` | `0` | The area dependence of `xrcrg1`, `pxrcrg1`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `xrcrg1` times square meters; default 0. |
| `pxrcrg2` | `0` | The area dependence of `xrcrg2`, `pxrcrg2`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `xrcrg2` times square meters; default 0. |
| `pxt` | `0` | The area dependence of `xt`, `pxt`: its coefficient of the reciprocal of the effective length times the effective width, in the unit of `xt` times square meters; default 0. |
| `rbdb` | `50` | The resistance from the drain's body node to the bulk, `rbdb`, in ohms; default 50. |
| `rbdbx0` | `100` | The scaling of `rbdbx` at `rbodymod` 2, `rbdbx0`, in ohms; default 100. |
| `rbdby0` | `100` | The scaling of `rbdby` at `rbodymod` 2, `rbdby0`, in ohms; default 100. |
| `rbpb` | `50` | The resistance from the channel's body node to the bulk, `rbpb`, in ohms; default 50. |
| `rbpbx0` | `100` | The scaling of `rbpbx` at `rbodymod` 2, `rbpbx0`, in ohms; default 100. |
| `rbpbxl` | `0` | The body resistance RBPBX L scaling, `rbpbxl`; default 0. |
| `rbpbxnf` | `0` | The body resistance RBPBX NF scaling, `rbpbxnf`; default 0. |
| `rbpbxw` | `0` | The body resistance RBPBX W scaling, `rbpbxw`; default 0. |
| `rbpby0` | `100` | The scaling of `rbpby` at `rbodymod` 2, `rbpby0`, in ohms; default 100. |
| `rbpbyl` | `0` | The body resistance RBPBY L scaling, `rbpbyl`; default 0. |
| `rbpbynf` | `0` | The body resistance RBPBY NF scaling, `rbpbynf`; default 0. |
| `rbpbyw` | `0` | The body resistance RBPBY W scaling, `rbpbyw`; default 0. |
| `rbpd` | `50` | The resistance from the channel's body node to the drain's, `rbpd`, in ohms; default 50. |
| `rbpd0` | `50` | The scaling of `rbpd` at `rbodymod` 2, `rbpd0`, in ohms; default 50. |
| `rbpdl` | `0` | The body resistance RBPD L scaling, `rbpdl`; default 0. |
| `rbpdnf` | `0` | The body resistance RBPD NF scaling, `rbpdnf`; default 0. |
| `rbpdw` | `0` | The body resistance RBPD W scaling, `rbpdw`; default 0. |
| `rbps` | `50` | The resistance from the channel's body node to the source's, `rbps`, in ohms; default 50. |
| `rbps0` | `50` | The scaling of `rbps` at `rbodymod` 2, `rbps0`, in ohms; default 50. |
| `rbpsl` | `0` | The body resistance RBPS L scaling, `rbpsl`; default 0. |
| `rbpsnf` | `0` | The body resistance RBPS NF scaling, `rbpsnf`; default 0. |
| `rbpsw` | `0` | The body resistance RBPS W scaling, `rbpsw`; default 0. |
| `rbsb` | `50` | The resistance from the source's body node to the bulk, `rbsb`, in ohms; default 50. |
| `rbsbx0` | `100` | The scaling of `rbsbx` at `rbodymod` 2, `rbsbx0`, in ohms; default 100. |
| `rbsby0` | `100` | The scaling of `rbsby` at `rbodymod` 2, `rbsby0`, in ohms; default 100. |
| `rbsdbxl` | `0` | The body resistance RBSDBX L scaling, `rbsdbxl`; default 0. |
| `rbsdbxnf` | `0` | The body resistance RBSDBX NF scaling, `rbsdbxnf`; default 0. |
| `rbsdbxw` | `0` | The body resistance RBSDBX W scaling, `rbsdbxw`; default 0. |
| `rbsdbyl` | `0` | The body resistance RBSDBY L scaling, `rbsdbyl`; default 0. |
| `rbsdbynf` | `0` | The body resistance RBSDBY NF scaling, `rbsdbynf`; default 0. |
| `rbsdbyw` | `0` | The body resistance RBSDBY W scaling, `rbsdbyw`; default 0. |
| `rdsw` | `200` | The source and drain resistance per width, `rdsw`, in ohms; default 200. |
| `rdswmin` | `0` | The source and drain resistance per width at high gate bias, `rdswmin`, in ohms; default 0. |
| `rdw` | `100` | The drain resistance per width, `rdw`, in ohms; default 100. |
| `rdwmin` | `0` | The drain resistance per width at high Vg, `rdwmin`, in ohms; default 0. |
| `rnoia` | `0.577` | The thermal noise coefficient, `rnoia`; default 0.577. |
| `rnoib` | `0.5164` | The thermal noise coefficient, `rnoib`; default 0.5164. |
| `rsh` | `0` | The sheet resistance of the source and drain diffusions, `rsh`, in ohms; default 0. |
| `rshg` | `0.1` | The gate sheet resistance, `rshg`, in ohms; default 0.1. |
| `rsw` | `100` | The source resistance per width, `rsw`, in ohms; default 100. |
| `rswmin` | `0` | The source resistance per width at high Vg, `rswmin`, in ohms; default 0. |
| `saref` | `1e-06` | The reference distance between OD edge to poly of one side, `saref`, in meters; default 1e-06. |
| `sbref` | `1e-06` | The reference distance between OD edge to poly of the other side, `sbref`, in meters; default 1e-06. |
| `scref` | `1e-06` | The reference distance to calculate SCA, SCB and SCC, `scref`, in meters; default 1e-06. |
| `source_area` |  | The source diffusion's area, `as`, in square meters; default derived from the fingers and `dmcg`, `dmci` and `dmcgt`. |
| `source_perimeter` |  | The source diffusion's perimeter, `ps`, in meters; default derived from the fingers and `dmcg`, `dmci` and `dmcgt`. |
| `source_squares` |  | The source diffusion's length in squares, `nrs`, behind which `rsh` sets a resistance where it is positive; default none, and no resistance. |
| `steta0` | `0` | The eta0 shift factor related to stress effect on vth, `steta0`, in meters; default 0. |
| `stk2` | `0` | The K2 shift factor related to stress effect on vth, `stk2`, in meters; default 0. |
| `tcj` | `0` | The temperature coefficient of cj, `tcj`, in per kelvin; default 0. |
| `tcjsw` | `0` | The temperature coefficient of cjsw, `tcjsw`, in per kelvin; default 0. |
| `tcjswg` | `0` | The temperature coefficient of cjswg, `tcjswg`, in per kelvin; default 0. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |
| `thermal_noise_model` | `0` | The channel's thermal noise model, `tnoimod`, 0 or 1; default 0. |
| `threshold_adjustment` | `0` | A shift of the zero-bias threshold, `delvto`, in volts; default 0. |
| `tku0` | `0` | The temperature coefficient of KU0, `tku0`; default 0. |
| `tnjts` | `0` | The temperature coefficient for NJTS, `tnjts`; default 0. |
| `tnjtssw` | `0` | The temperature coefficient for NJTSSW, `tnjtssw`; default 0. |
| `tnjtsswg` | `0` | The temperature coefficient for NJTSSWG, `tnjtsswg`; default 0. |
| `tnoia` | `1.5` | The thermal noise parameter, `tnoia`; default 1.5. |
| `tnoib` | `3.5` | The thermal noise parameter, `tnoib`; default 3.5. |
| `toxe` |  | The electrical thickness of the gate oxide, `toxe`, in meters; default 3 nm, or `toxp` plus `dtox` where `toxp` alone is given. |
| `toxm` |  | The oxide thickness the parameters were extracted at, `toxm`, in meters; default `toxe` where it is given, else 3 nm. |
| `toxp` |  | The physical thickness of the gate oxide, `toxp`, in meters; default `toxe` less `dtox` where `toxe` is given, else `toxe`. |
| `toxref` | `3e-09` | The oxide thickness of reference of the gate tunneling currents, `toxref`, in meters; default 3e-09. |
| `tpb` | `0` | The temperature coefficient of pb, `tpb`, in volts per kelvin; default 0. |
| `tpbsw` | `0` | The temperature coefficient of pbsw, `tpbsw`, in volts per kelvin; default 0. |
| `tpbswg` | `0` | The temperature coefficient of pbswg, `tpbswg`, in volts per kelvin; default 0. |
| `tvfbsdoff` | `0` | The temperature parameter for vfbsdoff, `tvfbsdoff`, in per kelvin; default 0. |
| `tvoff` | `0` | The temperature parameter for voff, `tvoff`, in per kelvin; default 0. |
| `u0` |  | The low-field mobility at Tnom, `u0`, in square meters per volt-second; default 0.067 m²/(V·s) for an n-channel and 0.025 m²/(V·s) for a p-channel. |
| `ua` |  | The linear gate dependence of mobility, `ua`, in meters per volt; default 1e-9 m/V, ngspice's at `mobmod` 0. |
| `ua1` | `1e-09` | The temperature coefficient of ua, `ua1`, in meters per volt; default 1e-09. |
| `ub` | `1e-19` | The quadratic gate dependence of mobility, `ub`, in square meters per square volt; default 1e-19. |
| `ub1` | `-1e-18` | The temperature coefficient of ub, `ub1`, in square meters per square volt; default -1e-18. |
| `uc` |  | The body-bias dependence of mobility, `uc`, in meters per square volt; default -4.65e-11 m/V², ngspice's at `mobmod` 0. |
| `uc1` |  | The temperature coefficient of uc, `uc1`, in meters per square volt; default -5.6e-11 m/V², ngspice's at `mobmod` 0. |
| `ud` | `100000000000000` | The coulomb scattering factor of mobility, `ud`, in per square meter; default 1e+14. |
| `ud1` | `0` | The temperature coefficient of ud, `ud1`, in per square meter; default 0. |
| `up` | `0` | The channel length linear factor of mobility, `up`, in per square meter; default 0. |
| `ute` | `-1.5` | The temperature coefficient of mobility, `ute`; default -1.5. |
| `vbm` | `-3` | The most negative bulk-source voltage the threshold's body effect reads, `vbm`, in volts; default -3. |
| `vbx` |  | The bulk-source voltage at which the depletion reaches `xt`, `vbx`, in volts; default derived from `ndep` and `xt`. |
| `vfb` |  | The flat-band voltage, `vfb`, in volts; default derived from `vth0`, or -1 V. |
| `vfbcv` | `-1` | The flat-band voltage of the charges at `capmod` 0, `vfbcv`, in volts; default -1. |
| `vfbsdoff` | `0` | The S/D flatband voltage offset, `vfbsdoff`, in volts; default 0. |
| `voff` | `-0.08` | The threshold voltage offset, `voff`, in volts; default -0.08. |
| `voffcv` | `0` | The C-V lateral-shift parameter, `voffcv`, in volts; default 0. |
| `voffl` | `0` | The channel-length dependence of the threshold offset, `voffl`, in volt meters; default 0. |
| `vsat` | `80000` | The saturation velocity at tnom, `vsat`, in meters per second; default 80000. |
| `vth0` |  | The zero-bias threshold voltage of a long, wide channel, negative for a p-channel, `vth0`, in volts; default derived from `vfb`, `phi` and `k1`. |
| `vtl` |  | The thermal velocity, `vtl`, in meters per second; default 2e5 m/s, and no limit of the source end's velocity where it is not given. |
| `vtsd` |  | The drain bottom trap-assisted voltage dependent parameter, `vtsd`, in volts; default `vtss`. |
| `vtss` | `10` | The source bottom trap-assisted voltage dependent parameter, `vtss`, in volts; default 10. |
| `vtsswd` |  | The drain STI sidewall trap-assisted voltage dependent parameter, `vtsswd`, in volts; default `vtssws`. |
| `vtsswgd` |  | The drain gate-edge sidewall trap-assisted voltage dependent parameter, `vtsswgd`, in volts; default `vtsswgs`. |
| `vtsswgs` | `10` | The source gate-edge sidewall trap-assisted voltage dependent parameter, `vtsswgs`, in volts; default 10. |
| `vtssws` | `10` | The source STI sidewall trap-assisted voltage dependent parameter, `vtssws`, in volts; default 10. |
| `w0` | `2.5e-06` | The narrow width effect parameter, `w0`, in meters; default 2.5e-06. |
| `wa0` | `0` | The width dependence of `a0`, `wa0`: its coefficient of the reciprocal of the effective width, in the unit of `a0` times meters; default 0. |
| `wa1` | `0` | The width dependence of `a1`, `wa1`: its coefficient of the reciprocal of the effective width, in the unit of `a1` times meters; default 0. |
| `wa2` | `0` | The width dependence of `a2`, `wa2`: its coefficient of the reciprocal of the effective width, in the unit of `a2` times meters; default 0. |
| `wacde` | `0` | The width dependence of `acde`, `wacde`: its coefficient of the reciprocal of the effective width, in the unit of `acde` times meters; default 0. |
| `wagidl` | `0` | The width dependence of `agidl`, `wagidl`: its coefficient of the reciprocal of the effective width, in the unit of `agidl` times meters; default 0. |
| `wags` | `0` | The width dependence of `ags`, `wags`: its coefficient of the reciprocal of the effective width, in the unit of `ags` times meters; default 0. |
| `waigbacc` | `0` | The width dependence of `aigbacc`, `waigbacc`: its coefficient of the reciprocal of the effective width, in the unit of `aigbacc` times meters; default 0. |
| `waigbinv` | `0` | The width dependence of `aigbinv`, `waigbinv`: its coefficient of the reciprocal of the effective width, in the unit of `aigbinv` times meters; default 0. |
| `waigc` | `0` | The width dependence of `aigc`, `waigc`: its coefficient of the reciprocal of the effective width, in the unit of `aigc` times meters; default 0. |
| `waigsd` | `0` | The width dependence of `aigsd`, `waigsd`: its coefficient of the reciprocal of the effective width, in the unit of `aigsd` times meters; default 0. |
| `walpha0` | `0` | The width dependence of `alpha0`, `walpha0`: its coefficient of the reciprocal of the effective width, in the unit of `alpha0` times meters; default 0. |
| `walpha1` | `0` | The width dependence of `alpha1`, `walpha1`: its coefficient of the reciprocal of the effective width, in the unit of `alpha1` times meters; default 0. |
| `wat` | `0` | The width dependence of `at`, `wat`: its coefficient of the reciprocal of the effective width, in the unit of `at` times meters; default 0. |
| `wb0` | `0` | The width dependence of `b0`, `wb0`: its coefficient of the reciprocal of the effective width, in the unit of `b0` times meters; default 0. |
| `wb1` | `0` | The width dependence of `b1`, `wb1`: its coefficient of the reciprocal of the effective width, in the unit of `b1` times meters; default 0. |
| `wbeta0` | `0` | The width dependence of `beta0`, `wbeta0`: its coefficient of the reciprocal of the effective width, in the unit of `beta0` times meters; default 0. |
| `wbgidl` | `0` | The width dependence of `bgidl`, `wbgidl`: its coefficient of the reciprocal of the effective width, in the unit of `bgidl` times meters; default 0. |
| `wbigbacc` | `0` | The width dependence of `bigbacc`, `wbigbacc`: its coefficient of the reciprocal of the effective width, in the unit of `bigbacc` times meters; default 0. |
| `wbigbinv` | `0` | The width dependence of `bigbinv`, `wbigbinv`: its coefficient of the reciprocal of the effective width, in the unit of `bigbinv` times meters; default 0. |
| `wbigc` | `0` | The width dependence of `bigc`, `wbigc`: its coefficient of the reciprocal of the effective width, in the unit of `bigc` times meters; default 0. |
| `wbigsd` | `0` | The width dependence of `bigsd`, `wbigsd`: its coefficient of the reciprocal of the effective width, in the unit of `bigsd` times meters; default 0. |
| `wcdsc` | `0` | The width dependence of `cdsc`, `wcdsc`: its coefficient of the reciprocal of the effective width, in the unit of `cdsc` times meters; default 0. |
| `wcdscb` | `0` | The width dependence of `cdscb`, `wcdscb`: its coefficient of the reciprocal of the effective width, in the unit of `cdscb` times meters; default 0. |
| `wcdscd` | `0` | The width dependence of `cdscd`, `wcdscd`: its coefficient of the reciprocal of the effective width, in the unit of `cdscd` times meters; default 0. |
| `wcf` | `0` | The width dependence of `cf`, `wcf`: its coefficient of the reciprocal of the effective width, in the unit of `cf` times meters; default 0. |
| `wcgdl` | `0` | The width dependence of `cgdl`, `wcgdl`: its coefficient of the reciprocal of the effective width, in the unit of `cgdl` times meters; default 0. |
| `wcgidl` | `0` | The width dependence of `cgidl`, `wcgidl`: its coefficient of the reciprocal of the effective width, in the unit of `cgidl` times meters; default 0. |
| `wcgsl` | `0` | The width dependence of `cgsl`, `wcgsl`: its coefficient of the reciprocal of the effective width, in the unit of `cgsl` times meters; default 0. |
| `wcigbacc` | `0` | The width dependence of `cigbacc`, `wcigbacc`: its coefficient of the reciprocal of the effective width, in the unit of `cigbacc` times meters; default 0. |
| `wcigbinv` | `0` | The width dependence of `cigbinv`, `wcigbinv`: its coefficient of the reciprocal of the effective width, in the unit of `cigbinv` times meters; default 0. |
| `wcigc` | `0` | The width dependence of `cigc`, `wcigc`: its coefficient of the reciprocal of the effective width, in the unit of `cigc` times meters; default 0. |
| `wcigsd` | `0` | The width dependence of `cigsd`, `wcigsd`: its coefficient of the reciprocal of the effective width, in the unit of `cigsd` times meters; default 0. |
| `wcit` | `0` | The width dependence of `cit`, `wcit`: its coefficient of the reciprocal of the effective width, in the unit of `cit` times meters; default 0. |
| `wckappad` | `0` | The width dependence of `ckappad`, `wckappad`: its coefficient of the reciprocal of the effective width, in the unit of `ckappad` times meters; default 0. |
| `wckappas` | `0` | The width dependence of `ckappas`, `wckappas`: its coefficient of the reciprocal of the effective width, in the unit of `ckappas` times meters; default 0. |
| `wclc` | `0` | The width dependence of `clc`, `wclc`: its coefficient of the reciprocal of the effective width, in the unit of `clc` times meters; default 0. |
| `wcle` | `0` | The width dependence of `cle`, `wcle`: its coefficient of the reciprocal of the effective width, in the unit of `cle` times meters; default 0. |
| `wdelta` | `0` | The width dependence of `delta`, `wdelta`: its coefficient of the reciprocal of the effective width, in the unit of `delta` times meters; default 0. |
| `wdrout` | `0` | The width dependence of `drout`, `wdrout`: its coefficient of the reciprocal of the effective width, in the unit of `drout` times meters; default 0. |
| `wdsub` | `0` | The width dependence of `dsub`, `wdsub`: its coefficient of the reciprocal of the effective width, in the unit of `dsub` times meters; default 0. |
| `wdvt0` | `0` | The width dependence of `dvt0`, `wdvt0`: its coefficient of the reciprocal of the effective width, in the unit of `dvt0` times meters; default 0. |
| `wdvt0w` | `0` | The width dependence of `dvt0w`, `wdvt0w`: its coefficient of the reciprocal of the effective width, in the unit of `dvt0w` times meters; default 0. |
| `wdvt1` | `0` | The width dependence of `dvt1`, `wdvt1`: its coefficient of the reciprocal of the effective width, in the unit of `dvt1` times meters; default 0. |
| `wdvt1w` | `0` | The width dependence of `dvt1w`, `wdvt1w`: its coefficient of the reciprocal of the effective width, in the unit of `dvt1w` times meters; default 0. |
| `wdvt2` | `0` | The width dependence of `dvt2`, `wdvt2`: its coefficient of the reciprocal of the effective width, in the unit of `dvt2` times meters; default 0. |
| `wdvt2w` | `0` | The width dependence of `dvt2w`, `wdvt2w`: its coefficient of the reciprocal of the effective width, in the unit of `dvt2w` times meters; default 0. |
| `wdvtp0` | `0` | The width dependence of `dvtp0`, `wdvtp0`: its coefficient of the reciprocal of the effective width, in the unit of `dvtp0` times meters; default 0. |
| `wdvtp1` | `0` | The width dependence of `dvtp1`, `wdvtp1`: its coefficient of the reciprocal of the effective width, in the unit of `dvtp1` times meters; default 0. |
| `wdwb` | `0` | The width dependence of `dwb`, `wdwb`: its coefficient of the reciprocal of the effective width, in the unit of `dwb` times meters; default 0. |
| `wdwg` | `0` | The width dependence of `dwg`, `wdwg`: its coefficient of the reciprocal of the effective width, in the unit of `dwg` times meters; default 0. |
| `web` | `0` | The coefficient for SCB, `web`; default 0. |
| `wec` | `0` | The coefficient for SCC, `wec`; default 0. |
| `wegidl` | `0` | The width dependence of `egidl`, `wegidl`: its coefficient of the reciprocal of the effective width, in the unit of `egidl` times meters; default 0. |
| `weigbinv` | `0` | The width dependence of `eigbinv`, `weigbinv`: its coefficient of the reciprocal of the effective width, in the unit of `eigbinv` times meters; default 0. |
| `weta0` | `0` | The width dependence of `eta0`, `weta0`: its coefficient of the reciprocal of the effective width, in the unit of `eta0` times meters; default 0. |
| `wetab` | `0` | The width dependence of `etab`, `wetab`: its coefficient of the reciprocal of the effective width, in the unit of `etab` times meters; default 0. |
| `weu` | `0` | The width dependence of `eu`, `weu`: its coefficient of the reciprocal of the effective width, in the unit of `eu` times meters; default 0. |
| `wfprout` | `0` | The width dependence of `fprout`, `wfprout`: its coefficient of the reciprocal of the effective width, in the unit of `fprout` times meters; default 0. |
| `wgamma1` | `0` | The width dependence of `gamma1`, `wgamma1`: its coefficient of the reciprocal of the effective width, in the unit of `gamma1` times meters; default 0. |
| `wgamma2` | `0` | The width dependence of `gamma2`, `wgamma2`: its coefficient of the reciprocal of the effective width, in the unit of `gamma2` times meters; default 0. |
| `width` | `5e-06` | The channel's drawn width, `w`, in meters, of all its fingers together; default 5 µm. |
| `wint` | `0` | The width lost on each side of the channel, `wint`, in meters; default 0. |
| `wk1` | `0` | The width dependence of `k1`, `wk1`: its coefficient of the reciprocal of the effective width, in the unit of `k1` times meters; default 0. |
| `wk2` | `0` | The width dependence of `k2`, `wk2`: its coefficient of the reciprocal of the effective width, in the unit of `k2` times meters; default 0. |
| `wk2we` | `0` | The width dependence of `k2we`, `wk2we`: its coefficient of the reciprocal of the effective width, in the unit of `k2we` times meters; default 0. |
| `wk3` | `0` | The width dependence of `k3`, `wk3`: its coefficient of the reciprocal of the effective width, in the unit of `k3` times meters; default 0. |
| `wk3b` | `0` | The width dependence of `k3b`, `wk3b`: its coefficient of the reciprocal of the effective width, in the unit of `k3b` times meters; default 0. |
| `wketa` | `0` | The width dependence of `keta`, `wketa`: its coefficient of the reciprocal of the effective width, in the unit of `keta` times meters; default 0. |
| `wkt1` | `0` | The width dependence of `kt1`, `wkt1`: its coefficient of the reciprocal of the effective width, in the unit of `kt1` times meters; default 0. |
| `wkt1l` | `0` | The width dependence of `kt1l`, `wkt1l`: its coefficient of the reciprocal of the effective width, in the unit of `kt1l` times meters; default 0. |
| `wkt2` | `0` | The width dependence of `kt2`, `wkt2`: its coefficient of the reciprocal of the effective width, in the unit of `kt2` times meters; default 0. |
| `wku0` | `0` | The width dependence of ku0, `wku0`, in the manual's units; default 0. |
| `wku0we` | `0` | The width dependence of `ku0we`, `wku0we`: its coefficient of the reciprocal of the effective width, in the unit of `ku0we` times meters; default 0. |
| `wkvth0` | `0` | The width dependence of kvth0, `wkvth0`, in the manual's units; default 0. |
| `wkvth0we` | `0` | The width dependence of `kvth0we`, `wkvth0we`: its coefficient of the reciprocal of the effective width, in the unit of `kvth0we` times meters; default 0. |
| `wl` | `0` | The coefficient of the length dependence of the width lost, `wl`, in the manual's units; default 0. |
| `wlambda` | `0` | The width dependence of `lambda`, `wlambda`: its coefficient of the reciprocal of the effective width, in the unit of `lambda` times meters; default 0. |
| `wlc` |  | The coefficient of the length dependence of the width lost in the charges, `wlc`, in the manual's units; default `wl`. |
| `wln` | `1` | The exponent of the length dependence of the width lost, `wln`; default 1. |
| `wlod` | `0` | The width parameter for stress effect, `wlod`, in meters; default 0. |
| `wlodku0` | `0` | The width parameter for u0 LOD effect, `wlodku0`; default 0. |
| `wlodvth` | `0` | The width parameter for vth LOD effect, `wlodvth`; default 0. |
| `wlp` | `0` | The width dependence of `lp`, `wlp`: its coefficient of the reciprocal of the effective width, in the unit of `lp` times meters; default 0. |
| `wlpe0` | `0` | The width dependence of `lpe0`, `wlpe0`: its coefficient of the reciprocal of the effective width, in the unit of `lpe0` times meters; default 0. |
| `wlpeb` | `0` | The width dependence of `lpeb`, `wlpeb`: its coefficient of the reciprocal of the effective width, in the unit of `lpeb` times meters; default 0. |
| `wminv` | `0` | The width dependence of `minv`, `wminv`: its coefficient of the reciprocal of the effective width, in the unit of `minv` times meters; default 0. |
| `wmoin` | `0` | The width dependence of `moin`, `wmoin`: its coefficient of the reciprocal of the effective width, in the unit of `moin` times meters; default 0. |
| `wndep` | `0` | The width dependence of `ndep`, `wndep`: its coefficient of the reciprocal of the effective width, in the unit of `ndep` times meters; default 0. |
| `wnfactor` | `0` | The width dependence of `nfactor`, `wnfactor`: its coefficient of the reciprocal of the effective width, in the unit of `nfactor` times meters; default 0. |
| `wngate` | `0` | The width dependence of `ngate`, `wngate`: its coefficient of the reciprocal of the effective width, in the unit of `ngate` times meters; default 0. |
| `wnigbacc` | `0` | The width dependence of `nigbacc`, `wnigbacc`: its coefficient of the reciprocal of the effective width, in the unit of `nigbacc` times meters; default 0. |
| `wnigbinv` | `0` | The width dependence of `nigbinv`, `wnigbinv`: its coefficient of the reciprocal of the effective width, in the unit of `nigbinv` times meters; default 0. |
| `wnigc` | `0` | The width dependence of `nigc`, `wnigc`: its coefficient of the reciprocal of the effective width, in the unit of `nigc` times meters; default 0. |
| `wnoff` | `0` | The width dependence of `noff`, `wnoff`: its coefficient of the reciprocal of the effective width, in the unit of `noff` times meters; default 0. |
| `wnsd` | `0` | The width dependence of `nsd`, `wnsd`: its coefficient of the reciprocal of the effective width, in the unit of `nsd` times meters; default 0. |
| `wnsub` | `0` | The width dependence of `nsub`, `wnsub`: its coefficient of the reciprocal of the effective width, in the unit of `nsub` times meters; default 0. |
| `wntox` | `0` | The width dependence of `ntox`, `wntox`: its coefficient of the reciprocal of the effective width, in the unit of `ntox` times meters; default 0. |
| `wpclm` | `0` | The width dependence of `pclm`, `wpclm`: its coefficient of the reciprocal of the effective width, in the unit of `pclm` times meters; default 0. |
| `wpdiblc1` | `0` | The width dependence of `pdiblc1`, `wpdiblc1`: its coefficient of the reciprocal of the effective width, in the unit of `pdiblc1` times meters; default 0. |
| `wpdiblc2` | `0` | The width dependence of `pdiblc2`, `wpdiblc2`: its coefficient of the reciprocal of the effective width, in the unit of `pdiblc2` times meters; default 0. |
| `wpdiblcb` | `0` | The width dependence of `pdiblcb`, `wpdiblcb`: its coefficient of the reciprocal of the effective width, in the unit of `pdiblcb` times meters; default 0. |
| `wpdits` | `0` | The width dependence of `pdits`, `wpdits`: its coefficient of the reciprocal of the effective width, in the unit of `pdits` times meters; default 0. |
| `wpditsd` | `0` | The width dependence of `pditsd`, `wpditsd`: its coefficient of the reciprocal of the effective width, in the unit of `pditsd` times meters; default 0. |
| `wphin` | `0` | The width dependence of `phin`, `wphin`: its coefficient of the reciprocal of the effective width, in the unit of `phin` times meters; default 0. |
| `wpigcd` | `0` | The width dependence of `pigcd`, `wpigcd`: its coefficient of the reciprocal of the effective width, in the unit of `pigcd` times meters; default 0. |
| `wpoxedge` | `0` | The width dependence of `poxedge`, `wpoxedge`: its coefficient of the reciprocal of the effective width, in the unit of `poxedge` times meters; default 0. |
| `wprt` | `0` | The width dependence of `prt`, `wprt`: its coefficient of the reciprocal of the effective width, in the unit of `prt` times meters; default 0. |
| `wprwb` | `0` | The width dependence of `prwb`, `wprwb`: its coefficient of the reciprocal of the effective width, in the unit of `prwb` times meters; default 0. |
| `wprwg` | `0` | The width dependence of `prwg`, `wprwg`: its coefficient of the reciprocal of the effective width, in the unit of `prwg` times meters; default 0. |
| `wpscbe1` | `0` | The width dependence of `pscbe1`, `wpscbe1`: its coefficient of the reciprocal of the effective width, in the unit of `pscbe1` times meters; default 0. |
| `wpscbe2` | `0` | The width dependence of `pscbe2`, `wpscbe2`: its coefficient of the reciprocal of the effective width, in the unit of `pscbe2` times meters; default 0. |
| `wpvag` | `0` | The width dependence of `pvag`, `wpvag`: its coefficient of the reciprocal of the effective width, in the unit of `pvag` times meters; default 0. |
| `wr` | `1` | The width exponent of the source and drain resistance, `wr`; default 1. |
| `wrdsw` | `0` | The width dependence of `rdsw`, `wrdsw`: its coefficient of the reciprocal of the effective width, in the unit of `rdsw` times meters; default 0. |
| `wrdw` | `0` | The width dependence of `rdw`, `wrdw`: its coefficient of the reciprocal of the effective width, in the unit of `rdw` times meters; default 0. |
| `wrsw` | `0` | The width dependence of `rsw`, `wrsw`: its coefficient of the reciprocal of the effective width, in the unit of `rsw` times meters; default 0. |
| `wtvfbsdoff` | `0` | The width dependence of `tvfbsdoff`, `wtvfbsdoff`: its coefficient of the reciprocal of the effective width, in the unit of `tvfbsdoff` times meters; default 0. |
| `wtvoff` | `0` | The width dependence of `tvoff`, `wtvoff`: its coefficient of the reciprocal of the effective width, in the unit of `tvoff` times meters; default 0. |
| `wu0` | `0` | The width dependence of `u0`, `wu0`: its coefficient of the reciprocal of the effective width, in the unit of `u0` times meters; default 0. |
| `wua` | `0` | The width dependence of `ua`, `wua`: its coefficient of the reciprocal of the effective width, in the unit of `ua` times meters; default 0. |
| `wua1` | `0` | The width dependence of `ua1`, `wua1`: its coefficient of the reciprocal of the effective width, in the unit of `ua1` times meters; default 0. |
| `wub` | `0` | The width dependence of `ub`, `wub`: its coefficient of the reciprocal of the effective width, in the unit of `ub` times meters; default 0. |
| `wub1` | `0` | The width dependence of `ub1`, `wub1`: its coefficient of the reciprocal of the effective width, in the unit of `ub1` times meters; default 0. |
| `wuc` | `0` | The width dependence of `uc`, `wuc`: its coefficient of the reciprocal of the effective width, in the unit of `uc` times meters; default 0. |
| `wuc1` | `0` | The width dependence of `uc1`, `wuc1`: its coefficient of the reciprocal of the effective width, in the unit of `uc1` times meters; default 0. |
| `wud` | `0` | The width dependence of `ud`, `wud`: its coefficient of the reciprocal of the effective width, in the unit of `ud` times meters; default 0. |
| `wud1` | `0` | The width dependence of `ud1`, `wud1`: its coefficient of the reciprocal of the effective width, in the unit of `ud1` times meters; default 0. |
| `wup` | `0` | The width dependence of `up`, `wup`: its coefficient of the reciprocal of the effective width, in the unit of `up` times meters; default 0. |
| `wute` | `0` | The width dependence of `ute`, `wute`: its coefficient of the reciprocal of the effective width, in the unit of `ute` times meters; default 0. |
| `wvbm` | `0` | The width dependence of `vbm`, `wvbm`: its coefficient of the reciprocal of the effective width, in the unit of `vbm` times meters; default 0. |
| `wvbx` | `0` | The width dependence of `vbx`, `wvbx`: its coefficient of the reciprocal of the effective width, in the unit of `vbx` times meters; default 0. |
| `wvfb` | `0` | The width dependence of `vfb`, `wvfb`: its coefficient of the reciprocal of the effective width, in the unit of `vfb` times meters; default 0. |
| `wvfbcv` | `0` | The width dependence of `vfbcv`, `wvfbcv`: its coefficient of the reciprocal of the effective width, in the unit of `vfbcv` times meters; default 0. |
| `wvfbsdoff` | `0` | The width dependence of `vfbsdoff`, `wvfbsdoff`: its coefficient of the reciprocal of the effective width, in the unit of `vfbsdoff` times meters; default 0. |
| `wvoff` | `0` | The width dependence of `voff`, `wvoff`: its coefficient of the reciprocal of the effective width, in the unit of `voff` times meters; default 0. |
| `wvoffcv` | `0` | The width dependence of `voffcv`, `wvoffcv`: its coefficient of the reciprocal of the effective width, in the unit of `voffcv` times meters; default 0. |
| `wvsat` | `0` | The width dependence of `vsat`, `wvsat`: its coefficient of the reciprocal of the effective width, in the unit of `vsat` times meters; default 0. |
| `wvth0` | `0` | The width dependence of `vth0`, `wvth0`: its coefficient of the reciprocal of the effective width, in the unit of `vth0` times meters; default 0. |
| `wvtl` | `0` | The width dependence of `vtl`, `wvtl`: its coefficient of the reciprocal of the effective width, in the unit of `vtl` times meters; default 0. |
| `ww` | `0` | The coefficient of the width dependence of the width lost, `ww`, in the manual's units; default 0. |
| `ww0` | `0` | The width dependence of `w0`, `ww0`: its coefficient of the reciprocal of the effective width, in the unit of `w0` times meters; default 0. |
| `wwc` |  | The coefficient of the width dependence of the width lost in the charges, `wwc`, in the manual's units; default `ww`. |
| `wwl` | `0` | The coefficient of the area dependence of the width lost, `wwl`, in the manual's units; default 0. |
| `wwlc` |  | The coefficient of the area dependence of the width lost in the charges, `wwlc`, in the manual's units; default `wwl`. |
| `wwn` | `1` | The exponent of the width dependence of the width lost, `wwn`; default 1. |
| `wwr` | `0` | The width dependence of `wr`, `wwr`: its coefficient of the reciprocal of the effective width, in the unit of `wr` times meters; default 0. |
| `wxj` | `0` | The width dependence of `xj`, `wxj`: its coefficient of the reciprocal of the effective width, in the unit of `xj` times meters; default 0. |
| `wxn` | `0` | The width dependence of `xn`, `wxn`: its coefficient of the reciprocal of the effective width, in the unit of `xn` times meters; default 0. |
| `wxrcrg1` | `0` | The width dependence of `xrcrg1`, `wxrcrg1`: its coefficient of the reciprocal of the effective width, in the unit of `xrcrg1` times meters; default 0. |
| `wxrcrg2` | `0` | The width dependence of `xrcrg2`, `wxrcrg2`: its coefficient of the reciprocal of the effective width, in the unit of `xrcrg2` times meters; default 0. |
| `wxt` | `0` | The width dependence of `xt`, `wxt`: its coefficient of the reciprocal of the effective width, in the unit of `xt` times meters; default 0. |
| `xgl` | `0` | The offset of the gate's drawn length, `xgl`, in meters; default 0. |
| `xgw` | `0` | The distance from gate contact center to device edge, `xgw`, in meters; default 0. |
| `xj` | `1.5e-07` | The source and drain junctions' depth, `xj`, in meters; default 1.5e-07. |
| `xjbvd` |  | The fitting parameter for drain diode breakdown current, `xjbvd`; default `xjbvs`. |
| `xjbvs` | `1` | The fitting parameter for source diode breakdown current, `xjbvs`; default 1. |
| `xl` | `0` | The offset of the channel's length from masking and etching, `xl`, in meters; default 0. |
| `xn` | `3` | The backscattering coefficient, `xn`; default 3. |
| `xrcrg1` | `12` | The first fitting parameter of the bias-dependent gate resistance, `xrcrg1`; default 12. |
| `xrcrg2` | `1` | The second fitting parameter of the bias-dependent gate resistance, `xrcrg2`; default 1. |
| `xt` | `1.55e-07` | The depth of the doping near the surface, `xt`, in meters; default 1.55e-07. |
| `xtid` |  | The drain junction current temperature exponent, `xtid`; default `xtis`. |
| `xtis` | `3` | The source junction current temperature exponent, `xtis`; default 3. |
| `xtsd` |  | The power dependence of JTSD on temperature, `xtsd`; default `xtss`. |
| `xtss` | `0.02` | The power dependence of JTSS on temperature, `xtss`; default 0.02. |
| `xtsswd` |  | The power dependence of JTSSWD on temperature, `xtsswd`; default `xtssws`. |
| `xtsswgd` |  | The power dependence of JTSSWGD on temperature, `xtsswgd`; default `xtsswgs`. |
| `xtsswgs` | `0.02` | The power dependence of JTSSWGS on temperature, `xtsswgs`; default 0.02. |
| `xtssws` | `0.02` | The power dependence of JTSSWS on temperature, `xtssws`; default 0.02. |
| `xw` | `0` | The offset of the channel's width from masking and etching, `xw`, in meters; default 0. |

## `capacitor`

| Parameter | Default | Description |
| --- | --- | --- |
| `capacitance` | required | The capacitance, in farads, at the nominal temperature; zero or positive. |
| `nominal_temperature` | `300.15` | The temperature the capacitance was measured at, in kelvins; default 27 °C. |
| `quadratic_temperature_coefficient` | `0` | The quadratic temperature coefficient `tc2`, per kelvin squared; default 0. |
| `quadratic_voltage_coefficient` | `0` | The quadratic voltage coefficient `vc2`, per volt squared; default 0. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_coefficient` | `0` | The linear temperature coefficient `tc1`, per kelvin; default 0. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |
| `voltage_coefficient` | `0` | The linear voltage coefficient `vc1`, per volt; default 0. |

## `coupled_inductors`

| Parameter | Default | Description |
| --- | --- | --- |
| `windings` | required | Each winding, as an inductor's parameters; at least two. |
| `windings.*.inductance` | required | The inductance, in henries, at the nominal temperature; positive. |
| `windings.*.nominal_temperature` | `300.15` | The temperature the inductance was measured at, in kelvins; default 27 °C. |
| `windings.*.quadratic_temperature_coefficient` | `0` | The quadratic temperature coefficient `tc2`, per kelvin squared; default 0. |
| `windings.*.temperature_coefficient` | `0` | The linear temperature coefficient `tc1`, per kelvin; default 0. |
| `couplings` | `[]` | The couplings between pairs of windings. |
| `couplings.*.coefficient` | required | The coupling coefficient, from -1 to 1. |
| `couplings.*.windings` | required | The two windings, by their numbers from 1. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |

## `current_controlled_current_source`

| Parameter | Default | Description |
| --- | --- | --- |
| `gain` | required | The ratio of the output current to the control current. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |

## `current_controlled_memristor`

| Parameter | Default | Description |
| --- | --- | --- |
| `charge` | required | The charge that carries the state across its whole range, in coulombs; positive. |
| `off_resistance` | required | The resistance fully off, at a state of 1, in ohms; above the on resistance. |
| `on_resistance` | required | The resistance fully on, at a state of 0, in ohms; positive. |
| `law` | `linear` | How the resistance follows the state; default linear. One of `linear`, `exponential`. |
| `state` | `1` | The state it was made in, from 0, fully on, to 1, fully off; default 1. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |
| `window` | `0.01` | The width of the window that slows the state at either bound, a fraction of its range; positive; default 0.01. |

## `current_controlled_switch`

| Parameter | Default | Description |
| --- | --- | --- |
| `off_current` | `0` | The control current at which the switch is off, in amperes; default 0. |
| `off_resistance` | `1000000` | The resistance with the control at or past the off current, in ohms; default 1e6. |
| `on_current` | `0.001` | The control current at which the switch is on, in amperes; default 1 mA. |
| `on_resistance` | `1` | The resistance with the control at or past the on current, in ohms; default 1. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |

## `current_controlled_voltage_source`

| Parameter | Default | Description |
| --- | --- | --- |
| `transresistance` | required | The ratio of the output voltage to the control current, in ohms. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |

## `current_source`

| Parameter | Default | Description |
| --- | --- | --- |
| `waveform` | required | The source's level over time, in volts or amperes. |
| `waveform.dc` |  | A constant level. |
| `waveform.pulse` |  | A trapezoidal pulse, repeated each period, if it has one. |
| `waveform.pulse.delay` | `0` | The time the first pulse starts its rise, in seconds. |
| `waveform.pulse.fall` | required | The time each pulse takes to fall, in seconds; positive. |
| `waveform.pulse.initial` | required | The level before the delay and between pulses. |
| `waveform.pulse.period` |  | The time from one pulse's start to the next's, in seconds; without it, a single pulse. |
| `waveform.pulse.pulsed` | required | The level at the top of each pulse. |
| `waveform.pulse.rise` | required | The time each pulse takes to rise, in seconds; positive. |
| `waveform.pulse.width` | required | The time each pulse holds its top, in seconds. |
| `waveform.sine` |  | A sine, damped from its delay on. |
| `waveform.sine.amplitude` | required | The sine's amplitude at its delay. |
| `waveform.sine.damping` | `0` | The rate at which the amplitude decays from the delay on, per second. |
| `waveform.sine.delay` | `0` | The time the sine starts, in seconds, before which it holds its value then. |
| `waveform.sine.frequency` | required | The sine's frequency, in hertz. |
| `waveform.sine.offset` | required | The level the sine swings about. |
| `waveform.sine.phase` | `0` | The phase at the delay, in radians. |
| `waveform.piecewise` |  | Straight lines between points, holding the first and last levels outside them. |
| `waveform.piecewise.points` | required | Each point's time and level, in strictly increasing time; at least one. |
| `waveform.exponential` |  | A rise toward a level and a fall back, each exponential, as SPICE's `EXP`. |
| `waveform.exponential.fall_delay` | required | The time the fall starts, in seconds; at or after the rise delay. |
| `waveform.exponential.fall_time_constant` | required | The fall's time constant, in seconds; positive. |
| `waveform.exponential.initial` | required | The level before the rise delay. |
| `waveform.exponential.pulsed` | required | The level the rise tends to. |
| `waveform.exponential.rise_delay` | `0` | The time the rise starts, in seconds. |
| `waveform.exponential.rise_time_constant` | required | The rise's time constant, in seconds; positive. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |

## `diode`

| Parameter | Default | Description |
| --- | --- | --- |
| `breakdown_current` | `0.001` | The current at the breakdown voltage, in amperes, at least 0; default 1e-3. |
| `breakdown_emission_coefficient` |  | The breakdown emission coefficient; the emission coefficient by default. |
| `breakdown_temperature_coefficient` | `0` | How fast the breakdown voltage falls with temperature: in volts per kelvin at temperature model 0, and per kelvin, relative to the breakdown voltage, at 1 and 2; default 0. |
| `breakdown_voltage` |  | The reverse breakdown voltage, in volts, at least 0; without it, the junction never breaks down. |
| `capacitance_temperature_model` | `0` | How the junction capacitances and potentials follow the temperature, SPICE's `TLEVC`; default 0. |
| `depletion_coefficient` | `0.5` | The fraction of the junction potential above which the depletion capacitance is extended linearly, below 1; default 0.5. |
| `emission_coefficient` | `1` | The emission coefficient; default 1. |
| `energy_gap` |  | The energy gap that scales the saturation currents with temperature, in volts; by default 1.11, silicon's, and at temperature model 2, where it is the gap at 0 K, 1.16. |
| `energy_gap_alpha` | `0.000702` | Varshni's `α` of the energy gap at temperature model 2, in volts per kelvin; default 7.02e-4, silicon's. |
| `energy_gap_beta` | `1108` | Varshni's `β` of the energy gap at temperature model 2, in kelvins; default 1108, silicon's. |
| `flicker_coefficient` | `0` | The flicker noise coefficient, `KF`; default 0, no flicker noise. |
| `flicker_exponent` | `1` | The power of the current in the flicker noise, `AF`; default 1. |
| `forward_knee_current` |  | The forward knee current, in amperes, above which high injection halves the junction current's slope; without it, none. |
| `grading_coefficient` | `0.5` | The grading coefficient of the junction, below 1; default 0.5. |
| `grading_coefficient_quadratic_temperature_coefficient` | `0` | The grading coefficient's quadratic temperature coefficient, per kelvin squared; default 0. |
| `grading_coefficient_temperature_coefficient` | `0` | The grading coefficient's linear temperature coefficient, per kelvin; default 0. |
| `junction_capacitance` | `0` | The zero-bias junction capacitance, in farads; default 0. |
| `junction_capacitance_temperature_coefficient` | `0` | The junction capacitance's linear temperature coefficient at capacitance temperature model 1, per kelvin; default 0. |
| `junction_potential` | `1` | The junction potential, in volts; default 1. |
| `junction_potential_temperature_coefficient` | `0` | How fast the junction potential falls with temperature at capacitance temperature model 1, in volts per kelvin; default 0. |
| `nominal_temperature` | `300.15` | The temperature the parameters were measured at, in kelvins; default 27 °C. |
| `recombination_emission_coefficient` | `2` | The recombination current's emission coefficient; default 2. |
| `recombination_saturation_current` |  | The saturation current of the recombination in the depletion region, in amperes; without it, no recombination current. |
| `reverse_knee_current` |  | The reverse knee current, in amperes, beyond which the reverse current's growth slows alike; without it, none. |
| `saturation_current` | `1e-14` | The saturation current, in amperes; default 1e-14. |
| `saturation_current_temperature_exponent` | `3` | The exponent of the temperature ratio in the saturation currents' scaling; default 3. |
| `series_resistance` | `0` | The series resistance, in ohms; default 0. |
| `series_resistance_quadratic_temperature_coefficient` | `0` | The series resistances' quadratic temperature coefficient, per kelvin squared; default 0. |
| `series_resistance_temperature_coefficient` | `0` | The series resistances' linear temperature coefficient, per kelvin; default 0. |
| `sidewall_depletion_coefficient` | `0.5` | The fraction of the sidewall junction's potential above which its depletion capacitance is extended linearly, below 1; default 0.5. |
| `sidewall_emission_coefficient` |  | The sidewall junction's emission coefficient; the emission coefficient by default. |
| `sidewall_forward_knee_current` |  | The sidewall junction's forward knee current, in amperes; without it, none. |
| `sidewall_grading_coefficient` | `0.33` | The sidewall junction's grading coefficient, below 1; default 0.33. |
| `sidewall_junction_capacitance` | `0` | The sidewall junction's zero-bias capacitance, in farads; default 0. |
| `sidewall_junction_capacitance_temperature_coefficient` | `0` | The sidewall junction capacitance's linear temperature coefficient at capacitance temperature model 1, per kelvin; default 0. |
| `sidewall_junction_potential` | `1` | The sidewall junction's potential, in volts; default 1. |
| `sidewall_junction_potential_temperature_coefficient` | `0` | How fast the sidewall junction potential falls with temperature at capacitance temperature model 1, in volts per kelvin; default 0. |
| `sidewall_saturation_current` | `0` | The sidewall junction's saturation current, in amperes; default 0, no sidewall current. |
| `sidewall_series_resistance` |  | The sidewall junction's own series resistance, in ohms, from the anode; without it, the sidewall sits beside the junction, behind the series resistance. |
| `sidewall_tunneling_saturation_current` | `0` | The sidewall's tunneling saturation current, in amperes; default 0. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_model` | `0` | How the saturation currents and the band gap follow the temperature, SPICE's `TLEV`; default 0. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |
| `thermal` | `false` | Whether the device heats by the power it dissipates, through a terminal `thermal` joined to a thermal net, whose temperature above the ambient it runs at. |
| `transit_time` | `0` | The transit time, in seconds; default 0. |
| `transit_time_quadratic_temperature_coefficient` | `0` | The transit time's quadratic temperature coefficient, per kelvin squared; default 0. |
| `transit_time_temperature_coefficient` | `0` | The transit time's linear temperature coefficient, per kelvin; default 0. |
| `tunneling_emission_coefficient` | `30` | The tunneling current's emission coefficient; default 30. |
| `tunneling_energy_gap_factor` | `1` | The factor of the energy gap in the tunneling saturation currents' scaling; default 1. |
| `tunneling_saturation_current` | `0` | The tunneling current's saturation current, in amperes; default 0, no tunneling. |
| `tunneling_saturation_current_temperature_exponent` | `3` | The exponent of the temperature ratio in the tunneling saturation currents' scaling; default 3. |

## `foundry_mosfet`

| Parameter | Default | Description |
| --- | --- | --- |
| `length` | required | The channel's length, in metres; positive. |
| `model` | required | The kit's model of the transistor, such as `sky130_fd_pr__nfet_01v8`. |
| `pdk` | required | The process design kit, such as `sky130`. |
| `width` | required | The channel's width, all its fingers together, in metres; positive. |
| `fingers` | `1` | The fingers the width is split into; 1 unless given. |
| `multiplier` | `1` | The transistors in parallel; 1 unless given. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |

## `heat_source`

| Parameter | Default | Description |
| --- | --- | --- |
| `power` | required | The heat flow, in watts. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |

## `inductor`

| Parameter | Default | Description |
| --- | --- | --- |
| `inductance` | required | The inductance, in henries, at the nominal temperature; positive. |
| `nominal_temperature` | `300.15` | The temperature the inductance was measured at, in kelvins; default 27 °C. |
| `quadratic_temperature_coefficient` | `0` | The quadratic temperature coefficient `tc2`, per kelvin squared; default 0. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_coefficient` | `0` | The linear temperature coefficient `tc1`, per kelvin; default 0. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |

## `jfet`

| Parameter | Default | Description |
| --- | --- | --- |
| `channel` | `n` | The channel; default `n`. One of `n`, `p`. |
| `channel_length_modulation` | `0` | The channel-length modulation, `λ`, per volt; default 0. |
| `depletion_coefficient` | `0.5` | The fraction of the junction potential above which the depletion capacitance is extended linearly, below 1; default 0.5. |
| `doping_tail` | `1` | The doping tail, `b`: 1 for the plain square law; default 1. |
| `drain_resistance` | `0` | The drain resistance, in ohms; default 0. |
| `energy_gap` | `1.11` | The energy gap that scales the saturation current with temperature, in volts; default 1.11, silicon's. |
| `flicker_coefficient` | `0` | The flicker noise coefficient of the drain current, `KF`; default 0, no flicker noise. |
| `flicker_exponent` | `1` | The power of the drain current in the flicker noise, `AF`; default 1. |
| `gate_drain_capacitance` | `0` | The zero-bias gate-drain capacitance, in farads; default 0. |
| `gate_source_capacitance` | `0` | The zero-bias gate-source capacitance, in farads; default 0. |
| `junction_potential` | `1` | The gate junctions' potential, `pb`, in volts; default 1. |
| `nominal_temperature` | `300.15` | The temperature the parameters were measured at, in kelvins; default 27 °C. |
| `saturation_current` | `1e-14` | The gate junctions' saturation current, in amperes; default 1e-14. |
| `saturation_current_temperature_exponent` | `0` | The exponent of the temperature ratio in the saturation current's scaling; default 0, as ngspice scales a JFET's by it only when it is given. |
| `source_resistance` | `0` | The source resistance, in ohms; default 0. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |
| `thermal` | `false` | Whether the device heats by the power it dissipates, through a terminal `thermal` joined to a thermal net, whose temperature above the ambient it runs at. |
| `threshold_voltage` | `-2` | The threshold voltage, `vto`, in volts; default −2. |
| `threshold_voltage_temperature_coefficient` | `0` | How fast the threshold voltage rises with temperature, in volts per kelvin; default 0. |
| `transconductance_parameter` | `0.0001` | The transconductance parameter, `β`, in amperes per square volt; default 1e-4. |
| `transconductance_parameter_temperature_exponent` | `0` | The exponent of the temperature ratio that scales the transconductance parameter; default 0. |

## `logic_cell`

| Parameter | Default | Description |
| --- | --- | --- |
| `inputs` | required | The inputs' names, in the order of the truth tables' bits. |
| `outputs` | required | The outputs, each with its truth table. |
| `outputs.*.name` | required | The output's name. |
| `outputs.*.table` | required | The output's level for each combination of the inputs, a `0` or a `1` per row. Row `k` holds the inputs whose bits make `k`, the first input the most significant: for inputs `a` and `b`, the rows are `ab` = 00, 01, 10 and 11, so a NAND is `1110`. |
| `timing` | required | How the outputs follow the inputs in time. |
| `timing.fixed` |  | The same delays after any input changes, whatever the load. |
| `timing.fixed.fall` | required | The delay of an output falling to 0, in seconds. |
| `timing.fixed.rise` | required | The delay of an output rising to 1, in seconds. |
| `timing.fixed.transition` | required | The outputs' transition, the time an edge takes to swing from ground to supply, in seconds. |
| `timing.characterized` |  | Delays and transitions that depend on the arc, the input transition and the load. |
| `timing.characterized.arcs` | required | Every arc: each change of an input that changes an output. |
| `timing.characterized.arcs.*.before` | required | Every input's level before the change, a `0` or a `1` each, in the order of the inputs. |
| `timing.characterized.arcs.*.delay` | required | The delay, in seconds, for each input transition, then for each load: one row per transition. |
| `timing.characterized.arcs.*.input` | required | The input that changes. |
| `timing.characterized.arcs.*.output` | required | The output that follows. |
| `timing.characterized.arcs.*.transition` | required | The output's transition, in seconds, laid out as the delay. |
| `timing.characterized.loads` | required | The output loads the tables were measured at, in farads, rising. |
| `timing.characterized.slews` | required | The input transitions the tables were measured at, in seconds, rising. |
| `timing.characterized.supply` |  | The voltage from ground to supply the tables were measured at, in volts. |
| `electrical` |  | With them, the cell has terminals `supply` and `ground`, and its pins may join electrical nets. |
| `electrical.ground` | `ground` | The name of the ground terminal; `ground` unless given. |
| `electrical.input_capacitances` | required | The capacitance of each input, in farads, in the order of the inputs. |
| `electrical.supply` | `supply` | The name of the supply terminal; `supply` unless given. |
| `electrical.threshold` | `0.5` | The fraction of the supply, from ground, at which an input switches; one half unless given. |

## `logic_source`

| Parameter | Default | Description |
| --- | --- | --- |
| `initial` | required | The level at time zero. One of `0`, `1`, `x`. |
| `transition` | required | The edges' transition, the time each takes to swing from ground to supply, in seconds. |
| `changes` | `[]` | Each later change: its time, in seconds, and the new level, in time order. |

## `mosfet`

| Parameter | Default | Description |
| --- | --- | --- |
| `body_effect` |  | The body-effect coefficient, `γ`, in square-root volts; default 0, or derived from the substrate doping. |
| `bottom_capacitance` | `0` | The bulk junctions' zero-bias bottom capacitance per area, `cj`, in farads per square meter; default 0. |
| `bottom_grading` | `0.5` | The bottom capacitance's grading coefficient, `mj`, below 1; default 0.5. |
| `bulk_drain_capacitance` |  | The zero-bias bulk-drain capacitance, in farads; default none, for the bottom capacitance's. |
| `bulk_source_capacitance` |  | The zero-bias bulk-source capacitance, in farads; default none, for the bottom capacitance's. |
| `channel` | `n` | The channel; default `n`. One of `n`, `p`. |
| `channel_length_modulation` |  | The channel-length modulation, `λ`, per volt, level 1's alone; default 0. |
| `depletion_coefficient` | `0.5` | The fraction of the junction potential above which the depletion capacitances are extended linearly, below 1; default 0.5. |
| `drain_area` | `0` | The drain diffusion's area, in square meters; default 0. |
| `drain_perimeter` | `0` | The drain diffusion's perimeter, in meters; default 0. |
| `drain_resistance` |  | The drain resistance, in ohms; default none, for the sheet resistance's. |
| `drain_squares` | `1` | The drain diffusion's length in squares, `nrd`; default 1. |
| `fast_surface_state_density` |  | Level 3: the fast surface state density, `nfs`, per square meter, which turns on weak inversion; default 0. |
| `flicker_coefficient` | `0` | The flicker noise coefficient of the drain current, `KF`; default 0, no flicker noise. |
| `flicker_exponent` | `1` | The power of the drain current in the flicker noise, `AF`; default 1. |
| `gate_bulk_overlap` | `0` | The gate-bulk overlap capacitance per length, in farads per meter; default 0. |
| `gate_drain_overlap` | `0` | The gate-drain overlap capacitance per width, in farads per meter; default 0. |
| `gate_material` | `opposite_doping` | The gate's material, which sets a derived `vto`; default `opposite_doping`. One of `opposite_doping`, `same_doping`, `aluminum`. |
| `gate_source_overlap` | `0` | The gate-source overlap capacitance per width, in farads per meter; default 0. |
| `junction_depth` |  | Level 3: the junction depth, `xj`, in meters, which a substrate doping turns into short channels' body effect; default 0. |
| `junction_potential` | `0.8` | The bulk junctions' potential, `pb`, in volts; default 0.8. |
| `lateral_diffusion` | `0` | The lateral diffusion, `ld`, by which the channel falls short of its drawn length at each end, in meters, or past it where negative, as ngspice takes it; default 0. |
| `length` | `0.0001` | The channel's drawn length, `l`, in meters; default 100 µm. |
| `length_adjustment` |  | Level 3: the length the mask adds, `xl`, in meters; default 0. |
| `level` | `1` | The model's level, 1 or 3; default 1. |
| `max_drift_velocity` |  | Level 3: the carriers' maximum drift velocity, `vmax`, in meters per second; default none, for no velocity saturation. |
| `mobility_modulation` |  | Level 3: the gate field's degradation of the mobility, `theta`, per volt; default 0. |
| `multiplier` | `1` | The number of identical transistors in parallel, `m`; default 1. |
| `narrow_width_factor` |  | Level 3: the width's effect on the threshold, `delta`; default 0. |
| `nominal_temperature` | `300.15` | The temperature the parameters were measured at, in kelvins; default 27 °C. |
| `oxide_thickness` |  | The gate oxide's thickness, `tox`, in meters; default none at level 1, which leaves the gate without Meyer's capacitances, and 0.1 µm at level 3 and for flicker noise. |
| `saturation_current` | `1e-14` | The bulk junctions' saturation current, in amperes; default 1e-14. |
| `saturation_current_density` | `0` | The bulk junctions' saturation current per area, in amperes per square meter, which replaces the saturation current when it and both areas are nonzero; default 0. |
| `saturation_field_factor` |  | Level 3: the saturation field factor, `kappa`; default 0.2, and 0 for no channel shortening, as ngspice reads it. |
| `sheet_resistance` | `0` | The drain and source diffusions' sheet resistance, in ohms per square; default 0. |
| `sidewall_capacitance` | `0` | The bulk junctions' zero-bias sidewall capacitance per length of perimeter, `cjsw`, in farads per meter; default 0. |
| `sidewall_grading` |  | The sidewall capacitance's grading coefficient, `mjsw`, below 1; default 0.5 at level 1 and 0.33 at level 3. |
| `source_area` | `0` | The source diffusion's area, in square meters; default 0. |
| `source_perimeter` | `0` | The source diffusion's perimeter, in meters; default 0. |
| `source_resistance` |  | The source resistance, in ohms; default none, for the sheet resistance's. |
| `source_squares` | `1` | The source diffusion's length in squares, `nrs`; default 1. |
| `static_feedback` |  | Level 3: the drain's static feedback on the threshold, `eta`; default 0. |
| `substrate_doping` |  | The substrate doping, `nsub`, per cubic meter, from which `vto`, `γ` and `φ` derive unless given; default none. It needs an oxide thickness. |
| `surface_mobility` | `0.06` | The surface mobility, `uo`, in square meters per volt-second, from which `kp` derives; default 0.06. |
| `surface_potential` |  | The surface potential, `φ`, in volts; default 0.6, or derived from the substrate doping. |
| `surface_state_density` | `0` | The surface state density, `nss`, per square meter, which shifts a derived `vto`; default 0. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |
| `thermal` | `false` | Whether the device heats by the power it dissipates, through a terminal `thermal` joined to a thermal net, whose temperature above the ambient it runs at. |
| `threshold_adjustment` |  | Level 3: a shift of the threshold, `delvto`, in volts; default 0. |
| `threshold_voltage` |  | The zero-bias threshold voltage, `vto`, in volts; default 0, or derived from the substrate doping. |
| `transconductance_parameter` |  | The transconductance parameter, `kp`, in amperes per square volt; default 2e-5, or the surface mobility times the oxide capacitance per area, given an oxide thickness. |
| `width` | `0.0001` | The channel's width, `w`, in meters; default 100 µm. |
| `width_adjustment` |  | Level 3: the width the mask adds, `xw`, in meters; default 0. |
| `width_narrowing` |  | Level 3: the width lost to diffusion at each side, `wd`, in meters; default 0. |

## `photocell`

| Parameter | Default | Description |
| --- | --- | --- |
| `resistance` | required | The resistance at the reference illuminance, in ohms; positive. |
| `dark_resistance` | `1000000` | The resistance in darkness, in ohms; positive; default 1e6, the GL5528's. |
| `gamma` | `0.6` | How steeply the resistance falls with the illuminance, `γ`, the slope of its logarithm against the illuminance's; positive; default 0.6, which with the default dark resistance is Senba's GL5528, the most common cell of its GL55 series. |
| `illuminance` | `10` | The reference illuminance, in lux; positive; default 10, as datasheets give it, such as Senba's of its GL55 series. |
| `response_time` | `0` | The time constant of the response, in seconds; zero or more; default 0. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |

## `resistor`

| Parameter | Default | Description |
| --- | --- | --- |
| `resistance` | required | The resistance, in ohms, at the nominal temperature; positive. |
| `exponential_temperature_coefficient` |  | The exponential temperature coefficient `tce`, in percent per kelvin: given, the resistance is multiplied by `1.01^(tce · ΔT)` instead of the polynomial. |
| `flicker_coefficient` | `0` | The flicker noise coefficient, `KF`; default 0, no flicker noise. |
| `flicker_exponent` | `1` | The power of the current in the flicker noise, `AF`; default 1. |
| `nominal_temperature` | `300.15` | The temperature the resistance was measured at, in kelvins; default 27 °C. |
| `quadratic_temperature_coefficient` | `0` | The quadratic temperature coefficient `tc2`, per kelvin squared; default 0. |
| `quadratic_voltage_coefficient` | `0` | The quadratic voltage coefficient `vc2`, per volt squared; default 0. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_coefficient` | `0` | The linear temperature coefficient `tc1`, per kelvin; default 0. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |
| `thermal` | `false` | Whether the device heats by the power it dissipates, through a terminal `thermal` joined to a thermal net, whose temperature above the ambient it runs at. |
| `voltage_coefficient` | `0` | The linear voltage coefficient `vc1`, per volt; default 0. |

## `thermal_capacitance`

| Parameter | Default | Description |
| --- | --- | --- |
| `capacitance` | required | The heat capacity, in joules per kelvin; zero or positive. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |

## `thermal_resistance`

| Parameter | Default | Description |
| --- | --- | --- |
| `resistance` | required | The resistance, in kelvins per watt; positive. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |

## `transmission_line`

| Parameter | Default | Description |
| --- | --- | --- |
| `characteristic_impedance` | required | The characteristic impedance `Z₀`, in ohms, SPICE's `Z0`; positive. |
| `delay` | required | The time a wave takes from one port to the other, in seconds, SPICE's `TD`; positive. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |

## `vbic`

| Parameter | Default | Description |
| --- | --- | --- |
| `activation_energy` | `1.12` | The activation energy of the transport saturation current and of the epi doping, `EA`, in volts; default 1.12. |
| `avalanche_coefficient` | `0` | The weak avalanche's coefficient, `AVC1`, per volt; default 0, no avalanche. |
| `avalanche_exponent_coefficient` | `0` | The weak avalanche's exponent coefficient, `AVC2`, in volts to the power `1 − MC`; default 0. |
| `avalanche_temperature_coefficient` | `0` | The linear temperature coefficient of the avalanche's exponent coefficient, `TAVC`, per kelvin; default 0. |
| `base_charge` | `vbic` | How the base charge is formed, as `QBM` selects it; default `vbic`. One of `vbic`, `gummel_poon`. |
| `base_collector_capacitance` | `0` | The zero-bias intrinsic base-collector depletion capacitance, `CJC`, in farads; default 0. |
| `base_collector_grading` | `0.33` | The base-collector grading coefficient, `MC`, below 1; default 0.33. |
| `base_collector_overlap_capacitance` | `0` | The base-collector overlap capacitance, `CBCO`, in farads, between the terminals; default 0. |
| `base_collector_potential` | `0.75` | The base-collector built-in potential, `PC`, in volts; default 0.75. |
| `base_collector_smoothing` | `-0.5` | The base-collector capacitances' smoothing, `AJC`, in volts; default −0.5. |
| `base_emitter_breakdown_current` | `1e-06` | The base-emitter breakdown current, `IBBE`, in amperes; default 1e-6. |
| `base_emitter_breakdown_emission_coefficient` | `1` | The base-emitter breakdown's emission coefficient, `NBBE`; default 1. |
| `base_emitter_breakdown_voltage` |  | The base-emitter breakdown voltage, `VBBE`, in volts; without it, the junction never breaks down. |
| `base_emitter_capacitance` | `0` | The zero-bias base-emitter depletion capacitance, `CJE`, in farads; default 0. |
| `base_emitter_grading` | `0.33` | The base-emitter grading coefficient, `ME`, below 1; default 0.33. |
| `base_emitter_overlap_capacitance` | `0` | The base-emitter overlap capacitance, `CBEO`, in farads, between the terminals; default 0. |
| `base_emitter_potential` | `0.75` | The base-emitter built-in potential, `PE`, in volts; default 0.75. |
| `base_emitter_smoothing` | `-0.5` | The base-emitter capacitance's smoothing, `AJE`, in volts: SPICE's linear extension above `FC · PE` at zero or below; default −0.5. |
| `breakdown_emission_temperature_coefficient` | `0` | The linear temperature coefficient of the breakdown's emission coefficient, `TNBBE`, per kelvin; default 0. |
| `breakdown_voltage_quadratic_temperature_coefficient` | `0` | The quadratic temperature coefficient of the base-emitter breakdown voltage, `TVBBE2`, per square kelvin; default 0. |
| `breakdown_voltage_temperature_coefficient` | `0` | The linear temperature coefficient of the base-emitter breakdown voltage, `TVBBE1`, per kelvin; default 0. |
| `collector_substrate_capacitance` | `0` | The zero-bias collector-substrate depletion capacitance, `CJCP`, in farads; default 0. |
| `collector_substrate_fixed_capacitance` | `0` | The fixed collector-substrate capacitance, `CCSO`, in farads; default 0. |
| `collector_substrate_grading` | `0.33` | The collector-substrate grading coefficient, `MS`, below 1; default 0.33. |
| `collector_substrate_potential` | `0.75` | The collector-substrate built-in potential, `PS`, in volts; default 0.75. |
| `collector_substrate_smoothing` | `-0.5` | The collector-substrate capacitance's smoothing, `AJS`, in volts; default −0.5. |
| `depletion_coefficient` | `0.9` | The fraction of each junction's potential where its depletion capacitance's smoothing sets in, `FC`, below 1; default 0.9. |
| `emission_coefficient_temperature_coefficient` | `0` | The linear temperature coefficient of the forward and reverse emission coefficients, `TNF`, per kelvin; default 0. |
| `emitter_resistance` | `0` | The emitter resistance, `RE`, in ohms; default 0. |
| `emitter_resistance_temperature_exponent` | `0` | The exponent of the temperature ratio that scales the emitter resistance, `XRE`; default 0. |
| `epi_charge` | `0` | The epi layer's charge, `QCO`, in coulombs; default 0. |
| `epi_doping_parameter` | `0` | The epi layer's doping parameter, `GAMM`; default 0. |
| `epi_saturation_voltage` |  | The epi layer's drift saturation voltage, `VO`, in volts; infinite by default, for no velocity saturation. |
| `epi_saturation_voltage_temperature_exponent` | `0` | The exponent of the temperature ratio that scales the epi saturation voltage, `XVO`; default 0. |
| `excess_phase_delay` | `0` | The forward excess-phase delay, `TD`, in seconds; default 0, no excess phase. |
| `extrinsic_base_collector_capacitance` | `0` | The zero-bias extrinsic base-collector depletion capacitance, `CJEP`, in farads, the parasitic transistor's base-emitter junction's; default 0. |
| `extrinsic_base_resistance` | `0` | The extrinsic base resistance, `RBX`, in ohms; default 0. |
| `extrinsic_base_resistance_temperature_exponent` | `0` | The exponent of the temperature ratio that scales the extrinsic base resistance, `XRBX`; default 0. |
| `extrinsic_collector_resistance` | `0` | The extrinsic collector resistance, `RCX`, in ohms; default 0. |
| `extrinsic_collector_resistance_temperature_exponent` | `0` | The exponent of the temperature ratio that scales the extrinsic collector resistance, `XRCX`; default 0. |
| `flicker_coefficient` | `0` | The flicker noise coefficient of the base-emitter currents, `KFN`; default 0, no flicker noise. |
| `flicker_exponent` | `1` | The power of the current in the flicker noise, `AFN`; default 1. |
| `flicker_frequency_exponent` | `1` | The power of the frequency the flicker noise falls with, `BFN`; default 1. |
| `forward_early_voltage` |  | The forward Early voltage, `VEF`, in volts; infinite by default. |
| `forward_emission_coefficient` | `1` | The forward emission coefficient, `NF`; default 1. |
| `forward_knee_current` |  | The forward knee current, `IKF`, in amperes; infinite by default. |
| `forward_knee_current_temperature_exponent` | `0` | The exponent of the temperature ratio that scales the forward knee current, `XIKF`; default 0. |
| `forward_transit_time` | `0` | The ideal forward transit time, `TF`, in seconds; default 0. |
| `high_current_collector_factor` | `1` | The high-current factor of the intrinsic collector resistance, `HRCF`, positive; default 1. |
| `ideal_base_collector_activation_energy` | `1.12` | The activation energy of the ideal base-collector currents and of their potential, `EAIC`, in volts; default 1.12. |
| `ideal_base_collector_current` | `1e-16` | The ideal base-collector saturation current, `IBCI`, in amperes; default 1e-16. |
| `ideal_base_collector_emission_coefficient` | `1` | The ideal base-collector emission coefficient, `NCI`; default 1. |
| `ideal_base_current_temperature_exponent` | `3` | The exponent of the temperature ratio in the ideal base currents' scaling, `XII`; default 3. |
| `ideal_base_emitter_activation_energy` | `1.12` | The activation energy of the ideal base-emitter current and of its potential, `EAIE`, in volts; default 1.12. |
| `ideal_base_emitter_current` | `1e-18` | The ideal base-emitter saturation current, `IBEI`, in amperes; default 1e-18. |
| `ideal_base_emitter_emission_coefficient` | `1` | The ideal base-emitter emission coefficient, `NEI`; default 1. |
| `ideal_collector_substrate_activation_energy` | `1.12` | The activation energy of the parasitic ideal base-collector current and of the collector-substrate potential, `EAIS`, in volts; default 1.12. |
| `intrinsic_base_emitter_fraction` | `1` | The fraction of the base-emitter currents and depletion charge at the intrinsic base, `WBE`, the rest at the extrinsic base, from 0 to 1; default 1. |
| `intrinsic_base_resistance` | `0.1` | The intrinsic base resistance, `RBI`, in ohms, positive; default 0.1. |
| `intrinsic_base_resistance_temperature_exponent` | `0` | The exponent of the temperature ratio that scales the intrinsic base resistance, `XRBI`; default 0. |
| `intrinsic_collector_resistance` | `0.1` | The intrinsic collector resistance, `RCI`, in ohms, positive; default 0.1. |
| `intrinsic_collector_resistance_temperature_exponent` | `0` | The exponent of the temperature ratio that scales the intrinsic collector resistance, `XRCI`; default 0. |
| `knee_rolloff` | `0.5` | The power of the high-injection roll-off, `NKF`, positive; default 0.5. |
| `nominal_temperature` | `300.15` | The temperature the parameters were measured at, `TNOM`, in kelvins; default 27 °C. |
| `nonideal_base_collector_activation_energy` | `1.12` | The activation energy of the non-ideal base-collector currents, `EANC`, in volts; default 1.12. |
| `nonideal_base_collector_current` | `0` | The non-ideal base-collector saturation current, `IBCN`, in amperes; default 0. |
| `nonideal_base_collector_emission_coefficient` | `2` | The non-ideal base-collector emission coefficient, `NCN`; default 2. |
| `nonideal_base_current_temperature_exponent` | `3` | The exponent of the temperature ratio in the non-ideal base currents' scaling, `XIN`; default 3. |
| `nonideal_base_emitter_activation_energy` | `1.12` | The activation energy of the non-ideal base-emitter current, `EANE`, in volts; default 1.12. |
| `nonideal_base_emitter_current` | `0` | The non-ideal base-emitter saturation current, `IBEN`, in amperes; default 0. |
| `nonideal_base_emitter_emission_coefficient` | `2` | The non-ideal base-emitter emission coefficient, `NEN`; default 2. |
| `nonideal_collector_substrate_activation_energy` | `1.12` | The activation energy of the parasitic non-ideal base-collector current, `EANS`, in volts; default 1.12. |
| `parasitic_activation_energy` | `1.12` | The activation energy of the parasitic transport saturation current, `EAP`, in volts; default 1.12. |
| `parasitic_base_resistance` | `0.1` | The parasitic base resistance, `RBP`, in ohms, positive; default 0.1. |
| `parasitic_base_resistance_temperature_exponent` | `0` | The exponent of the temperature ratio that scales the parasitic base resistance, `XRBP`; default 0. |
| `parasitic_extrinsic_fraction` | `1` | The fraction of the parasitic transport current the extrinsic base-collector junction drives, `WSP`, the rest the intrinsic one, from 0 to 1; default 1. |
| `parasitic_forward_emission_coefficient` | `1` | The parasitic transistor's forward emission coefficient, `NFP`; default 1. |
| `parasitic_ideal_base_collector_current` | `0` | The parasitic transistor's ideal base-collector saturation current, `IBCIP`, in amperes, across the collector-substrate junction; default 0. |
| `parasitic_ideal_base_collector_emission_coefficient` | `1` | The parasitic transistor's ideal base-collector emission coefficient, `NCIP`; default 1. |
| `parasitic_ideal_base_emitter_current` | `0` | The parasitic transistor's ideal base-emitter saturation current, `IBEIP`, in amperes, across the extrinsic base-collector junction; default 0. |
| `parasitic_knee_current` |  | The parasitic transistor's knee current, `IKP`, in amperes; infinite by default. |
| `parasitic_nonideal_base_collector_current` | `0` | The parasitic transistor's non-ideal base-collector saturation current, `IBCNP`, in amperes; default 0. |
| `parasitic_nonideal_base_collector_emission_coefficient` | `2` | The parasitic transistor's non-ideal base-collector emission coefficient, `NCNP`; default 2. |
| `parasitic_nonideal_base_emitter_current` | `0` | The parasitic transistor's non-ideal base-emitter saturation current, `IBENP`, in amperes; default 0. |
| `parasitic_saturation_current` | `0` | The parasitic transistor's transport saturation current, `ISP`, in amperes; default 0, no parasitic transport. |
| `polarity` | `npn` | The doping order, `NPN` or `PNP`; default `npn`. One of `npn`, `pnp`. |
| `reach_through_smoothing` | `0.1` | The smoothing of the reach-through, `ART`; default 0.1. |
| `reach_through_voltage` | `0` | The punch-through voltage of the base-collector junctions, `VRT`, in volts; default 0, no reach-through. |
| `reverse_activation_energy_shift` | `0` | The shift of the activation energy of the reverse transport current, `DEAR`, in volts; default 0. |
| `reverse_early_voltage` |  | The reverse Early voltage, `VER`, in volts; infinite by default. |
| `reverse_emission_coefficient` | `1` | The reverse emission coefficient, `NR`; default 1. |
| `reverse_knee_current` |  | The reverse knee current, `IKR`, in amperes; infinite by default. |
| `reverse_saturation_current_ratio` | `1` | The ratio of the reverse to the forward transport saturation current, `ISRR`; default 1. |
| `reverse_saturation_current_temperature_exponent` | `0` | The exponent of the temperature ratio in the reverse saturation current ratio's scaling, `XISR`; default 0. |
| `reverse_transit_time` | `0` | The ideal reverse transit time, `TR`, in seconds; default 0. |
| `saturation_current` | `1e-16` | The transport saturation current, `IS`, in amperes; default 1e-16. |
| `saturation_current_temperature_exponent` | `3` | The exponent of the temperature ratio in the transport saturation currents' scaling, `XIS`; default 3. |
| `substrate_resistance` | `0` | The substrate resistance, `RS`, in ohms; default 0. |
| `substrate_resistance_temperature_exponent` | `0` | The exponent of the temperature ratio that scales the substrate resistance, `XRS`; default 0. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |
| `thermal` | `false` | Whether the device heats by the power it dissipates, through a terminal `thermal` joined to a thermal net, whose temperature above the ambient it runs at. |
| `transit_time_base_width_coefficient` | `0` | How the forward transit time grows with the base charge's Early part, `QTF`; default 0. |
| `transit_time_bias_coefficient` | `0` | How much the forward transit time grows with bias, `XTF`; default 0. |
| `transit_time_current` | `0` | The current, `ITF`, above which the forward transit time grows, in amperes; default 0, for growth at any current. |
| `transit_time_voltage` |  | The base-collector voltage, `VTF`, that sets how the forward transit time grows with it, in volts; infinite by default. |

## `vdmos`

| Parameter | Default | Description |
| --- | --- | --- |
| `body_resistance` | `0` | The body diode's series resistance, `RB`, in ohms; default 0. |
| `body_resistance_quadratic_temperature_coefficient` | `0` | The body resistance's quadratic temperature coefficient, `TRB2`, per square kelvin; default 0. |
| `body_resistance_temperature_coefficient` | `0` | The body resistance's linear temperature coefficient, `TRB1`, per kelvin; default 0. |
| `breakdown_current` | `1e-10` | The current at the breakdown voltage, `IBV`, in amperes, whatever the multiplier; default 1e-10. |
| `breakdown_emission_coefficient` | `1` | The breakdown emission coefficient, `NBV`; default 1. |
| `breakdown_voltage` |  | The drain-source breakdown voltage, `BV`, in volts, positive; without it, the body diode never breaks down. |
| `channel` | `n` | The channel, `VDMOS` or `VDMOS pchan`; default `n`. One of `n`, `p`. |
| `channel_length_modulation` | `0` | The channel-length modulation, `LAMBDA`, per volt; default 0. |
| `depletion_coefficient` | `0.5` | The fraction of the body diode's junction potential above which its depletion capacitance is extended linearly, `FC`, below 1; default 0.5. |
| `drain_resistance` | `0` | The drain resistance, `RD`, in ohms; default 0. |
| `drain_resistance_quadratic_temperature_coefficient` | `0` | The drain resistance's quadratic temperature coefficient, `TRD2`, per square kelvin; default 0. |
| `drain_resistance_temperature_coefficient` | `0` | The drain resistance's linear temperature coefficient, `TRD1`, per kelvin; default 0. |
| `drain_resistance_temperature_exponent` |  | The exponent of the temperature ratio that scales the drain resistance, `TEXP0`; default none, for the drain resistance's temperature coefficients. |
| `drain_source_resistance` |  | The drain-source shunt resistance, `RDS`, in ohms; default none, for ngspice's conductance of 1 fS whatever the multiplier. |
| `emission_coefficient` | `1` | The body diode's emission coefficient, `N`; default 1. |
| `energy_gap` | `1.11` | The energy gap that scales the body diode's saturation current with temperature, `EG`, in volts; default 1.11, silicon's. |
| `flicker_coefficient` | `0` | The flicker noise coefficient of the drain current, `KF`; default 0, no flicker noise. |
| `flicker_exponent` | `1` | The power of the drain current in the flicker noise, `AF`; default 1. |
| `gate_drain_capacitance_coefficient` | `1` | How fast the gate-drain capacitance moves with the gate-drain voltage, `A`, per volt; default 1. |
| `gate_resistance` | `0` | The gate resistance, `RG`, in ohms; default 0. |
| `gate_resistance_quadratic_temperature_coefficient` | `0` | The gate resistance's quadratic temperature coefficient, `TRG2`, per square kelvin; default 0. |
| `gate_resistance_temperature_coefficient` | `0` | The gate resistance's linear temperature coefficient, `TRG1`, per kelvin; default 0. |
| `gate_source_capacitance` | `1.4e-09` | The gate-source capacitance, `CGS`, in farads; default 1.4 nF. |
| `grading_coefficient` | `0.5` | The body diode's grading coefficient, `M`, below 1; default 0.5. |
| `junction_capacitance` | `5e-10` | The body diode's zero-bias junction capacitance, `CJO`, in farads; default 0.5 nF. |
| `junction_potential` | `0.8` | The body diode's junction potential, `VJ`, in volts; default 0.8. |
| `max_gate_drain_capacitance` | `2e-09` | The gate-drain capacitance far above the drain, `CGDMAX`, in farads; default 2 nF. |
| `min_gate_drain_capacitance` | `2e-11` | The gate-drain capacitance far below the drain, `CGDMIN`, in farads; default 20 pF. |
| `mobility_modulation` | `0` | The gate field's degradation of the mobility, `THETA`, per volt; default 0. |
| `multiplier` | `1` | The number of identical transistors in parallel, `M`; default 1. |
| `nominal_temperature` | `300.15` | The temperature the parameters were measured at, `TNOM`, in kelvins; default 27 °C. |
| `quasi_saturation_resistance` |  | The quasi-saturation resistance, `RQ`, in ohms, which the drain resistance grows towards with the drain voltage; default none, for no quasi-saturation. |
| `quasi_saturation_temperature_exponent` | `0.3` | The exponent of the temperature ratio that scales the quasi-saturation resistance, `TEXP1`; default 0.3. |
| `quasi_saturation_voltage` |  | The quasi-saturation voltage, `VQ`, in volts, at which the drain resistance has grown by half the quasi-saturation resistance; default none, for no quasi-saturation. |
| `saturation_current` | `1e-14` | The body diode's saturation current, `IS`, in amperes; default 1e-14. |
| `saturation_current_temperature_exponent` | `3` | The exponent of the temperature ratio in the body diode's saturation current's scaling, `XTI`; default 3. |
| `source_resistance` | `0` | The source resistance, `RS`, in ohms; default 0. |
| `source_resistance_quadratic_temperature_coefficient` | `0` | The source resistance's quadratic temperature coefficient, `TRS2`, per square kelvin; default 0. |
| `source_resistance_temperature_coefficient` | `0` | The source resistance's linear temperature coefficient, `TRS1`, per kelvin; default 0. |
| `subthreshold_shift` | `0` | The shift of the softened overdrive along the gate voltage, `SUBSHIFT`, in volts; default 0. |
| `subthreshold_slope` | `0.1` | The slope of the overdrive's softening below the threshold, `KSUBTHRES`, in volts; default 0.1. |
| `subthreshold_slope_quadratic_temperature_coefficient` | `0` | The subthreshold slope's quadratic temperature coefficient, `TKSUBTHRES2`, per square kelvin; default 0. |
| `subthreshold_slope_temperature_coefficient` | `0` | The subthreshold slope's linear temperature coefficient, `TKSUBTHRES1`, per kelvin; default 0. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |
| `thermal` | `false` | Whether the device heats by the power it dissipates, through a terminal `thermal` joined to a thermal net, whose temperature above the ambient it runs at. |
| `threshold_voltage` |  | The threshold voltage, `VTO`, in volts; default 3 for an n-channel and −3 for a p-channel, an IRF540's and an IRF9540's. |
| `threshold_voltage_temperature_coefficient` | `0` | How fast the threshold voltage moves with temperature, `TCVTH` or `VTOTC`, in volts per kelvin; default 0. |
| `transconductance_parameter` |  | The transconductance parameter, `KP`, in amperes per square volt; default 20 for an n-channel and 10 for a p-channel. |
| `transconductance_parameter_temperature_exponent` | `-1.5` | The exponent of the temperature ratio that scales the transconductance parameter, `MU` or `BEX`; default −1.5. |
| `transit_time` | `0` | The body diode's transit time, `TT`, in seconds; default 0. |
| `triode_conductance_multiplier` | `1` | The factor on the drain-source voltage below saturation, `MTRIODE`; default 1. |

## `voltage_controlled_current_source`

| Parameter | Default | Description |
| --- | --- | --- |
| `transconductance` | required | The ratio of the output current to the control voltage, in siemens. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |

## `voltage_controlled_memristor`

| Parameter | Default | Description |
| --- | --- | --- |
| `off_rate` | required | The state's speed toward off at twice the off voltage, in full ranges per second; positive. |
| `off_resistance` | required | The resistance fully off, at a state of 1, in ohms; above the on resistance. |
| `off_voltage` | required | The voltage above which the state moves toward off, in volts; positive. |
| `on_rate` | required | The state's speed toward on at twice the on voltage, in full ranges per second; positive. |
| `on_resistance` | required | The resistance fully on, at a state of 0, in ohms; positive. |
| `on_voltage` | required | The voltage below which the state moves toward on, in volts; negative. |
| `law` | `linear` | How the resistance follows the state; default linear. One of `linear`, `exponential`. |
| `off_exponent` | `1` | How steeply the speed toward off grows past the off voltage, `α_off`; at least 1; default 1. |
| `on_exponent` | `1` | How steeply the speed toward on grows past the on voltage, `α_on`; at least 1; default 1. |
| `state` | `1` | The state it was made in, from 0, fully on, to 1, fully off; default 1. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |
| `window` | `0.01` | The width of the window that slows the state at either bound, a fraction of its range; positive; default 0.01. |

## `voltage_controlled_switch`

| Parameter | Default | Description |
| --- | --- | --- |
| `off_resistance` | `1000000` | The resistance with the control at or past the off voltage, in ohms; default 1e6. |
| `off_voltage` | `0` | The control voltage at which the switch is off, in volts; default 0. |
| `on_resistance` | `1` | The resistance with the control at or past the on voltage, in ohms; default 1. |
| `on_voltage` | `1` | The control voltage at which the switch is on, in volts; default 1. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |

## `voltage_controlled_voltage_source`

| Parameter | Default | Description |
| --- | --- | --- |
| `gain` | required | The ratio of the output voltage to the control voltage. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |

## `voltage_source`

| Parameter | Default | Description |
| --- | --- | --- |
| `waveform` | required | The source's level over time, in volts or amperes. |
| `waveform.dc` |  | A constant level. |
| `waveform.pulse` |  | A trapezoidal pulse, repeated each period, if it has one. |
| `waveform.pulse.delay` | `0` | The time the first pulse starts its rise, in seconds. |
| `waveform.pulse.fall` | required | The time each pulse takes to fall, in seconds; positive. |
| `waveform.pulse.initial` | required | The level before the delay and between pulses. |
| `waveform.pulse.period` |  | The time from one pulse's start to the next's, in seconds; without it, a single pulse. |
| `waveform.pulse.pulsed` | required | The level at the top of each pulse. |
| `waveform.pulse.rise` | required | The time each pulse takes to rise, in seconds; positive. |
| `waveform.pulse.width` | required | The time each pulse holds its top, in seconds. |
| `waveform.sine` |  | A sine, damped from its delay on. |
| `waveform.sine.amplitude` | required | The sine's amplitude at its delay. |
| `waveform.sine.damping` | `0` | The rate at which the amplitude decays from the delay on, per second. |
| `waveform.sine.delay` | `0` | The time the sine starts, in seconds, before which it holds its value then. |
| `waveform.sine.frequency` | required | The sine's frequency, in hertz. |
| `waveform.sine.offset` | required | The level the sine swings about. |
| `waveform.sine.phase` | `0` | The phase at the delay, in radians. |
| `waveform.piecewise` |  | Straight lines between points, holding the first and last levels outside them. |
| `waveform.piecewise.points` | required | Each point's time and level, in strictly increasing time; at least one. |
| `waveform.exponential` |  | A rise toward a level and a fall back, each exponential, as SPICE's `EXP`. |
| `waveform.exponential.fall_delay` | required | The time the fall starts, in seconds; at or after the rise delay. |
| `waveform.exponential.fall_time_constant` | required | The fall's time constant, in seconds; positive. |
| `waveform.exponential.initial` | required | The level before the rise delay. |
| `waveform.exponential.pulsed` | required | The level the rise tends to. |
| `waveform.exponential.rise_delay` | `0` | The time the rise starts, in seconds. |
| `waveform.exponential.rise_time_constant` | required | The rise's time constant, in seconds; positive. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |
