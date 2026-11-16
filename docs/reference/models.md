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
| `breakdown_current` | `0.001` | The current at the breakdown voltage, in amperes; default 1e-3. |
| `breakdown_emission_coefficient` |  | The breakdown emission coefficient; the emission coefficient by default. |
| `breakdown_temperature_coefficient` | `0` | How fast the breakdown voltage falls with temperature, in volts per kelvin; default 0. |
| `breakdown_voltage` |  | The reverse breakdown voltage, in volts, positive; without it, the junction never breaks down. |
| `depletion_coefficient` | `0.5` | The fraction of the junction potential above which the depletion capacitance is extended linearly, below 1; default 0.5. |
| `emission_coefficient` | `1` | The emission coefficient; default 1. |
| `energy_gap` | `1.11` | The energy gap that scales the saturation current with temperature, in volts; default 1.11, silicon's. |
| `flicker_coefficient` | `0` | The flicker noise coefficient, `KF`; default 0, no flicker noise. |
| `flicker_exponent` | `1` | The power of the current in the flicker noise, `AF`; default 1. |
| `grading_coefficient` | `0.5` | The grading coefficient of the junction, below 1; default 0.5. |
| `junction_capacitance` | `0` | The zero-bias junction capacitance, in farads; default 0. |
| `junction_potential` | `1` | The junction potential, in volts; default 1. |
| `nominal_temperature` | `300.15` | The temperature the parameters were measured at, in kelvins; default 27 °C. |
| `saturation_current` | `1e-14` | The saturation current, in amperes; default 1e-14. |
| `saturation_current_temperature_exponent` | `3` | The exponent of the temperature ratio in the saturation current's scaling; default 3. |
| `series_resistance` | `0` | The series resistance, in ohms; default 0. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |
| `thermal` | `false` | Whether the device heats by the power it dissipates, through a terminal `thermal` joined to a thermal net, whose temperature above the ambient it runs at. |
| `transit_time` | `0` | The transit time, in seconds; default 0. |

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
