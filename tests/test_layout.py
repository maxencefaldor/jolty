# Copyright 2026 Maxence Faldor

import itertools

from hypothesis import given, settings
from hypothesis import strategies as st

import jolty.layout
from jolty import Builder, Role, circuit, component, devices, ir
from jolty.layout import (
    Layout,
    Net,
    Orientation,
    Placed,
    Point,
    Segment,
    Shape,
    layout,
)
from jolty.units import V, kohm, nF, uS

TERMINALS = {
    "resistor": ("a", "b"),
    "capacitor": ("a", "b"),
    "inductor": ("a", "b"),
    "diode": ("anode", "cathode"),
    "voltage_source": ("positive", "negative"),
    "current_source": ("positive", "negative"),
    "voltage_controlled_voltage_source": (
        "positive",
        "negative",
        "control_positive",
        "control_negative",
    ),
    "voltage_controlled_current_source": (
        "positive",
        "negative",
        "control_positive",
        "control_negative",
    ),
}
"""The terminals of each model the layout draws as a symbol."""

STAGE = ir.Component(
    ports=(
        ir.Port(name="input", role=Role.SIGNAL),
        ir.Port(name="output", role=Role.SIGNAL),
        ir.Port(name="vdd", role=Role.SUPPLY),
        ir.Port(name="ground", role=Role.GROUND),
    )
)
"""A component the random circuits place as a box, its insides left out."""


@st.composite
def circuits(draw: st.DrawFn) -> ir.Circuit:
    """A circuit whose top places devices and boxes between a few random nets."""
    nets = ["ground", "vdd", *(f"n{index}" for index in range(draw(st.integers(1, 5))))]
    models = draw(st.lists(st.sampled_from([*TERMINALS, "stage"]), max_size=9))
    instances = []
    for index, model in enumerate(models):
        if model == "stage":
            of: ir.DeviceOf | ir.ComponentOf = ir.ComponentOf(component="stage")
            terminals = [port.name for port in STAGE.ports]
        else:
            of = ir.DeviceOf(model=model)
            terminals = list(TERMINALS[model])
        joined = {terminal: draw(st.sampled_from(nets)) for terminal in terminals}
        instances.append(ir.Instance(name=f"X{index}", of=of, terminals=joined))
    ports = [ir.Port(name="ground", role=Role.GROUND)]
    if draw(st.booleans()):
        ports.append(ir.Port(name="vdd", role=Role.SUPPLY))
    ports.extend(
        ir.Port(name=net, role=Role.SIGNAL) for net in nets[2:] if draw(st.booleans())
    )
    top = ir.Component(ports=tuple(ports), instances=tuple(instances))
    return ir.Circuit(top="top", components={"top": top, "stage": STAGE})


def points(segment: Segment) -> list[Point]:
    """Every grid point a segment runs through, its ends included."""
    (x1, y1), (x2, y2) = segment
    if x1 == x2:
        return [(x1, y) for y in range(min(y1, y2), max(y1, y2) + 1)]
    return [(x, y1) for x in range(min(x1, x2), max(x1, x2) + 1)]


def steps(net: Net) -> set[frozenset[Point]]:
    """The unit steps a net's wires take."""
    return {
        frozenset(pair)
        for segment in net.segments
        for pair in itertools.pairwise(points(segment))
    }


def ends(net: Net) -> set[Point]:
    """Where a net's wires end, bend or branch, and its pins."""
    return {end for segment in net.segments for end in segment} | set(net.pins)


def ways(net: Net, point: Point) -> set[str]:
    """Whether the net's wires run across (h) or up (v) through `point`."""
    return {
        "h" if segment[0][1] == segment[1][1] else "v"
        for segment in net.segments
        if point in points(segment)
    }


def connected(net: Net) -> bool:
    """Whether the net's wires join all its pins."""
    parent = {point: point for segment in net.segments for point in points(segment)}
    parent.update({pin: pin for pin in net.pins})

    def root(point: Point) -> Point:
        while parent[point] != point:
            point = parent[point]
        return point

    for segment in net.segments:
        for first, second in itertools.pairwise(points(segment)):
            parent[root(first)] = root(second)
    return len({root(pin) for pin in net.pins}) == 1


def assert_sound(drawn: Layout) -> None:
    """Checks every promise the layout makes about where things go."""
    bodies = [placed.body() for placed in drawn.instances]
    for ((left, bottom), (right, top)), ((x1, y1), (x2, y2)) in itertools.combinations(
        bodies, 2
    ):
        assert right < x1 or x2 < left or top < y1 or y2 < bottom, "bodies overlap"
    inside = {
        (x, y)
        for (left, bottom), (right, top) in bodies
        for x in range(left, right + 1)
        for y in range(bottom, top + 1)
    }
    owner: dict[Point, str] = {}
    for net in drawn.nets:
        for pin in net.pins:
            assert owner.setdefault(pin, net.name) == net.name, "nets share a pin"
    for net in drawn.nets:
        if net.role is Role.SIGNAL and len(set(net.pins)) > 1 and not net.labeled:
            assert connected(net), f"{net.name} is not connected"
        wired = {point for segment in net.segments for point in points(segment)}
        assert not wired & inside, f"{net.name} runs through a symbol"
        assert all(owner.get(point, net.name) == net.name for point in wired)
    for first, second in itertools.combinations(drawn.nets, 2):
        assert not steps(first) & steps(second), "wires run along each other"
        shared = {point for segment in first.segments for point in points(segment)} & {
            point for segment in second.segments for point in points(segment)
        }
        for point in shared:
            assert point not in ends(first) | ends(second), "wires meet"
            assert ways(first, point) | ways(second, point) == {"h", "v"}


@settings(max_examples=150, deadline=None)
@given(circuits())
def test_every_layout_keeps_its_nets_apart(drawn: ir.Circuit) -> None:
    assert_sound(layout(drawn, "top"))


@component
def stage(c: Builder) -> None:
    """A low-pass stage pulled up to its supply."""
    source = c.port("input")
    output = c.port("output")
    supply = c.port("vdd", Role.SUPPLY)
    ground = c.port("ground", Role.GROUND)
    pull = c.place(devices.resistor(10 * kohm), name="RP")
    series = c.place(devices.resistor(1 * kohm), name="R")
    cap = c.place(devices.capacitor(1 * nF), name="C")
    c.connect(source, series.a)
    c.connect(series.b, cap.a, pull.b, output)
    c.connect(pull.a, supply)
    c.connect(cap.b, ground)


@component
def chain(c: Builder) -> None:
    """Two stages in a row, fed from a source, on a rail of their own."""
    ground = c.port("ground", Role.GROUND)
    rail = c.place(devices.voltage_source(5 * V), name="VDD")
    source = c.place(devices.voltage_source(1 * V), name="V1")
    first = c.place(stage(), name="X1")
    second = c.place(stage(), name="X2")
    c.connect(rail.positive, first.vdd, second.vdd, name="vdd")
    c.connect(source.positive, first.input, name="in")
    c.connect(first.output, second.input, name="mid")
    c.connect(second.output, name="out")
    c.connect(rail.negative, source.negative, first.ground, second.ground, ground)


def test_a_net_plays_the_role_of_the_ports_on_it() -> None:
    drawn = layout(circuit(chain()), "chain")
    roles = {net.name: net.role for net in drawn.nets}
    assert roles == {
        "vdd": Role.SUPPLY,
        "in": Role.SIGNAL,
        "mid": Role.SIGNAL,
        "out": Role.SIGNAL,
        "ground": Role.GROUND,
    }
    assert not next(net for net in drawn.nets if net.name == "vdd").segments
    assert_sound(drawn)


def test_signal_flows_from_the_sources_and_rails_sink() -> None:
    drawn = layout(circuit(chain()), "chain")
    placed = {item.name: item for item in drawn.instances}
    assert [placed[id].column for id in ("V1", "X1", "X2")] == [0, 1, 2]
    assert placed["VDD"].column == 0
    assert placed["VDD"].origin[1] < placed["V1"].origin[1]
    assert placed["V1"].pin("positive")[1] == placed["X1"].pin("input")[1]


def test_without_sources_signal_flows_from_the_first_port() -> None:
    drawn = layout(circuit(stage()), "stage")
    placed = {item.name: item for item in drawn.instances}
    assert placed["R"].column == 0
    assert placed["R"].orientation == Orientation(turns=1)
    assert {flag.name: flag.side for flag in drawn.flags} == {
        "input": -1,
        "output": 1,
    }
    # The pull-up and the capacitor hang from the output, one above, one below.
    pull, cap = placed["RP"], placed["C"]
    assert pull.pin("b")[1] == cap.pin("a")[1] == placed["R"].pin("b")[1]
    assert pull.pin("a")[1] > pull.pin("b")[1] > cap.pin("b")[1]
    assert_sound(drawn)


@component
def transconductor(c: Builder) -> None:
    """A source driving a load through a transconductance from the ground."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(1 * V), name="V1")
    gain = c.place(devices.voltage_controlled_current_source(500 * uS), name="G1")
    load = c.place(devices.resistor(1 * kohm), name="R1")
    c.connect(source.positive, gain.control_positive, name="in")
    c.connect(gain.negative, load.a, name="out")
    c.connect(source.negative, gain.positive, gain.control_negative, load.b, ground)


def test_a_controlled_source_flips_to_keep_its_ground_below() -> None:
    drawn = layout(circuit(transconductor()), "transconductor")
    placed = {item.name: item for item in drawn.instances}
    gain = placed["G1"]
    assert gain.orientation == Orientation(flipped=True)
    assert gain.pin("positive")[1] < gain.pin("negative")[1]
    assert gain.pin("control_positive")[0] < gain.origin[0]
    assert_sound(drawn)


def test_an_unknown_model_or_component_draws_as_a_box() -> None:
    top = ir.Component(
        ports=(ir.Port(name="ground", role=Role.GROUND),),
        instances=(
            ir.Instance(
                name="U1",
                of=ir.DeviceOf(model="transistor"),
                terminals={"base": "in", "collector": "out", "emitter": "ground"},
            ),
            ir.Instance(
                name="X1",
                of=ir.ComponentOf(component="missing"),
                terminals={"a": "out", "b": "ground"},
            ),
        ),
    )
    drawn = layout(ir.Circuit(top="top", components={"top": top}), "top")
    placed = {item.name: item for item in drawn.instances}
    assert set(placed["U1"].shape.pins) == {"base", "collector", "emitter"}
    assert set(placed["X1"].shape.pins) == {"a", "b"}
    assert_sound(drawn)


def test_an_empty_component_lays_out_as_nothing() -> None:
    drawn = layout(ir.Circuit(top="top", components={"top": ir.Component()}), "top")
    assert drawn.instances == drawn.flags == drawn.nets == ()
    assert drawn.bounds() == ((0, 0), (0, 0))


def test_a_net_of_many_pins_is_wired_whole() -> None:
    loads = tuple(
        ir.Instance(
            name=f"R{index}",
            of=ir.DeviceOf(model="resistor"),
            terminals={"a": "bus", "b": f"n{index}"},
        )
        for index in range(24)
    )
    source = ir.Instance(
        name="V1",
        of=ir.DeviceOf(model="voltage_source"),
        terminals={"positive": "bus", "negative": "ground"},
    )
    top = ir.Component(
        ports=(ir.Port(name="ground", role=Role.GROUND),), instances=(source, *loads)
    )
    drawn = layout(ir.Circuit(top="top", components={"top": top}), "top")
    bus = next(net for net in drawn.nets if net.name == "bus")
    assert len(set(bus.pins)) == 25
    assert not bus.labeled
    assert_sound(drawn)


def test_wire_crossing_rollback_preserves_crossed_net() -> None:
    net_a = Net(name="A", role=Role.SIGNAL, pins=[(-2, 1), (6, 1)])
    net_b = Net(name="B", role=Role.SIGNAL, pins=[(2, -1), (2, 3), (12, 12)])
    net_c = Net(name="C", role=Role.SIGNAL, pins=[(1, 0), (3, 2)])
    cage = Placed(
        name="BLOCK",
        instance=ir.Instance(
            name="BLOCK", of=ir.DeviceOf(model="resistor"), terminals={}
        ),
        shape=Shape(pins={}, body=((-1, -1), (1, 1))),
        origin=(12, 12),
        orientation=Orientation(),
        column=0,
    )
    router = jolty.layout._Router(  # noqa: SLF001 - test crossing rollback
        [net_a, net_b, net_c], [cage]
    )
    router.route(net_a)
    assert router.through[(2, 1)] == ("A", "h")

    router.route(net_b)
    assert net_b.labeled
    assert not net_b.segments
    assert router.through[(2, 1)] == ("A", "h")
    assert (2, 1) not in router.ends
    assert not router.crossed
    assert router._crosses("C", (2, 1))  # noqa: SLF001 - verify A remains reserved
    assert not router._may_cross(  # noqa: SLF001 - verify same axis stays blocked
        "C", (2, 1), (1, 0), (3, 1)
    )
    assert not router._may_cross(  # noqa: SLF001 - verify ending on A stays blocked
        "C", (2, 1), (0, 1), (2, 1)
    )

    router.route(net_c)
    assert not net_c.labeled
    assert router.through[(2, 1)] == ("A", "h")
    assert all((2, 1) not in points(segment) for segment in net_c.segments)
    assert_sound(
        Layout(
            component="top",
            instances=(cage,),
            flags=(),
            nets=(net_a, net_b, net_c),
        )
    )
