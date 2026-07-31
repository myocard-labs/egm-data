"""Join a ClassifierBank back to the ``synthetic_bank`` that generated it.

The two banks are **parallel artifacts sharing a key**, not one derived
from the other: the ClassifierBank is a source-agnostic ML compression
(signal + label + join key) and the ``synthetic_bank`` carries the
generation config and the θ-spec. Phase 1.5 deliberately keeps θ *off*
the ClassifierBank, which leaves exactly one question open — how a
consumer that has features from one and θ from the other lines them up.
This module is that answer.

The join lives here rather than in each consumer because it is
I/O-shaped and shared: egm-studio's T4 views (STU1 / STU4 / STU5) and,
later, the FN-vs-θ diagnostic all need the same correspondence. Two
repos independently re-deriving a join is what the placement rules exist
to prevent. Design rationale:
``intracardiac-platform/project/investigations/synthetic_bank_source_of_truth.md``
section 12.

Two pieces:

- :func:`simulation_configs` — transpose the column-oriented
  ``Simulations`` model into one :class:`SimulationConfig` per
  simulation, keyed by ``simulation_id``. Useful on its own when you
  only want the config side.
- :func:`join_traces_with_simulations` — pair every ClassifierTrace with
  the config of the simulation that produced it.

**Scope.** θ values are *not* flattened out: consumers read them off the
typed config the same way they read any other field. Resolving
``TunedParam.path`` generically, and the predictions leg of the FN-vs-θ
join, are deliberately deferred (see the repo roadmap) — this is the
subset Phase 1.5 needs, not the whole correlation machinery.

**Synthetic-only by nature.** IAFDB traces have no ``simulation_id`` and
no generation config; a ClassifierBank built from one is rejected rather
than silently yielding an empty join.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from myocard_egm_contracts import synthetic_bank as _synthetic_bank_models

from .classifier_bank import ClassifierBank, ClassifierTrace

__all__ = [
    "SimulationConfig",
    "TraceWithSimulation",
    "join_traces_with_simulations",
    "simulation_configs",
]

#: The per-trace key that points back at a simulation. Kept in one place
#: so this module and the ClassifierBank writer's name guard cannot
#: drift apart.
SIMULATION_ID_KEY = "simulation_id"


@dataclass(frozen=True)
class SimulationConfig:
    """One simulation's generation config — a *row* view.

    The contracts ``Simulations`` model is column-oriented (parallel
    lists, one entry per simulation) because that is how the data sits
    in HDF5. Consumers reason about one simulation at a time, so this
    transposes it. The per-function objects are passed through **as the
    contracts models they already are** — this is a reshape, not a
    reinterpretation, and nothing here re-types or unwraps them.
    """

    simulation_id: int
    seed: int
    geometry: Any
    cell_model: Any
    substrate: Any
    substrate_summary: Any
    activation: Any
    electrodes: Any
    backend: Any
    label_policy: Any
    label_names: dict[str, str]


@dataclass(frozen=True)
class TraceWithSimulation:
    """A ClassifierTrace beside the config of the simulation behind it."""

    trace: ClassifierTrace
    simulation: SimulationConfig


def simulation_configs(
    synthetic_bank: _synthetic_bank_models.SyntheticBank,
) -> dict[int, SimulationConfig]:
    """Transpose ``simulations/`` into ``{simulation_id: SimulationConfig}``.

    Raises
    ------
    ValueError
        If two simulations share a ``simulation_id``. That would make
        the join ambiguous, and silently keeping the last one would
        attach half the traces to the wrong config.
    """
    sims = synthetic_bank.simulations
    configs: dict[int, SimulationConfig] = {}

    for index, raw_id in enumerate(sims.simulation_id):
        simulation_id = int(_unwrap(raw_id))
        if simulation_id in configs:
            raise ValueError(
                f"synthetic_bank has two simulations with simulation_id "
                f"{simulation_id}; ids must be unique or the trace join is "
                "ambiguous."
            )
        configs[simulation_id] = SimulationConfig(
            simulation_id=simulation_id,
            seed=int(_unwrap(sims.seed[index])),
            geometry=sims.geometry[index],
            cell_model=sims.cell_model[index],
            substrate=sims.substrate[index],
            substrate_summary=sims.substrate_summary[index],
            activation=sims.activation[index],
            electrodes=sims.electrodes[index],
            backend=sims.backend[index],
            label_policy=sims.label_policy[index],
            label_names=dict(sims.label_names[index]),
        )
    return configs


def join_traces_with_simulations(
    classifier_bank: ClassifierBank,
    synthetic_bank: _synthetic_bank_models.SyntheticBank,
) -> list[TraceWithSimulation]:
    """Pair each ClassifierTrace with its generating simulation's config.

    Parameters
    ----------
    classifier_bank
        A ClassifierBank converted from ``synthetic_bank``. Every trace
        must carry ``simulation_id`` in ``trace_metadata``.
    synthetic_bank
        The parallel source bank holding the per-simulation config and
        the θ-spec.

    Returns
    -------
    One :class:`TraceWithSimulation` per trace, in bank order.

    Raises
    ------
    ValueError
        If the two banks don't belong together, if a trace carries no
        ``simulation_id``, or if a trace points at a simulation the
        source bank doesn't have. All three are silent-wrong-answer
        failures otherwise: the join would return plausible pairs built
        from the wrong config.
    """
    _check_banks_correspond(classifier_bank, synthetic_bank)
    configs = simulation_configs(synthetic_bank)

    joined: list[TraceWithSimulation] = []
    for index, trace in enumerate(classifier_bank.traces):
        if SIMULATION_ID_KEY not in trace.trace_metadata:
            raise ValueError(
                f"ClassifierBank trace {index} has no "
                f"trace_metadata[{SIMULATION_ID_KEY!r}], so it cannot be "
                "joined to a simulation. This join is synthetic-only — "
                "IAFDB traces have no generation config."
            )
        simulation_id = int(trace.trace_metadata[SIMULATION_ID_KEY])
        config = configs.get(simulation_id)
        if config is None:
            raise ValueError(
                f"ClassifierBank trace {index} references simulation_id "
                f"{simulation_id}, which is absent from the synthetic bank "
                f"(known ids: {sorted(configs)}). Either the two banks are "
                "mismatched or the source bank is incomplete."
            )
        joined.append(TraceWithSimulation(trace=trace, simulation=config))
    return joined


def _check_banks_correspond(
    classifier_bank: ClassifierBank,
    synthetic_bank: _synthetic_bank_models.SyntheticBank,
) -> None:
    """Refuse to join a ClassifierBank against the wrong synthetic bank.

    Stable ids exist precisely so this is checkable. Without the check,
    joining against a *different* synthetic bank whose simulation ids
    happen to overlap — which they will, since ids start at 0 in every
    bank — produces a full set of confident, wrong pairings and no error
    at all. That is the worst failure mode available here, so it is
    worth a hard stop rather than a warning.
    """
    if synthetic_bank.bank_id is None:
        raise ValueError(
            "The synthetic bank carries no bank_id, so it cannot be matched "
            "against the ClassifierBank's source banks. Re-export it with a "
            "producer that stamps one."
        )
    source_ids = {entry.bank_id for entry in classifier_bank.banks}
    if synthetic_bank.bank_id not in source_ids:
        raise ValueError(
            f"synthetic bank {synthetic_bank.bank_id!r} is not among the "
            f"ClassifierBank's source banks ({sorted(source_ids)}). Joining "
            "mismatched banks would pair traces with the wrong generation "
            "config, since simulation ids restart at 0 in every bank."
        )


def _unwrap(value: Any) -> Any:
    """Unwrap a codegen constraint-root container (``.root`` accessor)."""
    return getattr(value, "root", value)
