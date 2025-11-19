# ONTOLOGY — DARWIN PBPK PLATFORM

## Purpose
Map the natural geometry of PBPK modeling before implementation. This ontology is the explicit phase space that every module must respect and expose.

## Entities
- **Organ**: structured tissue compartment (liver, kidney, scaffold niche)
- **BloodPool**: vascular transport manifold connecting organs
- **DrugSpecies**: molecular entity with physicochemical descriptors
- **ScaffoldMatrix**: porous biomaterial domain influencing diffusion
- **MetabolicRoute**: enzymatic pathway transforming DrugSpecies
- **Measurement**: observation (concentration vs time, imaging frame)
- **Hypothesis**: proposition relating structure → dynamics → observation

## Relations
- `Organ --perfuses--> Organ`
- `Organ --receives--> BloodPool`
- `DrugSpecies --diffuses_into--> ScaffoldMatrix`
- `DrugSpecies --metabolized_via--> MetabolicRoute`
- `Measurement --samples--> Organ`
- `Hypothesis --implies--> Trajectory`

## Geometry
- Compartments exist on a directed graph **G = (V, E)** with edge weights = flow rates.
- Drug distribution is piecewise-smooth; treat as trajectories in **C¹** manifold.
- Scaffold diffusion couples PDE domain (scaffold volume) with ODE nodes (organs).
- Phase space **Φ = (x, θ, σ)** where:
  - x: state vector (concentrations per compartment)
  - θ: parameter vector (permeability, partition coefficients, metabolic rates)
  - σ: structural descriptors (porosity distribution, curvature of scaffold)
- Curvature of σ modulates effective diffusion tensor; encode as Riemannian metric **g** on scaffold manifold.

## Invariants
- Mass conservation: ∑ inflow - ∑ outflow - metabolism = d/dt(mass)
- Non-negativity: concentrations ≥ 0, flow ≥ 0
- Topological consistency: scaffold curvature influences accessible surface area monotonically
- Parameter priors: θ follow hierarchies (population → subject → scaffold-adjusted)

## Axes of Variation
- **Temporal scale**: microseconds (binding) → months (scaffold degradation)
- **Spatial scale**: nanometer (pore) → organ-level centimeters
- **Uncertainty**: epistemic (model mis-specification), aleatoric (measurement noise)
- **Regulatory**: in vitro, in vivo, clinical trial phases

## Attractors (desired states)
- Calibration convergence (posterior stable)
- Predictive validity (cross-study generalization)
- Scaffold integration consistency (drug release vs. scaffold data alignment)

## Singularities (warning zones)
- Parameter non-identifiability (Jacobian rank drop)
- Scaffold topology mismatch (curvature contradicts diffusion assumptions)
- Conflicting measurements (mutual information collapse)

## Instrumentation linkage
- MLFlow experiment = Hypothesis trajectory
- Qdrant vector = semantic encoding of Hypothesis/Measurement pair
- Dashboard metric = entropy of posterior, curvature of scaffold manifold

## Implementation Guidance
- Every module exposes `phase_space()` returning coordinate representation.
- Experiment functions annotated with `@epistemically_logged` documenting hypothesis/theory/phenomenology/revision.
- Naming convention adopts philosophical lexicon (mnemosyne, eris, aletheia, kairos).

Update this ontology whenever new entities or relationships emerge. No code change proceeds without ontological alignment.
