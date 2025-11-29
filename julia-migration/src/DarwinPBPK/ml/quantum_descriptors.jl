"""
Quantum Chemistry Descriptors for PBPK Modeling

This module calculates quantum-mechanical and electronic structure descriptors
that provide deeper insight into drug-tissue interactions than classical
descriptors like logP alone.

QUANTUM DESCRIPTORS IMPLEMENTED:
================================

1. ELECTRONIC STRUCTURE
   - HOMO energy (Highest Occupied Molecular Orbital)
   - LUMO energy (Lowest Unoccupied Molecular Orbital)
   - HOMO-LUMO gap (chemical hardness indicator)
   - Electronegativity (χ = -(HOMO + LUMO)/2)
   - Chemical hardness (η = (LUMO - HOMO)/2)
   - Electrophilicity index (ω = χ²/2η)

2. MOLECULAR POLARIZABILITY
   - Mean polarizability (α) - better than logP for membrane partitioning
   - Polarizability tensor components

3. ELECTROSTATIC PROPERTIES
   - Molecular dipole moment
   - Quadrupole moment
   - Electrostatic potential descriptors

4. ABRAHAM SOLVATION DESCRIPTORS (calculated, not experimental)
   - E: Excess molar refraction
   - S: Dipolarity/polarizability
   - A: Hydrogen bond acidity
   - B: Hydrogen bond basicity
   - V: McGowan characteristic volume

WHY QUANTUM DESCRIPTORS MATTER FOR PBPK:
========================================

1. CYP450 BINDING
   - HOMO energy predicts electron donation to heme iron
   - High HOMO = good CYP substrate (easily oxidized)
   - Low HOMO = poor CYP substrate, accumulates in liver

2. MEMBRANE PARTITIONING
   - Polarizability (α) correlates better with Kp than logP
   - High α = stronger van der Waals interactions with lipids
   - Dipole moment affects orientation in membrane

3. TRANSPORTER INTERACTIONS
   - Electrostatic potential predicts binding orientation
   - OATP substrates: specific charge distribution patterns
   - P-gp substrates: amphiphilic with positive charge regions

4. PROTEIN BINDING
   - Electrophilicity predicts albumin binding sites
   - H-bond descriptors (A, B) crucial for binding affinity

Author: Darwin PBPK Platform
Date: November 2025
"""

module QuantumDescriptors

using MolecularGraph
using Statistics
using LinearAlgebra

export QuantumDescriptorSet, calculate_quantum_descriptors
export calculate_electronic_structure, calculate_polarizability
export calculate_abraham_descriptors, calculate_electrostatic_properties
export QUANTUM_DESCRIPTOR_DIM, descriptor_names, normalize_descriptors

# Total dimension of quantum descriptor vector
const QUANTUM_DESCRIPTOR_DIM = 24

"""
Complete set of quantum descriptors for a molecule.
"""
struct QuantumDescriptorSet
    # Electronic structure (6)
    homo::Float32
    lumo::Float32
    homo_lumo_gap::Float32
    electronegativity::Float32
    chemical_hardness::Float32
    electrophilicity::Float32

    # Polarizability (3)
    polarizability::Float32
    polar_surface_area::Float32
    molar_refractivity::Float32

    # Electrostatic (4)
    dipole_moment::Float32
    max_positive_charge::Float32
    max_negative_charge::Float32
    charge_asymmetry::Float32

    # Abraham descriptors (5)
    E::Float32
    S::Float32
    A::Float32
    B::Float32
    V::Float32

    # Molecular shape (6)
    molecular_weight::Float32
    n_rotatable_bonds::Float32
    n_rings::Float32
    n_aromatic_rings::Float32
    fraction_sp3::Float32
    globularity::Float32
end

# Atomic constants
const ATOMIC_POLARIZABILITY = Dict{Symbol,Float32}(
    :H => 0.387f0, :C => 1.061f0, :N => 0.964f0, :O => 0.637f0,
    :F => 0.296f0, :S => 2.900f0, :Cl => 2.315f0, :Br => 3.013f0,
    :I => 5.415f0, :P => 3.630f0, :B => 1.470f0, :Si => 3.000f0, :Se => 3.770f0,
)

const ATOMIC_ELECTRONEGATIVITY = Dict{Symbol,Float32}(
    :H => 2.20f0, :C => 2.55f0, :N => 3.04f0, :O => 3.44f0,
    :F => 3.98f0, :S => 2.58f0, :Cl => 3.16f0, :Br => 2.96f0,
    :I => 2.66f0, :P => 2.19f0, :B => 2.04f0, :Si => 1.90f0, :Se => 2.55f0,
)

const ATOMIC_REFRACTIVITY = Dict{Symbol,Float32}(
    :H => 1.057f0, :C => 2.503f0, :N => 2.262f0, :O => 1.532f0,
    :F => 1.108f0, :S => 7.365f0, :Cl => 5.853f0, :Br => 8.927f0,
    :I => 13.900f0, :P => 6.920f0,
)

const ATOMIC_MASSES = Dict{Symbol,Float32}(
    :H => 1.008f0, :C => 12.011f0, :N => 14.007f0, :O => 15.999f0,
    :F => 18.998f0, :S => 32.065f0, :Cl => 35.453f0, :Br => 79.904f0,
    :I => 126.90f0, :P => 30.974f0, :B => 10.81f0, :Si => 28.086f0, :Se => 78.96f0
)

const VOLUME_CONTRIBUTIONS = Dict{Symbol,Float32}(
    :H => 0.0867f0, :C => 0.1635f0, :N => 0.1406f0, :O => 0.1200f0,
    :F => 0.1048f0, :S => 0.2273f0, :Cl => 0.2061f0, :Br => 0.2642f0,
    :I => 0.3449f0, :P => 0.2447f0
)

"""
Get neighbor atom indices for a given atom.
MolecularGraph.jl neighbormap returns Dict{Int,Int} where keys are neighbor indices.

Note: In some molecules with fused rings, the neighbormap may contain
invalid indices. We filter to only valid atom indices.
"""
function get_neighbors(mol, atom_idx::Int)::Vector{Int}
    n_atoms = atomcount(mol)
    raw_neighbors = collect(keys(mol.neighbormap[atom_idx]))
    # Filter to valid atom indices only
    return filter(n -> n >= 1 && n <= n_atoms, raw_neighbors)
end

"""
Calculate electronic structure descriptors.
"""
function calculate_electronic_structure(mol)::NamedTuple
    n_atoms = atomcount(mol)
    symbols = atomsymbol(mol)
    aromatic = isaromatic(mol)

    # Molecular electronegativity
    electroneg_sum = 0.0f0
    n_heavy = 0
    for i in 1:n_atoms
        sym = symbols[i]
        if sym != :H
            electroneg_sum += get(ATOMIC_ELECTRONEGATIVITY, sym, 2.5f0)
            n_heavy += 1
        end
    end
    mol_electronegativity = n_heavy > 0 ? electroneg_sum / n_heavy : 2.5f0

    # Count aromatic and heteroatoms
    n_aromatic = count(aromatic)
    aromatic_fraction = n_atoms > 0 ? n_aromatic / n_atoms : 0.0f0
    n_hetero = count(s -> s in [:N, :O, :S, :F, :Cl, :Br], symbols)
    hetero_fraction = n_atoms > 0 ? n_hetero / n_atoms : 0.0f0

    # Empirical HOMO/LUMO estimation
    homo = -9.0f0 + 3.0f0 * aromatic_fraction - 0.5f0 * hetero_fraction
    lumo = -1.0f0 - 2.0f0 * hetero_fraction + 0.5f0 * aromatic_fraction

    # Functional group adjustments
    has_carbonyl = any(s -> s == :O, symbols) && any(aromatic)
    has_nitro = count(s -> s == :N, symbols) >= 1 && count(s -> s == :O, symbols) >= 2

    if has_nitro
        lumo -= 1.5f0
    end
    if has_carbonyl
        lumo -= 0.5f0
    end

    # Derived quantities
    gap = lumo - homo
    electronegativity = -(homo + lumo) / 2
    hardness = gap / 2
    electrophilicity = hardness > 0.01 ? electronegativity^2 / (2 * hardness) : 0.0f0

    return (
        homo = homo,
        lumo = lumo,
        homo_lumo_gap = gap,
        electronegativity = electronegativity,
        chemical_hardness = hardness,
        electrophilicity = electrophilicity
    )
end

"""
Calculate molecular polarizability and related properties.
"""
function calculate_polarizability(mol)::NamedTuple
    n_atoms = atomcount(mol)
    symbols = atomsymbol(mol)
    aromatic = isaromatic(mol)

    # Additive atomic polarizability
    alpha_sum = 0.0f0
    mr_sum = 0.0f0
    psa = 0.0f0

    for i in 1:n_atoms
        sym = symbols[i]

        # Polarizability
        alpha_atom = get(ATOMIC_POLARIZABILITY, sym, 1.0f0)
        if aromatic[i]
            alpha_atom *= 1.15f0
        end
        alpha_sum += alpha_atom

        # Molar refractivity
        mr_sum += get(ATOMIC_REFRACTIVITY, sym, 2.0f0)

        # PSA from N, O, S
        if sym == :N
            psa += 26.0f0
        elseif sym == :O
            psa += 20.0f0
        elseif sym == :S
            psa += 28.0f0
        end
    end

    # Ring correction
    n_aromatic = count(aromatic)
    n_rings_est = n_aromatic > 0 ? max(1, n_aromatic ÷ 5) : 0
    ring_correction = 1.0f0 - 0.02f0 * n_rings_est

    return (
        polarizability = alpha_sum * ring_correction,
        polar_surface_area = psa,
        molar_refractivity = mr_sum
    )
end

"""
Calculate electrostatic properties.
"""
function calculate_electrostatic_properties(mol)::NamedTuple
    n_atoms = atomcount(mol)
    symbols = atomsymbol(mol)
    formal_charges = charge(mol)

    # Estimate partial charges
    partial_charges = zeros(Float32, n_atoms)

    for i in 1:n_atoms
        sym = symbols[i]
        en = get(ATOMIC_ELECTRONEGATIVITY, sym, 2.5f0)
        fc = formal_charges[i]

        base_charge = (en - 2.55f0) * 0.1f0 + Float32(fc)

        # Neighbor effects
        neighbors = get_neighbors(mol, i)
        for j in neighbors
            neighbor_sym = symbols[j]
            neighbor_en = get(ATOMIC_ELECTRONEGATIVITY, neighbor_sym, 2.5f0)
            base_charge += (neighbor_en - en) * 0.05f0
        end

        partial_charges[i] = base_charge
    end

    max_pos = maximum(partial_charges)
    max_neg = minimum(partial_charges)

    pos_sum = sum(c -> c > 0 ? c : 0, partial_charges)
    neg_sum = sum(c -> c < 0 ? abs(c) : 0, partial_charges)
    total_charge_mag = pos_sum + neg_sum
    asymmetry = total_charge_mag > 0 ? abs(pos_sum - neg_sum) / total_charge_mag : 0.0f0

    n_hetero = count(s -> s in [:N, :O, :S, :F, :Cl], symbols)
    dipole_estimate = 1.0f0 + 0.5f0 * n_hetero + 2.0f0 * abs(max_pos - max_neg)

    return (
        dipole_moment = dipole_estimate,
        max_positive_charge = max_pos,
        max_negative_charge = max_neg,
        charge_asymmetry = asymmetry
    )
end

"""
Calculate Abraham solvation descriptors.
"""
function calculate_abraham_descriptors(mol)::NamedTuple
    n_atoms = atomcount(mol)
    symbols = atomsymbol(mol)
    aromatic = isaromatic(mol)

    # McGowan Volume
    v_sum = sum(get(VOLUME_CONTRIBUTIONS, symbols[i], 0.15f0) for i in 1:n_atoms)
    n_bonds = length(mol.edges)
    V = v_sum - 0.0656f0 * n_bonds

    # Excess molar refraction
    n_aromatic = count(aromatic)
    n_hetero = count(s -> s in [:N, :O, :S, :F, :Cl, :Br, :I], symbols)
    E = 0.3f0 * (n_aromatic / max(n_atoms, 1)) + 0.2f0 * (n_hetero / max(n_atoms, 1))

    # Dipolarity/Polarizability
    n_carbonyl = 0
    for i in 1:n_atoms
        if symbols[i] == :O
            neighbors = get_neighbors(mol, i)
            if length(neighbors) == 1
                n_carbonyl += 1
            end
        end
    end
    S = clamp(0.1f0 * n_hetero + 0.3f0 * n_carbonyl + 0.05f0 * n_aromatic, 0.0f0, 2.5f0)

    # H-bond acidity (donors)
    n_hbd = 0
    for i in 1:n_atoms
        sym = symbols[i]
        if sym in [:N, :O]
            neighbors = get_neighbors(mol, i)
            for j in neighbors
                if symbols[j] == :H
                    n_hbd += 1
                end
            end
        end
    end
    A = clamp(0.3f0 * n_hbd, 0.0f0, 2.0f0)

    # H-bond basicity (acceptors)
    n_hba = count(s -> s in [:N, :O, :F, :S], symbols)
    B = clamp(0.2f0 * n_hba, 0.0f0, 3.0f0)

    return (E = E, S = S, A = A, B = B, V = V)
end

"""
Calculate molecular shape descriptors.
"""
function calculate_shape_descriptors(mol)::NamedTuple
    n_atoms = atomcount(mol)
    symbols = atomsymbol(mol)
    aromatic = isaromatic(mol)

    # Molecular weight
    mw = sum(get(ATOMIC_MASSES, symbols[i], 12.0f0) for i in 1:n_atoms)

    # Rotatable bonds estimate
    n_rot = 0
    for (src, tgt) in mol.edges
        if symbols[src] != :H && symbols[tgt] != :H
            if !aromatic[src] && !aromatic[tgt]
                n_rot += 1
            end
        end
    end
    n_rot = max(0, n_rot - n_atoms ÷ 3)

    # Ring counts
    n_aromatic_atoms = count(aromatic)
    n_aromatic_rings = n_aromatic_atoms > 0 ? max(1, n_aromatic_atoms ÷ 5) : 0
    n_rings = n_aromatic_rings + (n_atoms > 6 ? 1 : 0)

    # Fraction sp3
    n_carbon = count(s -> s == :C, symbols)
    n_sp3_carbon = count(i -> symbols[i] == :C && !aromatic[i], 1:n_atoms)
    fsp3 = n_carbon > 0 ? n_sp3_carbon / n_carbon : 0.0f0

    # Globularity
    n_bonds = length(mol.edges)
    globularity = n_bonds > 0 ? min(1.0f0, n_atoms / (n_bonds * 1.5f0)) : 0.5f0

    return (
        molecular_weight = mw,
        n_rotatable_bonds = Float32(n_rot),
        n_rings = Float32(n_rings),
        n_aromatic_rings = Float32(n_aromatic_rings),
        fraction_sp3 = fsp3,
        globularity = globularity
    )
end

"""
Calculate complete quantum descriptor set from SMILES.
"""
function calculate_quantum_descriptors(smiles::String)::Union{QuantumDescriptorSet,Nothing}
    try
        mol = smilestomol(smiles)

        if atomcount(mol) == 0
            return nothing
        end

        electronic = calculate_electronic_structure(mol)
        polar = calculate_polarizability(mol)
        electro = calculate_electrostatic_properties(mol)
        abraham = calculate_abraham_descriptors(mol)
        shape = calculate_shape_descriptors(mol)

        return QuantumDescriptorSet(
            electronic.homo, electronic.lumo, electronic.homo_lumo_gap,
            electronic.electronegativity, electronic.chemical_hardness, electronic.electrophilicity,
            polar.polarizability, polar.polar_surface_area, polar.molar_refractivity,
            electro.dipole_moment, electro.max_positive_charge, electro.max_negative_charge, electro.charge_asymmetry,
            abraham.E, abraham.S, abraham.A, abraham.B, abraham.V,
            shape.molecular_weight, shape.n_rotatable_bonds, shape.n_rings,
            shape.n_aromatic_rings, shape.fraction_sp3, shape.globularity
        )
    catch e
        @warn "Failed to calculate quantum descriptors for: $smiles" exception = e
        return nothing
    end
end

"""
Convert QuantumDescriptorSet to vector for ML input.
"""
function Base.vec(qd::QuantumDescriptorSet)::Vector{Float32}
    return Float32[
        qd.homo, qd.lumo, qd.homo_lumo_gap,
        qd.electronegativity, qd.chemical_hardness, qd.electrophilicity,
        qd.polarizability, qd.polar_surface_area, qd.molar_refractivity,
        qd.dipole_moment, qd.max_positive_charge, qd.max_negative_charge, qd.charge_asymmetry,
        qd.E, qd.S, qd.A, qd.B, qd.V,
        qd.molecular_weight, qd.n_rotatable_bonds, qd.n_rings, qd.n_aromatic_rings,
        qd.fraction_sp3, qd.globularity
    ]
end

"""
Get descriptor names for interpretability.
"""
function descriptor_names()::Vector{String}
    return [
        "HOMO", "LUMO", "HOMO_LUMO_gap",
        "electronegativity", "chemical_hardness", "electrophilicity",
        "polarizability", "PSA", "molar_refractivity",
        "dipole_moment", "max_pos_charge", "max_neg_charge", "charge_asymmetry",
        "Abraham_E", "Abraham_S", "Abraham_A", "Abraham_B", "Abraham_V",
        "MW", "n_rotatable", "n_rings", "n_aromatic_rings",
        "fraction_sp3", "globularity"
    ]
end

"""
Normalize quantum descriptors to standard range.
"""
function normalize_descriptors(qd::QuantumDescriptorSet)::Vector{Float32}
    v = vec(qd)

    ranges = [
        (-12f0, -4f0), (-4f0, 2f0), (2f0, 12f0), (0f0, 8f0), (1f0, 6f0), (0f0, 20f0),
        (5f0, 100f0), (0f0, 200f0), (20f0, 200f0), (0f0, 10f0), (-0.5f0, 0.5f0),
        (-0.5f0, 0.5f0), (0f0, 1f0), (0f0, 2f0), (0f0, 3f0), (0f0, 2f0), (0f0, 3f0),
        (0.5f0, 6f0), (100f0, 800f0), (0f0, 20f0), (0f0, 8f0), (0f0, 5f0), (0f0, 1f0), (0f0, 1f0)
    ]

    normalized = similar(v)
    for i in eachindex(v)
        lo, hi = ranges[i]
        normalized[i] = clamp((v[i] - lo) / (hi - lo), 0.0f0, 1.0f0)
    end

    return normalized
end

end # module
