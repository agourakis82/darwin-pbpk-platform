# =============================================================================
# QUDT Unit Mappings - Darwin PBPK Semantic Layer
# =============================================================================
# Maps pharmacokinetic units to QUDT URIs with UO (Units Ontology) crosslinks.
#
# Dual Unit System:
# - QUDT: Primary for computational semantics, dimensional analysis
# - UO: Secondary for OBO Foundry interoperability
#
# References:
# - QUDT: http://qudt.org/
# - UO: http://purl.obolibrary.org/obo/uo.owl
#
# Author: Dr. Sounio Agourakis
# Date: December 2025
# =============================================================================

module QUDTUnits

export QUDTUnit, SemanticQuantity
export QUDT_UNITS, get_qudt_unit, validate_dimension
export to_jsonld, to_quantity_value
export dimension_compatible, convert_units

# =============================================================================
# QUDT Unit Structure
# =============================================================================

"""
    QUDTUnit

Represents a unit with QUDT and UO mappings.

# Fields
- `qudt_uri`: QUDT vocabulary URI (e.g., "unit:MilliGM-PER-L")
- `uo_id`: OBO Units Ontology ID (e.g., "UO:0000274")
- `symbol`: Display symbol (e.g., "mg/L")
- `label`: Human-readable label
- `dimension`: Dimensional formula (e.g., "[M][L]^-3")
- `conversion_factor`: Factor to convert to SI base units
- `quantity_kind`: QUDT quantity kind URI
"""
struct QUDTUnit
    qudt_uri::String
    uo_id::String
    symbol::String
    label::String
    dimension::String
    conversion_factor::Float64
    quantity_kind::String
end

# Convenience constructor
function QUDTUnit(qudt_uri::String, uo_id::String, symbol::String, dimension::String, conversion_factor::Float64)
    QUDTUnit(qudt_uri, uo_id, symbol, symbol, dimension, conversion_factor, "")
end

# =============================================================================
# Semantic Quantity Structure
# =============================================================================

"""
    SemanticQuantity

A numeric value with full semantic annotations.

# Fields
- `value`: Numeric value
- `unit`: QUDTUnit reference
- `iri`: Optional IRI for this quantity instance
- `pato_quality`: Optional PATO quality type (e.g., "PATO:0001025" for rate)
- `source`: Optional provenance URI
- `uncertainty`: Optional uncertainty (distribution, parameters)
"""
struct SemanticQuantity
    value::Float64
    unit::QUDTUnit
    iri::Union{String, Nothing}
    pato_quality::Union{String, Nothing}
    source::Union{String, Nothing}
    uncertainty::Union{NamedTuple, Nothing}
end

# Convenience constructors
function SemanticQuantity(value::Float64, unit::QUDTUnit)
    SemanticQuantity(value, unit, nothing, nothing, nothing, nothing)
end

function SemanticQuantity(value::Float64, unit_symbol::String)
    unit = get_qudt_unit(unit_symbol)
    SemanticQuantity(value, unit, nothing, nothing, nothing, nothing)
end

# =============================================================================
# QUDT Unit Registry
# =============================================================================

"""
Complete QUDT unit mappings for pharmacokinetic applications.
"""
const QUDT_UNITS = Dict{String, QUDTUnit}(
    # =========================================================================
    # Mass Units
    # =========================================================================
    "mg" => QUDTUnit(
        "unit:MilliGM",
        "UO:0000022",
        "mg",
        "milligram",
        "[M]",
        1e-6,
        "quantitykind:Mass"
    ),
    "g" => QUDTUnit(
        "unit:GM",
        "UO:0000021",
        "g",
        "gram",
        "[M]",
        1e-3,
        "quantitykind:Mass"
    ),
    "kg" => QUDTUnit(
        "unit:KiloGM",
        "UO:0000009",
        "kg",
        "kilogram",
        "[M]",
        1.0,
        "quantitykind:Mass"
    ),
    "ug" => QUDTUnit(
        "unit:MicroGM",
        "UO:0000023",
        "μg",
        "microgram",
        "[M]",
        1e-9,
        "quantitykind:Mass"
    ),
    "ng" => QUDTUnit(
        "unit:NanoGM",
        "UO:0000024",
        "ng",
        "nanogram",
        "[M]",
        1e-12,
        "quantitykind:Mass"
    ),

    # =========================================================================
    # Volume Units
    # =========================================================================
    "L" => QUDTUnit(
        "unit:L",
        "UO:0000099",
        "L",
        "liter",
        "[L]^3",
        1e-3,
        "quantitykind:Volume"
    ),
    "mL" => QUDTUnit(
        "unit:MilliL",
        "UO:0000098",
        "mL",
        "milliliter",
        "[L]^3",
        1e-6,
        "quantitykind:Volume"
    ),
    "uL" => QUDTUnit(
        "unit:MicroL",
        "UO:0000101",
        "μL",
        "microliter",
        "[L]^3",
        1e-9,
        "quantitykind:Volume"
    ),

    # =========================================================================
    # Time Units
    # =========================================================================
    "h" => QUDTUnit(
        "unit:HR",
        "UO:0000032",
        "h",
        "hour",
        "[T]",
        3600.0,
        "quantitykind:Time"
    ),
    "min" => QUDTUnit(
        "unit:MIN",
        "UO:0000031",
        "min",
        "minute",
        "[T]",
        60.0,
        "quantitykind:Time"
    ),
    "s" => QUDTUnit(
        "unit:SEC",
        "UO:0000010",
        "s",
        "second",
        "[T]",
        1.0,
        "quantitykind:Time"
    ),
    "d" => QUDTUnit(
        "unit:DAY",
        "UO:0000033",
        "d",
        "day",
        "[T]",
        86400.0,
        "quantitykind:Time"
    ),
    "wk" => QUDTUnit(
        "unit:WK",
        "UO:0000034",
        "wk",
        "week",
        "[T]",
        604800.0,
        "quantitykind:Time"
    ),

    # =========================================================================
    # Concentration Units
    # =========================================================================
    "mg/L" => QUDTUnit(
        "unit:MilliGM-PER-L",
        "UO:0000274",
        "mg/L",
        "milligram per liter",
        "[M][L]^-3",
        1.0,  # mg/L is common reference
        "quantitykind:MassConcentration"
    ),
    "ug/L" => QUDTUnit(
        "unit:MicroGM-PER-L",
        "UO:0000275",
        "μg/L",
        "microgram per liter",
        "[M][L]^-3",
        1e-3,
        "quantitykind:MassConcentration"
    ),
    "ng/mL" => QUDTUnit(
        "unit:NanoGM-PER-MilliL",
        "UO:0000276",
        "ng/mL",
        "nanogram per milliliter",
        "[M][L]^-3",
        1.0,  # ng/mL = ug/L
        "quantitykind:MassConcentration"
    ),
    "mg/dL" => QUDTUnit(
        "unit:MilliGM-PER-DeciL",
        "",  # No direct UO mapping
        "mg/dL",
        "milligram per deciliter",
        "[M][L]^-3",
        10.0,  # mg/dL * 10 = mg/L
        "quantitykind:MassConcentration"
    ),
    "mol/L" => QUDTUnit(
        "unit:MOL-PER-L",
        "UO:0000062",
        "mol/L",
        "molar",
        "[N][L]^-3",
        1.0,
        "quantitykind:AmountOfSubstanceConcentration"
    ),
    "mmol/L" => QUDTUnit(
        "unit:MilliMOL-PER-L",
        "UO:0000063",
        "mmol/L",
        "millimolar",
        "[N][L]^-3",
        1e-3,
        "quantitykind:AmountOfSubstanceConcentration"
    ),
    "umol/L" => QUDTUnit(
        "unit:MicroMOL-PER-L",
        "UO:0000064",
        "μmol/L",
        "micromolar",
        "[N][L]^-3",
        1e-6,
        "quantitykind:AmountOfSubstanceConcentration"
    ),

    # =========================================================================
    # Clearance Units (Volume/Time)
    # =========================================================================
    "L/h" => QUDTUnit(
        "unit:L-PER-HR",
        "UO:0000271",
        "L/h",
        "liter per hour",
        "[L]^3[T]^-1",
        2.778e-7,  # L/h to m³/s
        "quantitykind:VolumeFlowRate"
    ),
    "mL/min" => QUDTUnit(
        "unit:MilliL-PER-MIN",
        "UO:0000270",
        "mL/min",
        "milliliter per minute",
        "[L]^3[T]^-1",
        1.667e-8,  # mL/min to m³/s
        "quantitykind:VolumeFlowRate"
    ),
    "mL/h" => QUDTUnit(
        "unit:MilliL-PER-HR",
        "",
        "mL/h",
        "milliliter per hour",
        "[L]^3[T]^-1",
        2.778e-10,
        "quantitykind:VolumeFlowRate"
    ),
    "L/h/kg" => QUDTUnit(
        "unit:L-PER-HR-KiloGM",
        "",
        "L/h/kg",
        "liter per hour per kilogram",
        "[L]^3[T]^-1[M]^-1",
        2.778e-7,
        "quantitykind:SpecificVolumeFlowRate"
    ),
    "mL/min/kg" => QUDTUnit(
        "unit:MilliL-PER-MIN-KiloGM",
        "",
        "mL/min/kg",
        "milliliter per minute per kilogram",
        "[L]^3[T]^-1[M]^-1",
        1.667e-8,
        "quantitykind:SpecificVolumeFlowRate"
    ),

    # =========================================================================
    # Rate Constants (1/Time)
    # =========================================================================
    "1/h" => QUDTUnit(
        "unit:PER-HR",
        "UO:0000223",
        "h⁻¹",
        "per hour",
        "[T]^-1",
        2.778e-4,  # 1/h to 1/s
        "quantitykind:Frequency"
    ),
    "1/min" => QUDTUnit(
        "unit:PER-MIN",
        "UO:0000222",
        "min⁻¹",
        "per minute",
        "[T]^-1",
        1.667e-2,
        "quantitykind:Frequency"
    ),
    "1/d" => QUDTUnit(
        "unit:PER-DAY",
        "",
        "d⁻¹",
        "per day",
        "[T]^-1",
        1.157e-5,
        "quantitykind:Frequency"
    ),

    # =========================================================================
    # Dimensionless / Fraction
    # =========================================================================
    "%" => QUDTUnit(
        "unit:PERCENT",
        "UO:0000187",
        "%",
        "percent",
        "1",
        0.01,
        "quantitykind:DimensionlessRatio"
    ),
    "fraction" => QUDTUnit(
        "unit:UNITLESS",
        "UO:0000186",
        "",
        "unitless fraction",
        "1",
        1.0,
        "quantitykind:DimensionlessRatio"
    ),
    "unitless" => QUDTUnit(
        "unit:UNITLESS",
        "UO:0000186",
        "",
        "unitless",
        "1",
        1.0,
        "quantitykind:DimensionlessRatio"
    ),

    # =========================================================================
    # Molecular Weight
    # =========================================================================
    "g/mol" => QUDTUnit(
        "unit:GM-PER-MOL",
        "UO:0000088",
        "g/mol",
        "gram per mole",
        "[M][N]^-1",
        1e-3,  # to kg/mol
        "quantitykind:MolarMass"
    ),
    "Da" => QUDTUnit(
        "unit:Da",
        "UO:0000221",
        "Da",
        "dalton",
        "[M][N]^-1",
        1.661e-27,  # to kg
        "quantitykind:MolarMass"
    ),
    "kDa" => QUDTUnit(
        "unit:KiloDa",
        "",
        "kDa",
        "kilodalton",
        "[M][N]^-1",
        1.661e-24,
        "quantitykind:MolarMass"
    ),

    # =========================================================================
    # Area Under Curve (Concentration × Time)
    # =========================================================================
    "mg*h/L" => QUDTUnit(
        "unit:MilliGM-HR-PER-L",
        "",
        "mg·h/L",
        "milligram hour per liter",
        "[M][L]^-3[T]",
        3600.0,  # to mg·s/L
        "quantitykind:AreaUnderCurve"
    ),
    "ug*h/L" => QUDTUnit(
        "unit:MicroGM-HR-PER-L",
        "",
        "μg·h/L",
        "microgram hour per liter",
        "[M][L]^-3[T]",
        3.6,
        "quantitykind:AreaUnderCurve"
    ),
    "ng*h/mL" => QUDTUnit(
        "unit:NanoGM-HR-PER-MilliL",
        "",
        "ng·h/mL",
        "nanogram hour per milliliter",
        "[M][L]^-3[T]",
        3600.0,
        "quantitykind:AreaUnderCurve"
    ),

    # =========================================================================
    # Dose per Body Weight
    # =========================================================================
    "mg/kg" => QUDTUnit(
        "unit:MilliGM-PER-KiloGM",
        "UO:0000308",
        "mg/kg",
        "milligram per kilogram",
        "1",  # dimensionless ratio
        1e-6,
        "quantitykind:MassRatio"
    ),
    "ug/kg" => QUDTUnit(
        "unit:MicroGM-PER-KiloGM",
        "",
        "μg/kg",
        "microgram per kilogram",
        "1",
        1e-9,
        "quantitykind:MassRatio"
    ),

    # =========================================================================
    # Surface Area
    # =========================================================================
    "m^2" => QUDTUnit(
        "unit:M2",
        "UO:0000080",
        "m²",
        "square meter",
        "[L]^2",
        1.0,
        "quantitykind:Area"
    ),
    "mg/m^2" => QUDTUnit(
        "unit:MilliGM-PER-M2",
        "",
        "mg/m²",
        "milligram per square meter",
        "[M][L]^-2",
        1e-6,
        "quantitykind:MassPerArea"
    ),
)

# =============================================================================
# Unit Lookup Functions
# =============================================================================

"""
    get_qudt_unit(symbol::String) -> QUDTUnit

Get QUDT unit by symbol.

# Arguments
- `symbol`: Unit symbol (e.g., "mg/L", "L/h")

# Returns
- QUDTUnit struct

# Throws
- KeyError if unit not found
"""
function get_qudt_unit(symbol::String)::QUDTUnit
    # Normalize symbol
    normalized = strip(symbol)

    if haskey(QUDT_UNITS, normalized)
        return QUDT_UNITS[normalized]
    end

    # Try common variations
    variations = [
        replace(normalized, "μ" => "u"),
        replace(normalized, "·" => "*"),
        replace(normalized, " " => ""),
    ]

    for var in variations
        if haskey(QUDT_UNITS, var)
            return QUDT_UNITS[var]
        end
    end

    error("Unknown unit: $symbol. Available units: $(join(keys(QUDT_UNITS), ", "))")
end

"""
    validate_dimension(unit1::QUDTUnit, unit2::QUDTUnit) -> Bool

Check if two units have compatible dimensions.
"""
function dimension_compatible(unit1::QUDTUnit, unit2::QUDTUnit)::Bool
    return unit1.dimension == unit2.dimension
end

"""
    convert_units(value::Float64, from_unit::QUDTUnit, to_unit::QUDTUnit) -> Float64

Convert a value from one unit to another (if dimensions match).
"""
function convert_units(value::Float64, from_unit::QUDTUnit, to_unit::QUDTUnit)::Float64
    if !dimension_compatible(from_unit, to_unit)
        error("Cannot convert between incompatible dimensions: $(from_unit.dimension) and $(to_unit.dimension)")
    end

    # Convert to SI base, then to target unit
    si_value = value * from_unit.conversion_factor
    return si_value / to_unit.conversion_factor
end

# =============================================================================
# JSON-LD Serialization
# =============================================================================

"""
    to_jsonld(sq::SemanticQuantity) -> Dict{String, Any}

Convert SemanticQuantity to JSON-LD representation.
"""
function to_jsonld(sq::SemanticQuantity)::Dict{String, Any}
    result = Dict{String, Any}(
        "@type" => ["qudt:QuantityValue", "obo:IAO_0000032"],
        "numericValue" => sq.value,
        "unitRef" => sq.unit.qudt_uri,
    )

    # Add UO crosslink if available
    if !isempty(sq.unit.uo_id)
        result["darwin:unitOBO"] = Dict("@id" => sq.unit.uo_id)
    end

    # Add quantity kind if available
    if !isempty(sq.unit.quantity_kind)
        result["qudt:hasQuantityKind"] = Dict("@id" => sq.unit.quantity_kind)
    end

    # Add PATO quality if specified
    if sq.pato_quality !== nothing
        result["is_about"] = Dict("@id" => sq.pato_quality)
    end

    # Add provenance if specified
    if sq.source !== nothing
        result["prov:wasDerivedFrom"] = Dict("@id" => sq.source)
    end

    # Add uncertainty if specified
    if sq.uncertainty !== nothing
        result["darwin:uncertainty"] = Dict{String, Any}(
            "@type" => "obo:STATO_0000039",
            "darwin:distribution" => string(get(sq.uncertainty, :distribution, "Normal")),
        )
        if haskey(sq.uncertainty, :sd)
            result["darwin:uncertainty"]["darwin:standardDeviation"] = sq.uncertainty.sd
        end
        if haskey(sq.uncertainty, :cv)
            result["darwin:uncertainty"]["darwin:coefficientOfVariation"] = sq.uncertainty.cv
        end
    end

    return result
end

"""
    to_quantity_value(value::Float64, unit_symbol::String; kwargs...) -> Dict{String, Any}

Create a JSON-LD QuantityValue from a numeric value and unit symbol.

# Arguments
- `value`: Numeric value
- `unit_symbol`: Unit symbol (e.g., "mg/L")
- `iri`: Optional IRI for this quantity
- `pato`: Optional PATO quality type
- `source`: Optional provenance URI

# Returns
- JSON-LD dictionary representation
"""
function to_quantity_value(
    value::Float64,
    unit_symbol::String;
    iri::Union{String, Nothing} = nothing,
    pato::Union{String, Nothing} = nothing,
    source::Union{String, Nothing} = nothing,
    uncertainty::Union{NamedTuple, Nothing} = nothing
)::Dict{String, Any}

    unit = get_qudt_unit(unit_symbol)
    sq = SemanticQuantity(value, unit, iri, pato, source, uncertainty)
    return to_jsonld(sq)
end

end # module QUDTUnits
