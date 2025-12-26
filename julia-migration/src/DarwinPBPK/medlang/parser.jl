"""
MedLang DSL Parser for Darwin PBPK Platform

First Real Implementation of MedLang DSL (github.com/agourakis82/medlang)
Parses MedLang Track D PBPK model definitions into Julia structs.

Implements:
- MedLang-D grammar (minimal v0.1)
- Unit-typed parameters (Quantity<u, τ>)
- PBPK model definitions (14-compartment)
- Population variability (random effects)
- Dosing timelines

Grammar Reference: medlang_d_minimal_grammar_v0.md
Core Spec Reference: medlang_core_spec_v0.1.md

Author: Dr. Sounio Agourakis
Date: November 2025
"""

module MedLangParser

using ..ODEPBPKSolver: PBPKParams, PBPK_ORGANS, NUM_ORGANS

export parse_medlang, MedLangAST, ModelDef, StateDef, ParamDef, ODEEquation
export PopulationDef, TimelineDef, DoseEvent, ObserveEvent
export ParseError, validate_units
export AbsorptionDef, FirstPassDef, RouteType
export ROUTE_IV, ROUTE_ORAL, ROUTE_IM, ROUTE_SC, ROUTE_INFUSION
# SOTA v0.2 Neural-Symbolic exports
export CompoundDef, NeuralNetSpec, NeuralODEDef, MechanisticODEDef
export InferenceDef, PharmacodynamicsDef, TargetDef
export NeuralPredictDef, PriorDef, VirtualPopulationDef, SensitivityAnalysisDef

#=============================================================================
  Token Types
=============================================================================#

@enum TokenType begin
    # Keywords
    TOK_MODEL
    TOK_POPULATION
    TOK_MEASURE
    TOK_TIMELINE
    TOK_COHORT
    TOK_STATE
    TOK_PARAM
    TOK_OBS
    TOK_RAND
    TOK_INPUT
    TOK_AT
    TOK_DOSE
    TOK_OBSERVE
    TOK_TO
    TOK_USE_MEASURE
    TOK_BIND_PARAMS
    TOK_ORGAN
    TOK_CLEARANCE

    # SOTA Neural-Symbolic Keywords (v0.2)
    TOK_NEURAL_ODE
    TOK_MECHANISTIC_ODE
    TOK_NEURAL_PREDICT
    TOK_NEURAL_ENCODE
    TOK_COMPOUND
    TOK_INFERENCE
    TOK_LIKELIHOOD
    TOK_PRIOR
    TOK_POSTERIOR
    TOK_CONSTRAINT
    TOK_REGULARIZE
    TOK_SENSITIVITY
    TOK_IDENTIFIABILITY
    TOK_VIRTUAL_POPULATION
    TOK_PHARMACODYNAMICS
    TOK_TARGET
    TOK_DISSOLUTION
    TOK_PERMEABILITY
    TOK_PARTITION
    TOK_ABSORPTION
    TOK_FIRSTPASS
    TOK_ROUTE
    # Additional SOTA tokens
    TOK_SMILES
    TOK_EMBEDDING
    TOK_NETWORK
    TOK_LAYER
    TOK_ACTIVATION
    TOK_METHOD

    # Literals
    TOK_IDENT
    TOK_FLOAT
    TOK_INT
    TOK_STRING
    TOK_UNIT

    # Operators
    TOK_PLUS
    TOK_MINUS
    TOK_STAR
    TOK_SLASH
    TOK_CARET
    TOK_EQ
    TOK_EQEQ
    TOK_LT
    TOK_GT
    TOK_TILDE

    # Delimiters
    TOK_LPAREN
    TOK_RPAREN
    TOK_LBRACE
    TOK_RBRACE
    TOK_LBRACKET
    TOK_RBRACKET
    TOK_COLON
    TOK_COMMA
    TOK_DOT
    TOK_SEMICOLON

    # Special
    TOK_D_DT  # d/dt operator
    TOK_EOF
    TOK_ERROR
end

struct Token
    type::TokenType
    value::String
    line::Int
    col::Int
end

#=============================================================================
  AST Node Types
=============================================================================#

abstract type ASTNode end

"""Unit expression (e.g., mg, L/h, mg/L)"""
struct UnitExpr <: ASTNode
    base::String
    power::Int
    compound::Vector{Tuple{String,Int}}  # [(unit, power), ...]
end

UnitExpr(base::String) = UnitExpr(base, 1, Tuple{String,Int}[])

"""Type expression with optional unit"""
struct TypeExpr <: ASTNode
    name::String
    unit::Union{UnitExpr,Nothing}
end

"""Expression node"""
abstract type Expr <: ASTNode end

struct LiteralExpr <: Expr
    value::Union{Float64,Int,String}
    unit::Union{UnitExpr,Nothing}
end

struct IdentExpr <: Expr
    name::String
end

struct QualifiedExpr <: Expr
    parts::Vector{String}  # e.g., ["patient", "MAP"]
end

struct BinaryExpr <: Expr
    op::Symbol
    left::Expr
    right::Expr
end

struct UnaryExpr <: Expr
    op::Symbol
    operand::Expr
end

struct CallExpr <: Expr
    func::String
    args::Vector{Expr}
end

"""State declaration"""
struct StateDef <: ASTNode
    name::String
    type::TypeExpr
    initial::Union{Expr,Nothing}
end

"""Parameter declaration"""
struct ParamDef <: ASTNode
    name::String
    type::TypeExpr
    default::Union{Expr,Nothing}
end

"""Observable declaration"""
struct ObsDef <: ASTNode
    name::String
    type::TypeExpr
    expr::Expr
end

"""ODE equation: dX/dt = expr"""
struct ODEEquation <: ASTNode
    state::String
    rhs::Expr
end

"""Random effect declaration"""
struct RandomEffectDef <: ASTNode
    name::String
    type::TypeExpr
    distribution::CallExpr
end

"""Organ definition for PBPK"""
struct OrganDef <: ASTNode
    name::String
    volume::Expr           # Volume (L)
    blood_flow::Expr       # Blood flow (L/h)
    partition_coeff::Expr  # Kp
end

"""Clearance mechanism"""
struct ClearanceDef <: ASTNode
    mechanism::Symbol  # :hepatic, :renal, :biliary
    rate::Expr
end

"""Absorption parameters for oral dosing"""
struct AbsorptionDef <: ASTNode
    ka::Expr              # Absorption rate constant (1/h)
    f::Union{Expr,Nothing}  # Bioavailability (fraction)
    lag::Union{Expr,Nothing}  # Lag time (h)
end

"""First-pass metabolism parameters"""
struct FirstPassDef <: ASTNode
    fg::Expr  # Fraction escaping gut metabolism
    fh::Expr  # Fraction escaping hepatic first-pass
end

"""Route of administration"""
@enum RouteType begin
    ROUTE_IV
    ROUTE_ORAL
    ROUTE_IM
    ROUTE_SC
    ROUTE_INFUSION
end

#=============================================================================
  SOTA: Neural-Symbolic AST Nodes (v0.2)
=============================================================================#

"""Compound definition with molecular identity"""
struct CompoundDef <: ASTNode
    name::String
    smiles::String
    mw::Float64
    logP::Union{Float64,Nothing}
    pKa::Union{Float64,Nothing}
    embedding_model::Union{String,Nothing}
end

"""Neural network specification within DSL"""
struct NeuralNetSpec <: ASTNode
    input_features::Vector{String}
    hidden_layers::Vector{Int}
    activation::String
    output_dim::Int
end

"""Neural ODE block"""
struct NeuralODEDef <: ASTNode
    name::String
    state::String
    network::NeuralNetSpec
    constraints::Vector{Expr}
    regularization::Union{Expr,Nothing}
end

"""Mechanistic ODE block (explicit physics)"""
struct MechanisticODEDef <: ASTNode
    name::String
    equations::Vector{ODEEquation}
end

"""Neural prediction call"""
struct NeuralPredictDef <: ASTNode
    input_expr::Expr
    target::String
    model_name::String
end

"""Inference block for Bayesian estimation"""
struct InferenceDef <: ASTNode
    likelihood::Vector{Expr}
    method::String
    method_params::Dict{String,Any}
end

"""Prior distribution specification"""
struct PriorDef <: ASTNode
    param_name::String
    distribution::String
    params::Vector{Float64}
    source::Union{String,Nothing}
end

"""Pharmacodynamics block"""
struct PharmacodynamicsDef <: ASTNode
    name::String
    effect_equation::Expr
    states::Vector{StateDef}
    odes::Vector{ODEEquation}
end

"""Target (receptor/enzyme) definition for QSP"""
struct TargetDef <: ASTNode
    name::String
    expression::Dict{String,Float64}  # tissue -> expression level
    turnover::Float64
end

"""Virtual population definition"""
struct VirtualPopulationDef <: ASTNode
    name::String
    n_subjects::Int
    physiology::Dict{String,Any}
    enzyme_variability::Dict{String,Any}
    disease_modifiers::Dict{String,Any}
end

"""Sensitivity analysis specification"""
struct SensitivityAnalysisDef <: ASTNode
    model_name::String
    local_analysis::Bool
    global_analysis::Bool
    parameters::Vector{String}
    method::String
end

export CompoundDef, NeuralNetSpec, NeuralODEDef, MechanisticODEDef
export NeuralPredictDef, InferenceDef, PriorDef, PharmacodynamicsDef
export TargetDef, VirtualPopulationDef, SensitivityAnalysisDef

"""Model definition"""
struct ModelDef <: ASTNode
    name::String
    states::Vector{StateDef}
    params::Vector{ParamDef}
    organs::Vector{OrganDef}
    clearances::Vector{ClearanceDef}
    odes::Vector{ODEEquation}
    observables::Vector{ObsDef}
    absorption::Union{AbsorptionDef,Nothing}
    firstpass::Union{FirstPassDef,Nothing}
    route::RouteType
    # SOTA v0.2 fields
    compound::Union{CompoundDef,Nothing}
    neural_odes::Vector{NeuralODEDef}
    mechanistic_odes::Vector{MechanisticODEDef}
    inference::Union{InferenceDef,Nothing}
    pharmacodynamics::Vector{PharmacodynamicsDef}
    targets::Vector{TargetDef}
end

# Constructor with default values for backward compatibility (v0.1 API)
function ModelDef(name, states, params, organs, clearances, odes, observables)
    ModelDef(
        name, states, params, organs, clearances, odes, observables,
        nothing, nothing, ROUTE_IV,  # absorption, firstpass, route
        nothing, NeuralODEDef[], MechanisticODEDef[],  # compound, neural_odes, mechanistic_odes
        nothing, PharmacodynamicsDef[], TargetDef[]    # inference, pharmacodynamics, targets
    )
end

# Constructor with v0.1 absorption/firstpass/route (backward compatibility)
function ModelDef(name, states, params, organs, clearances, odes, observables, absorption, firstpass, route)
    ModelDef(
        name, states, params, organs, clearances, odes, observables,
        absorption, firstpass, route,
        nothing, NeuralODEDef[], MechanisticODEDef[],
        nothing, PharmacodynamicsDef[], TargetDef[]
    )
end

"""Dose event in timeline"""
struct DoseEvent <: ASTNode
    time::Expr
    amount::Expr
    target::String  # compartment name
end

"""Observe event in timeline"""
struct ObserveEvent <: ASTNode
    time::Expr
    observable::String
end

"""Timeline definition"""
struct TimelineDef <: ASTNode
    name::String
    events::Vector{Union{DoseEvent,ObserveEvent}}
end

"""Population definition"""
struct PopulationDef <: ASTNode
    name::String
    model::String
    params::Vector{ParamDef}
    random_effects::Vector{RandomEffectDef}
    inputs::Vector{ParamDef}
end

"""Full MedLang AST"""
struct MedLangAST <: ASTNode
    models::Vector{ModelDef}
    populations::Vector{PopulationDef}
    timelines::Vector{TimelineDef}
end

#=============================================================================
  Parse Error
=============================================================================#

struct ParseError <: Exception
    message::String
    line::Int
    col::Int
end

Base.showerror(io::IO, e::ParseError) = print(io, "ParseError at line $(e.line), col $(e.col): $(e.message)")

#=============================================================================
  Lexer
=============================================================================#

const KEYWORDS = Dict(
    # Core v0.1 keywords
    "model" => TOK_MODEL,
    "population" => TOK_POPULATION,
    "measure" => TOK_MEASURE,
    "timeline" => TOK_TIMELINE,
    "cohort" => TOK_COHORT,
    "state" => TOK_STATE,
    "param" => TOK_PARAM,
    "obs" => TOK_OBS,
    "rand" => TOK_RAND,
    "input" => TOK_INPUT,
    "at" => TOK_AT,
    "dose" => TOK_DOSE,
    "observe" => TOK_OBSERVE,
    "to" => TOK_TO,
    "use_measure" => TOK_USE_MEASURE,
    "bind_params" => TOK_BIND_PARAMS,
    "organ" => TOK_ORGAN,
    "clearance" => TOK_CLEARANCE,
    "partition" => TOK_PARTITION,
    "absorption" => TOK_ABSORPTION,
    "firstpass" => TOK_FIRSTPASS,
    "route" => TOK_ROUTE,
    # SOTA v0.2 Neural-Symbolic keywords
    "neural_ode" => TOK_NEURAL_ODE,
    "mechanistic_ode" => TOK_MECHANISTIC_ODE,
    "neural_predict" => TOK_NEURAL_PREDICT,
    "neural_encode" => TOK_NEURAL_ENCODE,
    "compound" => TOK_COMPOUND,
    "inference" => TOK_INFERENCE,
    "likelihood" => TOK_LIKELIHOOD,
    "prior" => TOK_PRIOR,
    "posterior" => TOK_POSTERIOR,
    "constraint" => TOK_CONSTRAINT,
    "regularize" => TOK_REGULARIZE,
    "sensitivity" => TOK_SENSITIVITY,
    "identifiability" => TOK_IDENTIFIABILITY,
    "virtual_population" => TOK_VIRTUAL_POPULATION,
    "pharmacodynamics" => TOK_PHARMACODYNAMICS,
    "target" => TOK_TARGET,
    "dissolution" => TOK_DISSOLUTION,
    "permeability" => TOK_PERMEABILITY,
    # Additional SOTA keywords
    "smiles" => TOK_SMILES,
    "embedding" => TOK_EMBEDDING,
    "network" => TOK_NETWORK,
    "layer" => TOK_LAYER,
    "activation" => TOK_ACTIVATION,
    "method" => TOK_METHOD,
)

# Standard units recognized by MedLang
const UNITS = Set([
    # Mass
    "mg", "g", "kg", "ug", "ng", "pg",
    # Volume
    "L", "mL", "uL", "dL",
    # Time
    "h", "min", "s", "d",
    # Concentration
    "mg/L", "ug/L", "ng/mL", "uM", "nM",
    # Clearance
    "L/h", "mL/min", "L/h/kg",
    # Rate
    "1/h", "1/min",
    # Other
    "mmHg", "kg/m2", "mol", "mmol",
])

mutable struct Lexer
    source::String
    pos::Int
    line::Int
    col::Int
    tokens::Vector{Token}
end

Lexer(source::String) = Lexer(source, 1, 1, 1, Token[])

function peek(lex::Lexer, offset::Int=0)::Char
    p = lex.pos + offset
    p <= length(lex.source) ? lex.source[p] : '\0'
end

function advance!(lex::Lexer)::Char
    c = peek(lex)
    lex.pos += 1
    if c == '\n'
        lex.line += 1
        lex.col = 1
    else
        lex.col += 1
    end
    return c
end

function skip_whitespace!(lex::Lexer)
    while !eof(lex)
        c = peek(lex)
        if c in (' ', '\t', '\n', '\r')
            advance!(lex)
        elseif c == '/' && peek(lex, 1) == '/'
            # Line comment
            while !eof(lex) && peek(lex) != '\n'
                advance!(lex)
            end
        elseif c == '/' && peek(lex, 1) == '*'
            # Block comment
            advance!(lex)
            advance!(lex)
            while !eof(lex)
                if peek(lex) == '*' && peek(lex, 1) == '/'
                    advance!(lex)
                    advance!(lex)
                    break
                end
                advance!(lex)
            end
        else
            break
        end
    end
end

eof(lex::Lexer) = lex.pos > length(lex.source)

function scan_number!(lex::Lexer)::Token
    start_line, start_col = lex.line, lex.col
    buf = IOBuffer()

    # Integer part
    while !eof(lex) && isdigit(peek(lex))
        write(buf, advance!(lex))
    end

    is_float = false

    # Decimal part
    if !eof(lex) && peek(lex) == '.' && isdigit(peek(lex, 1))
        is_float = true
        write(buf, advance!(lex))  # '.'
        while !eof(lex) && isdigit(peek(lex))
            write(buf, advance!(lex))
        end
    end

    # Exponent
    if !eof(lex) && peek(lex) in ('e', 'E')
        is_float = true
        write(buf, advance!(lex))  # 'e' or 'E'
        if !eof(lex) && peek(lex) in ('+', '-')
            write(buf, advance!(lex))
        end
        while !eof(lex) && isdigit(peek(lex))
            write(buf, advance!(lex))
        end
    end

    value = String(take!(buf))

    # Check for unit suffix (e.g., 100_mg)
    if !eof(lex) && peek(lex) == '_'
        advance!(lex)  # consume '_'
        unit_buf = IOBuffer()
        while !eof(lex) && (isletter(peek(lex)) || peek(lex) == '/')
            write(unit_buf, advance!(lex))
        end
        unit = String(take!(unit_buf))
        return Token(TOK_UNIT, value * "_" * unit, start_line, start_col)
    end

    return Token(is_float ? TOK_FLOAT : TOK_INT, value, start_line, start_col)
end

function scan_ident!(lex::Lexer)::Token
    start_line, start_col = lex.line, lex.col
    buf = IOBuffer()

    while !eof(lex) && (isletter(peek(lex)) || isdigit(peek(lex)) || peek(lex) == '_')
        write(buf, advance!(lex))
    end

    value = String(take!(buf))

    # Check for keyword
    tok_type = get(KEYWORDS, value, TOK_IDENT)

    return Token(tok_type, value, start_line, start_col)
end

function scan_string!(lex::Lexer)::Token
    start_line, start_col = lex.line, lex.col
    advance!(lex)  # consume opening quote
    buf = IOBuffer()

    while !eof(lex) && peek(lex) != '"'
        if peek(lex) == '\\'
            advance!(lex)
            c = advance!(lex)
            if c == 'n'
                write(buf, '\n')
            elseif c == 't'
                write(buf, '\t')
            else
                write(buf, c)
            end
        else
            write(buf, advance!(lex))
        end
    end

    if !eof(lex)
        advance!(lex)  # consume closing quote
    end

    return Token(TOK_STRING, String(take!(buf)), start_line, start_col)
end

function next_token!(lex::Lexer)::Token
    skip_whitespace!(lex)

    if eof(lex)
        return Token(TOK_EOF, "", lex.line, lex.col)
    end

    start_line, start_col = lex.line, lex.col
    c = peek(lex)

    # Numbers
    if isdigit(c)
        return scan_number!(lex)
    end

    # Special case: d/dt (must check BEFORE general identifiers!)
    if c == 'd' && peek(lex, 1) == '/'
        advance!(lex)  # 'd'
        advance!(lex)  # '/'
        if peek(lex) == 'd' && peek(lex, 1) == 't'
            advance!(lex)  # 'd'
            advance!(lex)  # 't'
            return Token(TOK_D_DT, "d/dt", start_line, start_col)
        end
        # Backtrack (not d/dt) - but we consumed '/', need to return slash
        # Actually we can't easily backtrack, so return the 'd' and let '/' be next
        return Token(TOK_IDENT, "d", start_line, start_col)
    end

    # Identifiers and keywords
    if isletter(c) || c == '_'
        return scan_ident!(lex)
    end

    # String literals
    if c == '"'
        return scan_string!(lex)
    end

    # Single/double character tokens
    advance!(lex)

    return if c == '+'
        Token(TOK_PLUS, "+", start_line, start_col)
    elseif c == '-'
        Token(TOK_MINUS, "-", start_line, start_col)
    elseif c == '*'
        Token(TOK_STAR, "*", start_line, start_col)
    elseif c == '/'
        Token(TOK_SLASH, "/", start_line, start_col)
    elseif c == '^'
        Token(TOK_CARET, "^", start_line, start_col)
    elseif c == '='
        if peek(lex) == '='
            advance!(lex)
            Token(TOK_EQEQ, "==", start_line, start_col)
        else
            Token(TOK_EQ, "=", start_line, start_col)
        end
    elseif c == '<'
        Token(TOK_LT, "<", start_line, start_col)
    elseif c == '>'
        Token(TOK_GT, ">", start_line, start_col)
    elseif c == '~'
        Token(TOK_TILDE, "~", start_line, start_col)
    elseif c == '('
        Token(TOK_LPAREN, "(", start_line, start_col)
    elseif c == ')'
        Token(TOK_RPAREN, ")", start_line, start_col)
    elseif c == '{'
        Token(TOK_LBRACE, "{", start_line, start_col)
    elseif c == '}'
        Token(TOK_RBRACE, "}", start_line, start_col)
    elseif c == '['
        Token(TOK_LBRACKET, "[", start_line, start_col)
    elseif c == ']'
        Token(TOK_RBRACKET, "]", start_line, start_col)
    elseif c == ':'
        Token(TOK_COLON, ":", start_line, start_col)
    elseif c == ','
        Token(TOK_COMMA, ",", start_line, start_col)
    elseif c == '.'
        Token(TOK_DOT, ".", start_line, start_col)
    elseif c == ';'
        Token(TOK_SEMICOLON, ";", start_line, start_col)
    else
        Token(TOK_ERROR, string(c), start_line, start_col)
    end
end

function tokenize(source::String)::Vector{Token}
    lex = Lexer(source)
    tokens = Token[]

    while true
        tok = next_token!(lex)
        push!(tokens, tok)
        if tok.type == TOK_EOF
            break
        end
    end

    return tokens
end

#=============================================================================
  Parser
=============================================================================#

mutable struct Parser
    tokens::Vector{Token}
    pos::Int
end

Parser(tokens::Vector{Token}) = Parser(tokens, 1)

current(p::Parser) = p.pos <= length(p.tokens) ? p.tokens[p.pos] : p.tokens[end]
peek_token(p::Parser, offset::Int=0) = p.pos + offset <= length(p.tokens) ? p.tokens[p.pos+offset] : p.tokens[end]

function advance!(p::Parser)::Token
    tok = current(p)
    p.pos += 1
    return tok
end

function expect!(p::Parser, type::TokenType)::Token
    tok = current(p)
    if tok.type != type
        throw(ParseError("Expected $(type), got $(tok.type)", tok.line, tok.col))
    end
    return advance!(p)
end

function match(p::Parser, types::TokenType...)::Bool
    return current(p).type in types
end

function check(p::Parser, type::TokenType)::Bool
    return current(p).type == type
end

#-----------------------------------------------------------------------------
# Expression Parsing (Pratt Parser)
#-----------------------------------------------------------------------------

function precedence(type::TokenType)::Int
    return if type == TOK_PLUS || type == TOK_MINUS
        1
    elseif type == TOK_STAR || type == TOK_SLASH
        2
    elseif type == TOK_CARET
        3
    else
        0
    end
end

function parse_primary(p::Parser)::Expr
    tok = current(p)

    if tok.type == TOK_FLOAT
        advance!(p)
        return LiteralExpr(parse(Float64, tok.value), nothing)
    elseif tok.type == TOK_INT
        advance!(p)
        return LiteralExpr(parse(Int, tok.value), nothing)
    elseif tok.type == TOK_UNIT
        advance!(p)
        # Parse "100_mg" format
        parts = split(tok.value, "_", limit=2)
        val = parse(Float64, parts[1])
        unit = UnitExpr(String(parts[2]))
        return LiteralExpr(val, unit)
    elseif tok.type == TOK_STRING
        advance!(p)
        return LiteralExpr(tok.value, nothing)
    elseif tok.type == TOK_IDENT
        name = advance!(p).value

        # Check for function call
        if check(p, TOK_LPAREN)
            advance!(p)  # '('
            args = Expr[]
            if !check(p, TOK_RPAREN)
                push!(args, parse_expr(p))
                while check(p, TOK_COMMA)
                    advance!(p)  # ','
                    push!(args, parse_expr(p))
                end
            end
            expect!(p, TOK_RPAREN)
            return CallExpr(name, args)
        end

        # Check for qualified name (a.b.c)
        if check(p, TOK_DOT)
            parts = [name]
            while check(p, TOK_DOT)
                advance!(p)  # '.'
                push!(parts, expect!(p, TOK_IDENT).value)
            end
            return QualifiedExpr(parts)
        end

        return IdentExpr(name)
    elseif tok.type == TOK_LPAREN
        advance!(p)  # '('
        expr = parse_expr(p)
        expect!(p, TOK_RPAREN)
        return expr
    elseif tok.type == TOK_MINUS
        advance!(p)
        return UnaryExpr(:-, parse_primary(p))
    else
        throw(ParseError("Unexpected token: $(tok.type)", tok.line, tok.col))
    end
end

function parse_expr(p::Parser, min_prec::Int=0)::Expr
    left = parse_primary(p)

    while true
        tok = current(p)
        prec = precedence(tok.type)

        if prec <= min_prec
            break
        end

        op = if tok.type == TOK_PLUS
            :+
        elseif tok.type == TOK_MINUS
            :-
        elseif tok.type == TOK_STAR
            :*
        elseif tok.type == TOK_SLASH
            :/
        elseif tok.type == TOK_CARET
            :^
        else
            break
        end

        advance!(p)
        right = parse_expr(p, prec)
        left = BinaryExpr(op, left, right)
    end

    return left
end

#-----------------------------------------------------------------------------
# Type Parsing
#-----------------------------------------------------------------------------

function parse_type(p::Parser)::TypeExpr
    name = expect!(p, TOK_IDENT).value

    # Check for unit annotation: Type<unit>
    if check(p, TOK_LT)
        advance!(p)  # '<'
        unit_name = expect!(p, TOK_IDENT).value
        # Handle compound units like L/h
        if check(p, TOK_SLASH)
            advance!(p)
            denom = expect!(p, TOK_IDENT).value
            unit_name *= "/" * denom
        end
        expect!(p, TOK_GT)  # '>'
        return TypeExpr(name, UnitExpr(unit_name))
    end

    return TypeExpr(name, nothing)
end

#-----------------------------------------------------------------------------
# Model Parsing
#-----------------------------------------------------------------------------

function parse_state_def(p::Parser)::StateDef
    expect!(p, TOK_STATE)
    name = expect!(p, TOK_IDENT).value
    expect!(p, TOK_COLON)
    type = parse_type(p)

    initial = nothing
    if check(p, TOK_EQ)
        advance!(p)
        initial = parse_expr(p)
    end

    return StateDef(name, type, initial)
end

function parse_param_def(p::Parser)::ParamDef
    expect!(p, TOK_PARAM)
    name = expect!(p, TOK_IDENT).value
    expect!(p, TOK_COLON)
    type = parse_type(p)

    default = nothing
    if check(p, TOK_EQ)
        advance!(p)
        default = parse_expr(p)
    end

    return ParamDef(name, type, default)
end

function parse_obs_def(p::Parser)::ObsDef
    expect!(p, TOK_OBS)
    name = expect!(p, TOK_IDENT).value
    expect!(p, TOK_COLON)
    type = parse_type(p)
    expect!(p, TOK_EQ)
    expr = parse_expr(p)

    return ObsDef(name, type, expr)
end

function parse_ode_equation(p::Parser)::ODEEquation
    # d/dt syntax: dX/dt = expr  OR  d/dt X = expr
    if check(p, TOK_D_DT)
        advance!(p)  # d/dt
        state = expect!(p, TOK_IDENT).value
    else
        # dX_dt = expr (alternative syntax)
        name = expect!(p, TOK_IDENT).value
        if startswith(name, "d") && endswith(name, "_dt")
            state = name[2:end-3]  # extract X from dX_dt
        else
            throw(ParseError("Expected ODE equation (dX/dt or dX_dt)", current(p).line, current(p).col))
        end
    end

    expect!(p, TOK_EQ)
    rhs = parse_expr(p)

    return ODEEquation(state, rhs)
end

function parse_organ_def(p::Parser)::OrganDef
    expect!(p, TOK_ORGAN)
    name = expect!(p, TOK_IDENT).value
    expect!(p, TOK_LBRACE)

    volume = LiteralExpr(1.0, UnitExpr("L"))
    blood_flow = LiteralExpr(0.0, UnitExpr("L/h"))
    partition_coeff = LiteralExpr(1.0, nothing)

    while !check(p, TOK_RBRACE)
        field = expect!(p, TOK_IDENT).value
        expect!(p, TOK_COLON)

        if field == "volume" || field == "V"
            volume = parse_expr(p)
        elseif field == "blood_flow" || field == "Q"
            blood_flow = parse_expr(p)
        elseif field == "partition" || field == "Kp"
            partition_coeff = parse_expr(p)
        end

        if check(p, TOK_COMMA)
            advance!(p)
        end
    end

    expect!(p, TOK_RBRACE)

    return OrganDef(name, volume, blood_flow, partition_coeff)
end

function parse_clearance_def(p::Parser)::ClearanceDef
    expect!(p, TOK_CLEARANCE)
    mechanism = Symbol(expect!(p, TOK_IDENT).value)  # hepatic, renal, etc.
    expect!(p, TOK_COLON)
    rate = parse_expr(p)

    return ClearanceDef(mechanism, rate)
end

function parse_absorption_def(p::Parser)::AbsorptionDef
    expect!(p, TOK_ABSORPTION)
    expect!(p, TOK_LBRACE)

    ka = LiteralExpr(1.0, UnitExpr("1/h"))  # Default Ka
    f = nothing   # Bioavailability (optional)
    lag = nothing # Lag time (optional)

    while !check(p, TOK_RBRACE)
        field = expect!(p, TOK_IDENT).value
        expect!(p, TOK_COLON)

        if field == "ka" || field == "Ka"
            ka = parse_expr(p)
        elseif field == "f" || field == "F" || field == "bioavailability"
            f = parse_expr(p)
        elseif field == "lag" || field == "tlag"
            lag = parse_expr(p)
        end

        if check(p, TOK_COMMA)
            advance!(p)
        end
    end

    expect!(p, TOK_RBRACE)

    return AbsorptionDef(ka, f, lag)
end

function parse_firstpass_def(p::Parser)::FirstPassDef
    expect!(p, TOK_FIRSTPASS)
    expect!(p, TOK_LBRACE)

    fg = LiteralExpr(1.0, nothing)  # Default: no gut metabolism
    fh = LiteralExpr(1.0, nothing)  # Default: no hepatic first-pass

    while !check(p, TOK_RBRACE)
        field = expect!(p, TOK_IDENT).value
        expect!(p, TOK_COLON)

        if field == "fg" || field == "Fg" || field == "gut"
            fg = parse_expr(p)
        elseif field == "fh" || field == "Fh" || field == "hepatic"
            fh = parse_expr(p)
        end

        if check(p, TOK_COMMA)
            advance!(p)
        end
    end

    expect!(p, TOK_RBRACE)

    return FirstPassDef(fg, fh)
end

function parse_route_def(p::Parser)::RouteType
    expect!(p, TOK_ROUTE)
    expect!(p, TOK_COLON)
    route_name = lowercase(expect!(p, TOK_IDENT).value)

    return if route_name == "iv" || route_name == "intravenous"
        ROUTE_IV
    elseif route_name == "oral" || route_name == "po"
        ROUTE_ORAL
    elseif route_name == "im" || route_name == "intramuscular"
        ROUTE_IM
    elseif route_name == "sc" || route_name == "subcutaneous"
        ROUTE_SC
    elseif route_name == "infusion"
        ROUTE_INFUSION
    else
        ROUTE_IV  # Default
    end
end

#=============================================================================
  SOTA v0.2 Neural-Symbolic Block Parsers
=============================================================================#

"""
Parse compound definition block:
    compound {
        smiles: "CC(=O)Oc1ccccc1C(=O)O"
        mw: 180.16 g/mol
        logP: 1.2
        pKa: [3.5, 13.4]
        embedding: neural_encode(smiles, model="ChemBERTa")
    }
"""
function parse_compound_def(p::Parser)::CompoundDef
    expect!(p, TOK_COMPOUND)
    expect!(p, TOK_LBRACE)

    name = "default"
    smiles = ""
    mw = 0.0
    logP = nothing
    pKa = nothing
    embedding_model = nothing

    while !check(p, TOK_RBRACE) && !check(p, TOK_EOF)
        if check(p, TOK_IDENT) || check(p, TOK_SMILES)
            field = advance!(p).value
            expect!(p, TOK_COLON)

            if field == "name"
                name = expect!(p, TOK_STRING).value
            elseif field == "smiles"
                smiles = expect!(p, TOK_STRING).value
            elseif field == "mw"
                mw_expr = parse_expr(p)
                mw = mw_expr isa LiteralExpr ? mw_expr.value : 0.0
            elseif field == "logP" || field == "logp"
                logP = parse_expr(p)
            elseif field == "pKa" || field == "pka"
                pKa = parse_expr(p)
            elseif field == "embedding"
                # Parse neural_encode(...) call
                embedding_model = parse_expr(p)
            end

            if check(p, TOK_COMMA)
                advance!(p)
            end
        else
            advance!(p)
        end
    end

    expect!(p, TOK_RBRACE)
    return CompoundDef(name, smiles, mw, logP, pKa, embedding_model)
end

"""
Parse neural network specification:
    network {
        layers: [64, 32, 16]
        activation: "swish"
        dropout: 0.1
    }
"""
function parse_network_spec(p::Parser)::NeuralNetSpec
    expect!(p, TOK_LBRACE)

    layers = Int[]
    activation = "tanh"
    dropout = 0.0

    while !check(p, TOK_RBRACE) && !check(p, TOK_EOF)
        if check(p, TOK_IDENT) || check(p, TOK_LAYER) || check(p, TOK_ACTIVATION)
            field = advance!(p).value
            expect!(p, TOK_COLON)

            if field == "layers"
                # Parse array [64, 32, 16]
                expect!(p, TOK_LBRACKET)
                while !check(p, TOK_RBRACKET)
                    if check(p, TOK_INT) || check(p, TOK_FLOAT)
                        push!(layers, Int(parse(Float64, advance!(p).value)))
                    end
                    if check(p, TOK_COMMA)
                        advance!(p)
                    end
                end
                expect!(p, TOK_RBRACKET)
            elseif field == "activation"
                activation = expect!(p, TOK_STRING).value
            elseif field == "dropout"
                dropout = parse(Float64, advance!(p).value)
            end

            if check(p, TOK_COMMA)
                advance!(p)
            end
        else
            advance!(p)
        end
    end

    expect!(p, TOK_RBRACE)
    return NeuralNetSpec(layers, activation, dropout)
end

"""
Parse neural ODE block:
    neural_ode tissue_dynamics {
        state: C_tissue
        network { layers: [64, 32], activation: "swish" }
        constraint: dC >= 0  # Physiological constraint
        regularize: L2(1e-4)
    }
"""
function parse_neural_ode_def(p::Parser)::NeuralODEDef
    expect!(p, TOK_NEURAL_ODE)
    name = expect!(p, TOK_IDENT).value
    expect!(p, TOK_LBRACE)

    state = ""
    network = NeuralNetSpec(Int[], "tanh", 0.0)
    constraints = Expr[]
    regularization = nothing

    while !check(p, TOK_RBRACE) && !check(p, TOK_EOF)
        if check(p, TOK_STATE)
            advance!(p)
            expect!(p, TOK_COLON)
            state = expect!(p, TOK_IDENT).value
        elseif check(p, TOK_NETWORK) || (check(p, TOK_IDENT) && current(p).value == "network")
            advance!(p)
            if check(p, TOK_COLON)
                advance!(p)
            end
            network = parse_network_spec(p)
        elseif check(p, TOK_CONSTRAINT) || (check(p, TOK_IDENT) && current(p).value == "constraint")
            advance!(p)
            expect!(p, TOK_COLON)
            push!(constraints, parse_expr(p))
        elseif check(p, TOK_REGULARIZE) || (check(p, TOK_IDENT) && current(p).value == "regularize")
            advance!(p)
            expect!(p, TOK_COLON)
            regularization = parse_expr(p)
        else
            advance!(p)
        end

        if check(p, TOK_COMMA)
            advance!(p)
        end
    end

    expect!(p, TOK_RBRACE)
    return NeuralODEDef(name, state, network, constraints, regularization)
end

"""
Parse mechanistic ODE block:
    mechanistic_ode elimination {
        d/dt(C_blood) = -CL/V * C_blood
        constraint: C_blood >= 0
    }
"""
function parse_mechanistic_ode_def(p::Parser)::MechanisticODEDef
    expect!(p, TOK_MECHANISTIC_ODE)
    name = expect!(p, TOK_IDENT).value
    expect!(p, TOK_LBRACE)

    equations = ODEEquation[]
    constraints = Expr[]

    while !check(p, TOK_RBRACE) && !check(p, TOK_EOF)
        if check(p, TOK_D_DT) || (check(p, TOK_IDENT) && startswith(current(p).value, "d"))
            push!(equations, parse_ode_equation(p))
        elseif check(p, TOK_CONSTRAINT) || (check(p, TOK_IDENT) && current(p).value == "constraint")
            advance!(p)
            expect!(p, TOK_COLON)
            push!(constraints, parse_expr(p))
        else
            advance!(p)
        end
    end

    expect!(p, TOK_RBRACE)
    return MechanisticODEDef(name, equations, constraints)
end

"""
Parse inference block (Bayesian):
    inference {
        likelihood {
            obs_Cp ~ Normal(C_blood, sigma)
        }
        prior {
            CL ~ LogNormal(log(5.0), 0.3)
            V ~ LogNormal(log(50.0), 0.2)
            sigma ~ HalfNormal(0.1)
        }
        method: NUTS(1000, 0.65)
    }
"""
function parse_inference_def(p::Parser)::InferenceDef
    expect!(p, TOK_INFERENCE)
    expect!(p, TOK_LBRACE)

    likelihood = Expr[]
    method = "NUTS"
    method_params = Dict{String,Any}()

    while !check(p, TOK_RBRACE) && !check(p, TOK_EOF)
        if check(p, TOK_LIKELIHOOD) || (check(p, TOK_IDENT) && current(p).value == "likelihood")
            advance!(p)
            expect!(p, TOK_LBRACE)
            while !check(p, TOK_RBRACE)
                push!(likelihood, parse_expr(p))
                if check(p, TOK_COMMA)
                    advance!(p)
                end
            end
            expect!(p, TOK_RBRACE)
        elseif check(p, TOK_PRIOR) || (check(p, TOK_IDENT) && current(p).value == "prior")
            # Prior block - parse but store in likelihood for now (unified)
            advance!(p)
            expect!(p, TOK_LBRACE)
            while !check(p, TOK_RBRACE)
                push!(likelihood, parse_expr(p))
                if check(p, TOK_COMMA)
                    advance!(p)
                end
            end
            expect!(p, TOK_RBRACE)
        elseif check(p, TOK_METHOD) || (check(p, TOK_IDENT) && current(p).value == "method")
            advance!(p)
            expect!(p, TOK_COLON)
            method = expect!(p, TOK_IDENT).value
            # Parse method params if present
            if check(p, TOK_LPAREN)
                advance!(p)
                param_idx = 1
                while !check(p, TOK_RPAREN)
                    if check(p, TOK_INT) || check(p, TOK_FLOAT)
                        method_params["arg$param_idx"] = parse(Float64, advance!(p).value)
                        param_idx += 1
                    end
                    if check(p, TOK_COMMA)
                        advance!(p)
                    end
                end
                expect!(p, TOK_RPAREN)
            end
        else
            advance!(p)
        end
    end

    expect!(p, TOK_RBRACE)
    return InferenceDef(likelihood, method, method_params)
end

"""
Parse pharmacodynamics block:
    pharmacodynamics effect {
        model: Emax
        E0: 0.0
        Emax: 100.0
        EC50: 10.0 ng/mL
        hill: 1.0
    }
"""
function parse_pharmacodynamics_def(p::Parser)::PharmacodynamicsDef
    expect!(p, TOK_PHARMACODYNAMICS)
    name = expect!(p, TOK_IDENT).value
    expect!(p, TOK_LBRACE)

    pd_model = "Emax"
    params = Dict{String,Expr}()

    while !check(p, TOK_RBRACE) && !check(p, TOK_EOF)
        if check(p, TOK_IDENT)
            field = advance!(p).value
            expect!(p, TOK_COLON)

            if field == "model"
                pd_model = expect!(p, TOK_IDENT).value
            else
                params[field] = parse_expr(p)
            end

            if check(p, TOK_COMMA)
                advance!(p)
            end
        else
            advance!(p)
        end
    end

    expect!(p, TOK_RBRACE)
    return PharmacodynamicsDef(name, pd_model, params)
end

"""
Parse target block:
    target receptor {
        type: GPCR
        Kd: 1.5 nM
        kon: 1e6 1/(M*s)
        koff: 1e-3 1/s
    }
"""
function parse_target_def(p::Parser)::TargetDef
    expect!(p, TOK_TARGET)
    name = expect!(p, TOK_IDENT).value
    expect!(p, TOK_LBRACE)

    target_type = "receptor"
    params = Dict{String,Expr}()

    while !check(p, TOK_RBRACE) && !check(p, TOK_EOF)
        if check(p, TOK_IDENT)
            field = advance!(p).value
            expect!(p, TOK_COLON)

            if field == "type"
                target_type = expect!(p, TOK_IDENT).value
            else
                params[field] = parse_expr(p)
            end

            if check(p, TOK_COMMA)
                advance!(p)
            end
        else
            advance!(p)
        end
    end

    expect!(p, TOK_RBRACE)
    return TargetDef(name, target_type, params)
end

#=============================================================================
  Model Parsing
=============================================================================#

function parse_model_def(p::Parser)::ModelDef
    expect!(p, TOK_MODEL)
    name = expect!(p, TOK_IDENT).value
    expect!(p, TOK_LBRACE)

    # v0.1 fields
    states = StateDef[]
    params = ParamDef[]
    organs = OrganDef[]
    clearances = ClearanceDef[]
    odes = ODEEquation[]
    observables = ObsDef[]
    absorption = nothing
    firstpass = nothing
    route = ROUTE_IV

    # SOTA v0.2 fields
    compound = nothing
    neural_odes = NeuralODEDef[]
    mechanistic_odes = MechanisticODEDef[]
    inference = nothing
    pharmacodynamics = PharmacodynamicsDef[]
    targets = TargetDef[]

    while !check(p, TOK_RBRACE) && !check(p, TOK_EOF)
        if check(p, TOK_STATE)
            push!(states, parse_state_def(p))
        elseif check(p, TOK_PARAM)
            push!(params, parse_param_def(p))
        elseif check(p, TOK_ORGAN)
            push!(organs, parse_organ_def(p))
        elseif check(p, TOK_CLEARANCE)
            push!(clearances, parse_clearance_def(p))
        elseif check(p, TOK_OBS)
            push!(observables, parse_obs_def(p))
        elseif check(p, TOK_ABSORPTION)
            absorption = parse_absorption_def(p)
        elseif check(p, TOK_FIRSTPASS)
            firstpass = parse_firstpass_def(p)
        elseif check(p, TOK_ROUTE)
            route = parse_route_def(p)
        # SOTA v0.2 blocks
        elseif check(p, TOK_COMPOUND)
            compound = parse_compound_def(p)
        elseif check(p, TOK_NEURAL_ODE)
            push!(neural_odes, parse_neural_ode_def(p))
        elseif check(p, TOK_MECHANISTIC_ODE)
            push!(mechanistic_odes, parse_mechanistic_ode_def(p))
        elseif check(p, TOK_INFERENCE)
            inference = parse_inference_def(p)
        elseif check(p, TOK_PHARMACODYNAMICS)
            push!(pharmacodynamics, parse_pharmacodynamics_def(p))
        elseif check(p, TOK_TARGET)
            push!(targets, parse_target_def(p))
        elseif check(p, TOK_D_DT) || (check(p, TOK_IDENT) && startswith(current(p).value, "d"))
            push!(odes, parse_ode_equation(p))
        else
            advance!(p)  # Skip unknown token
        end
    end

    expect!(p, TOK_RBRACE)

    return ModelDef(
        name, states, params, organs, clearances, odes, observables,
        absorption, firstpass, route,
        compound, neural_odes, mechanistic_odes,
        inference, pharmacodynamics, targets
    )
end

#-----------------------------------------------------------------------------
# Timeline Parsing
#-----------------------------------------------------------------------------

function parse_dose_event(p::Parser)::DoseEvent
    expect!(p, TOK_AT)
    time = parse_expr(p)
    expect!(p, TOK_COLON)
    expect!(p, TOK_DOSE)
    expect!(p, TOK_LBRACE)

    amount = LiteralExpr(0.0, UnitExpr("mg"))
    target = "blood"

    while !check(p, TOK_RBRACE)
        # Field name can be TOK_IDENT or TOK_TO (since 'to' is a keyword)
        if check(p, TOK_IDENT)
            field = advance!(p).value
        elseif check(p, TOK_TO)
            advance!(p)
            field = "to"
        else
            throw(ParseError("Expected field name in dose event", current(p).line, current(p).col))
        end
        expect!(p, TOK_EQ)

        if field == "amount"
            amount = parse_expr(p)
        elseif field == "to"
            target = expect!(p, TOK_IDENT).value
        end

        if check(p, TOK_COMMA)
            advance!(p)
        end
    end

    expect!(p, TOK_RBRACE)

    return DoseEvent(time, amount, target)
end

function parse_observe_event(p::Parser)::ObserveEvent
    expect!(p, TOK_AT)
    time = parse_expr(p)
    expect!(p, TOK_COLON)
    expect!(p, TOK_OBSERVE)
    observable = expect!(p, TOK_IDENT).value

    return ObserveEvent(time, observable)
end

function parse_timeline_def(p::Parser)::TimelineDef
    expect!(p, TOK_TIMELINE)
    name = expect!(p, TOK_IDENT).value
    expect!(p, TOK_LBRACE)

    events = Union{DoseEvent,ObserveEvent}[]

    while !check(p, TOK_RBRACE) && !check(p, TOK_EOF)
        if check(p, TOK_AT)
            # Look ahead to determine event type
            saved_pos = p.pos
            advance!(p)  # 'at'
            parse_expr(p)  # time
            expect!(p, TOK_COLON)

            if check(p, TOK_DOSE)
                p.pos = saved_pos
                push!(events, parse_dose_event(p))
            elseif check(p, TOK_OBSERVE)
                p.pos = saved_pos
                push!(events, parse_observe_event(p))
            else
                p.pos = saved_pos
                advance!(p)
            end
        else
            advance!(p)
        end
    end

    expect!(p, TOK_RBRACE)

    return TimelineDef(name, events)
end

#-----------------------------------------------------------------------------
# Population Parsing
#-----------------------------------------------------------------------------

function parse_random_effect(p::Parser)::RandomEffectDef
    expect!(p, TOK_RAND)
    name = expect!(p, TOK_IDENT).value
    expect!(p, TOK_COLON)
    type = parse_type(p)
    expect!(p, TOK_TILDE)

    # Distribution (e.g., Normal(0, 0.3))
    dist_name = expect!(p, TOK_IDENT).value
    expect!(p, TOK_LPAREN)
    args = Expr[]
    if !check(p, TOK_RPAREN)
        push!(args, parse_expr(p))
        while check(p, TOK_COMMA)
            advance!(p)
            push!(args, parse_expr(p))
        end
    end
    expect!(p, TOK_RPAREN)

    return RandomEffectDef(name, type, CallExpr(dist_name, args))
end

function parse_population_def(p::Parser)::PopulationDef
    expect!(p, TOK_POPULATION)
    name = expect!(p, TOK_IDENT).value
    expect!(p, TOK_LBRACE)

    model_name = ""
    params = ParamDef[]
    random_effects = RandomEffectDef[]
    inputs = ParamDef[]

    while !check(p, TOK_RBRACE) && !check(p, TOK_EOF)
        if check(p, TOK_MODEL)
            advance!(p)
            model_name = expect!(p, TOK_IDENT).value
        elseif check(p, TOK_PARAM)
            push!(params, parse_param_def(p))
        elseif check(p, TOK_RAND)
            push!(random_effects, parse_random_effect(p))
        elseif check(p, TOK_INPUT)
            advance!(p)
            inp_name = expect!(p, TOK_IDENT).value
            expect!(p, TOK_COLON)
            inp_type = parse_type(p)
            push!(inputs, ParamDef(inp_name, inp_type, nothing))
        else
            advance!(p)
        end
    end

    expect!(p, TOK_RBRACE)

    return PopulationDef(name, model_name, params, random_effects, inputs)
end

#-----------------------------------------------------------------------------
# Top-Level Parsing
#-----------------------------------------------------------------------------

function parse_program(p::Parser)::MedLangAST
    models = ModelDef[]
    populations = PopulationDef[]
    timelines = TimelineDef[]

    while !check(p, TOK_EOF)
        if check(p, TOK_MODEL)
            push!(models, parse_model_def(p))
        elseif check(p, TOK_POPULATION)
            push!(populations, parse_population_def(p))
        elseif check(p, TOK_TIMELINE)
            push!(timelines, parse_timeline_def(p))
        else
            advance!(p)  # Skip unknown top-level token
        end
    end

    return MedLangAST(models, populations, timelines)
end

#=============================================================================
  Public API
=============================================================================#

"""
Parse MedLang source code into an AST.

# Arguments
- `source::String`: MedLang source code

# Returns
- `MedLangAST`: Parsed abstract syntax tree

# Example
```julia
source = \"\"\"
model OneCmptOral {
    state A_gut : DoseMass = 0_mg
    state A_central : DoseMass = 0_mg
    param Ka : RateConst = 1.0_1/h
    param CL : Clearance = 10.0_L/h
    param V : Volume = 50.0_L

    d/dt A_gut = -Ka * A_gut
    d/dt A_central = Ka * A_gut - (CL / V) * A_central

    obs C_plasma : ConcMass = A_central / V
}
\"\"\"
ast = parse_medlang(source)
```
"""
function parse_medlang(source::String)::MedLangAST
    tokens = tokenize(source)
    parser = Parser(tokens)
    return parse_program(parser)
end

# ============================================================================
# Dimensional Analysis - Unit Validation System
# ============================================================================

"""
Canonical unit representation for dimensional analysis.
Uses base SI dimensions: [Mass, Length, Time, Amount]
"""
struct Dimension
    mass::Int      # kg
    length::Int    # m (for volume: m³)
    time::Int      # s (or h)
    amount::Int    # mol
end

Dimension() = Dimension(0, 0, 0, 0)

# Dimension arithmetic
Base.:+(d1::Dimension, d2::Dimension) = Dimension(d1.mass + d2.mass, d1.length + d2.length, d1.time + d2.time, d1.amount + d2.amount)
Base.:-(d1::Dimension, d2::Dimension) = Dimension(d1.mass - d2.mass, d1.length - d2.length, d1.time - d2.time, d1.amount - d2.amount)
Base.:*(d::Dimension, n::Int) = Dimension(d.mass * n, d.length * n, d.time * n, d.amount * n)
Base.:(==)(d1::Dimension, d2::Dimension) = d1.mass == d2.mass && d1.length == d2.length && d1.time == d2.time && d1.amount == d2.amount

"""
Map unit strings to dimensions.
"""
const UNIT_DIMENSIONS = Dict{String,Dimension}(
    # Mass units
    "mg" => Dimension(1, 0, 0, 0),
    "g" => Dimension(1, 0, 0, 0),
    "kg" => Dimension(1, 0, 0, 0),
    "ug" => Dimension(1, 0, 0, 0),
    "ng" => Dimension(1, 0, 0, 0),
    "pg" => Dimension(1, 0, 0, 0),

    # Volume units (length³)
    "L" => Dimension(0, 3, 0, 0),
    "mL" => Dimension(0, 3, 0, 0),
    "uL" => Dimension(0, 3, 0, 0),
    "dL" => Dimension(0, 3, 0, 0),

    # Time units
    "h" => Dimension(0, 0, 1, 0),
    "min" => Dimension(0, 0, 1, 0),
    "s" => Dimension(0, 0, 1, 0),
    "d" => Dimension(0, 0, 1, 0),

    # Amount
    "mol" => Dimension(0, 0, 0, 1),
    "mmol" => Dimension(0, 0, 0, 1),
    "umol" => Dimension(0, 0, 0, 1),

    # Dimensionless
    "" => Dimension(0, 0, 0, 0),
    "1" => Dimension(0, 0, 0, 0),
)

"""
Map semantic types to expected dimensions.
"""
const TYPE_DIMENSIONS = Dict{String,Dimension}(
    "DoseMass" => Dimension(1, 0, 0, 0),        # [M]
    "Clearance" => Dimension(0, 3, -1, 0),       # [L³/T] = L/h
    "Volume" => Dimension(0, 3, 0, 0),           # [L³]
    "RateConst" => Dimension(0, 0, -1, 0),       # [1/T]
    "Concentration" => Dimension(1, -3, 0, 0),   # [M/L³] = mg/L
    "ConcMass" => Dimension(1, -3, 0, 0),        # [M/L³]
    "ConcMolar" => Dimension(0, -3, 0, 1),       # [mol/L³]
    "Fraction" => Dimension(0, 0, 0, 0),         # dimensionless
    "Time" => Dimension(0, 0, 1, 0),             # [T]
    "MolWeight" => Dimension(1, 0, 0, -1),       # [M/mol] = g/mol
    "FlowRate" => Dimension(0, 3, -1, 0),        # [L³/T]
    "BloodFlow" => Dimension(0, 3, -1, 0),       # [L³/T]
    "PartitionCoeff" => Dimension(0, 0, 0, 0),   # dimensionless (Kp)
)

"""
Get dimension from UnitExpr.
"""
function get_dimension(unit::UnitExpr)::Dimension
    # Get base unit dimension
    base_dim = get(UNIT_DIMENSIONS, unit.base, nothing)
    if base_dim === nothing
        @warn "Unknown unit: $(unit.base)"
        return Dimension()
    end

    # Apply power
    result = base_dim * unit.power

    # Handle compound units (e.g., L/h, mg/L)
    for (comp_unit, comp_power) in unit.compound
        comp_dim = get(UNIT_DIMENSIONS, comp_unit, nothing)
        if comp_dim !== nothing
            result = result + (comp_dim * comp_power)
        else
            @warn "Unknown compound unit: $comp_unit"
        end
    end

    return result
end

"""
Get dimension from TypeExpr.
"""
function get_dimension(type::TypeExpr)::Union{Dimension,Nothing}
    # First check if type name maps to known dimension
    dim = get(TYPE_DIMENSIONS, type.name, nothing)
    if dim !== nothing
        return dim
    end

    # If type has explicit unit, use that
    if type.unit !== nothing
        return get_dimension(type.unit)
    end

    return nothing
end

"""
Infer dimension from expression.
"""
function infer_dimension(expr::Expr, context::Dict{String,Dimension}=Dict{String,Dimension}())::Union{Dimension,Nothing}
    if expr isa LiteralExpr
        if expr.unit !== nothing
            return get_dimension(expr.unit)
        end
        # Dimensionless literal
        return Dimension()

    elseif expr isa IdentExpr
        return get(context, expr.name, nothing)

    elseif expr isa QualifiedExpr
        # Look up in context with qualified name
        full_name = join(expr.parts, ".")
        return get(context, full_name, nothing)

    elseif expr isa BinaryExpr
        left_dim = infer_dimension(expr.left, context)
        right_dim = infer_dimension(expr.right, context)

        if left_dim === nothing || right_dim === nothing
            return nothing
        end

        if expr.op == :+ || expr.op == :-
            # Addition/subtraction requires same dimensions
            if left_dim == right_dim
                return left_dim
            else
                return nothing  # Dimension mismatch
            end
        elseif expr.op == :*
            # Multiplication adds dimensions
            return left_dim + right_dim
        elseif expr.op == :/
            # Division subtracts dimensions
            return left_dim - right_dim
        elseif expr.op == :^
            # Power: only valid if right is dimensionless integer
            if right_dim == Dimension() && expr.right isa LiteralExpr
                power = Int(expr.right.value)
                return left_dim * power
            end
            return nothing
        end

    elseif expr isa UnaryExpr
        return infer_dimension(expr.operand, context)

    elseif expr isa CallExpr
        # Most functions preserve or return specific dimensions
        if expr.func in ["exp", "log", "sin", "cos", "tan", "sqrt"]
            # exp, log, trig functions require dimensionless input
            arg_dim = infer_dimension(expr.args[1], context)
            if arg_dim == Dimension()
                return Dimension()  # Returns dimensionless
            elseif expr.func == "sqrt" && arg_dim !== nothing
                # sqrt halves dimensions
                return Dimension(arg_dim.mass ÷ 2, arg_dim.length ÷ 2, arg_dim.time ÷ 2, arg_dim.amount ÷ 2)
            end
        elseif expr.func == "pow" && length(expr.args) >= 2
            base_dim = infer_dimension(expr.args[1], context)
            if base_dim !== nothing && expr.args[2] isa LiteralExpr
                power = Int(expr.args[2].value)
                return base_dim * power
            end
        end
        return nothing
    end

    return nothing
end

"""
Validation result with detailed error messages.
"""
struct ValidationResult
    valid::Bool
    errors::Vector{String}
    warnings::Vector{String}
end

ValidationResult() = ValidationResult(true, String[], String[])

function add_error!(result::ValidationResult, msg::String)
    push!(result.errors, msg)
    return ValidationResult(false, result.errors, result.warnings)
end

function add_warning!(result::ValidationResult, msg::String)
    push!(result.warnings, msg)
    return result
end

"""
Validate unit consistency in expressions.

# Arguments
- `expr::Expr`: Expression to validate
- `expected::Union{Dimension, Nothing}`: Expected dimension
- `context::Dict{String, Dimension}`: Variable -> Dimension mapping

# Returns
- `ValidationResult`: Validation result with errors/warnings
"""
function validate_units(
    expr::Expr,
    expected::Union{Dimension,Nothing}=nothing,
    context::Dict{String,Dimension}=Dict{String,Dimension}()
)::ValidationResult
    result = ValidationResult()

    inferred = infer_dimension(expr, context)

    if inferred === nothing
        result = add_warning!(result, "Could not infer dimension for expression")
        return result
    end

    if expected !== nothing && inferred != expected
        result = add_error!(result,
            "Dimension mismatch: expected $(format_dimension(expected)), got $(format_dimension(inferred))")
    end

    # Recursively validate sub-expressions for addition/subtraction
    if expr isa BinaryExpr && (expr.op == :+ || expr.op == :-)
        left_dim = infer_dimension(expr.left, context)
        right_dim = infer_dimension(expr.right, context)

        if left_dim !== nothing && right_dim !== nothing && left_dim != right_dim
            result = add_error!(result,
                "Cannot $(expr.op == :+ ? "add" : "subtract") quantities with different dimensions: " *
                "$(format_dimension(left_dim)) vs $(format_dimension(right_dim))")
        end
    end

    return result
end

"""
Format dimension for human-readable output.
"""
function format_dimension(d::Dimension)::String
    parts = String[]

    if d.mass != 0
        push!(parts, d.mass == 1 ? "M" : "M^$(d.mass)")
    end
    if d.length != 0
        push!(parts, d.length == 1 ? "L" : "L^$(d.length)")
    end
    if d.time != 0
        push!(parts, d.time == 1 ? "T" : "T^$(d.time)")
    end
    if d.amount != 0
        push!(parts, d.amount == 1 ? "N" : "N^$(d.amount)")
    end

    if isempty(parts)
        return "dimensionless"
    end

    return "[" * join(parts, "·") * "]"
end

"""
Validate a complete model's dimensional consistency.
"""
function validate_model_units(model::ModelDef)::ValidationResult
    result = ValidationResult()
    context = Dict{String,Dimension}()

    # Build context from states and parameters
    for state in model.states
        dim = get_dimension(state.type)
        if dim !== nothing
            context[state.name] = dim
        end
    end

    for param in model.params
        dim = get_dimension(param.type)
        if dim !== nothing
            context[param.name] = dim
        end
    end

    # Validate ODE equations: d/dt [State] must have dimension [State]/[Time]
    for ode in model.odes
        if haskey(context, ode.state)
            state_dim = context[ode.state]
            expected_rhs_dim = state_dim - Dimension(0, 0, 1, 0)  # [State]/[T]

            ode_result = validate_units(ode.rhs, expected_rhs_dim, context)
            if !ode_result.valid
                for err in ode_result.errors
                    result = add_error!(result, "ODE d/dt $(ode.state): $err")
                end
            end
            for warn in ode_result.warnings
                result = add_warning!(result, "ODE d/dt $(ode.state): $warn")
            end
        end
    end

    # Validate observables
    for obs in model.observables
        obs_dim = get_dimension(obs.type)
        if obs_dim !== nothing
            obs_result = validate_units(obs.expr, obs_dim, context)
            if !obs_result.valid
                for err in obs_result.errors
                    result = add_error!(result, "Observable $(obs.name): $err")
                end
            end
        end
    end

    return result
end

# Export dimensional analysis functions
export Dimension, ValidationResult, validate_units, validate_model_units
export infer_dimension, get_dimension, format_dimension

end # module
