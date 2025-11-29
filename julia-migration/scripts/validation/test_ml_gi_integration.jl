# ===========================================================================
# ML → GI MODEL INTEGRATION TEST
# ===========================================================================
# Tests the integration of ML-based transporter prediction with
# the mechanistic GI absorption model.
#
# Workflow:
# 1. Input: Drug SMILES
# 2. ML: Multimodal encoder → Transporter prediction
# 3. Mechanistic: Predicted transporters → GI absorption simulation
# ===========================================================================

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))

using Printf

println("=" ^ 70)
println("ML → GI MODEL INTEGRATION TEST")
println("=" ^ 70)
println()

# ===========================================================================
# STEP 1: Load ML Transporter Predictor
# ===========================================================================

println("Loading ML modules...")

# Load transporter predictor (includes multimodal encoder)
include(joinpath(@__DIR__, "..", "..", "src", "DarwinPBPK", "ml", "transporter_predictor.jl"))
using .TransporterPredictor

println("  ✓ TransporterPredictor loaded")
println("  ✓ $(N_TRANSPORTERS) transporters defined: ", join(INTESTINAL_TRANSPORTERS, ", "))
println()

# ===========================================================================
# STEP 2: Load GI Absorption Model
# ===========================================================================

println("Loading GI absorption model...")

include(joinpath(@__DIR__, "..", "..", "src", "DarwinPBPK", "compartments", "gi_tract.jl"))
using .GITract

println("  ✓ GITract loaded")
println()

# ===========================================================================
# STEP 3: Test Drugs with SMILES
# ===========================================================================

# Test drugs with known transporter substrates
test_drugs = [
    (name = "Metformin", smiles = "CN(C)C(=N)NC(=N)N",
     expected_transporters = [:OCT1, :OCT3], observed_F = 0.55),

    (name = "Gabapentin", smiles = "NCC1(CCCCC1)CC(=O)O",
     expected_transporters = [:LAT1, :LAT2], observed_F = 0.60),

    (name = "Cephalexin", smiles = "CC1=C(N2C(SC1)C(C2=O)NC(=O)C(N)c3ccccc3)C(=O)O",
     expected_transporters = [:PEPT1], observed_F = 0.90),

    (name = "Digoxin", smiles = "CC1OC(CC(O)C1O)OC2C(O)CC(OC3C(O)CC(OC4CCC5(C)C(CCC6C5CCC7(C)C(CCC67)C8=CC(=O)OC8)C4)OC3C)OC2C",
     expected_transporters = [:PGP], observed_F = 0.75),

    (name = "Rosuvastatin", smiles = "CC(C)c1nc(N(C)S(C)(=O)=O)nc(c1/C=C/C(O)CC(O)CC(=O)O)c2ccc(F)cc2",
     expected_transporters = [:OATP2B1, :BCRP], observed_F = 0.20),

    (name = "Theophylline", smiles = "Cn1cnc2c1c(=O)[nH]c(=O)n2C",
     expected_transporters = [:ENT1], observed_F = 0.96),
]

println("=" ^ 70)
println("TEST 1: ML TRANSPORTER PREDICTION")
println("=" ^ 70)
println()

# Initialize model (random weights - not trained)
println("Initializing TransporterPredictorModel (untrained)...")
model = TransporterPredictorModel(use_gnn = true, use_quantum = false)  # Skip quantum for speed
println("  ✓ Model initialized")
println()

println("-" ^ 70)
@printf("%-15s %-25s %-25s\n", "Drug", "Expected Transporters", "Predicted (random)")
println("-" ^ 70)

for drug in test_drugs
    # Get ML predictions
    try
        raw_preds = model(drug.smiles)
        predictions = interpret_predictions(raw_preds)

        # Get predicted substrates (>50% probability)
        predicted = get_substrate_transporters(predictions, min_probability = 0.3)
        predicted_str = isempty(predicted) ? "none" : join(predicted, ", ")
        expected_str = join(drug.expected_transporters, ", ")

        @printf("%-15s %-25s %-25s\n", drug.name, expected_str, predicted_str)
    catch e
        @printf("%-15s %-25s ERROR: %s\n", drug.name, join(drug.expected_transporters, ", "), e)
    end
end

println("-" ^ 70)
println()
println("Note: Predictions are RANDOM because model is not trained.")
println("      With proper training data, ML would predict transporters from SMILES.")
println()

# ===========================================================================
# STEP 4: Integration with GI Model
# ===========================================================================

println("=" ^ 70)
println("TEST 2: ML → GI MODEL INTEGRATION")
println("=" ^ 70)
println()

println("Simulating absorption with ML-predicted transporters...")
println()

println("-" ^ 70)
@printf("%-15s %8s %8s %8s %10s\n", "Drug", "Obs F%", "ML→GI F%", "Error%", "P-gp ER")
println("-" ^ 70)

for drug in test_drugs
    try
        # Get ML predictions
        raw_preds = model(drug.smiles)
        predictions = interpret_predictions(raw_preds)
        gi_params = predictions_to_gi_params(predictions)

        # Extract drug properties (simplified - would come from ML too)
        # For demo, we use pre-computed values
        drug_props = Dict(
            "Metformin" => (logP = -2.6, MW = 129.2, pKa = 12.4, charge = :base),
            "Gabapentin" => (logP = -1.1, MW = 171.2, pKa = 3.7, charge = :zwitterion),
            "Cephalexin" => (logP = -0.7, MW = 347.4, pKa = 5.2, charge = :zwitterion),
            "Digoxin" => (logP = 1.3, MW = 780.9, pKa = nothing, charge = :neutral),
            "Rosuvastatin" => (logP = -0.3, MW = 481.5, pKa = 4.6, charge = :acid),
            "Theophylline" => (logP = -0.8, MW = 180.2, pKa = 8.6, charge = :neutral),
        )

        props = drug_props[drug.name]

        # Run GI simulation with ML-predicted P-gp ER
        sim = simulate_oral_absorption_enhanced(
            drug_name = drug.name,
            dose_mg = 100.0,
            logP = props.logP,
            MW = props.MW,
            solubility_mg_mL = 10.0,
            pKa = props.pKa,
            charge_type = props.charge,
            intrinsic_er = gi_params.pgp_efflux_ratio,  # From ML!
            simulation_time_h = 24.0
        )

        pred_F = sim.F
        obs_F = drug.observed_F
        error_pct = abs(pred_F - obs_F) / obs_F * 100

        @printf("%-15s %7.1f%% %7.1f%% %7.1f%% %9.1f\n",
                drug.name, obs_F * 100, pred_F * 100, error_pct, gi_params.pgp_efflux_ratio)

    catch e
        @printf("%-15s %7.1f%% ERROR    -         -\n", drug.name, drug.observed_F * 100)
        @warn "Error processing $(drug.name)" exception = e
    end
end

println("-" ^ 70)
println()

# ===========================================================================
# STEP 5: Architecture Summary
# ===========================================================================

println("=" ^ 70)
println("ARCHITECTURE SUMMARY: ML → MECHANISTIC INTEGRATION")
println("=" ^ 70)
println()
println("""
┌─────────────────────────────────────────────────────────────────────┐
│                        DRUG SMILES INPUT                            │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    MULTIMODAL ENCODER                               │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                │
│  │ SMILES GRU   │ │ GNN (GAT)    │ │ Quantum Desc │                │
│  │   768d       │ │   256d       │ │    128d      │                │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘                │
│         │                │                │                         │
│         └────────────────┼────────────────┘                         │
│                          ▼                                          │
│                ┌──────────────────┐                                 │
│                │ Cross-Attention  │                                 │
│                │ Fusion (512d)    │                                 │
│                └────────┬─────────┘                                 │
└─────────────────────────┼───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  TRANSPORTER PREDICTOR                              │
│                                                                     │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │ PEPT1   │ │ OCT1/3  │ │ OATP2B1 │ │ P-gp    │ │ BCRP    │ ...  │
│  │ prob    │ │ prob    │ │ prob    │ │ prob+Km │ │ prob    │       │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘       │
└───────┼──────────┼──────────┼──────────┼──────────┼─────────────────┘
        │          │          │          │          │
        └──────────┴──────────┴──────────┴──────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 MECHANISTIC GI MODEL                                │
│                                                                     │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐               │
│  │ Dissolution │ → │ Permeability│ → │ First-Pass  │               │
│  │ Noyes-Whit. │   │ Peff + Carr.│   │ Fg × Fh     │               │
│  └─────────────┘   └─────────────┘   └─────────────┘               │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │ F = Fa × Fg × Fh                                        │       │
│  │                                                         │       │
│  │ Peff = Peff_passive + Σ(Peff_carrier) × (1/ER_apparent) │       │
│  │                        ↑                    ↑           │       │
│  │                   FROM ML             FROM ML           │       │
│  └─────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      OUTPUT: Bioavailability                        │
│                      F = 85.7% (predicted)                          │
│                      With uncertainty quantification                │
└─────────────────────────────────────────────────────────────────────┘
""")

println()
println("=" ^ 70)
println("NEXT STEPS FOR PRODUCTION")
println("=" ^ 70)
println("""
1. TRAIN TRANSPORTER PREDICTOR
   - Dataset: DrugBank + ChEMBL transporter annotations
   - Labels: Substrate/inhibitor for each transporter
   - Multi-label classification with BCE loss

2. TRAIN PROPERTY PREDICTOR
   - Predict logP, pKa, solubility from SMILES
   - Use same multimodal encoder backbone
   - Enable fully SMILES-driven predictions

3. UNCERTAINTY QUANTIFICATION
   - Evidential learning for calibrated confidence
   - MC Dropout for model uncertainty
   - Ensemble for epistemic uncertainty

4. VALIDATION
   - Independent test set (different scaffolds)
   - Prospective validation on new drugs
   - Comparison with commercial tools (GastroPlus, Simcyp)
""")

println("=" ^ 70)
