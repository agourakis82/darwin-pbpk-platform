# Test DDI risk classification system

include("../src/DarwinPBPK/medlang/ddi_prediction.jl")
using .DDIPrediction

println("=" ^ 70)
println("DDI RISK CLASSIFICATION TEST")
println("=" ^ 70)

# Test individual risk assessments
println("\n1. INDIVIDUAL RISK ASSESSMENTS")
println("-" ^ 50)

test_pairs = [
    (:itraconazole, :midazolam, "Strong CYP3A4 inhibition"),
    (:ritonavir, :midazolam, "MBI - very strong"),
    (:fluconazole, :warfarin_s, "NTI drug interaction"),
    (:diltiazem, :simvastatin, "Moderate interaction"),
    (:cimetidine, :midazolam, "Weak inhibitor"),
]

for (perp, vic, desc) in test_pairs
    risk = classify_ddi_risk(perp, vic)
    level = uppercase(string(risk.risk_level))
    auc = round(risk.auc_ratio, digits=1)
    println("\n$perp + $vic ($desc)")
    println("  Risk: $level | AUC: $(auc)x | Monitor: $(risk.monitoring_required)")
    println("  Action: $(risk.clinical_action)")
    if !isempty(risk.alternative_drugs)
        println("  Alternatives: $(risk.alternative_drugs)")
    end
end

# Test polypharmacy screening
println("\n" * "=" ^ 70)
println("2. POLYPHARMACY SCREENING")
println("-" ^ 50)

# Elderly patient medication list
patient_meds = [:simvastatin, :diltiazem, :warfarin_s, :metformin, :omeprazole]
println("\nPatient medications: $patient_meds")

interactions = screen_drug_list(patient_meds)
println("\nIdentified interactions ($(length(interactions)) total):")

for risk in interactions
    level = uppercase(string(risk.risk_level))
    auc = round(risk.auc_ratio, digits=1)
    println("  [$level] $(risk.perpetrator) + $(risk.victim): $(auc)x")
end

# Test NTI drug handling
println("\n" * "=" ^ 70)
println("3. NARROW THERAPEUTIC INDEX (NTI) DRUG HANDLING")
println("-" ^ 50)

nti_tests = [
    (:fluconazole, :warfarin_s),  # Should be contraindicated (NTI + >2x)
    (:amiodarone, :digoxin),      # NTI
    (:cyclosporine, :tacrolimus), # Both NTI
]

for (perp, vic) in nti_tests
    risk = classify_ddi_risk(perp, vic)
    is_nti = vic in NTI_DRUGS ? "NTI" : "non-NTI"
    println("\n$perp + $vic ($is_nti)")
    println("  Risk: $(uppercase(string(risk.risk_level)))")
    println("  AUC: $(round(risk.auc_ratio, digits=1))x")
    println("  Dose: $(risk.dose_adjustment)")
end

# Print full report example
println("\n" * "=" ^ 70)
println("4. FULL RISK REPORT EXAMPLE")
println("-" ^ 50)

risk = classify_ddi_risk(:clarithromycin, :simvastatin)
print_risk_report(risk)

println("\n" * "=" ^ 70)
