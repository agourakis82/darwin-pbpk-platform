# =============================================================================
# FOOD AND HERBAL DRUG INTERACTIONS DATABASE
# =============================================================================
# Darwin PBPK Platform - Publication-Ready
#
# Sources: FDA, Natural Medicines Database, Memorial Sloan Kettering, PubMed
# Coverage: Common food-drug and herb-drug interactions
#
# Includes:
# - Grapefruit and other citrus juices
# - Dietary components (fiber, dairy, cruciferous vegetables)
# - Common herbal supplements
# - Caffeine and alcohol
# =============================================================================

# =============================================================================
# GRAPEFRUIT JUICE INTERACTIONS
# =============================================================================

"""
Grapefruit juice interactions - primarily intestinal CYP3A4 inhibition.
Mechanism: Furanocoumarins (bergamottin, 6',7'-dihydroxybergamottin) cause
irreversible inhibition of intestinal CYP3A4. Effect can last 24-72 hours.
"""
const GRAPEFRUIT_JUICE_INTERACTIONS = Dict{Symbol, NamedTuple}(
    # === High Risk (AUC increase >= 5x) ===
    :lovastatin => (
        auc_ratio=15.3, cmax_ratio=12.0, risk=:high,
        mechanism=:cyp3a4_intestinal, onset_hours=1.0, duration_hours=72,
        recommendation="Avoid grapefruit entirely", pmid=8606630
    ),
    :simvastatin => (
        auc_ratio=3.6, cmax_ratio=2.0, risk=:high,
        mechanism=:cyp3a4_intestinal, onset_hours=1.0, duration_hours=72,
        recommendation="Avoid grapefruit", pmid=9610513
    ),
    :buspirone => (
        auc_ratio=9.2, cmax_ratio=4.5, risk=:high,
        mechanism=:cyp3a4_intestinal, onset_hours=1.0, duration_hours=48,
        recommendation="Avoid grapefruit", pmid=9618529
    ),
    :felodipine => (
        auc_ratio=2.8, cmax_ratio=2.3, risk=:moderate,
        mechanism=:cyp3a4_intestinal, onset_hours=0.5, duration_hours=24,
        recommendation="Separate by 4 hours or avoid", pmid=2063878
    ),
    :nisoldipine => (
        auc_ratio=4.0, cmax_ratio=3.0, risk=:high,
        mechanism=:cyp3a4_intestinal, onset_hours=0.5, duration_hours=24,
        recommendation="Avoid grapefruit", pmid=7752772
    ),
    :nifedipine => (
        auc_ratio=1.4, cmax_ratio=1.3, risk=:low,
        mechanism=:cyp3a4_intestinal, onset_hours=0.5, duration_hours=24,
        recommendation="Generally safe with moderate consumption", pmid=8606634
    ),
    :triazolam => (
        auc_ratio=2.5, cmax_ratio=1.5, risk=:moderate,
        mechanism=:cyp3a4_intestinal, onset_hours=1.0, duration_hours=48,
        recommendation="Avoid or reduce dose", pmid=8841161
    ),
    :midazolam_oral => (
        auc_ratio=1.5, cmax_ratio=1.5, risk=:moderate,
        mechanism=:cyp3a4_intestinal, onset_hours=1.0, duration_hours=24,
        recommendation="Monitor for increased sedation", pmid=8957169
    ),
    :cyclosporine => (
        auc_ratio=1.6, cmax_ratio=1.4, risk=:moderate,
        mechanism=:cyp3a4_intestinal, onset_hours=1.0, duration_hours=48,
        recommendation="Avoid due to narrow therapeutic index", pmid=8841158
    ),
    :tacrolimus_gfj => (
        auc_ratio=1.4, cmax_ratio=1.3, risk=:moderate,
        mechanism=:cyp3a4_intestinal, onset_hours=1.0, duration_hours=48,
        recommendation="Avoid due to narrow therapeutic index", pmid=10223779
    ),
    :sildenafil => (
        auc_ratio=1.2, cmax_ratio=1.1, risk=:low,
        mechanism=:cyp3a4_intestinal, onset_hours=1.0, duration_hours=24,
        recommendation="Generally safe", pmid=11302564
    ),
    :atorvastatin => (
        auc_ratio=1.8, cmax_ratio=1.6, risk=:moderate,
        mechanism=:cyp3a4_intestinal, onset_hours=1.0, duration_hours=48,
        recommendation="Limit grapefruit intake", pmid=12139083
    ),
    :amiodarone => (
        auc_ratio=1.5, cmax_ratio=1.3, risk=:moderate,
        mechanism=:cyp3a4_intestinal, onset_hours=1.0, duration_hours=72,
        recommendation="Avoid due to potential QT effects", pmid=9649360
    ),
    :carbamazepine => (
        auc_ratio=1.4, cmax_ratio=1.3, risk=:moderate,
        mechanism=:cyp3a4_intestinal, onset_hours=1.0, duration_hours=48,
        recommendation="Monitor levels if consuming grapefruit", pmid=8606635
    ),
    :erythromycin => (
        auc_ratio=1.5, cmax_ratio=1.4, risk=:low,
        mechanism=:cyp3a4_intestinal, onset_hours=1.0, duration_hours=24,
        recommendation="Generally safe", pmid=9357905
    ),
)

# =============================================================================
# OTHER CITRUS JUICE INTERACTIONS
# =============================================================================

const CITRUS_JUICE_INTERACTIONS = Dict{Symbol, NamedTuple}(
    # === Seville Orange (Bitter Orange) - Contains furanocoumarins ===
    :seville_orange_felodipine => (
        citrus=:seville_orange, drug=:felodipine,
        auc_ratio=2.5, cmax_ratio=2.0, risk=:moderate,
        mechanism=:cyp3a4_intestinal,
        recommendation="Similar to grapefruit - avoid"
    ),
    :seville_orange_cyclosporine => (
        citrus=:seville_orange, drug=:cyclosporine,
        auc_ratio=1.6, cmax_ratio=1.4, risk=:moderate,
        mechanism=:cyp3a4_intestinal,
        recommendation="Avoid"
    ),

    # === Pomelo - Contains furanocoumarins ===
    :pomelo_cyclosporine => (
        citrus=:pomelo, drug=:cyclosporine,
        auc_ratio=1.3, cmax_ratio=1.2, risk=:low,
        mechanism=:cyp3a4_intestinal,
        recommendation="Limit consumption"
    ),

    # === Orange Juice (Sweet) - Generally safe ===
    :orange_juice_felodipine => (
        citrus=:sweet_orange, drug=:felodipine,
        auc_ratio=1.0, cmax_ratio=1.0, risk=:none,
        mechanism=:none,
        recommendation="Safe"
    ),
    :orange_juice_fexofenadine => (
        citrus=:sweet_orange, drug=:fexofenadine,
        auc_ratio=0.4, cmax_ratio=0.5, risk=:moderate,
        mechanism=:oatp_inhibition,
        recommendation="Separate by 4 hours - reduces absorption"
    ),

    # === Apple Juice - OATP inhibition ===
    :apple_juice_fexofenadine => (
        citrus=:apple, drug=:fexofenadine,
        auc_ratio=0.3, cmax_ratio=0.4, risk=:moderate,
        mechanism=:oatp_inhibition,
        recommendation="Separate by 4 hours"
    ),
    :apple_juice_atenolol => (
        citrus=:apple, drug=:atenolol,
        auc_ratio=0.5, cmax_ratio=0.6, risk=:moderate,
        mechanism=:oatp_inhibition,
        recommendation="Separate by 4 hours"
    ),
)

# =============================================================================
# HERBAL SUPPLEMENT INTERACTIONS
# =============================================================================

"""
St. John's Wort (Hypericum perforatum) interactions.
Mechanism: Strong CYP3A4 and P-gp inducer via PXR activation.
Effect takes 1-2 weeks to reach maximum, persists 1-2 weeks after discontinuation.
"""
const ST_JOHNS_WORT_INTERACTIONS = Dict{Symbol, NamedTuple}(
    :sjw_midazolam => (
        auc_ratio=0.35, cmax_ratio=0.45, risk=:high,
        mechanism=:cyp3a4_induction, onset_days=7, washout_days=14,
        recommendation="Avoid combination", pmid=10561903
    ),
    :sjw_cyclosporine => (
        auc_ratio=0.48, cmax_ratio=0.55, risk=:very_high,
        mechanism=[:cyp3a4_induction, :pgp_induction], onset_days=7, washout_days=14,
        recommendation="Contraindicated - rejection risk", pmid=11723197
    ),
    :sjw_tacrolimus => (
        auc_ratio=0.35, cmax_ratio=0.40, risk=:very_high,
        mechanism=[:cyp3a4_induction, :pgp_induction], onset_days=7, washout_days=14,
        recommendation="Contraindicated - rejection risk", pmid=12683478
    ),
    :sjw_oral_contraceptive => (
        auc_ratio=0.60, cmax_ratio=0.70, risk=:high,
        mechanism=:cyp3a4_induction, onset_days=7, washout_days=14,
        recommendation="Use alternative contraception", pmid=12139082
    ),
    :sjw_digoxin => (
        auc_ratio=0.75, cmax_ratio=0.80, risk=:moderate,
        mechanism=:pgp_induction, onset_days=7, washout_days=14,
        recommendation="Avoid or monitor levels", pmid=10561904
    ),
    :sjw_simvastatin => (
        auc_ratio=0.50, cmax_ratio=0.55, risk=:moderate,
        mechanism=:cyp3a4_induction, onset_days=7, washout_days=14,
        recommendation="Avoid combination", pmid=11568986
    ),
    :sjw_imatinib => (
        auc_ratio=0.30, cmax_ratio=0.35, risk=:very_high,
        mechanism=:cyp3a4_induction, onset_days=7, washout_days=14,
        recommendation="Contraindicated", pmid=15226332
    ),
    :sjw_indinavir => (
        auc_ratio=0.43, cmax_ratio=0.50, risk=:very_high,
        mechanism=[:cyp3a4_induction, :pgp_induction], onset_days=7, washout_days=14,
        recommendation="Contraindicated - treatment failure risk", pmid=10561905
    ),
    :sjw_warfarin => (
        auc_ratio=0.80, cmax_ratio=0.85, risk=:moderate,
        mechanism=[:cyp2c9_induction, :cyp3a4_induction], onset_days=7, washout_days=14,
        recommendation="Monitor INR, may need dose increase", pmid=10223780
    ),
    :sjw_ssri => (
        auc_ratio=1.0, cmax_ratio=1.0, risk=:high,
        mechanism=:pharmacodynamic, onset_days=1, washout_days=7,
        recommendation="Avoid - serotonin syndrome risk", pmid=11302565
    ),
)

"""
Other common herbal supplement interactions.
"""
const HERBAL_INTERACTIONS = Dict{Symbol, NamedTuple}(
    # === Ginkgo biloba ===
    :ginkgo_warfarin => (
        herb=:ginkgo_biloba, drug=:warfarin,
        effect=:increased_bleeding, mechanism=:antiplatelet,
        risk=:moderate, recommendation="Use caution, monitor INR"
    ),
    :ginkgo_aspirin => (
        herb=:ginkgo_biloba, drug=:aspirin,
        effect=:increased_bleeding, mechanism=:antiplatelet,
        risk=:moderate, recommendation="Use caution"
    ),
    :ginkgo_ssri => (
        herb=:ginkgo_biloba, drug=:ssri,
        effect=:increased_bleeding, mechanism=:antiplatelet,
        risk=:low, recommendation="Monitor for bleeding"
    ),
    :ginkgo_alprazolam => (
        herb=:ginkgo_biloba, drug=:alprazolam,
        auc_ratio=0.7, mechanism=:cyp3a4_induction,
        risk=:low, recommendation="May reduce alprazolam effect"
    ),

    # === Ginseng (Panax) ===
    :ginseng_warfarin => (
        herb=:panax_ginseng, drug=:warfarin,
        effect=:decreased_inr, mechanism=:unknown,
        risk=:moderate, recommendation="Monitor INR"
    ),
    :ginseng_phenelzine => (
        herb=:panax_ginseng, drug=:phenelzine,
        effect=:hypertension_mania, mechanism=:pharmacodynamic,
        risk=:high, recommendation="Avoid combination"
    ),
    :ginseng_insulin => (
        herb=:panax_ginseng, drug=:insulin,
        effect=:hypoglycemia, mechanism=:pharmacodynamic,
        risk=:moderate, recommendation="Monitor blood glucose"
    ),

    # === Echinacea ===
    :echinacea_caffeine => (
        herb=:echinacea, drug=:caffeine,
        auc_ratio=1.3, mechanism=:cyp1a2_inhibition,
        risk=:low, recommendation="May increase caffeine effects"
    ),
    :echinacea_midazolam => (
        herb=:echinacea, drug=:midazolam,
        auc_ratio=0.8, mechanism=:cyp3a4_induction,
        risk=:low, recommendation="Minor effect"
    ),

    # === Kava ===
    :kava_alprazolam => (
        herb=:kava, drug=:alprazolam,
        effect=:increased_sedation, mechanism=:pharmacodynamic,
        risk=:high, recommendation="Avoid combination"
    ),
    :kava_levodopa => (
        herb=:kava, drug=:levodopa,
        effect=:decreased_efficacy, mechanism=:dopamine_antagonism,
        risk=:moderate, recommendation="Avoid combination"
    ),
    :kava_hepatotoxic_drugs => (
        herb=:kava, drug=:hepatotoxic,
        effect=:additive_hepatotoxicity, mechanism=:pharmacodynamic,
        risk=:high, recommendation="Avoid due to kava hepatotoxicity"
    ),

    # === Valerian ===
    :valerian_benzodiazepines => (
        herb=:valerian, drug=:benzodiazepines,
        effect=:increased_sedation, mechanism=:gaba_enhancement,
        risk=:moderate, recommendation="Use caution"
    ),
    :valerian_opioids => (
        herb=:valerian, drug=:opioids,
        effect=:increased_sedation, mechanism=:cns_depression,
        risk=:moderate, recommendation="Use caution"
    ),

    # === Garlic ===
    :garlic_warfarin => (
        herb=:garlic, drug=:warfarin,
        effect=:increased_bleeding, mechanism=:antiplatelet,
        risk=:low, recommendation="Dietary amounts likely safe"
    ),
    :garlic_saquinavir => (
        herb=:garlic, drug=:saquinavir,
        auc_ratio=0.50, mechanism=:pgp_induction,
        risk=:high, recommendation="Avoid combination"
    ),

    # === Green Tea ===
    :green_tea_nadolol => (
        herb=:green_tea, drug=:nadolol,
        auc_ratio=0.6, mechanism=:oatp_inhibition,
        risk=:moderate, recommendation="Separate administration"
    ),
    :green_tea_warfarin => (
        herb=:green_tea, drug=:warfarin,
        effect=:decreased_inr, mechanism=:vitamin_k,
        risk=:low, recommendation="Consistent intake recommended"
    ),

    # === Milk Thistle (Silymarin) ===
    :milk_thistle_indinavir => (
        herb=:milk_thistle, drug=:indinavir,
        auc_ratio=1.0, mechanism=:minimal,
        risk=:low, recommendation="Generally safe"
    ),
    :milk_thistle_metronidazole => (
        herb=:milk_thistle, drug=:metronidazole,
        auc_ratio=0.8, mechanism=:cyp3a4_induction,
        risk=:low, recommendation="Minor effect"
    ),

    # === Goldenseal ===
    :goldenseal_midazolam => (
        herb=:goldenseal, drug=:midazolam,
        auc_ratio=1.4, mechanism=:cyp3a4_inhibition,
        risk=:moderate, recommendation="Monitor for increased sedation"
    ),
    :goldenseal_cyclosporine => (
        herb=:goldenseal, drug=:cyclosporine,
        auc_ratio=1.3, mechanism=:cyp3a4_inhibition,
        risk=:moderate, recommendation="Monitor levels"
    ),

    # === Cannabis/CBD ===
    :cbd_clobazam => (
        herb=:cannabidiol, drug=:clobazam,
        auc_ratio=3.0, mechanism=:cyp2c19_inhibition,
        risk=:high, recommendation="Reduce clobazam dose"
    ),
    :cbd_warfarin => (
        herb=:cannabidiol, drug=:warfarin,
        auc_ratio=1.5, mechanism=:cyp2c9_inhibition,
        risk=:moderate, recommendation="Monitor INR closely"
    ),
    :cbd_tacrolimus => (
        herb=:cannabidiol, drug=:tacrolimus,
        auc_ratio=1.5, mechanism=:cyp3a4_inhibition,
        risk=:moderate, recommendation="Monitor levels"
    ),
)

# =============================================================================
# DIETARY COMPONENT INTERACTIONS
# =============================================================================

const DIETARY_INTERACTIONS = Dict{Symbol, NamedTuple}(
    # === High-Fat Meals ===
    :fat_griseofulvin => (
        food=:high_fat_meal, drug=:griseofulvin,
        auc_ratio=2.0, mechanism=:increased_absorption,
        recommendation="Take with fatty meal"
    ),
    :fat_itraconazole_capsule => (
        food=:high_fat_meal, drug=:itraconazole_capsule,
        auc_ratio=1.5, mechanism=:increased_absorption,
        recommendation="Take with food"
    ),
    :fat_atovaquone => (
        food=:high_fat_meal, drug=:atovaquone,
        auc_ratio=3.0, mechanism=:increased_absorption,
        recommendation="Take with fatty meal"
    ),
    :fat_ziprasidone => (
        food=:high_fat_meal, drug=:ziprasidone,
        auc_ratio=2.0, mechanism=:increased_absorption,
        recommendation="Take with food (>=500 calories)"
    ),

    # === Dairy/Calcium ===
    :dairy_tetracycline => (
        food=:dairy, drug=:tetracycline,
        auc_ratio=0.5, mechanism=:chelation,
        recommendation="Separate by 2 hours"
    ),
    :dairy_ciprofloxacin => (
        food=:dairy, drug=:ciprofloxacin,
        auc_ratio=0.6, mechanism=:chelation,
        recommendation="Separate by 2 hours"
    ),
    :dairy_levothyroxine => (
        food=:dairy, drug=:levothyroxine,
        auc_ratio=0.7, mechanism=:reduced_absorption,
        recommendation="Take on empty stomach"
    ),
    :calcium_bisphosphonates => (
        food=:calcium, drug=:alendronate,
        auc_ratio=0.1, mechanism=:chelation,
        recommendation="Take with water only, 30 min before food"
    ),

    # === Fiber ===
    :fiber_digoxin => (
        food=:high_fiber, drug=:digoxin,
        auc_ratio=0.8, mechanism=:binding,
        recommendation="Consistent fiber intake recommended"
    ),
    :fiber_levothyroxine => (
        food=:high_fiber, drug=:levothyroxine,
        auc_ratio=0.8, mechanism=:binding,
        recommendation="Consistent fiber intake recommended"
    ),

    # === Cruciferous Vegetables (CYP1A2 inducers) ===
    :cruciferous_theophylline => (
        food=:cruciferous, drug=:theophylline,
        auc_ratio=0.75, mechanism=:cyp1a2_induction,
        recommendation="Consistent intake recommended"
    ),
    :cruciferous_caffeine => (
        food=:cruciferous, drug=:caffeine,
        auc_ratio=0.8, mechanism=:cyp1a2_induction,
        recommendation="Minor effect"
    ),

    # === Charbroiled/Smoked Foods (CYP1A2 inducers) ===
    :charbroiled_theophylline => (
        food=:charbroiled, drug=:theophylline,
        auc_ratio=0.65, mechanism=:cyp1a2_induction,
        recommendation="Avoid large amounts"
    ),
    :charbroiled_clozapine => (
        food=:charbroiled, drug=:clozapine,
        auc_ratio=0.7, mechanism=:cyp1a2_induction,
        recommendation="Monitor if diet changes significantly"
    ),

    # === Vitamin K Rich Foods ===
    :vitamin_k_warfarin => (
        food=:vitamin_k_rich, drug=:warfarin,
        effect=:decreased_inr, mechanism=:pharmacodynamic,
        recommendation="Consistent intake of vitamin K foods"
    ),

    # === Tyramine-Rich Foods (with MAOIs) ===
    :tyramine_maoi => (
        food=:tyramine_rich, drug=:maoi,
        effect=:hypertensive_crisis, mechanism=:pharmacodynamic,
        risk=:very_high, recommendation="Avoid aged cheese, wine, fermented foods"
    ),
)

# =============================================================================
# CAFFEINE AND ALCOHOL INTERACTIONS
# =============================================================================

const CAFFEINE_ALCOHOL_INTERACTIONS = Dict{Symbol, NamedTuple}(
    # === Caffeine Interactions ===
    :caffeine_ciprofloxacin => (
        substance=:caffeine, drug=:ciprofloxacin,
        caffeine_auc_ratio=1.5, mechanism=:cyp1a2_inhibition,
        recommendation="May need to reduce caffeine intake"
    ),
    :caffeine_fluvoxamine => (
        substance=:caffeine, drug=:fluvoxamine,
        caffeine_auc_ratio=5.0, mechanism=:cyp1a2_inhibition,
        recommendation="Significantly reduce caffeine"
    ),
    :caffeine_oral_contraceptive => (
        substance=:caffeine, drug=:oral_contraceptive,
        caffeine_auc_ratio=1.5, mechanism=:cyp1a2_inhibition,
        recommendation="May need to reduce caffeine"
    ),
    :caffeine_mexiletine => (
        substance=:caffeine, drug=:mexiletine,
        caffeine_auc_ratio=1.5, mechanism=:cyp1a2_inhibition,
        recommendation="Reduce caffeine intake"
    ),
    :caffeine_theophylline => (
        substance=:caffeine, drug=:theophylline,
        effect=:additive_stimulation, mechanism=:pharmacodynamic,
        recommendation="Reduce caffeine, monitor for tachycardia"
    ),

    # === Alcohol Interactions ===
    :alcohol_metronidazole => (
        substance=:alcohol, drug=:metronidazole,
        effect=:disulfiram_reaction, mechanism=:aldh_inhibition,
        risk=:high, recommendation="Avoid alcohol during and 3 days after"
    ),
    :alcohol_disulfiram => (
        substance=:alcohol, drug=:disulfiram,
        effect=:severe_reaction, mechanism=:aldh_inhibition,
        risk=:very_high, recommendation="Absolute contraindication"
    ),
    :alcohol_acetaminophen => (
        substance=:alcohol, drug=:acetaminophen,
        effect=:hepatotoxicity, mechanism=:cyp2e1_induction,
        risk=:high, recommendation="Limit acetaminophen, avoid chronic heavy drinking"
    ),
    :alcohol_benzodiazepines => (
        substance=:alcohol, drug=:benzodiazepines,
        effect=:increased_sedation, mechanism=:cns_depression,
        risk=:high, recommendation="Avoid combination"
    ),
    :alcohol_opioids => (
        substance=:alcohol, drug=:opioids,
        effect=:respiratory_depression, mechanism=:cns_depression,
        risk=:very_high, recommendation="Avoid combination"
    ),
    :alcohol_warfarin_acute => (
        substance=:alcohol, drug=:warfarin,
        effect=:increased_inr, mechanism=:cyp2c9_inhibition,
        risk=:moderate, recommendation="Avoid binge drinking"
    ),
    :alcohol_warfarin_chronic => (
        substance=:alcohol, drug=:warfarin,
        effect=:decreased_inr, mechanism=:cyp2c9_induction,
        risk=:moderate, recommendation="Avoid chronic heavy drinking"
    ),
    :alcohol_metformin => (
        substance=:alcohol, drug=:metformin,
        effect=:lactic_acidosis, mechanism=:gluconeogenesis_inhibition,
        risk=:moderate, recommendation="Moderate consumption only"
    ),
    :alcohol_nsaids => (
        substance=:alcohol, drug=:nsaids,
        effect=:gi_bleeding, mechanism=:additive_gastric_irritation,
        risk=:moderate, recommendation="Use caution"
    ),
    :alcohol_antidepressants => (
        substance=:alcohol, drug=:antidepressants,
        effect=:increased_sedation, mechanism=:cns_depression,
        risk=:moderate, recommendation="Use caution"
    ),
    :alcohol_antihypertensives => (
        substance=:alcohol, drug=:antihypertensives,
        effect=:hypotension, mechanism=:additive_vasodilation,
        risk=:moderate, recommendation="Use caution, especially initially"
    ),
)

# =============================================================================
# EXPORT ALL FOOD/HERB DATABASES
# =============================================================================

export GRAPEFRUIT_JUICE_INTERACTIONS, CITRUS_JUICE_INTERACTIONS
export ST_JOHNS_WORT_INTERACTIONS, HERBAL_INTERACTIONS
export DIETARY_INTERACTIONS, CAFFEINE_ALCOHOL_INTERACTIONS
