//! PBPK Simulation using Darwin stdlib
//! Generated from MedLang model: OneCompPK

module test_pbpk_simulation

// Darwin PBPK stdlib - PRODUCTION READY
import darwin_pbpk.simulation::{SimulationConfig, SimulationResult, run_pbpk_simulation}
import darwin_pbpk.core.pbpk_params::{PBPKParams, PatientData, DrugProperties}
import std.io::{println}

fn main() -> effect[IO] {
    println!("=== Darwin PBPK Platform ===")
    println!("Testing one-compartment PK model with darwin_pbpk stdlib\n")

    // Define patient (adult male, 70 kg)
    let patient = PatientData {
        body_weight: 70.0@kg,
        age: 35.0@year,
        sex: 0,  // 0 = male
        height: 175.0@cm,
        bmi: 22.9@kg_per_m2,
    }

    // Define drug properties (example: oral drug with moderate clearance)
    let drug = DrugProperties {
        molecular_weight: 250.0@g_per_mol,
        logp: 2.5@dimensionless,
        fu_plasma: 0.1@dimensionless,     // 10% unbound in plasma
        bp_ratio: 1.0@dimensionless,
        pka_base: 8.0@dimensionless,
        fa: 0.9@dimensionless,            // 90% absorbed
        ka: 1.0@per_h,                    // Absorption rate
        clint_hepatic: 50.0@uL_per_min_per_mg_protein,
        clr: 5.0@L_per_h,                 // Renal clearance
    }

    // Simulation configuration
    let config = SimulationConfig {
        t_end: 24.0@h,
        dt: 0.1@h,
        dose: 100.0@mg,
        route: 1,  // 1 = oral
        n_doses: 1,
        dosing_interval: 24.0@h,
    }

    println!("Patient: {:.1} kg, {} years old", patient.body_weight, patient.age)
    println!("Drug: MW = {:.1} g/mol, LogP = {:.2}", drug.molecular_weight, drug.logp)
    println!("Dose: {} mg (oral)", config.dose)
    println!("Simulation time: {} hours\n", config.t_end)

    // Run PBPK simulation
    println!("Running PBPK simulation...")
    let result = run_pbpk_simulation(config, patient, drug)

    if result.success {
        println!("\n✓ Simulation completed successfully!")
        println!("\n=== Pharmacokinetic Parameters ===")
        println!("Cmax (plasma):     {:.2} mg/L", result.cmax_plasma)
        println!("Tmax:              {:.2} hours", result.tmax)
        println!("AUC(0-∞):          {:.2} mg·h/L", result.auc_0_inf)
        println!("Half-life:         {:.2} hours", result.half_life)
        println!("Clearance (CL):    {:.2} L/h", result.clearance)
        println!("Vdss:              {:.2} L", result.vdss)
    } else {
        println!("\n✗ Simulation failed!")
    }
}
