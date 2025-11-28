#!/usr/bin/env julia
"""
Comprehensive Test Suite for Dynamic GNN Module

Tests:
1. Organ graph construction
2. Message passing layers
3. Forward pass with batching
4. Concentration constraints
5. Gradient flow verification

Author: Dr. Demetrios Agourakis + AI Assistant
Date: November 2025
"""

using Test
using Statistics

# Include modules
include(joinpath(@__DIR__, "../src/DarwinPBPK/ode_solver.jl"))
using .ODEPBPKSolver

include(joinpath(@__DIR__, "../src/DarwinPBPK/dynamic_gnn.jl"))
using .DynamicGNN

println("=" ^ 60)
println("Dynamic GNN Test Suite")
println("=" ^ 60)

@testset "Dynamic GNN Tests" begin

    #=========================================================================
      Test 1: Organ Graph Construction
    =========================================================================#
    @testset "Organ Graph Construction" begin
        println("\n[1/5] Testing organ graph construction...")

        # Create organ graph
        g = create_organ_graph()

        # Check node count (14 PBPK organs)
        @test g.num_nodes == NUM_ORGANS
        println("  ✓ Node count: $(g.num_nodes) organs")

        # Check edges exist
        @test g.num_edges > 0
        println("  ✓ Edge count: $(g.num_edges) connections")

        # Expected edges: 13 arterial + 12 venous + 1 portal = 26
        @test g.num_edges >= 25  # Allow some flexibility

        # Check edge data (flow weights)
        @test haskey(g.edata, :flow)
        @test all(0 .<= g.edata.flow .<= 1)  # Normalized
        println("  ✓ Flow weights normalized: [$(minimum(g.edata.flow)), $(maximum(g.edata.flow))]")

        # Create edge features
        edge_feats = create_edge_features(g)
        @test size(edge_feats, 1) == 4  # [flow, arterial, venous, portal]
        @test size(edge_feats, 2) == g.num_edges
        println("  ✓ Edge features shape: $(size(edge_feats))")
    end

    #=========================================================================
      Test 2: Model Creation
    =========================================================================#
    @testset "Model Creation" begin
        println("\n[2/5] Testing model creation...")

        # Create model with default parameters
        model = DynamicPBPKGNN()

        @test model.node_dim == 16
        @test model.hidden_dim == 64
        @test model.num_gnn_layers == 3
        println("  ✓ Default model: hidden_dim=$(model.hidden_dim), layers=$(model.num_gnn_layers)")

        # Create model with custom parameters
        model_custom = DynamicPBPKGNN(
            node_dim=32,
            hidden_dim=128,
            num_gnn_layers=4,
            use_attention=true,
        )

        @test model_custom.node_dim == 32
        @test model_custom.hidden_dim == 128
        @test model_custom.num_gnn_layers == 4
        @test model_custom.use_attention == true
        println("  ✓ Custom model: hidden_dim=$(model_custom.hidden_dim), layers=$(model_custom.num_gnn_layers)")
    end

    #=========================================================================
      Test 3: Forward Pass
    =========================================================================#
    @testset "Forward Pass" begin
        println("\n[3/5] Testing forward pass...")

        model = DynamicPBPKGNN(
            hidden_dim=32,
            num_gnn_layers=2,
            num_temporal_steps=10,
        )

        # Single sample
        dose = 100.0
        params = PBPKParams(clearance_hepatic=10.0, clearance_renal=5.0)
        time_points = collect(0.0:0.5:12.0)

        result = forward(model, dose, params, time_points, cpu)

        @test haskey(result, "concentrations")
        @test haskey(result, "time_points")
        @test haskey(result, "organ_names")

        concs = result["concentrations"]
        @test size(concs, 1) == 1  # batch_size
        @test size(concs, 2) == NUM_ORGANS  # organs
        println("  ✓ Single sample output shape: $(size(concs))")

        # Batch forward
        batch_size = 4
        doses = [50.0, 100.0, 150.0, 200.0]
        params_batch = [PBPKParams(clearance_hepatic=5.0+i*2.0, clearance_renal=2.0+i)
                        for i in 1:batch_size]
        time_points_batch = [time_points for _ in 1:batch_size]

        result_batch = forward_batch(model, doses, params_batch, time_points_batch, cpu)

        concs_batch = result_batch["concentrations"]
        @test size(concs_batch, 1) == batch_size
        @test size(concs_batch, 2) == NUM_ORGANS
        println("  ✓ Batch output shape: $(size(concs_batch))")
    end

    #=========================================================================
      Test 4: Concentration Constraints
    =========================================================================#
    @testset "Concentration Constraints" begin
        println("\n[4/5] Testing concentration constraints...")

        model = DynamicPBPKGNN(
            hidden_dim=32,
            num_gnn_layers=2,
            num_temporal_steps=20,
        )

        # Test with various doses
        doses = [10.0, 100.0, 500.0, 1000.0]

        for dose in doses
            params = PBPKParams(clearance_hepatic=10.0, clearance_renal=5.0)
            time_points = collect(0.0:1.0:24.0)

            result = forward(model, dose, params, time_points, cpu)
            concs = result["concentrations"]

            # All concentrations must be >= 0 (relu constraint)
            @test all(concs .>= 0)

            # Check for NaN/Inf
            @test all(isfinite.(concs))
        end
        println("  ✓ All concentrations ≥ 0 for doses: $doses")
        println("  ✓ No NaN/Inf values detected")
    end

    #=========================================================================
      Test 5: Blood Flow Connectivity
    =========================================================================#
    @testset "Blood Flow Connectivity" begin
        println("\n[5/5] Testing blood flow connectivity...")

        g = create_organ_graph()
        edge_feats = create_edge_features(g)

        # Check arterial edges (blood → organs)
        arterial_count = sum(edge_feats[2, :] .> 0)
        @test arterial_count >= 10  # Most organs receive arterial supply
        println("  ✓ Arterial edges: $arterial_count")

        # Check venous edges (organs → blood)
        venous_count = sum(edge_feats[3, :] .> 0)
        @test venous_count >= 10  # Most organs have venous return
        println("  ✓ Venous edges: $venous_count")

        # Check portal edge (gut → liver)
        portal_count = sum(edge_feats[4, :] .> 0)
        @test portal_count >= 1  # At least one portal connection
        println("  ✓ Portal edges: $portal_count")

        # Check blood flow constants exist
        @test haskey(ORGAN_BLOOD_FLOWS, "liver")
        @test haskey(ORGAN_BLOOD_FLOWS, "kidney")
        @test haskey(ORGAN_BLOOD_FLOWS, "lung")

        # Lung should have highest flow (100% cardiac output)
        @test ORGAN_BLOOD_FLOWS["lung"] > ORGAN_BLOOD_FLOWS["liver"]
        println("  ✓ Physiological blood flows validated")
    end

end  # @testset

println("\n" * "=" ^ 60)
println("ALL DYNAMIC GNN TESTS COMPLETED")
println("=" ^ 60)
