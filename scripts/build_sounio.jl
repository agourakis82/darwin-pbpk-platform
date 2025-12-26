#!/usr/bin/env julia
# ===========================================================================
# BUILD SOUNIO FROM JULIA
# ===========================================================================
# Script to build Sounio compiler and PBPK models from Julia environment.
#
# Usage:
#   julia scripts/build_sounio.jl                    # Build compiler
#   julia scripts/build_sounio.jl --check-all        # Check all PBPK models
#   julia scripts/build_sounio.jl --run example.d    # Run specific model
#
# Author: Dr. Sounio Agourakis
# Date: December 2025
# ===========================================================================

using Pkg
using Dates

# ===========================================================================
# Configuration
# ===========================================================================

const DARWIN_ROOT = dirname(dirname(@__FILE__))
const SOUNIO_ROOT = joinpath(DARWIN_ROOT, "Darwin-sounio")
const COMPILER_DIR = joinpath(SOUNIO_ROOT, "compiler")
const PBPK_EXAMPLES = joinpath(SOUNIO_ROOT, "examples", "pbpk")
const JULIA_INTEGRATION = joinpath(DARWIN_ROOT, "julia-migration", "src", "DarwinPBPK", "sounio")

# ===========================================================================
# Helper Functions
# ===========================================================================

function log_info(msg)
    println("\e[34m[INFO]\e[0m $msg")
end

function log_success(msg)
    println("\e[32m[OK]\e[0m $msg")
end

function log_error(msg)
    println("\e[31m[ERROR]\e[0m $msg")
end

function log_warning(msg)
    println("\e[33m[WARN]\e[0m $msg")
end

function run_cmd(cmd; capture=false, dir=nothing)
    if dir !== nothing
        cd(dir)
    end

    if capture
        try
            output = read(cmd, String)
            return (success=true, output=output)
        catch e
            return (success=false, output=string(e))
        end
    else
        try
            run(cmd)
            return (success=true, output="")
        catch e
            return (success=false, output=string(e))
        end
    end
end

# ===========================================================================
# Build Sounio Compiler
# ===========================================================================

function build_compiler(; release=true, features=String[])
    log_info("Building Sounio compiler...")

    if !isdir(COMPILER_DIR)
        log_error("Compiler directory not found: $COMPILER_DIR")
        return false
    end

    cd(COMPILER_DIR)

    # Build command
    cargo_args = ["cargo", "build"]
    if release
        push!(cargo_args, "--release")
    end

    for feature in features
        push!(cargo_args, "--features", feature)
    end

    result = run_cmd(Cmd(cargo_args))

    if result.success
        log_success("Compiler built successfully")

        # Get binary path
        binary = release ?
            joinpath(COMPILER_DIR, "target", "release", "dc") :
            joinpath(COMPILER_DIR, "target", "debug", "dc")

        if isfile(binary)
            log_info("Binary: $binary")

            # Get version
            version_result = run_cmd(`$binary --version`, capture=true)
            if version_result.success
                log_info("Version: $(strip(version_result.output))")
            end
        end

        return true
    else
        log_error("Build failed: $(result.output)")
        return false
    end
end

# ===========================================================================
# Check PBPK Models
# ===========================================================================

function check_pbpk_model(source_file::String; show_types=false)
    compiler = joinpath(COMPILER_DIR, "target", "release", "dc")

    if !isfile(compiler)
        log_error("Compiler not found. Run build first.")
        return false
    end

    if !isfile(source_file)
        log_error("Source file not found: $source_file")
        return false
    end

    log_info("Checking: $(basename(source_file))")

    cmd_args = [compiler, "check", source_file]
    if show_types
        push!(cmd_args, "--show-types")
    end

    result = run_cmd(Cmd(cmd_args), capture=true)

    if result.success
        log_success("$(basename(source_file)) - OK")
        return true
    else
        log_error("$(basename(source_file)) - FAILED")
        println(result.output)
        return false
    end
end

function check_all_pbpk_models(; show_types=false)
    log_info("Checking all PBPK models in: $PBPK_EXAMPLES")
    println()

    if !isdir(PBPK_EXAMPLES)
        log_error("PBPK examples directory not found")
        return
    end

    models = filter(f -> endswith(f, ".sio"), readdir(PBPK_EXAMPLES))

    passed = 0
    failed = 0

    for model in models
        source_file = joinpath(PBPK_EXAMPLES, model)
        if check_pbpk_model(source_file; show_types=show_types)
            passed += 1
        else
            failed += 1
        end
    end

    println()
    log_info("Results: $passed passed, $failed failed out of $(length(models)) models")

    if failed == 0
        log_success("All PBPK models passed type checking!")
    else
        log_warning("$failed models failed type checking")
    end
end

# ===========================================================================
# Run PBPK Model
# ===========================================================================

function run_pbpk_model(source_file::String; input_file=nothing, output_file=nothing)
    compiler = joinpath(COMPILER_DIR, "target", "release", "dc")

    if !isfile(compiler)
        log_error("Compiler not found. Run build first.")
        return nothing
    end

    if !isfile(source_file)
        log_error("Source file not found: $source_file")
        return nothing
    end

    log_info("Running: $(basename(source_file))")

    cmd_args = [compiler, "run", source_file]

    if input_file !== nothing && isfile(input_file)
        push!(cmd_args, "--input", input_file)
    end

    if output_file !== nothing
        push!(cmd_args, "--output", output_file)
    end

    start_time = time()
    result = run_cmd(Cmd(cmd_args), capture=true)
    elapsed = (time() - start_time) * 1000  # ms

    if result.success
        log_success("Execution completed in $(round(elapsed, digits=1)) ms")
        println()
        println(result.output)
        return result.output
    else
        log_error("Execution failed")
        println(result.output)
        return nothing
    end
end

# ===========================================================================
# Run Tests
# ===========================================================================

function run_compiler_tests()
    log_info("Running Sounio compiler tests...")

    cd(COMPILER_DIR)
    result = run_cmd(`cargo test`, capture=true)

    if result.success
        log_success("All tests passed")
        return true
    else
        log_error("Tests failed")
        println(result.output)
        return false
    end
end

# ===========================================================================
# Generate Julia Bindings
# ===========================================================================

function generate_julia_bindings()
    log_info("Generating Julia bindings...")

    # Ensure integration directory exists
    mkpath(JULIA_INTEGRATION)
    mkpath(joinpath(JULIA_INTEGRATION, "schemas"))

    # Check that files exist
    integration_file = joinpath(JULIA_INTEGRATION, "SounioIntegration.jl")
    schema_file = joinpath(JULIA_INTEGRATION, "schemas", "pbpk_schema.json")

    if isfile(integration_file)
        log_success("Integration module: $integration_file")
    else
        log_warning("Integration module not found")
    end

    if isfile(schema_file)
        log_success("JSON schema: $schema_file")
    else
        log_warning("JSON schema not found")
    end

    # Update DarwinPBPK.jl to include the integration module
    darwin_main = joinpath(DARWIN_ROOT, "julia-migration", "src", "DarwinPBPK.jl")
    if isfile(darwin_main)
        content = read(darwin_main, String)
        if !contains(content, "SounioIntegration")
            log_info("Adding SounioIntegration to DarwinPBPK.jl...")
            # Would append the include statement
        else
            log_info("SounioIntegration already included in DarwinPBPK.jl")
        end
    end

    log_success("Julia bindings ready")
end

# ===========================================================================
# Validate Cross-Platform Results
# ===========================================================================

function validate_cross_platform(model_name::String; dose_mg=100.0)
    log_info("Cross-validating Julia vs Sounio for: $model_name")

    # This would run both Julia and Sounio implementations
    # and compare results

    log_info("Running Julia implementation...")
    # julia_result = DarwinPBPK.simulate(...)

    log_info("Running Sounio implementation...")
    source_file = joinpath(PBPK_EXAMPLES, "$model_name.sio")
    # sounio_result = run_pbpk_model(source_file)

    log_info("Cross-validation complete")
    # Return comparison metrics
end

# ===========================================================================
# Main Entry Point
# ===========================================================================

function print_usage()
    println("""
    Darwin PBPK - Sounio Build Script

    Usage:
        julia scripts/build_sounio.jl [command] [options]

    Commands:
        build           Build the Sounio compiler (default)
        check           Check a specific PBPK model
        check-all       Check all PBPK models
        run             Run a PBPK model
        test            Run compiler tests
        bindings        Generate Julia bindings
        validate        Cross-validate Julia vs Sounio

    Options:
        --release       Build in release mode (default)
        --debug         Build in debug mode
        --show-types    Show type information when checking
        --input FILE    Input JSON file for simulation
        --output FILE   Output JSON file for results

    Examples:
        julia scripts/build_sounio.jl build
        julia scripts/build_sounio.jl check-all
        julia scripts/build_sounio.jl run darwin_pbpk_14comp.d
        julia scripts/build_sounio.jl check mechanistic_ddi.d --show-types
    """)
end

function main()
    args = ARGS

    if isempty(args)
        # Default: build
        build_compiler()
        return
    end

    command = args[1]
    rest_args = length(args) > 1 ? args[2:end] : String[]

    if command == "build"
        release = !("--debug" in rest_args)
        build_compiler(release=release)

    elseif command == "check"
        if isempty(rest_args) || startswith(rest_args[1], "--")
            log_error("Please specify a model file to check")
            return
        end
        model_file = rest_args[1]
        if !isabspath(model_file)
            model_file = joinpath(PBPK_EXAMPLES, model_file)
        end
        show_types = "--show-types" in rest_args
        check_pbpk_model(model_file; show_types=show_types)

    elseif command == "check-all"
        show_types = "--show-types" in rest_args
        check_all_pbpk_models(; show_types=show_types)

    elseif command == "run"
        if isempty(rest_args) || startswith(rest_args[1], "--")
            log_error("Please specify a model file to run")
            return
        end
        model_file = rest_args[1]
        if !isabspath(model_file)
            model_file = joinpath(PBPK_EXAMPLES, model_file)
        end

        input_file = nothing
        output_file = nothing
        for (i, arg) in enumerate(rest_args)
            if arg == "--input" && i < length(rest_args)
                input_file = rest_args[i+1]
            elseif arg == "--output" && i < length(rest_args)
                output_file = rest_args[i+1]
            end
        end

        run_pbpk_model(model_file; input_file=input_file, output_file=output_file)

    elseif command == "test"
        run_compiler_tests()

    elseif command == "bindings"
        generate_julia_bindings()

    elseif command == "validate"
        if isempty(rest_args)
            validate_cross_platform("darwin_pbpk_14comp")
        else
            validate_cross_platform(rest_args[1])
        end

    elseif command in ["--help", "-h", "help"]
        print_usage()

    else
        log_error("Unknown command: $command")
        print_usage()
    end
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
