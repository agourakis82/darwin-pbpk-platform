#!/usr/bin/env julia
# Explore PK-Sim SQLite Database

using SQLite, DataFrames

db_path = joinpath(@__DIR__, "..", "data", "external_pk_datasets", "PKSimDB.sqlite")
db = SQLite.DB(db_path)

println("=" ^ 70)
println("PK-Sim Database Explorer")
println("=" ^ 70)

# Species
println("\n[SPECIES]")
species = DBInterface.execute(db, "SELECT * FROM tab_species") |> DataFrame
println(species)

# Populations
println("\n[POPULATIONS]")
populations = DBInterface.execute(db, "SELECT * FROM tab_populations") |> DataFrame
println(first(populations, 10))

# Genders
println("\n[GENDERS]")
genders = DBInterface.execute(db, "SELECT * FROM tab_genders") |> DataFrame
println(genders)

# Container names (organs)
println("\n[CONTAINERS/ORGANS - Sample]")
containers = DBInterface.execute(db, "SELECT * FROM tab_container_names LIMIT 30") |> DataFrame
println(containers)

# Parameters available
println("\n[PARAMETERS - Sample]")
params = DBInterface.execute(db, "SELECT * FROM tab_parameters LIMIT 20") |> DataFrame
println(params)

# Container parameter values - the key table
println("\n[CONTAINER PARAMETER VALUES - Schema]")
schema = DBInterface.execute(db, "PRAGMA table_info(tab_container_parameter_values)") |> DataFrame
println(schema)

# Sample values
println("\n[CONTAINER PARAMETER VALUES - Sample]")
values = DBInterface.execute(db, "SELECT * FROM tab_container_parameter_values LIMIT 10") |> DataFrame
println(values)

# Look for human physiological values
println("\n[HUMAN ORGAN DATA]")
human_data = DBInterface.execute(db, """
    SELECT * FROM tab_container_parameter_values
    WHERE species = 1
    LIMIT 30
""") |> DataFrame
println(human_data)

println("\n" * "=" ^ 70)
println("Database exploration complete!")
println("=" ^ 70)

DBInterface.close!(db)
