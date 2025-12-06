#!/usr/bin/env julia
# Extract PK-Sim Reference Values for PBPK Validation

using SQLite, DataFrames

db_path = joinpath(@__DIR__, "..", "data", "external_pk_datasets", "PKSimDB.sqlite")
db = SQLite.DB(db_path)

println("=" ^ 70)
println("PK-Sim REFERENCE VALUES FOR DARWIN PBPK VALIDATION")
println("=" ^ 70)

# Count
count_q = DBInterface.execute(db, """
    SELECT COUNT(*) as n FROM tab_container_parameter_values WHERE species = 'Human'
""") |> DataFrame
println("\nTotal human physiological parameters: $(count_q[1, :n])")

# pH Values
println("\n" * "-" ^ 50)
println("pH VALUES (Human)")
println("-" ^ 50)
ph = DBInterface.execute(db, """
    SELECT container_name, parameter_name, default_value
    FROM tab_container_parameter_values
    WHERE species = 'Human' AND parameter_name LIKE 'pH%'
    ORDER BY container_name
""") |> DataFrame
for i in 1:nrow(ph)
    println("  $(ph[i, :container_name]): $(ph[i, :parameter_name]) = $(ph[i, :default_value])")
end

# Tissue composition fractions
println("\n" * "-" ^ 50)
println("TISSUE COMPOSITION FRACTIONS")
println("-" ^ 50)
fractions = DBInterface.execute(db, """
    SELECT container_name, parameter_name, default_value
    FROM tab_container_parameter_values
    WHERE species = 'Human' AND parameter_name LIKE 'Fraction%'
    ORDER BY container_name, parameter_name
    LIMIT 50
""") |> DataFrame
for i in 1:nrow(fractions)
    println("  $(fractions[i, :container_name]): $(fractions[i, :parameter_name]) = $(fractions[i, :default_value])")
end

# Blood flow parameters
println("\n" * "-" ^ 50)
println("BLOOD FLOW PARAMETERS")
println("-" ^ 50)
flow = DBInterface.execute(db, """
    SELECT container_name, parameter_name, default_value
    FROM tab_container_parameter_values
    WHERE species = 'Human' AND parameter_name LIKE '%flow%'
    ORDER BY container_name
    LIMIT 40
""") |> DataFrame
for i in 1:nrow(flow)
    println("  $(flow[i, :container_name]): $(flow[i, :parameter_name]) = $(flow[i, :default_value])")
end

# Density values
println("\n" * "-" ^ 50)
println("TISSUE DENSITY")
println("-" ^ 50)
density = DBInterface.execute(db, """
    SELECT container_name, parameter_name, default_value
    FROM tab_container_parameter_values
    WHERE species = 'Human' AND parameter_name LIKE 'Density%'
    ORDER BY container_name
""") |> DataFrame
for i in 1:nrow(density)
    println("  $(density[i, :container_name]): $(density[i, :default_value])")
end

# Summary for key organs
println("\n" * "=" ^ 70)
println("KEY ORGAN PARAMETERS SUMMARY")
println("=" ^ 70)

key_organs = ["Liver", "Kidney", "Heart", "Brain", "Lung", "Muscle", "Fat", "Spleen"]

for organ in key_organs
    organ_data = DBInterface.execute(db, """
        SELECT parameter_name, default_value
        FROM tab_container_parameter_values
        WHERE species = 'Human' AND container_name = '$organ'
        ORDER BY parameter_name
    """) |> DataFrame

    if nrow(organ_data) > 0
        println("\n[$organ]")
        for i in 1:nrow(organ_data)
            println("  $(organ_data[i, :parameter_name]): $(organ_data[i, :default_value])")
        end
    end
end

# Export key values to CSV
println("\n" * "=" ^ 70)
println("EXPORTING REFERENCE VALUES")
println("=" ^ 70)

all_human = DBInterface.execute(db, """
    SELECT container_name, parameter_name, default_value, min_value, max_value
    FROM tab_container_parameter_values
    WHERE species = 'Human'
    ORDER BY container_name, parameter_name
""") |> DataFrame

output_path = joinpath(@__DIR__, "..", "data", "external_pk_datasets", "PKSim_Human_Reference_Values.csv")
CSV.write(output_path, all_human)
println("Exported $(nrow(all_human)) parameters to: $output_path")

DBInterface.close!(db)

println("\n" * "=" ^ 70)
println("DONE!")
println("=" ^ 70)
