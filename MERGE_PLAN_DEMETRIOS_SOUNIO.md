# Demetrios → Sounio Rebrand Migration Plan

**Status**: PLANNING
**Date**: December 26, 2025
**Scope**: Rebrand migration - Demetrios is being renamed to Sounio

---

## Executive Summary

**Sounio IS Demetrios** - this is a rebrand, not a merge of two separate codebases.

```
Demetrios v0.83.0  ──────rebrand──────▶  Sounio
     │                                      │
     └── Same language                      │
     └── Same compiler                      │
     └── Same stdlib                        │
     └── New name, new identity ────────────┘
```

**Author**: Demetrios Chiuratto Agourakis
**Repositories**:
- Legacy: `github.com/chiuratto-AI/demetrios`
- New: `github.com/sounio-lang/sounio`

---

## Migration Scope

### What Changes

| Component | Demetrios | Sounio | Action |
|-----------|-----------|--------|--------|
| Language name | Demetrios | Sounio | Rename |
| File extension | `.d` | `.sou` (TBD) | Convert all |
| Compiler binary | `dc` | `souc` (TBD) | Rename |
| CLI commands | `dc build` | `souc build` | Update |
| Module imports | `import demetrios.*` | `import sounio.*` | Sed replace |
| Stdlib path | `darwin_pbpk/` | Keep or rename? | TBD |

### What Stays the Same

- ✅ Type system (phantom types, M·L·T units)
- ✅ Algebraic effects (`effect[IO, Mut, Alloc]`)
- ✅ Refinement types (SMT-verified)
- ✅ Knowledge<T> / EpistemicValue<T>
- ✅ ODE solvers (Tsit5)
- ✅ PBPK stdlib (~150KB)
- ✅ Compiler architecture (Rust, logos, nom, miette)

---

## Migration Work Packages

### Package 1: darwin-pbpk-platform Updates

**Scope**: Update all Demetrios references in this repository

#### 1.1 Documentation Updates

| File | Action |
|------|--------|
| `CLAUDE.md` | Update Demetrios → Sounio references |
| `DEMETRIOS_INTEGRATION.md` | Rename to `SOUNIO_INTEGRATION.md` |
| `DEMETRIOS_STDLIB_DISCOVERED.md` | Rename to `SOUNIO_STDLIB.md` |

#### 1.2 Julia Integration Updates

| File | Action |
|------|--------|
| `julia-migration/src/DarwinPBPK/demetrios/` | Rename to `sounio/` |
| `DemetriosIntegration.jl` | Rename to `SounioIntegration.jl` |
| `medlang_demetrios_compiler.jl` | Rename to `medlang_sounio_compiler.jl` |

**Code changes in Julia files**:
```julia
# Before
struct DemetriosCompiler ... end
compile_demetrios(source_file)
run(`dc build model.d`)

# After
struct SounioCompiler ... end
compile_sounio(source_file)
run(`souc build model.sou`)
```

#### 1.3 MedLang Compiler Updates

| File | Action |
|------|--------|
| `Darwin-medlang/compiler/src/codegen/demetrios.rs` | Rename to `sounio.rs` |
| `Darwin-medlang/compiler/src/codegen/mod.rs` | Update exports |
| `Darwin-medlang/compiler/src/bin/mlc.rs` | Update CLI (`--backend sounio`) |

**CLI change**:
```bash
# Before
mlc compile model.medlang --backend demetrios -o model.d

# After
mlc compile model.medlang --backend sounio -o model.sou
```

#### 1.4 Example File Conversions

| Current | New |
|---------|-----|
| `test_simple_pk.d` | `test_simple_pk.sou` |
| `test_two_comp_pk.d` | `test_two_comp_pk.sou` |
| `test_enhanced_pk.d` | `test_enhanced_pk.sou` |
| `test_epistemic_pk.d` | `test_epistemic_pk.sou` |
| `test_medlang_to_demetrios.d` | `test_medlang_to_sounio.sou` |

#### 1.5 Submodule Update

```bash
# Remove old submodule
git submodule deinit Darwin-demetrios
git rm Darwin-demetrios

# Add new submodule
git submodule add https://github.com/sounio-lang/sounio Darwin-sounio
```

---

### Package 2: Sounio Repository Setup (github.com/sounio-lang/sounio)

#### 2.1 Repository Structure

```
sounio/
├── compiler/
│   ├── src/
│   │   ├── lexer/
│   │   ├── parser/
│   │   ├── typechecker/
│   │   ├── ir/
│   │   └── codegen/
│   ├── benches/
│   ├── tests/
│   └── Cargo.toml
├── stdlib/
│   ├── std/           # Core library
│   ├── epistemic/     # Knowledge<T>, uncertainty
│   ├── medlang/       # PK/PD models
│   ├── darwin_pbpk/   # PBPK stdlib
│   ├── fmri/          # Neuroimaging
│   ├── causal/        # Causal inference
│   ├── signal/        # DSP
│   └── gpu/           # CUDA kernels
├── runtime/
├── spec/              # Language specification
├── examples/
├── docs/
└── editors/
    └── vscode/        # Syntax highlighting
```

#### 2.2 Compiler Binary

```bash
# Build
cd compiler && cargo build --release

# Install
cargo install --path .

# Verify
souc --version
# Sounio Compiler v1.0.0 (formerly Demetrios v0.83.0)
```

#### 2.3 File Extension Registration

- Extension: `.sou`
- MIME type: `text/x-sounio`
- VSCode language ID: `sounio`

---

### Package 3: Search & Replace Operations

#### 3.1 Global Renames

```bash
# In darwin-pbpk-platform
find . -type f \( -name "*.jl" -o -name "*.md" -o -name "*.rs" -o -name "*.toml" \) \
  -exec sed -i 's/Demetrios/Sounio/g' {} \;

find . -type f \( -name "*.jl" -o -name "*.md" -o -name "*.rs" -o -name "*.toml" \) \
  -exec sed -i 's/demetrios/sounio/g' {} \;

find . -type f -name "*.d" -exec rename 's/\.d$/.sou/' {} \;
```

#### 3.2 Specific Replacements

| Pattern | Replacement |
|---------|-------------|
| `Demetrios` | `Sounio` |
| `demetrios` | `sounio` |
| `DEMETRIOS` | `SOUNIO` |
| `.d` (extension) | `.sou` |
| `dc` (compiler) | `souc` |
| `Darwin-demetrios` | `Darwin-sounio` |

---

## Timeline

| Phase | Duration | Tasks |
|-------|----------|-------|
| Phase 1 | 1 day | Rename files and directories |
| Phase 2 | 1 day | Search & replace content |
| Phase 3 | 1 day | Update submodules |
| Phase 4 | 1 day | Test compilation & examples |
| Phase 5 | 1 day | Update documentation |

**Total**: ~1 week (vs. 4.5 months for a full merge)

---

## Open Questions

1. **File extension**: Is `.sou` correct, or different?
2. **Compiler binary**: Is `souc` the name, or something else?
3. **Stdlib namespace**: Keep `darwin_pbpk` or rename to `sounio.pbpk`?
4. **Version**: Start at v1.0.0 or continue from v0.83.x?

---

## Checklist

### darwin-pbpk-platform Repository

- [ ] Rename `DEMETRIOS_INTEGRATION.md` → `SOUNIO_INTEGRATION.md`
- [ ] Rename `DEMETRIOS_STDLIB_DISCOVERED.md` → `SOUNIO_STDLIB.md`
- [ ] Update `CLAUDE.md` references
- [ ] Rename `julia-migration/src/DarwinPBPK/demetrios/` → `sounio/`
- [ ] Rename `DemetriosIntegration.jl` → `SounioIntegration.jl`
- [ ] Update Julia code (struct names, function names)
- [ ] Rename `medlang_demetrios_compiler.jl` → `medlang_sounio_compiler.jl`
- [ ] Update MedLang codegen (`demetrios.rs` → `sounio.rs`)
- [ ] Update MedLang CLI (`--backend sounio`)
- [ ] Convert all `.d` files to `.sou`
- [ ] Update submodule reference
- [ ] Run tests
- [ ] Update version numbers

### Sounio Repository (github.com/sounio-lang/sounio)

- [ ] Verify compiler builds
- [ ] Verify stdlib compiles
- [ ] Update README
- [ ] Create migration guide from Demetrios
- [ ] Tag v1.0.0 release
- [ ] Archive Demetrios repository

---

## Migration Script (Draft)

```bash
#!/bin/bash
# migrate_demetrios_to_sounio.sh

set -e

echo "=== Demetrios → Sounio Migration ==="

# 1. Rename documentation
mv DEMETRIOS_INTEGRATION.md SOUNIO_INTEGRATION.md
mv DEMETRIOS_STDLIB_DISCOVERED.md SOUNIO_STDLIB.md

# 2. Rename Julia directories
mv julia-migration/src/DarwinPBPK/demetrios julia-migration/src/DarwinPBPK/sounio
mv julia-migration/src/DarwinPBPK/sounio/DemetriosIntegration.jl \
   julia-migration/src/DarwinPBPK/sounio/SounioIntegration.jl

# 3. Rename MedLang codegen
mv Darwin-medlang/compiler/src/codegen/demetrios.rs \
   Darwin-medlang/compiler/src/codegen/sounio.rs

# 4. Convert .d files to .sou
find . -name "*.d" -type f | while read f; do
  mv "$f" "${f%.d}.sou"
done

# 5. Global search & replace
find . -type f \( -name "*.jl" -o -name "*.md" -o -name "*.rs" \) \
  -exec sed -i 's/Demetrios/Sounio/g' {} \;
find . -type f \( -name "*.jl" -o -name "*.md" -o -name "*.rs" \) \
  -exec sed -i 's/demetrios/sounio/g' {} \;
find . -type f \( -name "*.jl" -o -name "*.md" -o -name "*.rs" \) \
  -exec sed -i 's/\.d"/.sou"/g' {} \;

# 6. Update submodule
git submodule deinit -f Darwin-demetrios || true
git rm -f Darwin-demetrios || true
git submodule add https://github.com/sounio-lang/sounio Darwin-sounio

echo "=== Migration Complete ==="
echo "Run tests to verify: julia --project=julia-migration -e 'using Pkg; Pkg.test()'"
```

---

**Document Version**: 2.0
**Change**: Revised from "merge plan" to "rebrand migration plan"
**Authors**: Claude Code
**Repository**: darwin-pbpk-platform
**Branch**: claude/plan-demetrios-sounio-merge-UTnPQ
