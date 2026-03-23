import os
import subprocess

def run(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# 1. Delete legacy and WIP
run("rm -rf models")
run("rm -rf oneura-terrarium")

# 2. Rename main folders
run("mv oneuro-metal core")
run("mv oneuro-3d desktop")
run("mv oneuro-wasm web")

# 3. Create CLI crate
os.makedirs("cli/src", exist_ok=True)
run("mv core/src/bin cli/src/")

# Move web assets
run("mv core/web/* web/")
run("rm -rf core/web")

# 4. Create Workspace Cargo.toml
with open("Cargo.toml", "w") as f:
    f.write("""[workspace]
members = [
    "core",
    "cli",
    "desktop",
    "web"
]
resolver = "2"
""")

# 5. Fix core/Cargo.toml
with open("core/Cargo.toml", "r") as f:
    core_toml = f.read()
core_toml = core_toml.replace('name = "oneuro-metal"', 'name = "oneura-core"')
with open("core/Cargo.toml", "w") as f:
    f.write(core_toml)

# 6. Create cli/Cargo.toml
cli_toml = core_toml.replace('name = "oneura-core"', 'name = "oneura-cli"')
cli_toml += "\noneura-core = { path = \"../core\" }\n"
with open("cli/Cargo.toml", "w") as f:
    f.write(cli_toml)

# 7. Fix desktop/Cargo.toml
with open("desktop/Cargo.toml", "r") as f:
    desktop_toml = f.read()
desktop_toml = desktop_toml.replace('name = "oneuro-3d"', 'name = "oneura-desktop"')
desktop_toml = desktop_toml.replace('oneuro-metal = { path = "../oneuro-metal" }', 'oneura-core = { path = "../core" }')
with open("desktop/Cargo.toml", "w") as f:
    f.write(desktop_toml)

# 8. Fix web/Cargo.toml
with open("web/Cargo.toml", "r") as f:
    web_toml = f.read()
web_toml = web_toml.replace('name = "oneuro-wasm"', 'name = "oneura-web"')
web_toml = web_toml.replace('oneuro-metal = { path = "../oneuro-metal" }', 'oneura-core = { path = "../core" }')
with open("web/Cargo.toml", "w") as f:
    f.write(web_toml)

# 9. Global replace in .rs files: oneuro_metal -> oneura_core
run("find . -type f -name '*.rs' -exec sed -i '' 's/oneuro_metal/oneura_core/g' {} +")
run("find . -type f -name '*.rs' -exec sed -i '' 's/oneuro-metal/oneura-core/g' {} +")

# 10. Update imports in HTML/JS/MD files if any
run("find web -type f -name '*.html' -exec sed -i '' 's/oneuro-metal/oneura-core/g' {} +")
run("find web -type f -name '*.js' -exec sed -i '' 's/oneuro-metal/oneura-core/g' {} + 2>/dev/null || true")
