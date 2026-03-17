#!/usr/bin/env python3
"""Test-compile with all terrarium_advanced modules ungated.
Does NOT commit — just shows what errors we'd get.
"""
import subprocess, re, sys, shutil

# Backup originals
shutil.copy("src/terrarium_world.rs", "/tmp/tw_backup.rs")

tw = open("src/terrarium_world.rs").read()

# 1. Remove ALL terrarium_advanced feature gates
tw = tw.replace('#[cfg(feature = "terrarium_advanced")]\nmod snapshot;', 'mod snapshot;')
tw = tw.replace('#[cfg(feature = "terrarium_advanced")]\nmod biomechanics;', 'mod biomechanics;')
tw = tw.replace('#[cfg(feature = "terrarium_advanced")]\nmod explicit_microbe_impl;', 'mod explicit_microbe_impl;')

# 2. Ungate add_explicit_microbe method
tw = tw.replace(
    '    #[cfg(feature = "terrarium_advanced")]\n    pub(crate) fn add_explicit_microbe(',
    '    pub(crate) fn add_explicit_microbe(',
)

# 3. Remove inline snapshot() — it conflicts with snapshot.rs
tw = re.sub(
    r'    pub fn snapshot\(&self\) -> TerrariumWorldSnapshot \{.*?\n        \}\n    \}\n',
    '',
    tw, count=1, flags=re.DOTALL
)

open("src/terrarium_world.rs", "w").write(tw)

# 4. Compile
print("Compiling with all advanced modules ungated...")
result = subprocess.run(
    ["cargo", "check", "--no-default-features", "--lib"],
    capture_output=True, text=True, timeout=180
)

# Restore backup
shutil.copy("/tmp/tw_backup.rs", "src/terrarium_world.rs")

# Report
if result.returncode == 0:
    print("BUILD OK — all modules compile!")
    print(result.stderr.count("warning"))
else:
    errors = [l for l in result.stderr.splitlines() if l.startswith("error")]
    print(f"\n{len(errors)} errors:")
    for e in errors:
        print(f"  {e}")

    # Also show detailed error info
    print("\n--- Detailed errors ---")
    lines = result.stderr.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("error"):
            # Print context (a few lines before and after)
            start = max(0, i-2)
            end = min(len(lines), i+5)
            for j in range(start, end):
                print(f"  {lines[j]}")
            print()
