#[allow(dead_code)]
#[path = "../terrarium_ascii.rs"]
mod terrarium_terminal_3d;

use std::process::ExitCode;

fn main() -> ExitCode {
    terrarium_terminal_3d::run_terminal_3d("terrarium_3d")
}
