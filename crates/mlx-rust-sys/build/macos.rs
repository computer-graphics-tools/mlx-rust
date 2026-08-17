use std::{env, process::Command};

/// MLX gates the NAX (Metal 4 tensor-op) quantized kernels behind this
/// deployment target; below it, MLX compiles with `MLX_METAL_NO_NAX` and those
/// kernels silently do not exist.
pub const NAX_MIN_DEPLOYMENT_TARGET: (u32, u32) = (26, 2);

const DEFAULT_DEPLOYMENT_TARGET: &str = "26.4";

pub fn deployment_target() -> String {
    env::var("MACOSX_DEPLOYMENT_TARGET")
        .ok()
        .filter(|target| !target.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_DEPLOYMENT_TARGET.to_string())
}

fn parse_version(version: &str) -> (u32, u32) {
    let mut parts = version.trim().split('.');
    let mut next = || parts.next().and_then(|part| part.parse().ok());
    (next().unwrap_or(0), next().unwrap_or(0))
}

pub fn supports_nax(deployment_target: &str) -> bool {
    parse_version(deployment_target) >= NAX_MIN_DEPLOYMENT_TARGET
}

pub fn sdk_version() -> Option<String> {
    let output =
        Command::new("xcrun").args(["--show-sdk-version"]).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Whether the SDK supports NAX, which it gates independently of the deployment
/// target.
pub fn sdk_supports_nax() -> Option<bool> {
    sdk_version()
        .map(|version| parse_version(&version) >= NAX_MIN_DEPLOYMENT_TARGET)
}
