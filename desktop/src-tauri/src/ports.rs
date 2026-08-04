//! Loopback port selection.
//!
//! The dev workflow (`make dev`) owns 8000/3000. The desktop app must be able
//! to run beside it, so both child servers get ephemeral ports instead of the
//! fixed ones.

use std::net::{Ipv4Addr, SocketAddrV4, TcpListener};

use anyhow::{Context, Result};

/// Ask the kernel for a free loopback port by binding to port 0 and reading
/// back the assignment.
///
/// This is the standard "ephemeral port" trick and carries its standard race:
/// the port is released when the listener drops and could in principle be
/// taken before the child server binds it. In practice the window is
/// microseconds, and the alternative — parsing the port out of uvicorn's and
/// Next's startup logs — is far more fragile than the race it would avoid.
pub fn free_port() -> Result<u16> {
    let listener = TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0))
        .context("failed to bind a loopback port")?;
    let port = listener
        .local_addr()
        .context("failed to read the bound port")?
        .port();
    Ok(port)
}

/// Two distinct free ports, for the backend and the Next server.
pub fn free_port_pair() -> Result<(u16, u16)> {
    // Hold the first listener open while picking the second so the kernel
    // cannot hand out the same port twice.
    let first = TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0))
        .context("failed to bind a loopback port")?;
    let second = free_port()?;
    let first_port = first.local_addr()?.port();
    drop(first);
    Ok((first_port, second))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_port_returns_a_usable_port() {
        let port = free_port().unwrap();
        assert!(port > 0);
        // The port must actually be bindable after selection.
        TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, port)).unwrap();
    }

    #[test]
    fn free_port_pair_returns_distinct_ports() {
        let (a, b) = free_port_pair().unwrap();
        assert_ne!(a, b, "backend and frontend must not share a port");
    }
}
