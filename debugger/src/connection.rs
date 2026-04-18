//! TCP client for the NC debug socket protocol.
//! Length-prefixed bincode over TCP.

use std::io::{Read, Write};
use std::net::TcpStream;
use isis_runtime::nc_socket::{DebugRequest, DebugResponse};

pub struct DebugClient {
    stream: TcpStream,
}

impl DebugClient {
    pub fn connect(addr: &str) -> std::io::Result<Self> {
        let stream = TcpStream::connect(addr)?;
        stream.set_nodelay(true)?;
        stream.set_read_timeout(Some(std::time::Duration::from_secs(2)))?;
        Ok(Self { stream })
    }

    pub fn request(&mut self, req: &DebugRequest) -> std::io::Result<DebugResponse> {
        // Send
        let data = bincode::serialize(req)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let len = data.len() as u32;
        self.stream.write_all(&len.to_le_bytes())?;
        self.stream.write_all(&data)?;
        self.stream.flush()?;

        // Receive
        let mut len_buf = [0u8; 4];
        self.stream.read_exact(&mut len_buf)?;
        let len = u32::from_le_bytes(len_buf) as usize;
        if len > 10_000_000 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "response too large"));
        }
        let mut buf = vec![0u8; len];
        self.stream.read_exact(&mut buf)?;
        bincode::deserialize(&buf)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}
