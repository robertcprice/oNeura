//! BMP screenshot export (F12 key).

use std::fs::File;
use std::io::Write;

/// Save a pixel buffer as a 24-bit BMP file. Returns the filename on success.
pub fn save_screenshot(buffer: &[u32], width: usize, height: usize) -> Result<String, String> {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let filename = format!("terrarium_3d_{}.bmp", timestamp);

    let row_size = ((width * 3 + 3) / 4) * 4; // pad rows to 4-byte boundary
    let data_size = row_size * height;
    let file_size = 54 + data_size;

    let mut file = File::create(&filename).map_err(|e| format!("{}", e))?;

    // BMP file header (14 bytes) + DIB header (40 bytes)
    let mut header = vec![0u8; 54];
    header[0] = b'B';
    header[1] = b'M';
    header[2..6].copy_from_slice(&(file_size as u32).to_le_bytes());
    header[10..14].copy_from_slice(&54u32.to_le_bytes()); // pixel data offset
    header[14..18].copy_from_slice(&40u32.to_le_bytes()); // DIB header size
    header[18..22].copy_from_slice(&(width as u32).to_le_bytes());
    header[22..26].copy_from_slice(&(height as u32).to_le_bytes());
    header[26..28].copy_from_slice(&1u16.to_le_bytes()); // color planes
    header[28..30].copy_from_slice(&24u16.to_le_bytes()); // bits per pixel
    header[34..38].copy_from_slice(&(data_size as u32).to_le_bytes());
    file.write_all(&header).map_err(|e| format!("{}", e))?;

    // BMP pixels are stored bottom-to-top in BGR order
    let mut row_buf = vec![0u8; row_size];
    for y in (0..height).rev() {
        for x in 0..width {
            let c = buffer[y * width + x];
            let offset = x * 3;
            row_buf[offset] = (c & 0xff) as u8; // B
            row_buf[offset + 1] = ((c >> 8) & 0xff) as u8; // G
            row_buf[offset + 2] = ((c >> 16) & 0xff) as u8; // R
        }
        // Zero padding bytes
        for i in (width * 3)..row_size {
            row_buf[i] = 0;
        }
        file.write_all(&row_buf).map_err(|e| format!("{}", e))?;
    }

    Ok(filename)
}
