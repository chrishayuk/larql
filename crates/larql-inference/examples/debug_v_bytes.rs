//! Debug: examine raw V projection Q6_K bytes

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vd = std::path::PathBuf::from("output/gemma3-4b-v2.vindex");
    let mut index =
        larql_vindex::VectorIndex::load_vindex(&vd, &mut larql_vindex::SilentLoadCallbacks)?;
    let _ = index.load_attn_kquant(&vd);

    if let Some([_q, _k, v, _o]) = index.attn_kquant_layer_data(0) {
        let data = v.0;
        println!("V data: {} bytes, format={}", data.len(), v.1);

        // First superblock (210 bytes)
        let sb = &data[0..210];
        let ql = &sb[0..128];
        let qh = &sb[128..192];
        let scales = &sb[192..208];
        let d_bytes = &sb[208..210];

        let d = larql_models::quant::half::f16_to_f32(u16::from_le_bytes([d_bytes[0], d_bytes[1]]));
        println!("\nFirst superblock:");
        println!(
            "  d (f16 scale): {:.6} (bytes: {:02x} {:02x})",
            d, d_bytes[0], d_bytes[1]
        );
        println!("  ql first 10: {:?}", &ql[..10]);
        println!("  qh first 10: {:?}", &qh[..10]);
        println!("  scales: {:?}", scales);

        // Check if d is zero (would make everything zero)
        println!("\n  d == 0: {}", d == 0.0);
        println!("  scales all zero: {}", scales.iter().all(|&s| s == 0));
        println!("  ql all 0x00: {}", ql.iter().all(|&b| b == 0));
        println!("  ql all 0xFF: {}", ql.iter().all(|&b| b == 0xFF));

        // Check multiple superblocks
        let n_sb = data.len() / 210;
        let mut zero_d_count = 0;
        let mut zero_scales_count = 0;
        for i in 0..n_sb.min(100) {
            let sb = &data[i * 210..(i + 1) * 210];
            let d = larql_models::quant::half::f16_to_f32(u16::from_le_bytes([sb[208], sb[209]]));
            if d == 0.0 {
                zero_d_count += 1;
            }
            let scales = &sb[192..208];
            if scales.iter().all(|&s| s == 0) {
                zero_scales_count += 1;
            }
        }
        println!(
            "\n  First 100 superblocks: d=0 in {}/100, scales=0 in {}/100",
            zero_d_count, zero_scales_count
        );
    }

    Ok(())
}
