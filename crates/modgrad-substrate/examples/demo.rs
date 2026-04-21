fn main() {
    match modgrad_substrate::Snapshot::take() {
        Ok(s) => {
            println!("governor           = {}", s.governor);
            println!("cpus               = {}", s.cpu_freqs_khz.len());
            if let Some(mean) = s.mean_freq_khz() {
                println!("mean cur freq      = {} MHz", mean / 1000);
            }
            if let Some(ratio) = s.mean_freq_ratio() {
                println!(
                    "mean cur/hw_max    = {:.2} (1.00 = unthrottled)",
                    ratio
                );
            }
            if let Some(max_t) = s.max_temp_c() {
                println!("max cpu temp       = {:.1} C", max_t);
            }
            println!("total throttle cnt = {}", s.throttle_total());
            println!("temp sensors:");
            for (label, c) in &s.cpu_temps_c {
                println!("  {:28} {:.1} C", label, c);
            }
        }
        Err(e) => eprintln!("substrate: {e}"),
    }
}
