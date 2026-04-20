//! Reproducible harness for the Hoel/LSD dream-pretraining experiments.
//!
//! Runs the mazes binary across a fixed set of pretraining conditions
//! and prints one comparison table at the end. Use this to reproduce
//! the empirical claims in VisualRetina::lsd and train_dream docs.
//!
//! Prereq:
//!   cargo build -p mazes --release
//!
//! Run:
//!   cargo run -p dream_bench --release
//!   cargo run -p dream_bench --release -- --cliff   # include 0.8/0.9/0.95
//!   cargo run -p dream_bench --release -- --quick   # 500 steps, for iteration
//!
//! The harness shells out to ./target/release/mazes rather than
//! embedding the training loop, so results here are bit-identical to
//! what a human would see running the same command line.

use std::process::Command;
use std::time::Instant;

struct Condition {
    name: &'static str,
    flags: Vec<String>,
}

struct Result {
    name: String,
    train_loss: f32,
    train_acc_pct: f32,
    id_first_pct: f32,
    id_per_step_pct: f32,
    ood_first_pct: f32,
    ood_per_step_pct: f32,
    wall_seconds: f32,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut steps = 1500usize;
    let mut size = 11usize;
    let mut ood_size = 21usize;
    let mut seed = 7u64;
    let mut cliff = false;
    let mut batch = 8usize;
    let mut d_model = 128usize;
    let mut route_len = 10usize;
    let mut ticks = 8usize;
    let mut lr = 1e-3f32;
    let mut samples = 500usize;
    let mut epochs = 2usize;
    let mut out_dir = String::from("/tmp/dream_bench");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--steps" => { steps = args[i+1].parse().unwrap(); i += 2; }
            "--size" => { size = args[i+1].parse().unwrap(); i += 2; }
            "--ood-size" => { ood_size = args[i+1].parse().unwrap(); i += 2; }
            "--seed" => { seed = args[i+1].parse().unwrap(); i += 2; }
            "--batch" => { batch = args[i+1].parse().unwrap(); i += 2; }
            "--d-model" => { d_model = args[i+1].parse().unwrap(); i += 2; }
            "--route-len" => { route_len = args[i+1].parse().unwrap(); i += 2; }
            "--ticks" => { ticks = args[i+1].parse().unwrap(); i += 2; }
            "--lr" => { lr = args[i+1].parse().unwrap(); i += 2; }
            "--samples" => { samples = args[i+1].parse().unwrap(); i += 2; }
            "--epochs" => { epochs = args[i+1].parse().unwrap(); i += 2; }
            "--out-dir" => { out_dir = args[i+1].clone(); i += 2; }
            "--cliff" => { cliff = true; i += 1; }
            "--quick" => { steps = 500; i += 1; }
            "--help" | "-h" => {
                eprintln!(
"Usage: dream_bench [--steps N] [--size N] [--ood-size N] [--seed N]
                   [--batch N] [--d-model N] [--route-len N] [--ticks N]
                   [--lr F] [--samples N] [--epochs N]
                   [--out-dir PATH] [--cliff] [--quick]

Runs mazes across pretraining conditions and prints a comparison.

Default conditions (without --cliff):
  A  Baseline          (no pretraining)
  B  Hebbian only      (--hebbian-epochs 2)
  C  Dream integ=1.0   (legacy train_dream, known-broken)
  D  Hebbian+Dream=1.0 (combo at broken integration)
  E  LSD integ=0.1     (wear-off 90%)
  F  LSD integ=0.3     (wear-off 70%)
  G  LSD integ=0.5     (wear-off 50%)
  H  LSD integ=0.7     (validated best on mazes)

With --cliff, additionally runs:
  I  LSD integ=0.80
  J  LSD integ=0.90
  K  LSD integ=0.95
");
                return;
            }
            _ => { i += 1; }
        }
    }

    std::fs::create_dir_all(&out_dir).expect("could not create out-dir");

    let base_flags = vec![
        "--size".into(), size.to_string(),
        "--ticks".into(), ticks.to_string(),
        "--steps".into(), steps.to_string(),
        "--batch".into(), batch.to_string(),
        "--d-model".into(), d_model.to_string(),
        "--route-len".into(), route_len.to_string(),
        "--lr".into(), lr.to_string(),
        "--seed".into(), seed.to_string(),
        "--ood-size".into(), ood_size.to_string(),
    ];
    let pretrain_args = vec![
        "--hebbian-samples".into(), samples.to_string(),
    ];
    let dream_args = vec![
        "--dream-epochs".into(), epochs.to_string(),
    ];
    let hebbian_args = vec![
        "--hebbian-epochs".into(), epochs.to_string(),
    ];

    let mut conditions: Vec<Condition> = vec![
        Condition { name: "A baseline", flags: vec![] },
        Condition { name: "B hebbian",
            flags: [hebbian_args.clone(), pretrain_args.clone()].concat() },
        Condition { name: "C dream integ=1.0 (legacy)",
            flags: [dream_args.clone(), pretrain_args.clone()].concat() },
        Condition { name: "D hebbian+dream integ=1.0",
            flags: [hebbian_args.clone(), dream_args.clone(), pretrain_args.clone()].concat() },
    ];
    for integ in [0.1f32, 0.3, 0.5, 0.7] {
        conditions.push(Condition {
            name: lsd_label(integ),
            flags: lsd_flags(integ, &pretrain_args, &dream_args),
        });
    }
    if cliff {
        for integ in [0.80f32, 0.90, 0.95] {
            conditions.push(Condition {
                name: lsd_label(integ),
                flags: lsd_flags(integ, &pretrain_args, &dream_args),
            });
        }
    }

    println!("dream_bench: {} conditions, steps={steps}, size={size}→ood={ood_size}, seed={seed}",
        conditions.len());
    println!("(each run ~5-10 min on a single-socket CPU; {} runs total)",
        conditions.len());
    println!();

    let mut results = Vec::new();
    for (idx, cond) in conditions.iter().enumerate() {
        print!("[{}/{}] {:32}", idx + 1, conditions.len(), cond.name);
        let _ = std::io::Write::flush(&mut std::io::stdout());
        let log_path = format!("{}/run_{:02}_{}.log", out_dir, idx,
            cond.name.replace(' ', "_").replace('/', "_").replace('=', "_"));
        let mut all_args = base_flags.clone();
        all_args.extend(cond.flags.iter().cloned());
        let t0 = Instant::now();
        let log_file = std::fs::File::create(&log_path).expect("log file");
        let status = Command::new("./target/release/mazes")
            .args(&all_args)
            .stdout(log_file.try_clone().unwrap())
            .stderr(log_file)
            .status()
            .expect("failed to spawn mazes — did you `cargo build -p mazes --release`?");
        let secs = t0.elapsed().as_secs_f32();

        if !status.success() {
            println!(" FAILED (exit {}) — see {}", status.code().unwrap_or(-1), log_path);
            continue;
        }

        let log = std::fs::read_to_string(&log_path).unwrap_or_default();
        match parse_run(&log, cond.name, secs) {
            Some(r) => {
                println!(" done ({:.1}s) id={:.1}% ood={:.1}%",
                    r.wall_seconds, r.id_per_step_pct, r.ood_per_step_pct);
                results.push(r);
            }
            None => {
                println!(" parse failed — see {}", log_path);
            }
        }
    }

    println!();
    print_table(&results);
}

fn lsd_label(integ: f32) -> &'static str {
    match integ {
        x if (x - 0.1).abs() < 1e-3 => "E LSD integ=0.1",
        x if (x - 0.3).abs() < 1e-3 => "F LSD integ=0.3",
        x if (x - 0.5).abs() < 1e-3 => "G LSD integ=0.5",
        x if (x - 0.7).abs() < 1e-3 => "H LSD integ=0.7 (validated)",
        x if (x - 0.80).abs() < 1e-3 => "I LSD integ=0.80",
        x if (x - 0.90).abs() < 1e-3 => "J LSD integ=0.90",
        x if (x - 0.95).abs() < 1e-3 => "K LSD integ=0.95",
        _ => "? LSD integ=?",
    }
}

fn lsd_flags(integ: f32, pretrain_args: &[String], dream_args: &[String]) -> Vec<String> {
    let mut v = dream_args.to_vec();
    v.extend(pretrain_args.iter().cloned());
    v.push("--lsd-integration".into());
    v.push(format!("{integ}"));
    v
}

fn parse_run(log: &str, name: &str, wall_seconds: f32) -> Option<Result> {
    // Pull the final "step ...: loss=... route_acc=...%" line before
    // the first eval separator. That's the endpoint train metric.
    let pre_eval = log.split("--- Evaluation").next().unwrap_or("");
    let mut train_loss = 0.0f32;
    let mut train_acc_pct = 0.0f32;
    for line in pre_eval.lines() {
        if line.trim_start().starts_with("step ") && line.contains("loss=") {
            train_loss = extract_after(line, "loss=").unwrap_or(train_loss);
            train_acc_pct = extract_pct_after(line, "route_acc=").unwrap_or(train_acc_pct);
        }
    }

    // Partition into ID / OOD sections.
    let mut splits = log.split("--- ");
    splits.next(); // discard prelude
    let mut id_first = None;
    let mut id_per_step = None;
    let mut ood_first = None;
    let mut ood_per_step = None;
    for section in splits {
        let is_ood = section.starts_with("OOD");
        if !section.starts_with("Evaluation") && !is_ood { continue; }
        let mut first: Option<f32> = None;
        let mut per_step: Option<f32> = None;
        for line in section.lines() {
            let t = line.trim_start();
            if t.starts_with("First step acc:") {
                first = extract_pct_in_parens(t);
            } else if t.starts_with("Per-step acc:") {
                per_step = extract_pct_in_parens(t);
            }
        }
        if is_ood {
            ood_first = first;
            ood_per_step = per_step;
        } else {
            id_first = first;
            id_per_step = per_step;
        }
    }

    Some(Result {
        name: name.to_string(),
        train_loss,
        train_acc_pct,
        id_first_pct: id_first?,
        id_per_step_pct: id_per_step?,
        ood_first_pct: ood_first.unwrap_or(f32::NAN),
        ood_per_step_pct: ood_per_step.unwrap_or(f32::NAN),
        wall_seconds,
    })
}

fn extract_after(s: &str, prefix: &str) -> Option<f32> {
    let idx = s.find(prefix)?;
    let rest = &s[idx + prefix.len()..];
    let end = rest.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

fn extract_pct_after(s: &str, prefix: &str) -> Option<f32> {
    extract_after(s, prefix).map(|v| v * 100.0).or_else(|| {
        let idx = s.find(prefix)?;
        let rest = &s[idx + prefix.len()..];
        // form: route_acc=24.7%
        let end = rest.find('%')?;
        rest[..end].parse().ok()
    })
}

fn extract_pct_in_parens(s: &str) -> Option<f32> {
    // form: "First step acc:     74/200 (37.0%)"
    let open = s.find('(')?;
    let close = s.find('%')?;
    if close <= open { return None; }
    s[open + 1..close].parse().ok()
}

fn print_table(results: &[Result]) {
    println!("{:<32} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Condition", "TrainLoss", "TrainAcc", "ID 1st", "ID /step", "OOD 1st", "OOD /step", "Δfirst", "Δperstep");
    println!("{}", "-".repeat(32 + 9 * 11));
    // Use the baseline as the reference for gaps if present.
    let baseline_per = results.iter()
        .find(|r| r.name.contains("baseline"))
        .map(|r| r.ood_per_step_pct);
    for r in results {
        let gap_first = r.id_first_pct - r.ood_first_pct;
        let gap_per = r.id_per_step_pct - r.ood_per_step_pct;
        println!("{:<32} {:>10.3} {:>9.1}% {:>9.1}% {:>9.1}% {:>9.1}% {:>9.1}% {:>+9.1} {:>+9.1}",
            r.name, r.train_loss, r.train_acc_pct,
            r.id_first_pct, r.id_per_step_pct,
            r.ood_first_pct, r.ood_per_step_pct,
            gap_first, gap_per);
    }
    println!();
    if let Some(base_ood) = baseline_per {
        println!("Reference: baseline OOD per-step = {base_ood:.1}%. Configurations that");
        println!("both beat this on OOD and keep the gap small are Hoel-regularizing.");
    }
    println!();
    println!("Best by OOD per-step: {}",
        results.iter()
            .filter(|r| !r.ood_per_step_pct.is_nan())
            .max_by(|a, b| a.ood_per_step_pct.partial_cmp(&b.ood_per_step_pct).unwrap())
            .map(|r| format!("{} ({:.1}%)", r.name, r.ood_per_step_pct))
            .unwrap_or_else(|| "—".to_string()));
    println!("Best by OOD first-step: {}",
        results.iter()
            .filter(|r| !r.ood_first_pct.is_nan())
            .max_by(|a, b| a.ood_first_pct.partial_cmp(&b.ood_first_pct).unwrap())
            .map(|r| format!("{} ({:.1}%)", r.name, r.ood_first_pct))
            .unwrap_or_else(|| "—".to_string()));
}
