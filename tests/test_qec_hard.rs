//! Hard QEC: d=9 and d=11 where decoding actually matters.
//! cargo test --release --test test_qec_hard -- --nocapture --test-threads=1

use modgrad::ctm::{Ctm, CtmConfig, CtmWeights, CtmSession, Linear, LayerConfig, forward_split, SimpleRng};
use modgrad::linalg;
use fusion_blossom::util::{SolverInitializer, SyndromePattern, VertexIndex, Weight};
use rayon::prelude::*;

fn parse(path: &str) -> Vec<(Vec<f32>, usize)> {
    std::fs::read_to_string(path).unwrap().lines()
        .filter_map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).ok()?;
            let syn: Vec<f32> = v["syndrome"].as_array()?
                .iter().filter_map(|x| Some(x.as_f64()? as f32)).collect();
            Some((syn, v["label"].as_u64()? as usize))
        }).collect()
}

fn isis_eval(cfg: &CtmConfig, train: &[(Vec<f32>, usize)], test: &[(Vec<f32>, usize)]) -> f32 {
    let ctm = Ctm::new(cfg.clone());
    let (weights, _) = ctm.into_split();
    let proprio = vec![0.0f32; cfg.d_input];
    let train_feats: Vec<(Vec<f32>, usize)> = train.par_iter().map(|(syn, label)| {
        let mut s = CtmSession::new(&weights.config);
        let mut t = weights.init_tick_state();
        let _ = forward_split(&weights, &mut s, &mut t, syn, &proprio, false);
        (t.activations.clone(), *label)
    }).collect();

    let fd = train_feats[0].0.len();
    // If fd > train samples, use random projection to avoid overfit
    let (proj, proj_dim) = if fd > train.len() / 2 {
        let pd = 256.min(fd);
        let mut rng = SimpleRng::new(31337);
        let sc = (1.0 / pd as f32).sqrt();
        let p: Vec<f32> = (0..pd * fd).map(|_| rng.next_normal() * sc).collect();
        (Some(p), pd)
    } else {
        (None, fd)
    };

    let project = |feat: &[f32]| -> Vec<f32> {
        if let Some(ref p) = proj {
            (0..proj_dim).map(|i| p[i*fd..(i+1)*fd].iter().zip(feat).map(|(a,b)| a*b).sum::<f32>()).collect()
        } else {
            feat.to_vec()
        }
    };

    let projected: Vec<(Vec<f32>, usize)> = train_feats.iter()
        .map(|(f, l)| (project(f), *l)).collect();

    let mut xtx = vec![0.0f32; proj_dim * proj_dim];
    let mut xty = vec![0.0f32; proj_dim * 2];
    for (f, l) in &projected {
        for r in 0..proj_dim { for c in 0..proj_dim { xtx[r*proj_dim+c] += f[r]*f[c]; } xty[r*2+*l] += f[r]; }
    }

    let mut best_lam = 1.0f32;
    let mut best_val = 0.0f32;
    let vs = projected.len() * 4 / 5;
    for &lam in &[1e-4, 1e-2, 0.1, 1.0, 10.0] {
        let mut xr = xtx.clone();
        for i in 0..proj_dim { xr[i*proj_dim+i] += lam; }
        if let Some(l) = linalg::cholesky(&xr, proj_dim) {
            let mut rd = Linear::new(proj_dim, 2);
            for c in 0..2 {
                let rhs: Vec<f32> = (0..proj_dim).map(|r| xty[r*2+c]).collect();
                let z = linalg::forward_solve(&l, &rhs, proj_dim);
                let w = linalg::backward_solve(&l, &z, proj_dim);
                for r in 0..proj_dim { rd.weight[c*rd.in_dim+r] = w[r]; }
            }
            let ok: usize = projected[vs..].iter()
                .map(|(f,l)| if rd.forward(f).iter().enumerate().max_by(|a,b|a.1.partial_cmp(b.1).unwrap()).unwrap().0==*l{1}else{0}).sum();
            let a = ok as f32 / (projected.len()-vs) as f32;
            if a > best_val { best_val = a; best_lam = lam; }
        }
    }

    let mut xr = xtx;
    for i in 0..proj_dim { xr[i*proj_dim+i] += best_lam; }
    if let Some(l) = linalg::cholesky(&xr, proj_dim) {
        let mut rd = Linear::new(proj_dim, 2);
        for c in 0..2 {
            let rhs: Vec<f32> = (0..proj_dim).map(|r| xty[r*2+c]).collect();
            let z = linalg::forward_solve(&l, &rhs, proj_dim);
            let w = linalg::backward_solve(&l, &z, proj_dim);
            for r in 0..proj_dim { rd.weight[c*rd.in_dim+r] = w[r]; }
        }
        let ok: usize = test.par_iter().map(|(syn,label)| {
            let mut s = CtmSession::new(&weights.config);
            let mut t = weights.init_tick_state();
            let _ = forward_split(&weights,&mut s,&mut t,syn,&proprio,false);
            let feat = project(&t.activations);
            if rd.forward(&feat).iter().enumerate().max_by(|a,b|a.1.partial_cmp(b.1).unwrap()).unwrap().0==*label{1}else{0}
        }).sum();
        ok as f32 / test.len() as f32
    } else { 0.5 }
}

fn mwpm_eval(graph_file: &str, test: &[(Vec<f32>, usize)]) -> f32 {
    let g: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(graph_file).unwrap()).unwrap();
    let n_det = g["n_det"].as_u64().unwrap() as usize;
    let mut we = Vec::new();
    let mut eo: Vec<Vec<usize>> = Vec::new();
    for e in g["edges"].as_array().unwrap() {
        let d0 = e["d0"].as_u64().unwrap() as VertexIndex;
        let d1 = e["d1"].as_u64().unwrap() as VertexIndex;
        let p: f64 = e["p"].as_f64().unwrap();
        let w = ((-1000.0*(p/(1.0-p)).ln()).abs() as Weight).max(2);
        we.push((d0,d1,(w/2)*2));
        eo.push(e["obs"].as_array().unwrap().iter().map(|v|v.as_u64().unwrap() as usize).collect());
    }
    let vv = n_det as VertexIndex;
    for b in g["boundary"].as_array().unwrap() {
        let d = b["d"].as_u64().unwrap() as VertexIndex;
        let p: f64 = b["p"].as_f64().unwrap();
        let w = ((-1000.0*(p/(1.0-p)).ln()).abs() as Weight).max(2);
        we.push((d,vv,(w/2)*2));
        eo.push(b["obs"].as_array().unwrap().iter().map(|v|v.as_u64().unwrap() as usize).collect());
    }
    let init = SolverInitializer::new((n_det+1) as VertexIndex, we, vec![vv]);
    let mut ok = 0usize;
    for (syn,label) in test {
        let defects: Vec<VertexIndex> = syn.iter().enumerate().filter(|(_,v)|**v>0.5).map(|(i,_)|i as VertexIndex).collect();
        let matched = fusion_blossom::fusion_mwpm(&init, &SyndromePattern::new_vertices(defects));
        let mut flip = 0usize;
        for pair in matched.chunks(2) {
            if pair.len()==2 {
                let (v0,v1)=(pair[0] as usize,pair[1] as usize);
                for (ei,&(a,b,_)) in init.weighted_edges.iter().enumerate() {
                    if (a as usize==v0&&b as usize==v1)||(a as usize==v1&&b as usize==v0) {
                        if ei<eo.len() { for &o in &eo[ei] { if o==0 { flip^=1; }}} break;
                    }
                }
            }
        }
        if flip==*label { ok+=1; }
    }
    ok as f32/test.len() as f32
}

#[test]
fn hard_qec() {
    eprintln!("\n  ╔═══════════════════════════════════════════════════════╗");
    eprintln!("  ║  Hard QEC: d=9 and d=11 (where decoding matters)      ║");
    eprintln!("  ╚═══════════════════════════════════════════════════════╝\n");

    for (name, d_in, train_f, test_f, graph_f) in [
        ("d=9 (720 det)", 720, "data/qec/surface_d9_depol_train.jsonl", "data/qec/surface_d9_depol_test.jsonl", "data/qec/surface_d9_depol_graph.json"),
        ("d=11 (1320 det)", 1320, "data/qec/surface_d11_depol_train.jsonl", "data/qec/surface_d11_depol_test.jsonl", "data/qec/surface_d11_depol_graph.json"),
    ] {
        let train = parse(train_f);
        let test = parse(test_f);
        let err_rate = test.iter().filter(|(_,l)|*l==1).count() as f32 / test.len() as f32;

        eprintln!("  --- {} ---", name);
        eprintln!("  {} train, {} test, {:.1}% error rate\n", train.len(), test.len(), err_rate * 100.0);

        let cfg = CtmConfig {
            iterations: 8, d_input: d_in, n_sync_out: 128,
            input_layer: LayerConfig { n_neurons: 128, ..Default::default() },
            attention_layer: LayerConfig { n_neurons: 128, ..Default::default() },
            output_layer: LayerConfig { n_neurons: 128, ..Default::default() },
            motor_layer: LayerConfig { n_neurons: 128, ..Default::default() },
            ..CtmConfig::default()
        };

        let t0 = std::time::Instant::now();
        let isis_acc = isis_eval(&cfg, &train, &test);
        let isis_time = t0.elapsed().as_secs_f64();

        let t0 = std::time::Instant::now();
        let mwpm_acc = mwpm_eval(graph_f, &test);
        let mwpm_time = t0.elapsed().as_secs_f64();

        let maj = test.iter().filter(|(_,l)|*l==0).count() as f32 / test.len() as f32;

        eprintln!("  {:>15} {:>8} {:>8}", "Decoder", "Accuracy", "Time");
        eprintln!("  {}", "-".repeat(35));
        eprintln!("  {:>15} {:>7.1}% {:>7.1}s", "isis CTM", isis_acc * 100.0, isis_time);
        eprintln!("  {:>15} {:>7.1}% {:>7.1}s", "MWPM", mwpm_acc * 100.0, mwpm_time);
        eprintln!("  {:>15} {:>7.1}%", "majority", maj * 100.0);
        eprintln!("  isis - MWPM = {:+.1}pp\n", (isis_acc - mwpm_acc) * 100.0);
    }
}
