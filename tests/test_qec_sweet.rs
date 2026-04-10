//! QEC at sweet spot noise: where decoding actually matters.
//! cargo test --release --test test_qec_sweet -- --nocapture --test-threads=1

use modgrad::ctm::{Ctm, CtmConfig, CtmWeights, CtmSession, Linear, LayerConfig, forward_split, SimpleRng};
use modgrad::linalg;
use fusion_blossom::util::{SolverInitializer, SyndromePattern, VertexIndex, Weight};
use rayon::prelude::*;

fn parse(path: &str) -> Vec<(Vec<f32>, usize)> {
    std::fs::read_to_string(path).unwrap().lines()
        .filter_map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).ok()?;
            let syn: Vec<f32> = v["syndrome"].as_array()?.iter().filter_map(|x| Some(x.as_f64()? as f32)).collect();
            Some((syn, v["label"].as_u64()? as usize))
        }).collect()
}

fn isis_eval(d_input: usize, train: &[(Vec<f32>, usize)], test: &[(Vec<f32>, usize)]) -> f32 {
    let cfg = CtmConfig {
        iterations: 8, d_input, n_sync_out: 128,
        input_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        attention_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        output_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        motor_layer: LayerConfig { n_neurons: 128, ..Default::default() },
        ..CtmConfig::default()
    };
    let ctm = Ctm::new(cfg.clone());
    let (weights, _) = ctm.into_split();
    let proprio = vec![0.0f32; d_input];

    let feats: Vec<(Vec<f32>, usize)> = train.par_iter().map(|(s, l)| {
        let mut ses = CtmSession::new(&weights.config);
        let mut t = weights.init_tick_state();
        let _ = forward_split(&weights, &mut ses, &mut t, s, &proprio, false);
        (t.activations.clone(), *l)
    }).collect();

    let fd = feats[0].0.len();
    let mut xtx = vec![0.0f32; fd*fd];
    let mut xty = vec![0.0f32; fd*2];
    for (f,l) in &feats { for r in 0..fd { for c in 0..fd { xtx[r*fd+c]+=f[r]*f[c]; } xty[r*2+*l]+=f[r]; }}

    let mut best_lam = 1.0f32;
    let mut best_v = 0.0f32;
    let vs = feats.len()*4/5;
    for &lam in &[1e-4,1e-2,0.1,1.0,10.0] {
        let mut xr = xtx.clone();
        for i in 0..fd { xr[i*fd+i]+=lam; }
        if let Some(l) = linalg::cholesky(&xr, fd) {
            let mut rd = Linear::new(fd,2);
            for c in 0..2 { let rhs:Vec<f32>=(0..fd).map(|r|xty[r*2+c]).collect(); let z=linalg::forward_solve(&l,&rhs,fd); let w=linalg::backward_solve(&l,&z,fd); for r in 0..fd{rd.weight[c*rd.in_dim+r]=w[r];}}
            let ok:usize=feats[vs..].iter().map(|(f,l)|if rd.forward(f).iter().enumerate().max_by(|a,b|a.1.partial_cmp(b.1).unwrap()).unwrap().0==*l{1}else{0}).sum();
            let a=ok as f32/(feats.len()-vs)as f32; if a>best_v{best_v=a;best_lam=lam;}
        }
    }
    let mut xr=xtx; for i in 0..fd{xr[i*fd+i]+=best_lam;}
    if let Some(l)=linalg::cholesky(&xr,fd) {
        let mut rd=Linear::new(fd,2);
        for c in 0..2{let rhs:Vec<f32>=(0..fd).map(|r|xty[r*2+c]).collect();let z=linalg::forward_solve(&l,&rhs,fd);let w=linalg::backward_solve(&l,&z,fd);for r in 0..fd{rd.weight[c*rd.in_dim+r]=w[r];}}
        let ok:usize=test.par_iter().map(|(s,l)|{let mut ses=CtmSession::new(&weights.config);let mut t=weights.init_tick_state();let _=forward_split(&weights,&mut ses,&mut t,s,&proprio,false);if rd.forward(&t.activations).iter().enumerate().max_by(|a,b|a.1.partial_cmp(b.1).unwrap()).unwrap().0==*l{1}else{0}}).sum();
        ok as f32/test.len()as f32
    } else {0.5}
}

fn mwpm_eval(graph_file: &str, test: &[(Vec<f32>, usize)]) -> f32 {
    let g:serde_json::Value=serde_json::from_str(&std::fs::read_to_string(graph_file).unwrap()).unwrap();
    let nd=g["n_det"].as_u64().unwrap()as usize;
    let mut we=Vec::new();let mut eo:Vec<Vec<usize>>=Vec::new();
    for e in g["edges"].as_array().unwrap(){let d0=e["d0"].as_u64().unwrap()as VertexIndex;let d1=e["d1"].as_u64().unwrap()as VertexIndex;let p:f64=e["p"].as_f64().unwrap();let w=((-1000.0*(p/(1.0-p)).ln()).abs()as Weight).max(2);we.push((d0,d1,(w/2)*2));eo.push(e["obs"].as_array().unwrap().iter().map(|v|v.as_u64().unwrap()as usize).collect());}
    let vv=nd as VertexIndex;
    for b in g["boundary"].as_array().unwrap(){let d=b["d"].as_u64().unwrap()as VertexIndex;let p:f64=b["p"].as_f64().unwrap();let w=((-1000.0*(p/(1.0-p)).ln()).abs()as Weight).max(2);we.push((d,vv,(w/2)*2));eo.push(b["obs"].as_array().unwrap().iter().map(|v|v.as_u64().unwrap()as usize).collect());}
    let init=SolverInitializer::new((nd+1)as VertexIndex,we,vec![vv]);
    let mut ok=0usize;
    for(s,l)in test{let def:Vec<VertexIndex>=s.iter().enumerate().filter(|(_,v)|**v>0.5).map(|(i,_)|i as VertexIndex).collect();let m=fusion_blossom::fusion_mwpm(&init,&SyndromePattern::new_vertices(def));let mut flip=0usize;for pair in m.chunks(2){if pair.len()==2{let(v0,v1)=(pair[0]as usize,pair[1]as usize);for(ei,&(a,b,_))in init.weighted_edges.iter().enumerate(){if(a as usize==v0&&b as usize==v1)||(a as usize==v1&&b as usize==v0){if ei<eo.len(){for&o in&eo[ei]{if o==0{flip^=1;}}}break;}}}}if flip==*l{ok+=1;}}
    ok as f32/test.len()as f32
}

#[test]
fn sweet_spot() {
    eprintln!("\n  ╔═══════════════════════════════════════════════════════════╗");
    eprintln!("  ║  QEC Sweet Spot: Where Decoding Actually Matters          ║");
    eprintln!("  ╚═══════════════════════════════════════════════════════════╝\n");

    let configs = [
        ("d=5 p=2%", 120, "data/qec/sweet_d5_p2", 87.5, 62.4),
        ("d=7 p=2%", 336, "data/qec/sweet_d7_p2", 81.8, 52.6),
        ("d=7 p=1%", 336, "data/qec/sweet_d7_p1", 96.7, 61.8),
    ];

    eprintln!("  {:>12} {:>8} {:>8} {:>8}", "Config", "isis", "MWPM", "majority");
    eprintln!("  {}", "-".repeat(42));

    for (name, d_in, prefix, mwpm_ref, maj_ref) in configs {
        let train = parse(&format!("{prefix}_train.jsonl"));
        let test = parse(&format!("{prefix}_test.jsonl"));
        let graph = format!("{prefix}_graph.json");

        let isis = isis_eval(d_in, &train, &test);
        let mwpm = mwpm_eval(&graph, &test);
        let maj = test.iter().filter(|(_,l)|*l==0).count() as f32 / test.len() as f32;

        eprintln!("  {:>12} {:>7.1}% {:>7.1}% {:>7.1}%", name, isis*100.0, mwpm*100.0, maj.max(1.0-maj)*100.0);
    }
    eprintln!();
}
