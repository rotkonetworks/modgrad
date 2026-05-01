//! End-to-end BLT lifecycle: TRAIN → SAVE → LOAD → GENERATE.
//!
//! What this binary proves
//! -----------------------
//! Training results no longer disappear at process exit. The four
//! existing BLT smoke binaries each prove one phase in isolation
//! (forward, backward, train-loop, generate); this one chains them
//! through the checkpoint primitive that just landed and verifies the
//! property that matters in practice: a generation run from a freshly
//! loaded checkpoint produces *byte-identical* output to a generation
//! run from the in-memory trained model. That equality is the
//! `hdevalence`-rigor proof that the checkpoint format faithfully
//! preserves model behaviour, not merely weight bit-patterns (the
//! latter is already covered by `blt_checkpoint_roundtrip` in
//! `modgrad-blt`'s own tests). If a weight buffer were silently dropped
//! from the checkpoint, the existing roundtrip test would still pass
//! (because each side would compare against itself), but greedy
//! generation would diverge.
//!
//! HIP is required at runtime — exits cleanly with status 0 when no
//! HIP device is reachable, matching the rest of the smoke suite.
//!
//! Run:
//!   cargo run -p blt_lifecycle --release

use std::time::Instant;

use modgrad_blt::byteify::ByteifyRecipe;
use modgrad_blt::checkpoint::{load_blt_model_from_path, save_blt_model};
use modgrad_blt::decoder::LocalDecoderConfig;
use modgrad_blt::encoder::LocalEncoderConfig;
use modgrad_blt::model::{BltConfig, BltLatentConfig, BltModel, BltScratch};
use modgrad_transformer::config::WindowPattern;
use modgrad_blt::trainer::{BltModelTrainer, BltTrainerConfig};
use modgrad_device::backend::HipBatch;
use modgrad_device::backend::rocm::ffi::runtime_available;

// ── Model dim constants ──────────────────────────────────────────
//
// Same shape as `blt_train_real_text` and `blt_generate` — small
// enough to train 5000 steps in a couple of minutes on a contended
// 7600M XT, large enough to exercise the full encoder→latent→decoder
// chain. Cross-checks (must hold for `BltConfig::validate`):
//   encoder.patch_dim == latent.patch_dim == decoder.patch_dim = 64
//   encoder.byte_dim  == decoder.byte_dim                       = 32
//   encoder.n_heads * encoder.head_dim == byte_dim   (4 *  8 = 32)
//   latent.n_heads  * latent.head_dim  == patch_dim  (4 * 16 = 64)
//   decoder.n_heads * decoder.head_dim == byte_dim   (4 *  8 = 32)
const BYTE_DIM: usize = 32;
const N_BYTE_HEADS: usize = 4;
const BYTE_HEAD_DIM: usize = BYTE_DIM / N_BYTE_HEADS; // 8

const PATCH_DIM: usize = 64;
const N_PATCH_HEADS: usize = 4;
const PATCH_HEAD_DIM: usize = PATCH_DIM / N_PATCH_HEADS; // 16

const MAX_SEQ: usize = 32;
const MAX_PATCHES: usize = 8;

// ── Training constants ──────────────────────────────────────────
const N_TRAIN_STEPS: usize = 5_000;
const PRINT_BIN_SIZE: usize = 250;
const N_NEW_BYTES: usize = 32;
const TEMPERATURE: f32 = 0.7;

/// Deterministic seed for both training-window sampling AND generation
/// (greedy is deterministic by construction, but temperature sampling
/// needs a fixed seed to make the pre-save / post-load comparison
/// honest if it ever extends to non-greedy mode).
const RNG_SEED: u64 = 0xB17_CAFE_u64;
/// Distinct seed for window sampling so the loss curve is reproducible
/// run-to-run on the same machine (modulo non-determinism inside
/// hipBLAS, which is small).
const TRAIN_RNG_SEED: u64 = 0xC0DE_FACE_u64;

const CKPT_PATH: &str = "/tmp/blt_lifecycle.ckpt";

// ── Corpus ──────────────────────────────────────────────────────
//
// Opening of Jane Austen, *Pride and Prejudice* (1813). Public
// domain. Same ~8 KB excerpt used by `blt_train_real_text` —
// keeping the corpus identical means a regression in the lifecycle
// chain can't be blamed on a different distribution.
const CORPUS: &str = "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters. \"My dear Mr. Bennet,\" said his lady to him one day, \"have you heard that Netherfield Park is let at last?\" Mr. Bennet replied that he had not. \"But it is,\" returned she; \"for Mrs. Long has just been here, and she told me all about it.\" Mr. Bennet made no answer. \"Do not you want to know who has taken it?\" cried his wife impatiently. \"You want to tell me, and I have no objection to hearing it.\" This was invitation enough. \"Why, my dear, you must know, Mrs. Long says that Netherfield is taken by a young man of large fortune from the north of England; that he came down on Monday in a chaise and four to see the place, and was so much delighted with it, that he agreed with Mr. Morris immediately; that he is to take possession before Michaelmas, and some of his servants are to be in the house by the end of next week.\" \"What is his name?\" \"Bingley.\" \"Is he married or single?\" \"Oh! Single, my dear, to be sure! A single man of large fortune; four or five thousand a year. What a fine thing for our girls!\" \"How so? How can it affect them?\" \"My dear Mr. Bennet,\" replied his wife, \"how can you be so tiresome! You must know that I am thinking of his marrying one of them.\" \"Is that his design in settling here?\" \"Design! Nonsense, how can you talk so! But it is very likely that he may fall in love with one of them, and therefore you must visit him as soon as he comes.\" \"I see no occasion for that. You and the girls may go, or you may send them by themselves, which perhaps will be still better, for as you are as handsome as any of them, Mr. Bingley may like you the best of the party.\" \"My dear, you flatter me. I certainly have had my share of beauty, but I do not pretend to be any thing extraordinary now. When a woman has five grown up daughters, she ought to give over thinking of her own beauty.\" \"In such cases, a woman has not often much beauty to think of.\" \"But, my dear, you must indeed go and see Mr. Bingley when he comes into the neighbourhood.\" \"It is more than I engage for, I assure you.\" \"But consider your daughters. Only think what an establishment it would be for one of them. Sir William and Lady Lucas are determined to go, merely on that account, for in general you know they visit no new comers. Indeed you must go, for it will be impossible for us to visit him, if you do not.\" \"You are over scrupulous surely. I dare say Mr. Bingley will be very glad to see you; and I will send a few lines by you to assure him of my hearty consent to his marrying which ever he chuses of the girls; though I must throw in a good word for my little Lizzy.\" \"I desire you will do no such thing. Lizzy is not a bit better than the others; and I am sure she is not half so handsome as Jane, nor half so good humoured as Lydia. But you are always giving her the preference.\" \"They have none of them much to recommend them,\" replied he; \"they are all silly and ignorant like other girls; but Lizzy has something more of quickness than her sisters.\" \"Mr. Bennet, how can you abuse your own children in such way? You take delight in vexing me. You have no compassion on my poor nerves.\" \"You mistake me, my dear. I have a high respect for your nerves. They are my old friends. I have heard you mention them with consideration these twenty years at least.\" \"Ah! you do not know what I suffer.\" \"But I hope you will get over it, and live to see many young men of four thousand a year come into the neighbourhood.\" \"It will be no use to us, if twenty such should come since you will not visit them.\" \"Depend upon it, my dear, that when there are twenty, I will visit them all.\" Mr. Bennet was so odd a mixture of quick parts, sarcastic humour, reserve, and caprice, that the experience of three and twenty years had been insufficient to make his wife understand his character. Her mind was less difficult to develop. She was a woman of mean understanding, little information, and uncertain temper. When she was discontented she fancied herself nervous. The business of her life was to get her daughters married; its solace was visiting and news. Mr. Bennet was among the earliest of those who waited on Mr. Bingley. He had always intended to visit him, though to the last always assuring his wife that he should not go; and till the evening after the visit was paid, she had no knowledge of it. It was then disclosed in the following manner. Observing his second daughter employed in trimming a hat, he suddenly addressed her with, \"I hope Mr. Bingley will like it Lizzy.\" \"We are not in a way to know what Mr. Bingley likes,\" said her mother resentfully, \"since we are not to visit.\" \"But you forget, mama,\" said Elizabeth, \"that we shall meet him at the assemblies, and that Mrs. Long has promised to introduce him.\" \"I do not believe Mrs. Long will do any such thing. She has two nieces of her own. She is a selfish, hypocritical woman, and I have no opinion of her.\" \"No more have I,\" said Mr. Bennet; \"and I am glad to find that you do not depend on her serving you.\" Mrs. Bennet deigned not to make any reply; but unable to contain herself, began scolding one of her daughters. \"Don't keep coughing so, Kitty, for heaven's sake! Have a little compassion on my nerves. You tear them to pieces.\" \"Kitty has no discretion in her coughs,\" said her father; \"she times them ill.\" \"I do not cough for my own amusement,\" replied Kitty fretfully. \"When is your next ball to be, Lizzy?\" \"To-morrow fortnight.\" \"Aye, so it is,\" cried her mother, \"and Mrs. Long does not come back till the day before; so, it will be impossible for her to introduce him, for she will not know him herself.\" \"Then, my dear, you may have the advantage of your friend, and introduce Mr. Bingley to her.\" \"Impossible, Mr. Bennet, impossible, when I am not acquainted with him myself; how can you be so teazing?\" \"I honour your circumspection. A fortnight's acquaintance is certainly very little. One cannot know what a man really is by the end of a fortnight. But if we do not venture, somebody else will; and after all, Mrs. Long and her neices must stand their chance; and therefore, as she will think it an act of kindness, if you decline the office, I will take it on myself.\" The girls stared at their father. Mrs. Bennet said only, \"Nonsense, nonsense!\" \"What can be the meaning of that emphatic exclamation?\" cried he. \"Do you consider the forms of introduction, and the stress that is laid on them, as nonsense? I cannot quite agree with you there. What say you, Mary? for you are a young lady of deep reflection I know, and read great books, and make extracts.\" Mary wished to say something very sensible, but knew not how. \"While Mary is adjusting her ideas,\" he continued, \"let us return to Mr. Bingley.\" \"I am sick of Mr. Bingley,\" cried his wife. \"I am sorry to hear that; but why did not you tell me so before? If I had known as much this morning, I certainly would not have called on him. It is very unlucky; but as I have actually paid the visit, we cannot escape the acquaintance now.\" The astonishment of the ladies was just what he wished; that of Mrs. Bennet perhaps surpassing the rest; though when the first tumult of joy was over, she began to declare that it was what she had expected all the while. \"How good it was in you, my dear Mr. Bennet! But I knew I should persuade you at last. I was sure you loved your girls too well to neglect such an acquaintance. Well, how pleased I am! and it is such a good joke, too, that you should have gone this morning, and never said a word about it till now.\" \"Now, Kitty, you may cough as much as you chuse,\" said Mr. Bennet; and, as he spoke, he left the room, fatigued with the raptures of his wife.";

fn build_config() -> BltConfig {
    BltConfig {
        encoder: LocalEncoderConfig {
            n_layers: 1,
            byte_dim: BYTE_DIM,
            patch_dim: PATCH_DIM,
            n_heads: N_BYTE_HEADS,
            head_dim: BYTE_HEAD_DIM,
            mlp_dim: BYTE_DIM * 2,
            norm_eps: 1e-5,
            rope_base: 10_000.0,
            max_seq_len: MAX_SEQ,
            ngram_min_n: 3,
            ngram_max_n: 5,
            ngram_vocab_per_n: 256,
            window_pattern: WindowPattern::Full,
        },
        latent: BltLatentConfig {
            n_layers: 2,
            patch_dim: PATCH_DIM,
            n_heads: N_PATCH_HEADS,
            head_dim: PATCH_HEAD_DIM,
            mlp_dim: PATCH_DIM * 2,
            norm_eps: 1e-5,
            rope_base: 10_000.0,
            max_patches: MAX_PATCHES,
        },
        decoder: LocalDecoderConfig {
            n_layers: 1,
            byte_dim: BYTE_DIM,
            patch_dim: PATCH_DIM,
            n_heads: N_BYTE_HEADS,
            head_dim: BYTE_HEAD_DIM,
            mlp_dim: BYTE_DIM * 2,
            norm_eps: 1e-5,
            rope_base: 10_000.0,
            max_seq_len: MAX_SEQ,
            window_pattern: WindowPattern::Full,
        },
    }
}

/// Format a byte slice for human-readable display: ASCII-printable
/// bytes pass through; everything else escapes as `\xNN`.
fn format_bytes(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 4);
    for &b in bytes {
        if b == b'\\' {
            s.push_str("\\\\");
        } else if b == b'"' {
            s.push_str("\\\"");
        } else if (0x20..=0x7E).contains(&b) {
            s.push(b as char);
        } else if b == b'\n' {
            s.push_str("\\n");
        } else if b == b'\t' {
            s.push_str("\\t");
        } else {
            s.push_str(&format!("\\x{:02x}", b));
        }
    }
    s
}

fn main() {
    if !runtime_available() {
        eprintln!("BLT lifecycle: HIP runtime unavailable, skipping");
        std::process::exit(0);
    }

    // ─────────────────────────────────────────────────────────────
    // PHASE 1: TRAIN
    // ─────────────────────────────────────────────────────────────
    let blt_cfg = build_config();
    blt_cfg.validate().expect("tiny BLT config validates");

    let model = BltModel::new(blt_cfg.clone()).expect("BltModel::new");

    let mbs = MAX_SEQ;
    let trainer_cfg = BltTrainerConfig {
        micro_batch_size: mbs,
        seq_len: mbs,
        ..BltTrainerConfig::default()
    };
    let mut trainer = BltModelTrainer::new(
        model,
        trainer_cfg,
        ByteifyRecipe::global_predicate(),
    )
    .expect("BltModelTrainer::new");

    // Sliding 32-byte windows at stride 16 — every n-gram appears in
    // at least two distinct contexts. Same shape as
    // `blt_train_real_text`.
    let corpus = CORPUS.as_bytes();
    let stride = 16usize;
    assert!(corpus.len() >= mbs, "corpus must be at least one window long");
    let n_windows = (corpus.len() - mbs) / stride + 1;
    let windows: Vec<&[u8]> = (0..n_windows)
        .map(|w| {
            let start = w * stride;
            &corpus[start..start + mbs]
        })
        .collect();

    // 8 patches × 4 bytes = 32 — fixed-stride patcher. Boundaries
    // are constant across windows.
    let train_boundaries: Vec<usize> = (0..=MAX_PATCHES).map(|p| p * 4).collect();
    assert_eq!(train_boundaries[0], 0);
    assert_eq!(*train_boundaries.last().unwrap(), mbs);

    eprintln!(
        "blt_lifecycle: corpus_bytes={} n_windows={} train_steps={} \
         dims byte_dim={} patch_dim={} max_seq={} max_patches={}",
        corpus.len(), n_windows, N_TRAIN_STEPS,
        BYTE_DIM, PATCH_DIM, MAX_SEQ, MAX_PATCHES,
    );

    // Numerical Recipes LCG — full-period u64, seeded at TRAIN_RNG_SEED.
    let mut rng_state: u64 = TRAIN_RNG_SEED;
    let mut next_window_idx = || -> usize {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (rng_state >> 33) as usize % n_windows
    };

    eprintln!("─── PHASE 1: TRAIN ({} steps) ───────────────────", N_TRAIN_STEPS);
    let t_train = Instant::now();
    let mut losses: Vec<f32> = Vec::with_capacity(N_TRAIN_STEPS);
    for step in 0..N_TRAIN_STEPS {
        let widx = next_window_idx();
        let bytes = windows[widx];
        let loss = trainer
            .train_step(bytes, &train_boundaries)
            .unwrap_or_else(|e| panic!("train_step failed at step {step}: {e:?}"));
        assert!(loss.is_finite(), "loss not finite at step {step}: {loss}");
        losses.push(loss);

        // Print at the end of each bin for a tidy progress view.
        if (step + 1) % PRINT_BIN_SIZE == 0 {
            let lo = step + 1 - PRINT_BIN_SIZE;
            let hi = step + 1;
            let mean = losses[lo..hi].iter().copied().sum::<f32>()
                / PRINT_BIN_SIZE as f32;
            eprintln!(
                "  steps [{lo:>4}..{hi:>4}): mean loss = {mean:.4}",
            );
        }
    }
    let train_secs = t_train.elapsed().as_secs_f64();

    let bin_mean = |bin: usize| -> f32 {
        let lo = bin * PRINT_BIN_SIZE;
        let hi = lo + PRINT_BIN_SIZE;
        losses[lo..hi].iter().copied().sum::<f32>() / PRINT_BIN_SIZE as f32
    };
    let n_bins = N_TRAIN_STEPS / PRINT_BIN_SIZE;
    let first_bin_mean = bin_mean(0);
    let last_bin_mean = bin_mean(n_bins - 1);
    let final_loss = *losses.last().unwrap();
    eprintln!(
        "PHASE 1 done: {:.1}s ({} steps; first 250-bin mean {:.4}, \
         last 250-bin mean {:.4}, final {:.4})",
        train_secs, N_TRAIN_STEPS, first_bin_mean, last_bin_mean, final_loss,
    );

    // ── Capture pre-save greedy output for the load-bearing assertion ──
    //
    // Same prompt, same seed, same scratch — the only thing that should
    // differ across save/load is the path the weights took through the
    // checkpoint. If a buffer is silently dropped from the checkpoint
    // its `BltModel::new` initialiser (random or zero) survives load,
    // which will skew the argmax at some step → byte divergence.
    let pre_save_prompt = b"It is a truth";
    let model_ref = trainer.model_mut();
    let mut scratch_pre = BltScratch::new(&blt_cfg).expect("BltScratch::new (pre)");
    let batch_pre = HipBatch::new();
    let pre_save_greedy = model_ref
        .generate(&batch_pre, pre_save_prompt, N_NEW_BYTES, 0.0, RNG_SEED, &mut scratch_pre)
        .expect("generate (pre-save greedy)");
    eprintln!(
        "pre-save greedy ({} bytes for prompt {:?}): {}",
        pre_save_greedy.len(),
        std::str::from_utf8(pre_save_prompt).unwrap_or("<non-utf8>"),
        format_bytes(&pre_save_greedy),
    );

    // ─────────────────────────────────────────────────────────────
    // PHASE 2: SAVE
    // ─────────────────────────────────────────────────────────────
    eprintln!("─── PHASE 2: SAVE ─────────────────────────────────");
    let t_save = Instant::now();
    {
        let model_ref = trainer.model();
        save_blt_model(model_ref, CKPT_PATH).expect("save_blt_model");
    }
    let save_secs = t_save.elapsed().as_secs_f64();
    let ckpt_size_bytes = std::fs::metadata(CKPT_PATH)
        .expect("stat checkpoint")
        .len();
    let ckpt_size_mb = ckpt_size_bytes as f64 / (1024.0 * 1024.0);
    println!(
        "Saved checkpoint: {:.2} MB at {} ({:.2}s)",
        ckpt_size_mb, CKPT_PATH, save_secs,
    );

    // ── Drop the trained model handle ───────────────────────────
    //
    // Explicit teardown so any latent in-memory aliasing of the trained
    // weights is impossible — anything we observe post-load came off
    // disk. `trainer` owns `model`, so dropping the trainer drops the
    // model too.
    drop(scratch_pre);
    drop(batch_pre);
    drop(trainer);

    // ─────────────────────────────────────────────────────────────
    // PHASE 3: LOAD
    // ─────────────────────────────────────────────────────────────
    eprintln!("─── PHASE 3: LOAD ─────────────────────────────────");
    let t_load = Instant::now();
    let mut loaded_model = load_blt_model_from_path(CKPT_PATH)
        .expect("load_blt_model_from_path");
    let load_secs = t_load.elapsed().as_secs_f64();

    // Approximate parameter count from checkpoint file size — each
    // weight is a raw f32 (4 bytes), so n_params ≈ ckpt_bytes / 4
    // minus a small header (~few hundred bytes). Good enough for a
    // diagnostic line.
    let n_params_approx = (ckpt_size_bytes as usize).saturating_sub(512) / 4;
    println!(
        "Loaded checkpoint: layers={} params≈{} ({:.2}s)",
        loaded_model.n_layers(),
        n_params_approx,
        load_secs,
    );

    // ─────────────────────────────────────────────────────────────
    // PHASE 4: GENERATE
    // ─────────────────────────────────────────────────────────────
    eprintln!("─── PHASE 4: GENERATE ─────────────────────────────");
    let mut scratch = BltScratch::new(&blt_cfg).expect("BltScratch::new (post-load)");
    let batch = HipBatch::new();

    let prompts: &[&str] = &["It is a truth", "Mr. Bennet", "the quick"];
    let t_gen = Instant::now();
    println!();
    for prompt in prompts {
        let prompt_bytes = prompt.as_bytes();
        let greedy = loaded_model
            .generate(&batch, prompt_bytes, N_NEW_BYTES, 0.0, RNG_SEED, &mut scratch)
            .expect("generate (greedy)");
        let temp = loaded_model
            .generate(&batch, prompt_bytes, N_NEW_BYTES, TEMPERATURE, RNG_SEED, &mut scratch)
            .expect("generate (temperature)");
        assert_eq!(greedy.len(), N_NEW_BYTES, "greedy length mismatch ({prompt:?})");
        assert_eq!(temp.len(), N_NEW_BYTES, "temperature length mismatch ({prompt:?})");

        println!("prompt: {prompt:?}");
        println!("  greedy:    {}", format_bytes(&greedy));
        println!("  temp=0.7:  {}", format_bytes(&temp));
        println!();
    }
    let gen_secs = t_gen.elapsed().as_secs_f64();

    // ─────────────────────────────────────────────────────────────
    // hdevalence-rigor assertion: pre-save greedy == post-load greedy
    // ─────────────────────────────────────────────────────────────
    //
    // Re-run the SAME prompt with the SAME seed against the freshly
    // loaded model. Bit-for-bit equality is the contract: any
    // divergence means a weight buffer was missed by the checkpoint
    // (the freshly-allocated value survived load). The existing
    // `blt_checkpoint_roundtrip` test compares weight bytes, but a
    // dropped buffer would fool it (each side would compare against
    // itself); generation equality across the save/load boundary is
    // the load-bearing observable.
    let post_load_greedy = loaded_model
        .generate(&batch, pre_save_prompt, N_NEW_BYTES, 0.0, RNG_SEED, &mut scratch)
        .expect("generate (post-load greedy)");
    let equal = pre_save_greedy == post_load_greedy;
    println!(
        "pre/post save equality check (prompt {:?}, {} bytes greedy):",
        std::str::from_utf8(pre_save_prompt).unwrap_or("<non-utf8>"),
        N_NEW_BYTES,
    );
    println!("  pre-save:  {}", format_bytes(&pre_save_greedy));
    println!("  post-load: {}", format_bytes(&post_load_greedy));
    println!("  byte-equal: {}", equal);
    assert!(
        equal,
        "BLT lifecycle: greedy generation diverged across save/load — \
         a weight buffer is missing from the checkpoint format. \
         pre-save  = {pre:?}, post-load = {post:?}",
        pre = format_bytes(&pre_save_greedy),
        post = format_bytes(&post_load_greedy),
    );

    // ── Wall-clock summary ──────────────────────────────────────
    let total_secs = train_secs + save_secs + load_secs + gen_secs;
    println!();
    println!(
        "wall-clock: train {:.1}s | save {:.2}s | load {:.2}s | generate {:.2}s | total {:.1}s",
        train_secs, save_secs, load_secs, gen_secs, total_secs,
    );

    println!();
    println!("PASS: BLT lifecycle complete (train → save → load → generate; greedy preserved)");

    // Best-effort cleanup. Failing to remove the file is not a fatal
    // condition for this binary — `/tmp` cleans itself eventually.
    let _ = std::fs::remove_file(CKPT_PATH);
}
