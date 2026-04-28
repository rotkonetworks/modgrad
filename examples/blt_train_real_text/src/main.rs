//! Proof-at-scale: a tiny BLT trains end-to-end on ~10 KB of real
//! English prose (the opening of *Pride and Prejudice*, public domain)
//! for 500 steps and the loss measurably decreases. Asserts the mean
//! of the final 5% of step losses is at least 15% lower than the mean
//! of the first 5%.
//!
//! Why this complements `blt_train_smoke`:
//!   * `blt_train_smoke` cycles 8 windows of one 45-byte string for 50
//!     steps. It proves the backward chain is wired but says nothing
//!     about whether training generalises across distinct n-grams.
//!   * This example tiles a 10 KB corpus with ~620 overlapping windows
//!     and runs 10× more steps, so a working BLT must learn real
//!     conditional byte distributions, not memorise a single span.
//!
//! Threshold rationale (15%):
//!   * `blt_train_smoke` uses 5% (loose, because 50 steps on a tiny
//!     model is barely past initialisation noise).
//!   * On this 10× longer run the model has time for real progress;
//!     5% would be too loose to catch a regression. 50% would be too
//!     tight — a 32-byte-context model on real prose with 500 steps
//!     and no LR schedule is not unrestricted training. 15% reliably
//!     separates "gradients flow and loss decreases" from "stuck".
//!
//! HIP is required at runtime — the binary exits cleanly with status 0
//! when no HIP device is reachable, matching the existing examples.
//!
//! Run:
//!   cargo run --release -p blt_train_real_text

use modgrad_blt::byteify::ByteifyRecipe;
use modgrad_blt::decoder::LocalDecoderConfig;
use modgrad_blt::encoder::LocalEncoderConfig;
use modgrad_blt::model::{BltConfig, BltLatentConfig, BltModel};
use modgrad_blt::trainer::{BltModelTrainer, BltTrainerConfig};
use modgrad_device::backend::rocm::ffi::runtime_available;

// ── Corpus ───────────────────────────────────────────────────────
//
// Opening of Jane Austen, *Pride and Prejudice* (1813). Public
// domain in every jurisdiction this code will ever land in. The
// first byte is the start of Chapter 1's first sentence — a real
// sentence boundary, not mid-word — which makes byte-level training
// both correct and easy to inspect by eye.
//
// Total length is targeted at ~10 KB. With 32-byte windows at stride
// 16 that gives ~620 windows, each overlapping its neighbours by 16
// bytes so every n-gram appears in two distinct contexts.
const CORPUS: &str = "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters. \"My dear Mr. Bennet,\" said his lady to him one day, \"have you heard that Netherfield Park is let at last?\" Mr. Bennet replied that he had not. \"But it is,\" returned she; \"for Mrs. Long has just been here, and she told me all about it.\" Mr. Bennet made no answer. \"Do not you want to know who has taken it?\" cried his wife impatiently. \"You want to tell me, and I have no objection to hearing it.\" This was invitation enough. \"Why, my dear, you must know, Mrs. Long says that Netherfield is taken by a young man of large fortune from the north of England; that he came down on Monday in a chaise and four to see the place, and was so much delighted with it, that he agreed with Mr. Morris immediately; that he is to take possession before Michaelmas, and some of his servants are to be in the house by the end of next week.\" \"What is his name?\" \"Bingley.\" \"Is he married or single?\" \"Oh! Single, my dear, to be sure! A single man of large fortune; four or five thousand a year. What a fine thing for our girls!\" \"How so? How can it affect them?\" \"My dear Mr. Bennet,\" replied his wife, \"how can you be so tiresome! You must know that I am thinking of his marrying one of them.\" \"Is that his design in settling here?\" \"Design! Nonsense, how can you talk so! But it is very likely that he may fall in love with one of them, and therefore you must visit him as soon as he comes.\" \"I see no occasion for that. You and the girls may go, or you may send them by themselves, which perhaps will be still better, for as you are as handsome as any of them, Mr. Bingley may like you the best of the party.\" \"My dear, you flatter me. I certainly have had my share of beauty, but I do not pretend to be any thing extraordinary now. When a woman has five grown up daughters, she ought to give over thinking of her own beauty.\" \"In such cases, a woman has not often much beauty to think of.\" \"But, my dear, you must indeed go and see Mr. Bingley when he comes into the neighbourhood.\" \"It is more than I engage for, I assure you.\" \"But consider your daughters. Only think what an establishment it would be for one of them. Sir William and Lady Lucas are determined to go, merely on that account, for in general you know they visit no new comers. Indeed you must go, for it will be impossible for us to visit him, if you do not.\" \"You are over scrupulous surely. I dare say Mr. Bingley will be very glad to see you; and I will send a few lines by you to assure him of my hearty consent to his marrying which ever he chuses of the girls; though I must throw in a good word for my little Lizzy.\" \"I desire you will do no such thing. Lizzy is not a bit better than the others; and I am sure she is not half so handsome as Jane, nor half so good humoured as Lydia. But you are always giving her the preference.\" \"They have none of them much to recommend them,\" replied he; \"they are all silly and ignorant like other girls; but Lizzy has something more of quickness than her sisters.\" \"Mr. Bennet, how can you abuse your own children in such way? You take delight in vexing me. You have no compassion on my poor nerves.\" \"You mistake me, my dear. I have a high respect for your nerves. They are my old friends. I have heard you mention them with consideration these twenty years at least.\" \"Ah! you do not know what I suffer.\" \"But I hope you will get over it, and live to see many young men of four thousand a year come into the neighbourhood.\" \"It will be no use to us, if twenty such should come since you will not visit them.\" \"Depend upon it, my dear, that when there are twenty, I will visit them all.\" Mr. Bennet was so odd a mixture of quick parts, sarcastic humour, reserve, and caprice, that the experience of three and twenty years had been insufficient to make his wife understand his character. Her mind was less difficult to develop. She was a woman of mean understanding, little information, and uncertain temper. When she was discontented she fancied herself nervous. The business of her life was to get her daughters married; its solace was visiting and news. Mr. Bennet was among the earliest of those who waited on Mr. Bingley. He had always intended to visit him, though to the last always assuring his wife that he should not go; and till the evening after the visit was paid, she had no knowledge of it. It was then disclosed in the following manner. Observing his second daughter employed in trimming a hat, he suddenly addressed her with, \"I hope Mr. Bingley will like it Lizzy.\" \"We are not in a way to know what Mr. Bingley likes,\" said her mother resentfully, \"since we are not to visit.\" \"But you forget, mama,\" said Elizabeth, \"that we shall meet him at the assemblies, and that Mrs. Long has promised to introduce him.\" \"I do not believe Mrs. Long will do any such thing. She has two nieces of her own. She is a selfish, hypocritical woman, and I have no opinion of her.\" \"No more have I,\" said Mr. Bennet; \"and I am glad to find that you do not depend on her serving you.\" Mrs. Bennet deigned not to make any reply; but unable to contain herself, began scolding one of her daughters. \"Don't keep coughing so, Kitty, for heaven's sake! Have a little compassion on my nerves. You tear them to pieces.\" \"Kitty has no discretion in her coughs,\" said her father; \"she times them ill.\" \"I do not cough for my own amusement,\" replied Kitty fretfully. \"When is your next ball to be, Lizzy?\" \"To-morrow fortnight.\" \"Aye, so it is,\" cried her mother, \"and Mrs. Long does not come back till the day before; so, it will be impossible for her to introduce him, for she will not know him herself.\" \"Then, my dear, you may have the advantage of your friend, and introduce Mr. Bingley to her.\" \"Impossible, Mr. Bennet, impossible, when I am not acquainted with him myself; how can you be so teazing?\" \"I honour your circumspection. A fortnight's acquaintance is certainly very little. One cannot know what a man really is by the end of a fortnight. But if we do not venture, somebody else will; and after all, Mrs. Long and her neices must stand their chance; and therefore, as she will think it an act of kindness, if you decline the office, I will take it on myself.\" The girls stared at their father. Mrs. Bennet said only, \"Nonsense, nonsense!\" \"What can be the meaning of that emphatic exclamation?\" cried he. \"Do you consider the forms of introduction, and the stress that is laid on them, as nonsense? I cannot quite agree with you there. What say you, Mary? for you are a young lady of deep reflection I know, and read great books, and make extracts.\" Mary wished to say something very sensible, but knew not how. \"While Mary is adjusting her ideas,\" he continued, \"let us return to Mr. Bingley.\" \"I am sick of Mr. Bingley,\" cried his wife. \"I am sorry to hear that; but why did not you tell me so before? If I had known as much this morning, I certainly would not have called on him. It is very unlucky; but as I have actually paid the visit, we cannot escape the acquaintance now.\" The astonishment of the ladies was just what he wished; that of Mrs. Bennet perhaps surpassing the rest; though when the first tumult of joy was over, she began to declare that it was what she had expected all the while. \"How good it was in you, my dear Mr. Bennet! But I knew I should persuade you at last. I was sure you loved your girls too well to neglect such an acquaintance. Well, how pleased I am! and it is such a good joke, too, that you should have gone this morning, and never said a word about it till now.\" \"Now, Kitty, you may cough as much as you chuse,\" said Mr. Bennet; and, as he spoke, he left the room, fatigued with the raptures of his wife.";

fn main() {
    if !runtime_available() {
        eprintln!("BLT real-text training: HIP runtime unavailable, skipping");
        std::process::exit(0);
    }

    // ── Tiny BLT — same dimensional shape as `blt_train_smoke` ────
    //
    // Cross-checks (must hold for `BltConfig::validate`):
    //   encoder.patch_dim == latent.patch_dim == decoder.patch_dim = 64
    //   encoder.byte_dim  == decoder.byte_dim                       = 32
    //   encoder.n_heads * encoder.head_dim == byte_dim   (4 *  8 = 32)
    //   latent.n_heads  * latent.head_dim  == patch_dim  (4 * 16 = 64)
    //   decoder.n_heads * decoder.head_dim == byte_dim   (4 *  8 = 32)
    //   max_patches = n_patches  (LocalDecoder backward asserts equality)
    let byte_dim = 32usize;
    let n_byte_heads = 4usize;
    let byte_head_dim = byte_dim / n_byte_heads; // 8
    let patch_dim = 64usize;
    let n_patch_heads = 4usize;
    let patch_head_dim = patch_dim / n_patch_heads; // 16
    let max_seq = 32usize;
    let max_patches = 8usize;

    let blt_cfg = BltConfig {
        encoder: LocalEncoderConfig {
            n_layers: 1,
            byte_dim,
            patch_dim,
            n_heads: n_byte_heads,
            head_dim: byte_head_dim,
            mlp_dim: byte_dim * 2,
            norm_eps: 1e-5,
            rope_base: 10_000.0,
            max_seq_len: max_seq,
            ngram_min_n: 3,
            ngram_max_n: 5,
            ngram_vocab_per_n: 256,
        },
        latent: BltLatentConfig {
            n_layers: 2,
            patch_dim,
            n_heads: n_patch_heads,
            head_dim: patch_head_dim,
            mlp_dim: patch_dim * 2,
            norm_eps: 1e-5,
            rope_base: 10_000.0,
            max_patches,
        },
        decoder: LocalDecoderConfig {
            n_layers: 1,
            byte_dim,
            patch_dim,
            n_heads: n_byte_heads,
            head_dim: byte_head_dim,
            mlp_dim: byte_dim * 2,
            norm_eps: 1e-5,
            rope_base: 10_000.0,
            max_seq_len: max_seq,
        },
    };
    blt_cfg.validate().expect("tiny BLT config validates");

    let model = BltModel::new(blt_cfg).expect("BltModel::new");

    // ── Trainer ──────────────────────────────────────────────────
    //
    // `BltModelTrainer::train_step` requires `bytes.len() ==
    // micro_batch_size` (per train_step's contract — last position has
    // no target, first n-1 contribute to CE). Use 32 so each step
    // covers a full 32-byte window.
    let mbs = max_seq;
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

    // ── Corpus → sliding 32-byte windows at stride 16 ─────────────
    //
    // Stride 16 means adjacent windows overlap by 16 bytes, so every
    // n-gram in the corpus appears in (at least) two distinct context
    // alignments. That's the property that turns this from "memorise
    // one span" into "learn real conditional distributions".
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
    for (i, w) in windows.iter().enumerate() {
        assert_eq!(
            w.len(),
            mbs,
            "window {i} must be exactly {mbs} bytes — sliding-window math is wrong"
        );
    }

    // 8 patches × 4 bytes = 32 — matches `blt_train_smoke`. Same
    // fixed-stride patcher contract: `boundaries[0] == 0`,
    // `boundaries.last() == mbs`, n_patches == max_patches.
    let boundaries: Vec<usize> = (0..=8).map(|p| p * 4).collect();
    assert_eq!(boundaries[0], 0);
    assert_eq!(*boundaries.last().unwrap(), mbs);

    eprintln!(
        "blt_train_real_text: corpus_bytes={} n_windows={} stride={} \
         model dims byte_dim={} patch_dim={} lE=1 lL=2 lD=1 mbs={}",
        corpus.len(),
        n_windows,
        stride,
        byte_dim,
        patch_dim,
        mbs,
    );
    let preview_n = corpus.len().min(100);
    let preview = std::str::from_utf8(&corpus[..preview_n]).unwrap_or("<non-utf8>");
    eprintln!("blt_train_real_text: corpus[0..{preview_n}] = {preview:?}");

    // ── Loop ─────────────────────────────────────────────────────
    //
    // 500 steps. Each step samples one window with a deterministic LCG
    // seeded at 0xC0DE_FACE, so the loss curve is reproducible run to
    // run on the same machine (modulo non-determinism inside hipBLAS,
    // which is small).
    let n_steps = 500usize;
    let print_every = 25usize;
    let bin_size = 50usize;

    // Numerical Recipes LCG — full-period u64, no `rand` dep needed.
    let mut rng_state: u64 = 0xC0DE_FACE_u64;
    let mut next_window_idx = || -> usize {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (rng_state >> 33) as usize % n_windows
    };

    let mut losses: Vec<f32> = Vec::with_capacity(n_steps);
    for step in 0..n_steps {
        let widx = next_window_idx();
        let bytes = windows[widx];
        let loss = trainer
            .train_step(bytes, &boundaries)
            .unwrap_or_else(|e| panic!("train_step failed at step {step} (window {widx}): {e:?}"));
        assert!(
            loss.is_finite(),
            "loss not finite at step {step} (window {widx}): {loss}"
        );
        losses.push(loss);
        if step % print_every == 0 || step == n_steps - 1 {
            eprintln!("step {step:>3}: window {widx:>3} loss = {loss:.4}");
        }
    }

    // ── Binned curve (every 50 steps) ────────────────────────────
    let mean = |xs: &[f32]| xs.iter().copied().sum::<f32>() / xs.len() as f32;
    let n_bins = n_steps / bin_size;
    let bins: Vec<f32> = (0..n_bins)
        .map(|b| mean(&losses[b * bin_size..(b + 1) * bin_size]))
        .collect();
    eprintln!("blt_train_real_text: binned loss curve (bin = {bin_size} steps):");
    for (i, b) in bins.iter().enumerate() {
        let lo = i * bin_size;
        let hi = (i + 1) * bin_size;
        eprintln!("  steps [{lo:>3}..{hi:>3}): mean = {b:.4}");
    }

    // ── Assertion ────────────────────────────────────────────────
    //
    // Mean of last 5% (steps 475..500) must be at least 15% below mean
    // of first 5% (steps 0..25). See file-header rationale.
    let pct = n_steps / 20; // 5% of n_steps = 25
    let mean_first = mean(&losses[..pct]);
    let mean_last = mean(&losses[n_steps - pct..]);
    let threshold = mean_first * 0.85;
    let drop_pct = (mean_first - mean_last) / mean_first * 100.0;

    eprintln!(
        "blt_train_real_text: mean(first 5%) = {mean_first:.4}, \
         mean(last 5%) = {mean_last:.4}, threshold (15% drop) = {threshold:.4}, \
         measured drop = {drop_pct:.1}%"
    );

    if mean_last < threshold {
        println!(
            "PASS: BLT trains on real-text corpus (mean loss {mean_first:.2} \
             → {mean_last:.2}, drop = {drop_pct:.0}%)"
        );
    } else {
        eprintln!("FAIL: binned curve above; full loss history:");
        for (i, l) in losses.iter().enumerate() {
            eprintln!("  step {i:>3}: {l:.4}");
        }
        panic!(
            "BLT real-text training: loss did not drop ≥15% \
             (first 5% mean {mean_first:.4} → last 5% mean {mean_last:.4}, \
             drop {drop_pct:.1}% < 15%) — backward chain or optimiser may be regressing"
        );
    }
}
