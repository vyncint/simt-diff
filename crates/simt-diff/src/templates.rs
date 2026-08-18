//! Proof-carrying kernel templates (brief §10).
//!
//! Stage 1 is the *mask pair* rather than the divergent barrier the brief
//! proposes, because the baseline measured the barrier producing no dynamic
//! evidence at all on sm_86 (§9.3/§9.4) while the mask pair produces a real,
//! checkable value discrepancy (§9.5). It is the better first vertical
//! slice: same pipeline, but with something at the end of it.
//!
//! Every template declares its semantic class and, where it can, computes
//! what the kernel should produce. The kernel text is emitted verbatim into
//! both the analysis crate (cuda-device only, analyzable anywhere) and the
//! runner crate (adds the host runtime, GPU only), so the two are provably
//! the same program.


use crate::oracle::ConstructionOracle;
use crate::records::{GeneratorRecord, Launch, ReferenceModel};

pub const GENERATOR_VERSION: &str = concat!("simt-diff/", env!("CARGO_PKG_VERSION"));

/// The `#[kernel]` body plus the declared semantics around it.
pub struct Template {
    pub id: &'static str,
    pub oracle: ConstructionOracle,
    pub oracle_reason: &'static str,
    /// Doc comment placed on the generated kernel.
    pub doc: &'static str,
    /// Body of `pub fn probe(mut out: DisjointSlice<u32>)`.
    pub body: &'static str,
    /// Extra `use` items the body needs, beyond the common set.
    pub extra_uses: &'static [&'static str],
    /// Expected value per lane, given a launch. `None` = no reference model.
    pub reference: fn(Launch) -> Option<ReferenceModel>,
    /// The analyzer's own documented reason for not reporting this class,
    /// quoted, when one exists.
    pub documented_limitation: Option<&'static str>,
}

/// Every lane observes the full-warp ballot, which is what a mask naming all
/// 32 lanes must produce when every lane passes a true predicate.
fn full_ballot(launch: Launch) -> Option<ReferenceModel> {
    Some(ReferenceModel {
        description: "ballot_sync(0xffff_ffff, true) with all 32 lanes present \
                      names every lane, so every lane reads 0xffffffff"
            .to_string(),
        expected: (0..launch.block.0).map(|l| (l, 0xffff_ffffu32)).collect(),
        launch,
    })
}

/// The mask names lanes 0..=15 only. Every lane still executes the call, so
/// the operation is undefined by the PTX rules -- but the *defined* reading
/// of the mask it was given is a ballot over the named lanes, i.e. 0x0000ffff.
///
/// Recording that as the reference is what makes the bug visible: the
/// baseline measured hardware returning 0xffffffff, identical to the valid
/// case, so nothing except this comparison distinguishes them.
fn shrunk_ballot(launch: Launch) -> Option<ReferenceModel> {
    Some(ReferenceModel {
        description: "the supplied mask names lanes 0..=15, so a ballot that \
                      honoured it would read 0x0000ffff; every lane executing \
                      the call makes the operation undefined, and the observed \
                      value is the evidence of that"
            .to_string(),
        expected: (0..launch.block.0).map(|l| (l, 0x0000_ffffu32)).collect(),
        launch,
    })
}

fn no_reference(_: Launch) -> Option<ReferenceModel> {
    None
}

/// Stage 1: the mask pair, plus the barrier pair kept for contrast.
pub const TEMPLATES: &[Template] = &[
    Template {
        id: "mask_full_convergent",
        oracle: ConstructionOracle::KnownMaskValid,
        oracle_reason: "the collective is called at a convergent point and the \
                        mask names exactly the lanes that participate",
        doc: "ORACLE: KNOWN_MASK_VALID. Convergent call, mask == participants.",
        body: "        let i = thread::index_1d();\n\
               \x20       let b = warp::ballot_sync(0xffff_ffff, true);\n\
               \x20       if let Some(e) = out.get_mut(i) { *e = b; }",
        extra_uses: &["warp"],
        reference: full_ballot,
        documented_limitation: None,
    },
    Template {
        id: "mask_shrunk_convergent",
        oracle: ConstructionOracle::KnownMaskInvalid,
        oracle_reason: "all 32 lanes execute the collective while the supplied \
                        mask names only lanes 0..=15, so lanes 16..=31 \
                        participate unnamed",
        doc: "ORACLE: KNOWN_MASK_INVALID. Convergent call, mask names 16 of 32 \
              participating lanes. reconverge's own corpus records this class \
              (`shrinkmask`) at 0% static recall in v1, by design.",
        body: "        let i = thread::index_1d();\n\
               \x20       let b = warp::ballot_sync(0x0000_ffff, true);\n\
               \x20       if let Some(e) = out.get_mut(i) { *e = b; }",
        extra_uses: &["warp"],
        reference: shrunk_ballot,
        documented_limitation: Some(
            "conformance/MUTATION.md, shrinkmask row: \"a shrunk full mask at \
             a *convergent* call site is a mask-lane mismatch only under launch \
             shapes the static engine does not model (RC002 v1 checks \
             convergence; it does not do mask arithmetic against launch \
             shapes)\" -- expected recall 0 in v1",
        ),
    },
    Template {
        id: "barrier_uniform",
        oracle: ConstructionOracle::KnownSafe,
        oracle_reason: "the barrier is not under any conditional, so every \
                        thread of the block reaches it",
        doc: "ORACLE: KNOWN_SAFE. Every thread of the block reaches the barrier.",
        body: "        let i = thread::index_1d();\n\
               \x20       thread::sync_threads();\n\
               \x20       if let Some(e) = out.get_mut(i) { *e = 1; }",
        extra_uses: &[],
        documented_limitation: None,
        reference: |launch| {
            Some(ReferenceModel {
                description: "every lane writes 1".to_string(),
                expected: (0..launch.block.0).map(|l| (l, 1u32)).collect(),
                launch,
            })
        },
    },
    Template {
        id: "barrier_divergent_intra_warp",
        oracle: ConstructionOracle::KnownUnsafe,
        oracle_reason: "the barrier is guarded by a thread-index-derived \
                        predicate, so only the even lanes reach it",
        doc: "ORACLE: KNOWN_UNSAFE (RC001). Only the even lanes reach the \
              barrier. Measured on sm_86: completes, synccheck clean -- see \
              docs/research-baseline.md §9.3.",
        body: "        let i = thread::index_1d();\n\
               \x20       if i.get() % 2 == 0 {\n\
               \x20           thread::sync_threads();\n\
               \x20       }\n\
               \x20       if let Some(e) = out.get_mut(i) { *e = 1; }",
        extra_uses: &[],
        // Deliberately none: the kernel's observable output is identical
        // whether or not the barrier misbehaves, so claiming a reference
        // model here would invent evidence.
        reference: no_reference,
        documented_limitation: None,
    },
];

pub fn find(id: &str) -> Option<&'static Template> {
    TEMPLATES.iter().find(|t| t.id == id)
}

impl Template {
    /// The kernel crate's `lib.rs`: analyzable with cuda-device alone, which
    /// is the shape reconverge's own fixtures use, so it needs no GPU.
    pub fn kernel_source(&self) -> String {
        self.kernel_source_for(Launch::one_block(32))
    }

    /// The analyzed kernel, with the launch declared in source.
    ///
    /// The contract is not cosmetic. Without it reconverge emits RC005
    /// ("calls `index_1d()` without a launch contract"), which is correct and
    /// would be noise in every generated case; declaring it also makes the
    /// launch a property of the program the analyzer sees, not just of the
    /// runner that executes it.
    pub fn kernel_source_for(&self, launch: Launch) -> String {
        let mut uses = vec!["DisjointSlice", "kernel", "launch_contract", "thread"];
        uses.extend_from_slice(self.extra_uses);
        uses.sort_unstable();
        format!(
            "//! Generated by {GENERATOR_VERSION}. Do not edit.\n\
             //!\n\
             //! {}\n\
             \n\
             use cuda_device::{{{}}};\n\
             \n\
             #[kernel]\n\
             #[launch_contract(domain = 1, coordinates = u32, block = ({}, {}, {}))]\n\
             pub fn probe(mut out: DisjointSlice<u32>) {{\n\
             {}\n\
             }}\n",
            self.doc,
            uses.join(", "),
            launch.block.0,
            launch.block.1,
            launch.block.2,
            self.body
        )
    }

    pub fn record(&self, seed: u64, launches: Vec<Launch>) -> GeneratorRecord {
        let primary = launches.first().copied().unwrap_or(Launch::one_block(32));
        let source = self.kernel_source_for(primary);
        let reference = launches.first().copied().and_then(self.reference);
        GeneratorRecord {
            template_id: self.id.to_string(),
            generator_version: GENERATOR_VERSION.to_string(),
            seed,
            oracle: self.oracle,
            oracle_reason: self.oracle_reason.to_string(),
            kernel_sha256: sha256_hex(source.as_bytes()),
            kernel_source: source,
            kernel_name: "probe".to_string(),
            launches,
            reference_model: reference,
            documented_limitation: self.documented_limitation.map(str::to_string),
        }
    }
}

/// Content-derived identity (brief §24), so the same case has the same name
/// on every machine.
pub fn case_id(record: &GeneratorRecord) -> String {
    let mut material = String::new();
    material.push_str(&record.generator_version);
    material.push('\n');
    material.push_str(&record.template_id);
    material.push('\n');
    material.push_str(&record.seed.to_string());
    material.push('\n');
    material.push_str(&record.kernel_sha256);
    material.push('\n');
    for l in &record.launches {
        material.push_str(&format!(
            "{:?}x{:?}+{}\n",
            l.grid, l.block, l.shared_mem_bytes
        ));
    }
    sha256_hex(material.as_bytes())[..16].to_string()
}

/// SHA-256, implemented here so the crate has no dependency for it.
fn sha256_hex(input: &[u8]) -> String {
    let h = Sha256::digest(input);
    h.iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------- sha256 ---
// FIPS 180-4. Vendored rather than depended upon: identity must be stable
// for the life of the corpus, and this is 40 lines.
struct Sha256;

impl Sha256 {
    fn digest(input: &[u8]) -> [u8; 32] {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
            0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
            0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
            0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
            0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
        ];
        let mut h: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c,
            0x1f83d9ab, 0x5be0cd19,
        ];
        let mut message = input.to_vec();
        let bit_len = (input.len() as u64) * 8;
        message.push(0x80);
        while message.len() % 64 != 56 {
            message.push(0);
        }
        message.extend_from_slice(&bit_len.to_be_bytes());

        for chunk in message.chunks_exact(64) {
            let mut w = [0u32; 64];
            for (i, word) in chunk.chunks_exact(4).enumerate() {
                w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
            }
            for i in 16..64 {
                let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
                let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16]
                    .wrapping_add(s0)
                    .wrapping_add(w[i - 7])
                    .wrapping_add(s1);
            }
            let mut v = h;
            for i in 0..64 {
                let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
                let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
                let t1 = v[7]
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K[i])
                    .wrapping_add(w[i]);
                let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
                let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
                let t2 = s0.wrapping_add(maj);
                v = [
                    t1.wrapping_add(t2), v[0], v[1], v[2],
                    v[3].wrapping_add(t1), v[4], v[5], v[6],
                ];
            }
            for (dst, src) in h.iter_mut().zip(v.iter()) {
                *dst = dst.wrapping_add(*src);
            }
        }
        let mut out = [0u8; 32];
        for (i, word) in h.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_the_published_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // Crosses the padding boundary at 56 bytes.
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn case_identity_is_content_derived_and_stable() {
        let t = find("mask_shrunk_convergent").unwrap();
        let a = t.record(7, vec![Launch::one_block(32)]);
        let b = t.record(7, vec![Launch::one_block(32)]);
        assert_eq!(case_id(&a), case_id(&b), "same inputs, same identity");

        let other_seed = t.record(8, vec![Launch::one_block(32)]);
        assert_ne!(case_id(&a), case_id(&other_seed));

        let other_launch = t.record(7, vec![Launch::one_block(64)]);
        assert_ne!(
            case_id(&a),
            case_id(&other_launch),
            "launch configuration is part of the case, per brief §24"
        );
    }

    #[test]
    fn the_mask_pair_differs_only_in_the_mask_literal() {
        let valid = find("mask_full_convergent").unwrap().kernel_source();
        let invalid = find("mask_shrunk_convergent").unwrap().kernel_source();
        assert!(valid.contains("0xffff_ffff"));
        assert!(invalid.contains("0x0000_ffff"));
        let strip = |s: &str| s.replace("0xffff_ffff", "M").replace("0x0000_ffff", "M");
        // The bodies must be identical modulo the mask, so any difference in
        // evidence is attributable to the mask alone.
        let body = |s: String| {
            s.lines()
                .filter(|l| !l.starts_with("//!"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        assert_eq!(body(strip(&valid)), body(strip(&invalid)));
    }

    #[test]
    fn reference_models_disagree_exactly_where_the_masks_do() {
        let launch = Launch::one_block(32);
        let valid = (find("mask_full_convergent").unwrap().reference)(launch).unwrap();
        let invalid = (find("mask_shrunk_convergent").unwrap().reference)(launch).unwrap();
        assert_eq!(valid.expected[&0], 0xffff_ffff);
        assert_eq!(invalid.expected[&0], 0x0000_ffff);
    }

    #[test]
    fn the_divergent_barrier_template_ships_no_reference_model() {
        // Its observable output is identical whether or not the barrier
        // misbehaves; a reference model would be invented evidence.
        let t = find("barrier_divergent_intra_warp").unwrap();
        assert!((t.reference)(Launch::one_block(32)).is_none());
    }

    #[test]
    fn every_template_declares_a_reason_for_its_oracle() {
        for t in TEMPLATES {
            assert!(
                t.oracle_reason.len() > 20,
                "{} needs a real justification, not a label",
                t.id
            );
        }
    }
}
