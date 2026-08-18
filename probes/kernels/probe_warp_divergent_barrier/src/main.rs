//! ORACLE: KNOWN_UNSAFE (RC001), constructed for WHOLE-WARP divergence.
//!
//! The canonical `i % 2 == 0` sample diverges *inside* a warp, and a block
//! barrier counts arrivals per warp -- so every warp still arrives and the
//! launch completes. Here the predicate splits on a warp boundary instead:
//! at block=64, warp 0 reaches the barrier and warp 1 never does.
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

const KERNEL: &str = "probe_warp_divergent_barrier";

#[cuda_module]
mod kernels {
    use super::*;
    #[kernel]
    pub fn probe(mut out: DisjointSlice<u32>) {
        let i = thread::index_1d();
        if i.get() < 32 {
            thread::sync_threads();
        }
        if let Some(e) = out.get_mut(i) { *e = 1; }
    }
}

fn main() {
    let block: u32 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(64);
    let n = block as usize;
    eprintln!("probe: kernel={} block={} grid=1", KERNEL, block);

    let ctx = CudaContext::new(0).expect("context");
    let stream = ctx.default_stream();
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, n).unwrap();
    let module = kernels::load(&ctx).expect("load embedded module");
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: one block of `block` threads; `out` covers every lane's write.
    unsafe { module.probe(&stream, cfg, &mut out) }.expect("launch");
    let host = out.to_host_vec(&stream).unwrap();
    let written = host.iter().filter(|v| **v != 0).count();
    println!("RESULT kernel={} block={} written={}/{}", KERNEL, block, written, n);
}
